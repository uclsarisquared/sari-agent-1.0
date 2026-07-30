"""Renders an attempt's replay clip, and optionally posts it to Discord.

This is a worker thread rather than a call inside the watcher's diff, for two reasons.

* **The diff runs on an HTTP handler thread.** `WatchState._notify` is reached from `GET /api/state`,
  so an inline ffmpeg pass over a few hundred frames would hang the dashboard - the very thing you
  opened to find out what the fleet is doing - for as long as the encode takes.
* **One worker means one encode at a time.** The watcher usually shares a host with the runner and its
  agent subprocesses. A battery where eight attempts finish together must not answer with eight
  concurrent ffmpeg processes competing with the agents still running.

The queue is bounded and every failure is swallowed: a halt is always announced, with the clip if there
is one and without it if the render did not work out.

Two kinds of job share the one worker, because both reasons above apply to both:

* **announce** - a halt was detected, render a clip and post it. Rendering is best-effort.
* **render** - an attempt finished (or a reviewer requested an older replay). Nothing is posted; the
  clip lands in the run dir and `GET /api/attempt/<key>/replay.mp4` serves it on the next poll.

Rendering is therefore enabled independently of Discord: the dashboard's verdict flow needs the
encoder with no webhook configured at all.
"""

from __future__ import annotations

import contextlib
import itertools
import queue
import threading
from pathlib import Path
from typing import Any

from sari_bench import video
from sari_bench.watch import notify

# Deep enough to absorb a burst of simultaneous finishes, shallow enough that a wedged encode fails
# loudly instead of hoarding attempt dicts forever.
QUEUE_MAXSIZE = 32
ANNOUNCE_PRIORITY = 0
RENDER_PRIORITY = 10
SHUTDOWN_PRIORITY = -1

_SHUTDOWN = object()

# What `request()` tells the HTTP layer about a clip.
READY = "ready"
RENDERING = "rendering"
UNAVAILABLE = "unavailable"


def _log(message: str) -> None:
    print(f"[sari-bench replay] {message}", flush=True)


class ReplayNotifier(threading.Thread):
    """Serialises clip renders - and the Discord post that some of them carry - off the request path."""

    daemon = True

    def __init__(self, discord: notify.Discord, *, enabled: bool = True,
                 max_bytes: int = video.DISCORD_BUDGET_BYTES,
                 width: int = video.UPLOAD_WIDTH, fps: float = video.UPLOAD_FPS,
                 review_fps: float = video.DEFAULT_FPS,
                 review_max_frames: int | None = video.MAX_REPLAY_FRAMES) -> None:
        super().__init__(name="replay-notifier")
        self.discord = discord
        # Two flags, not one. The dashboard renders clips for review whether or not a webhook is
        # configured; only the posting half depends on Discord.
        self.render_enabled = bool(enabled)
        self.announce_enabled = bool(enabled and discord.enabled)
        self.max_bytes = max_bytes
        self.width = width
        self.fps = fps
        self.review_fps = review_fps
        self.review_max_frames = review_max_frames
        # A single queue retains join/task_done lifecycle semantics while making time-sensitive
        # notification clips jump ahead of archival dashboard renders already waiting in the burst.
        self._queue: queue.PriorityQueue[tuple[int, int, Any]] = queue.PriorityQueue(
            maxsize=QUEUE_MAXSIZE
        )
        self._sequence = itertools.count()
        self._claimed: set[str] = set()
        self._requested: set[str] = set()
        self._failed: set[str] = set()
        self._claim_lock = threading.Lock()
        self._stop = threading.Event()

    # -- called from the HTTP thread -------------------------------------------------------

    def submit(self, attempt: dict[str, Any]) -> bool:
        """Takes ownership of one attempt's notification. Never blocks, never raises.

        True means this worker will announce the attempt; False means the caller should, because the
        worker is off, the queue is backed up, or the key was already claimed (in which case
        `Discord.attempt_finished` will no-op on its own dedupe and nothing is sent twice).
        """
        if attempt.get("end_reason") == "already_successful":
            return True
        if not self.announce_enabled:
            return False
        key = attempt.get("key") or ""
        if not key:
            return False
        with self._claim_lock:
            if key in self._claimed:
                return True
            self._claimed.add(key)
        try:
            # Copy: the view dict is cached on WatchState and re-read by other handlers.
            self._put_nowait(
                {"attempt": dict(attempt), "announce": True},
                priority=ANNOUNCE_PRIORITY,
            )
        except queue.Full:
            # Let the HTTP caller post immediately without a clip. Keeping the claim here used to
            # suppress that fallback and silently lose the finish notification as well as its video.
            with self._claim_lock:
                self._claimed.discard(key)
            _log(f"queue full, dropping replay for {key}")
            return False
        return True

    def enqueue(self, key: str, run_dir: Path) -> str:
        """Queues the full dashboard replay. Returns READY, RENDERING or UNAVAILABLE. Never blocks.

        READY means the full review clip is already on disk.
        """
        if not key or not run_dir.is_dir():
            return UNAVAILABLE
        clip = run_dir / video.REPLAY_NAME
        try:
            if clip.is_file() and video.is_complete_mp4(clip):
                return READY
        except OSError:
            pass
        if not self.render_enabled:
            return UNAVAILABLE
        with self._claim_lock:
            # A render that already failed for this attempt (no frames, no ffmpeg, over budget) is
            # not retried on every 2s poll - it would pin a core for as long as the tab is open.
            if key in self._failed:
                return UNAVAILABLE
            if key in self._requested:
                return RENDERING
            self._requested.add(key)
        try:
            self._put_nowait(
                {"attempt": {"key": key, "run_dir": str(run_dir)}, "announce": False},
                priority=RENDER_PRIORITY,
            )
        except queue.Full:
            with self._claim_lock:
                self._requested.discard(key)
            return RENDERING  # the reviewer's next poll re-queues it
        return RENDERING

    def request(self, key: str, run_dir: Path) -> str:
        """Compatibility/read-path wrapper: requesting a missing older replay also queues it."""
        return self.enqueue(key, run_dir)

    def seed(self, attempts: list[dict[str, Any]]) -> None:
        """Marks the finishes already on disk as handled, without rendering or posting any of them.

        Called on the watcher's first look at a battery. Without it, restarting the watcher mid-run
        would re-announce every attempt that had finished so far - up to a whole battery's worth of
        uploads, all of it stale news.
        """
        keys = {a.get("key") or "" for a in attempts if a.get("state") == "finished"}
        keys.discard("")
        if not keys:
            return
        with self._claim_lock:
            self._claimed.update(keys)
        self.discord.suppress_finished(keys)
        _log(f"seeded {len(keys)} finished attempt(s) as already announced")

    def invalidate(self, key: str) -> None:
        """Lets the next `enqueue` render this key again, without touching its announcement claim.

        For the one case where a clip on disk is known to be wrong rather than absent: a replay a
        reviewer asked for while the run was still going. `forget_attempt` is the wrong tool - it
        also clears the Discord dedupe, which would re-announce a halt already posted.
        """
        with self._claim_lock:
            self._requested.discard(key)
            self._failed.discard(key)

    def forget_attempt(self, key: str) -> None:
        """Drops per-key caches before a replacement execution reuses the logical key."""
        with self._claim_lock:
            self._claimed.discard(key)
            self._requested.discard(key)
            self._failed.discard(key)
        self.discord.forget_attempt(key)

    # -- worker ----------------------------------------------------------------------------

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                _, _, item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if item is _SHUTDOWN:
                    return
                self._handle(item)
            except Exception as error:  # noqa: BLE001 - one bad attempt must not kill the worker
                _log(f"failed on {item!r}: {error!r}")
            finally:
                self._queue.task_done()

    def _handle(self, job: dict[str, Any]) -> None:
        attempt = job.get("attempt") or {}
        announce = bool(job.get("announce"))
        key = str(attempt.get("key") or "")
        run_dir = Path(attempt.get("run_dir") or "")

        clip = None
        if run_dir.is_dir():
            if announce:
                clip = video.render_for_upload(run_dir, max_bytes=self.max_bytes,
                                               fps=self.fps, width=self.width)
            else:
                clip = video.render(
                    run_dir,
                    run_dir / video.REPLAY_NAME,
                    fps=self.review_fps,
                    width=max(self.width, 1280),
                    max_frames=self.review_max_frames,
                )
        elif announce:
            _log(f"no run dir for {key}, posting without a clip")

        if not announce:
            # A review render. Remember a terminal failure so `request()` can answer UNAVAILABLE
            # instead of re-queueing the same doomed encode every time the page polls.
            with self._claim_lock:
                self._requested.discard(key)
                if clip is None:
                    self._failed.add(key)
            if clip is None:
                _log(f"no clip could be rendered for {key}")
            return

        # Unconditional: the notification is the point, the clip is the illustration.
        self.discord.attempt_finished(attempt, video=clip)

    def stop(self) -> None:
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self._put_nowait(_SHUTDOWN, priority=SHUTDOWN_PRIORITY)

    def _put_nowait(self, item: Any, *, priority: int) -> None:
        self._queue.put_nowait((priority, next(self._sequence), item))
