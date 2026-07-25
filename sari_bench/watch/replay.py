"""Renders an attempt's replay clip when it halts, then posts it to Discord.

This is a worker thread rather than a call inside the watcher's diff, for two reasons.

* **The diff runs on an HTTP handler thread.** `WatchState._notify` is reached from `GET /api/state`,
  so an inline ffmpeg pass over a few hundred frames would hang the dashboard - the very thing you
  opened to find out what the fleet is doing - for as long as the encode takes.
* **One worker means one encode at a time.** The watcher usually shares a host with the runner and its
  agent subprocesses. A battery where eight attempts finish together must not answer with eight
  concurrent ffmpeg processes competing with the agents still running.

The queue is bounded and every failure is swallowed: a halt is always announced, with the clip if there
is one and without it if the render did not work out.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from pathlib import Path
from typing import Any

from sari_bench import video
from sari_bench.watch import notify

# Deep enough to absorb a burst of simultaneous finishes, shallow enough that a wedged encode fails
# loudly instead of hoarding attempt dicts forever.
QUEUE_MAXSIZE = 32

_SHUTDOWN = object()


def _log(message: str) -> None:
    print(f"[sari-bench replay] {message}", flush=True)


class ReplayNotifier(threading.Thread):
    """Serialises render-then-post for finished attempts, off the request path."""

    daemon = True

    def __init__(self, discord: notify.Discord, *, enabled: bool = True,
                 max_bytes: int = video.DISCORD_BUDGET_BYTES,
                 width: int = video.UPLOAD_WIDTH, fps: float = video.UPLOAD_FPS) -> None:
        super().__init__(name="replay-notifier")
        self.discord = discord
        self.enabled = bool(enabled and discord.enabled)
        self.max_bytes = max_bytes
        self.width = width
        self.fps = fps
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._claimed: set[str] = set()
        self._claim_lock = threading.Lock()
        self._stop = threading.Event()

    # -- called from the HTTP thread -------------------------------------------------------

    def submit(self, attempt: dict[str, Any]) -> bool:
        """Takes ownership of one attempt's notification. Never blocks, never raises.

        True means this worker will announce the attempt; False means the caller should, because the
        worker is off, the queue is backed up, or the key was already claimed (in which case
        `Discord.attempt_finished` will no-op on its own dedupe and nothing is sent twice).
        """
        if not self.enabled:
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
            self._queue.put_nowait(dict(attempt))
        except queue.Full:
            # Keep the claim. Thirty-two deep is already minutes of backlog, and dropping one clip
            # beats growing the queue without bound while the fleet keeps finishing.
            _log(f"queue full, dropping replay for {key}")
        return True

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

    # -- worker ----------------------------------------------------------------------------

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
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

    def _handle(self, attempt: dict[str, Any]) -> None:
        run_dir = Path(attempt.get("run_dir") or "")
        clip = None
        if run_dir.is_dir():
            clip = video.render_for_upload(run_dir, max_bytes=self.max_bytes,
                                           fps=self.fps, width=self.width)
        else:
            _log(f"no run dir for {attempt.get('key')}, posting without a clip")
        # Unconditional: the notification is the point, the clip is the illustration.
        self.discord.attempt_finished(attempt, video=clip)

    def stop(self) -> None:
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(_SHUTDOWN)
