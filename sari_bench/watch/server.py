"""The live benchmark dashboard: `python -m sari_bench watch`.

Runs beside the RUNNER, not the coordinator. Screenshots and step logs are written by agent
subprocesses the runner spawns, so they are on the runner's local disk; the coordinator is allowed
to be a third machine entirely.

It reads the filesystem, and optionally opens ONE read-only connection to the coordinator's /bench
route to show the sandbox pool. That connection only ever sends `bench.status`, never
`bench.acquire`, so it cannot take a lease away from a worker.

Stdlib only - ThreadingHTTPServer plus one HTML file, polled. This is a handful of tiles refreshing
every couple of seconds; it does not warrant a web framework, and pyproject.toml is deliberately
kept to what is actually imported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sari_bench import video
from sari_bench.protocol import DEFAULT_COORDINATOR_PORT
from sari_bench.watch import health, notify, replay as replay_mod, scan

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BENCH_ROOT = REPO_ROOT / "bench_runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

POOL_REFRESH_SECONDS = 5.0
LOG_TAIL_LINES = 25


def _log(message: str) -> None:
    print(f"[sari-bench watch] {message}", flush=True)


class WatchState:
    """Everything the HTTP handlers read. Rebuilt on demand, at most once per `min_interval`."""

    def __init__(
        self,
        *,
        bench_root: Path,
        fixed_battery: Path | None,
        discord: notify.Discord,
        replay: replay_mod.ReplayNotifier | None = None,
        backfill: bool = False,
        min_interval: float = 1.0,
    ) -> None:
        self.bench_root = bench_root
        self.fixed_battery = fixed_battery
        self.discord = discord
        self.replay = replay
        self.backfill = backfill
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._notify_lock = threading.Lock()
        self._seeded = False
        self._cached: dict[str, Any] = {}
        self._cached_at = 0.0
        self._pool: list[dict[str, Any]] = []
        self._pool_error = ""
        self._announced_start = False
        self._announced_done = False
        self.battery: Path | None = None

    def resolve_battery(self) -> Path | None:
        """Auto-discovery with a single-battery override.

        `--run-dir` pins one battery; otherwise the newest under `bench_root` wins and the watcher
        follows the fleet from battery to battery without a restart.
        """
        if self.fixed_battery is not None:
            return self.fixed_battery
        batteries = scan.find_batteries(self.bench_root)
        return batteries[0] if batteries else None

    def snapshot(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            if not force and self._cached and now - self._cached_at < self.min_interval:
                return self._cached

            battery = self.resolve_battery()
            if battery is None:
                self._cached = {
                    "battery_id": None,
                    "error": f"no battery found under {self.bench_root}",
                    "attempts": [],
                    "counts": {},
                    "pool": self._pool,
                    "pool_error": self._pool_error,
                    "discovered": [],
                    "now": now,
                }
                self._cached_at = now
                return self._cached

            if battery != self.battery:
                _log(f"watching battery {battery}")
                self.battery = battery
                self._announced_start = False
                self._announced_done = False
                self._seeded = False

            discovered = [] if self.fixed_battery else scan.find_batteries(self.bench_root)
            view = scan.scan_battery(battery, now, discovered=discovered).as_dict()
            view["pool"] = self._pool
            view["pool_error"] = self._pool_error
            view["now"] = now
            view["bench_root"] = str(self.bench_root)
            view["mode"] = "pinned" if self.fixed_battery else "auto"
            self._cached = view
            self._cached_at = now

        self._notify(view)
        return view

    def _notify(self, view: dict[str, Any]) -> None:
        """Diffs the fresh snapshot against what has already been announced. Outside the lock: a
        slow webhook must not block the HTTP handlers."""
        if not self.discord.enabled:
            return
        # One notifier at a time. Two concurrent /api/state polls could otherwise both clear the
        # already-announced checks and double-post; harmless when a post was a line of text, wasteful
        # now that it is an upload. Non-blocking, because a skipped pass costs nothing - this is a pure
        # diff and the next poll re-derives it.
        if not self._notify_lock.acquire(blocking=False):
            return
        try:
            attempts = view.get("attempts") or []
            if not self._announced_start and attempts:
                self._announced_start = True
                self.discord.battery_started(view, len(self._pool))

            if not self._seeded:
                self._seeded = True
                if self.replay is not None and not self.backfill:
                    self.replay.seed(attempts)

            for attempt in attempts:
                if attempt.get("state") == "finished":
                    # The worker renders the replay and posts; it only declines when it is off or
                    # backed up, in which case announce the halt here without a clip.
                    if self.replay is None or not self.replay.submit(attempt):
                        self.discord.attempt_finished(attempt)
                elif (attempt.get("health") or {}).get("level") == health.LEVEL_ALERT:
                    frame = attempt.get("frame")
                    path = (self.battery / frame) if (self.battery and frame) else None
                    self.discord.collapse(attempt, path)

            planned = (view.get("battery") or {}).get("planned_attempts")
            finished = sum(1 for a in attempts if a.get("state") in {"finished", "requeued"})
            live = sum(1 for a in attempts if a.get("state") in {"starting", "running"})
            if planned and finished >= planned and not live and not self._announced_done:
                self._announced_done = True
                self.discord.battery_finished(view)
        finally:
            self._notify_lock.release()

    def set_pool(self, pool: list[dict[str, Any]], error: str = "") -> None:
        with self._lock:
            self._pool = pool
            self._pool_error = error

    # -- actions ---------------------------------------------------------------------------

    def kill(self, key: str) -> dict[str, Any]:
        """Stops one attempt, and then gets out of the way.

        The watcher signals the agent's process group and does nothing else: the runner's own
        `process.wait()` returns non-zero, it records the attempt, and its `finally` releases the
        lease so the coordinator resets and re-pools the sandbox. No second path to maintain, and
        the watcher never has to talk to the coordinator about it. The `killed_by` stamp is what
        stops the row being scored as an agent crash.
        """
        battery = self.resolve_battery()
        if battery is None:
            return {"ok": False, "error": "no battery"}
        run_dir = _safe_run_dir(battery, key)
        if run_dir is None:
            return {"ok": False, "error": "unknown attempt"}

        manifest_path = run_dir / scan.ATTEMPT_MANIFEST
        manifest = scan._read_json(manifest_path)
        pid = manifest.get("pid")
        if manifest.get("state") == "finished":
            return {"ok": False, "error": "attempt already finished"}
        if not pid:
            return {"ok": False, "error": "no pid recorded"}

        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError) as error:
            return {"ok": False, "error": f"{error!r}"}

        _stamp(manifest_path, {"killed_by": "watcher",
                               "killed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        _log(f"killed {key} (pid {pid}); the runner will release its lease")
        return {"ok": True, "pid": pid}

    def verdict(self, key: str, success: bool, *, note: str = "", by: str = "") -> dict[str, Any]:
        """Records a human's pass/fail for one halted attempt.

        The predicates that end a run with `halt_granted` are the ones that grant on state they
        cannot ground - several of them say `[unverified]` in their own reason string. This is the
        check on them, so the verdict is stamped BESIDE `success`, never over it: a measured pass with
        a verified fail is exactly the discrepancy worth collecting, and overwriting `success` would
        erase it.

        The eligibility check is repeated here rather than trusted from the UI: the button is only
        rendered on a verifiable card, but the route is reachable without the page.
        """
        battery = self.resolve_battery()
        if battery is None:
            return {"ok": False, "error": "no battery"}
        run_dir = _safe_run_dir(battery, key)
        if run_dir is None:
            return {"ok": False, "error": "unknown attempt"}

        manifest_path = run_dir / scan.ATTEMPT_MANIFEST
        with self._lock:
            manifest = scan._read_json(manifest_path)
            state = str(manifest.get("state") or "")
            end_reason = str(manifest.get("end_reason") or "")
            if not scan.is_verifiable(state, end_reason):
                return {"ok": False,
                        "error": f"not reviewable (state={state or '?'}, end={end_reason or '?'}); "
                                 f"only {'/'.join(sorted(scan.VERIFIABLE_END_REASONS))} can be judged"}
            fields = {
                "verified_success": bool(success),
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "verified_by": by or os.environ.get("USER") or "watcher",
                "verified_note": note,
            }
            _stamp(manifest_path, fields)
            # The cached snapshot predates the stamp, so the next poll would show the old badge for
            # up to `min_interval`. Drop it and let the reviewer see their own click land.
            self._cached_at = 0.0

        _log(f"verdict on {key}: {'SUCCESS' if success else 'FAIL'} by {fields['verified_by']}"
             f"{' (predicate disagreed)' if bool(manifest.get('success')) != bool(success) else ''}")
        return {"ok": True, **fields}

    def clear_verdict(self, key: str) -> dict[str, Any]:
        """Un-reviews an attempt, for a misclick. Leaves no trace, so the row reads as never looked at
        rather than as a verdict of False."""
        battery = self.resolve_battery()
        if battery is None:
            return {"ok": False, "error": "no battery"}
        run_dir = _safe_run_dir(battery, key)
        if run_dir is None:
            return {"ok": False, "error": "unknown attempt"}

        manifest_path = run_dir / scan.ATTEMPT_MANIFEST
        with self._lock:
            manifest = scan._read_json(manifest_path)
            if "verified_success" not in manifest:
                return {"ok": True, "cleared": False}
            for field in ("verified_success", "verified_at", "verified_by", "verified_note"):
                manifest.pop(field, None)
            _write_json(manifest_path, manifest)
            self._cached_at = 0.0
        _log(f"verdict cleared on {key}")
        return {"ok": True, "cleared": True}

    def replay_status(self, key: str) -> tuple[str, Path | None]:
        """(status, clip path). Renders on demand, on the replay worker - never on this thread."""
        if self.replay is None:
            return replay_mod.UNAVAILABLE, None
        battery = self.resolve_battery()
        if battery is None:
            return replay_mod.UNAVAILABLE, None
        run_dir = _safe_run_dir(battery, key)
        if run_dir is None:
            return replay_mod.UNAVAILABLE, None
        status = self.replay.request(key, run_dir)
        clip = run_dir / video.UPLOAD_NAME
        return status, (clip if status == replay_mod.READY and clip.is_file() else None)

    def log_tail(self, key: str, lines: int = LOG_TAIL_LINES) -> dict[str, Any]:
        battery = self.resolve_battery()
        if battery is None:
            return {"lines": []}
        run_dir = _safe_run_dir(battery, key)
        if run_dir is None:
            return {"lines": []}
        path = run_dir / "agent.log"
        if not path.exists():
            return {"lines": []}
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 16_384))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError as error:
            return {"lines": [f"<unreadable: {error!r}>"]}
        return {"lines": text.splitlines()[-lines:]}

    def frame_path(self, key: str) -> Path | None:
        battery = self.resolve_battery()
        if battery is None:
            return None
        run_dir = _safe_run_dir(battery, key)
        if run_dir is None:
            return None
        return scan._latest_frame(run_dir)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic replace, so a poller mid-read never sees a half-written manifest."""
    try:
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    except OSError as error:  # noqa: BLE001
        _log(f"could not write {path}: {error!r}")


def _stamp(path: Path, fields: dict[str, Any]) -> None:
    payload = scan._read_json(path)
    payload.update(fields)
    _write_json(path, payload)


def _safe_run_dir(battery: Path, key: str) -> Path | None:
    """Resolves an attempt key to a run dir, refusing anything that escapes the battery dir.

    The key arrives from an HTTP request, so `../../` must not reach the filesystem.
    """
    candidate = (battery / key).resolve()
    try:
        candidate.relative_to(battery.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_dir() else None


class _PoolPoller(threading.Thread):
    """Keeps the sandbox pool fresh on its own event loop.

    Read-only: it sends `bench.status` and nothing else, so it never competes with a worker for a
    sandbox. A coordinator that is down is reported in the UI, not raised.
    """

    daemon = True

    def __init__(self, state: WatchState, coordinator_url: str) -> None:
        super().__init__(name="pool-poller")
        self.state = state
        self.coordinator_url = coordinator_url
        self._stop = threading.Event()

    def run(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        from sari_bench.client import CoordinatorClient

        while not self._stop.is_set():
            try:
                async with CoordinatorClient(self.coordinator_url) as client:
                    self.state.set_pool(await client.pool())
            except Exception as error:  # noqa: BLE001 - the dashboard survives a dead coordinator
                self.state.set_pool([], f"{error!r}")
            await asyncio.sleep(POOL_REFRESH_SECONDS)

    def stop(self) -> None:
        self._stop.set()


class Handler(BaseHTTPRequestHandler):
    state: WatchState = None  # type: ignore[assignment]

    def log_message(self, *_args: Any) -> None:  # noqa: D102 - silence per-request stderr spam
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode("utf-8"), "application/json")

    def _send_file(self, path: Path, content_type: str) -> None:
        """Serves a file, honouring a single byte range.

        `<video>` needs this. Without `Accept-Ranges` a browser treats the clip as unseekable and the
        reviewer can only watch it start to finish - which defeats the point of attaching it to a
        verdict. The clips are capped at a few megabytes, so the whole file is still read at once and
        only the slice and the headers differ.
        """
        try:
            body = path.read_bytes()
        except OSError as error:
            self._send(404, f"unreadable: {error!r}".encode("utf-8"), "text/plain")
            return

        total = len(body)
        start, end = 0, total - 1
        partial = False
        header = self.headers.get("Range", "")
        if header.startswith("bytes=") and "," not in header and total:
            first, _, last = header[len("bytes="):].partition("-")
            try:
                if first:
                    start = int(first)
                    end = int(last) if last else total - 1
                elif last:  # a suffix range: the LAST n bytes
                    start = max(0, total - int(last))
                partial = True
            except ValueError:
                partial = False
        end = min(end, total - 1)

        if partial and (start > end or start >= total):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        chunk = body[start:end + 1] if partial else body
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        # Not `no-store` like the JSON routes: a browser that may not keep the clip re-fetches on
        # every seek, and some refuse to seek at all. A rendered clip only changes if it is rendered
        # again, so a short window is safe and makes scrubbing usable.
        self.send_header("Cache-Control", "private, max-age=60")
        self.end_headers()
        self.wfile.write(chunk)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in {"/", "/index.html"}:
            page = STATIC_DIR / "dashboard.html"
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/state":
            self._json(self.state.snapshot())
            return

        if path.startswith("/api/attempt/"):
            rest = path[len("/api/attempt/"):]
            key, _, action = rest.rpartition("/")
            if action == "frame.png":
                frame = self.state.frame_path(key)
                if frame is None or not frame.exists():
                    self._send(404, b"no frame yet", "text/plain")
                    return
                self._send(200, frame.read_bytes(), "image/png")
                return
            if action == "log":
                self._json(self.state.log_tail(key))
                return
            if action == "replay.mp4":
                status, clip = self.state.replay_status(key)
                if clip is not None:
                    self._send_file(clip, "video/mp4")
                elif status == replay_mod.RENDERING:
                    # 202: the encode is queued on the replay worker. The page polls until 200.
                    self._json({"status": status}, code=202)
                else:
                    self._json({"status": replay_mod.UNAVAILABLE,
                                "reason": "no frames, no ffmpeg, or the clip busts the size budget"},
                               code=409)
                return

        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/attempt/"):
            rest = path[len("/api/attempt/"):]
            if rest.endswith("/verdict/clear"):
                result = self.state.clear_verdict(rest[:-len("/verdict/clear")])
                self._json(result, code=200 if result.get("ok") else 400)
                return
            if rest.endswith("/verdict"):
                body = self._body()
                if not isinstance(body.get("success"), bool):
                    self._json({"ok": False, "error": "body needs a boolean 'success'"}, code=400)
                    return
                result = self.state.verdict(
                    rest[:-len("/verdict")],
                    body["success"],
                    note=str(body.get("note") or ""),
                    by=str(body.get("by") or ""),
                )
                self._json(result, code=200 if result.get("ok") else 400)
                return
            if rest.endswith("/kill"):
                result = self.state.kill(rest[:-len("/kill")])
                self._json(result, code=200 if result.get("ok") else 400)
                return
        self._send(404, b"not found", "text/plain")


def serve(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sari_bench watch", description="Live dashboard for a running prompt battery."
    )
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT,
                        help="Directory holding battery dirs. The newest is followed automatically.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Pin one battery dir instead of auto-discovering the newest.")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. 0.0.0.0 exposes the dashboard - including its kill "
                             "endpoint - to the network.")
    parser.add_argument("--coordinator", default=f"ws://localhost:{DEFAULT_COORDINATOR_PORT}",
                        help="Coordinator to read the sandbox pool from. --no-pool to skip.")
    parser.add_argument("--no-pool", action="store_true", help="Do not connect to the coordinator.")
    parser.add_argument("--discord", action="store_true",
                        help=f"Send Discord notifications (needs {notify.WEBHOOK_ENV}).")
    parser.add_argument("--no-replay", action="store_true",
                        help="Announce halts without rendering and attaching a replay clip.")
    parser.add_argument("--replay-fps", type=float, default=video.UPLOAD_FPS)
    parser.add_argument("--replay-width", type=int, default=video.UPLOAD_WIDTH)
    parser.add_argument("--replay-max-mb", type=float,
                        default=video.DISCORD_BUDGET_BYTES / 1e6,
                        help="Size budget for an attached clip.")
    parser.add_argument("--replay-backfill", action="store_true",
                        help="Also announce attempts that had already finished when the watcher "
                             "started. Off by default: a restart mid-battery would otherwise replay "
                             "the whole run into the channel.")
    args = parser.parse_args(argv)

    _load_api_env()

    discord = notify.Discord(enabled=args.discord)
    if args.discord and not discord.enabled:
        _log(f"--discord given but {notify.WEBHOOK_ENV} is unset; notifications are OFF")

    replay = replay_mod.ReplayNotifier(
        discord,
        enabled=not args.no_replay,
        max_bytes=int(args.replay_max_mb * 1e6),
        width=args.replay_width,
        fps=args.replay_fps,
    )
    # Started whenever rendering is on, not only when Discord is: the dashboard's review flow asks
    # this same worker for clips, and it must be running with no webhook configured.
    if replay.render_enabled:
        if shutil.which("ffmpeg") is None:
            _log("replay clips are ON but ffmpeg is not on PATH; halts will post without one "
                 "and the dashboard cannot show replays")
        replay.start()

    state = WatchState(
        bench_root=args.bench_root.resolve(),
        fixed_battery=args.run_dir.resolve() if args.run_dir else None,
        discord=discord,
        replay=replay,
        backfill=args.replay_backfill,
    )

    battery = state.resolve_battery()
    if args.run_dir:
        _log(f"PINNED to battery dir {args.run_dir} (single-battery mode)")
        if not args.run_dir.exists():
            _log("  ...which does not exist yet; it will appear once the runner creates it")
    elif battery is not None:
        _log(f"auto-discovery under {state.bench_root}: watching {battery.name} (newest of "
             f"{len(scan.find_batteries(state.bench_root))})")
    else:
        _log(f"auto-discovery under {state.bench_root}: no battery dirs yet, waiting for one")

    poller = None
    if not args.no_pool:
        poller = _PoolPoller(state, args.coordinator)
        poller.start()

    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"dashboard on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("stopping")
    finally:
        if poller is not None:
            poller.stop()
        replay.stop()
        server.server_close()
    return 0


def _load_api_env() -> None:
    """Loads api.env from the repo root, the same file every other module reads."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = REPO_ROOT / "api.env"
    if env_path.exists():
        load_dotenv(env_path)


def main(argv: list[str] | None = None) -> int:
    return serve(argv)


if __name__ == "__main__":
    raise SystemExit(main())
