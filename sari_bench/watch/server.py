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
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sari_bench.protocol import DEFAULT_COORDINATOR_PORT
from sari_bench.watch import health, notify, scan

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
        min_interval: float = 1.0,
    ) -> None:
        self.bench_root = bench_root
        self.fixed_battery = fixed_battery
        self.discord = discord
        self.min_interval = min_interval
        self._lock = threading.Lock()
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
        attempts = view.get("attempts") or []
        if not self._announced_start and attempts:
            self._announced_start = True
            self.discord.battery_started(view, len(self._pool))

        for attempt in attempts:
            if attempt.get("state") == "finished":
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


def _stamp(path: Path, fields: dict[str, Any]) -> None:
    try:
        payload = scan._read_json(path)
        payload.update(fields)
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    except OSError as error:  # noqa: BLE001
        _log(f"could not stamp {path}: {error!r}")


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

        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/attempt/") and path.endswith("/kill"):
            key = path[len("/api/attempt/"):-len("/kill")]
            result = self.state.kill(key)
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
    args = parser.parse_args(argv)

    _load_api_env()

    discord = notify.Discord(enabled=args.discord)
    if args.discord and not discord.enabled:
        _log(f"--discord given but {notify.WEBHOOK_ENV} is unset; notifications are OFF")

    state = WatchState(
        bench_root=args.bench_root.resolve(),
        fixed_battery=args.run_dir.resolve() if args.run_dir else None,
        discord=discord,
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
