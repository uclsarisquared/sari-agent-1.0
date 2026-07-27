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
import contextlib
import json
import os
import shutil
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from sari_bench import video
from sari_bench.protocol import DEFAULT_COORDINATOR_PORT
from sari_bench.storage import edit_json_locked
from sari_bench.watch import health, notify, replay as replay_mod, scan

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BENCH_ROOT = REPO_ROOT / "bench_runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

POOL_REFRESH_SECONDS = 5.0
LOG_TAIL_LINES = 25
LOG_MAX_LINES = 2000
LOG_BOOTSTRAP_BYTES = 16_384
ALREADY_SUCCESSFUL = "already_successful"


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
        coordinator_url: str | None = None,
        retry_agent_entry: str | None = None,
        retry_agent_cwd: Path | None = None,
    ) -> None:
        self.bench_root = bench_root
        self.fixed_battery = fixed_battery
        self.discord = discord
        self.replay = replay
        self.backfill = backfill
        self.min_interval = min_interval
        self.coordinator_url = coordinator_url
        self.retry_agent_entry = retry_agent_entry
        self.retry_agent_cwd = retry_agent_cwd
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
        self._retry_jobs: dict[str, dict[str, Any]] = {}

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
            self._merge_retry_jobs(view)
            self._cached = view
            self._cached_at = now

        self._notify(view)
        return view

    def _merge_retry_jobs(self, view: dict[str, Any]) -> None:
        """Keeps a logical try visible while its old directory is gone and replacement is queued."""
        attempts = view.get("attempts")
        if not isinstance(attempts, list):
            attempts = []
            view["attempts"] = attempts
        by_key = {str(attempt.get("key") or ""): attempt for attempt in attempts}
        for key, job in self._retry_jobs.items():
            current = by_key.get(key)
            logical = job["attempt"]
            related = [
                attempt for attempt in attempts
                if attempt.get("prompt_id") == logical.get("prompt_id")
                and attempt.get("attempt") == logical.get("attempt")
            ]
            for attempt in related:
                attempt["retry_state"] = job["state"]
                attempt["retry_error"] = job.get("error", "")
            if current is not None:
                continue
            placeholder = dict(job["attempt"])
            placeholder.update({
                "key": key,
                "run_id": job["run_id"],
                "state": "retrying",
                "retry_state": job["state"],
                "retry_error": job.get("error", ""),
                "alive": False,
                "verifiable": False,
                "verified": False,
                "frame": "",
                "log_bytes": 0,
            })
            attempts.append(placeholder)

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
                    if attempt.get("end_reason") == ALREADY_SUCCESSFUL:
                        # Administrative sibling cancellation is bookkeeping, not an attempt halt:
                        # do not spend encoder capacity or post one Discord message per skipped try.
                        self.discord.suppress_finished([attempt.get("key") or ""])
                        continue
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

    def retry(self, key: str) -> dict[str, Any]:
        """Schedules a destructive replacement of one logical prompt/try."""
        battery = self.resolve_battery()
        if battery is None:
            return {"ok": False, "error": "no battery"}
        selected = _safe_run_dir(battery, key)
        previous_error: dict[str, Any] | None = None
        if selected is None:
            with self._lock:
                candidate = self._retry_jobs.get(key)
                if candidate and candidate.get("state") == "error":
                    previous_error = dict(candidate)
            if previous_error is None:
                return {"ok": False, "error": "unknown attempt"}
            manifest = dict(previous_error["source_manifest"])
            prompt_id = str(manifest.get("prompt_id") or "")
            attempt = int(manifest.get("attempt") or 0)
            canonical_key = f"{prompt_id}/try{attempt:02d}"
            if key != canonical_key:
                return {"ok": False, "error": "invalid retry key"}
            attempt_view = dict(previous_error["attempt"])
        else:
            manifest = scan._read_json(selected / scan.ATTEMPT_MANIFEST)
            prompt_id = str(manifest.get("prompt_id") or selected.parent.name)
            try:
                attempt = int(manifest.get("attempt") or selected.name[3:].split(".", 1)[0])
            except (TypeError, ValueError):
                return {"ok": False, "error": "attempt has no valid try number"}
            if attempt < 1 or prompt_id != selected.parent.name:
                return {"ok": False, "error": "invalid attempt metadata"}
            canonical_key = f"{prompt_id}/try{attempt:02d}"
            attempt_view = scan.scan_attempt(selected, battery, time.time()).as_dict()

        with self._lock:
            existing = self._retry_jobs.get(canonical_key)
            if existing and existing.get("state") != "error":
                return {"ok": False, "error": "retry already in progress", "key": canonical_key}
            job = {
                "key": canonical_key,
                "run_id": f"retry-{uuid.uuid4().hex}",
                "state": "stopping",
                "error": "",
                "attempt": attempt_view,
                "source_manifest": manifest,
            }
            self._retry_jobs[canonical_key] = job
            self._cached_at = 0.0
            self._announced_done = False
            if self.replay is not None:
                self.replay.forget_attempt(canonical_key)
            else:
                self.discord.forget_attempt(canonical_key)

        threading.Thread(
            target=self._retry_worker,
            args=(battery, prompt_id, attempt, manifest, job["run_id"]),
            name=f"retry-{prompt_id}-{attempt}",
            daemon=True,
        ).start()
        _log(f"retry requested for {canonical_key}; prior history will be deleted")
        return {"ok": True, "key": canonical_key, "retry_state": "stopping"}

    def _set_retry_state(self, key: str, state: str, error: str = "") -> None:
        with self._lock:
            job = self._retry_jobs.get(key)
            if job is None:
                return
            job["state"] = state
            job["error"] = error
            self._cached_at = 0.0

    def _retry_worker(
        self,
        battery: Path,
        prompt_id: str,
        attempt: int,
        source_manifest: dict[str, Any],
        retry_run_id: str,
    ) -> None:
        key = f"{prompt_id}/try{attempt:02d}"
        try:
            config = self._retry_config(battery, source_manifest)
            self._stop_logical_try(battery, prompt_id, attempt)
            self._set_retry_state(key, "cleaning")
            self._clear_prompt_winner(battery, prompt_id)
            self._delete_logical_try(battery, prompt_id, attempt)

            from sari_bench.runner import (
                BenchmarkRunner,
                ORCHESTRATOR_ENTRY,
                OVERHAUL_DIR,
                Prompt,
                purge_attempt_records,
            )

            purge_attempt_records(battery, prompt_id, attempt)
            with contextlib.suppress(OSError):
                (battery / "summary.json").unlink()

            self._set_retry_state(key, "queued")
            runner = BenchmarkRunner(
                prompts=[Prompt(
                    id=prompt_id,
                    prompt=str(source_manifest.get("prompt") or ""),
                    family=str(source_manifest.get("family") or ""),
                    looking_for=str(source_manifest.get("looking_for") or ""),
                )],
                coordinator_url=config["coordinator"],
                output_dir=battery,
                tries=max(attempt, 1),
                time_limit_minutes=config["time_limit_minutes"],
                per_leg_minutes=config["per_leg_minutes"],
                concurrency=1,
                max_steps=config["max_steps"],
                arm=config["arm"],
                map_dir=config["map_dir"],
                leg_retries=config["leg_retries"],
                timeout_grace=config["timeout_grace"],
                sandbox_startup_timeout=config["sandbox_startup_timeout"],
                capture_interval=config["capture_interval"],
                python_executable=sys.executable,
                agent_entry=self.retry_agent_entry or ORCHESTRATOR_ENTRY,
                agent_cwd=self.retry_agent_cwd or OVERHAUL_DIR,
                work_items=[(prompt_id, attempt)],
                initialize_battery=False,
            )
            self._set_retry_state(key, "running")
            asyncio.run(runner.run())
        except Exception as error:  # noqa: BLE001 - surfaced on the tile; watcher stays alive
            _log(f"retry failed for {key}: {error!r}")
            self._set_retry_state(key, "error", repr(error))
            return

        with self._lock:
            current = self._retry_jobs.get(key)
            if current and current.get("run_id") == retry_run_id:
                self._retry_jobs.pop(key, None)
            self._cached_at = 0.0
        _log(f"retry completed for {key}")

    def _retry_config(
        self, battery: Path, source_manifest: dict[str, Any]
    ) -> dict[str, Any]:
        plan = scan._read_json(battery / scan.BATTERY_MANIFEST)
        command = source_manifest.get("command")
        command = command if isinstance(command, list) else []

        def option(name: str, default: Any) -> Any:
            try:
                return command[command.index(name) + 1]
            except (ValueError, IndexError):
                return default

        coordinator = str(plan.get("coordinator") or self.coordinator_url or "")
        if not coordinator:
            raise RuntimeError("battery does not record its coordinator")
        return {
            "coordinator": coordinator,
            "time_limit_minutes": float(
                source_manifest.get("time_limit_minutes")
                or plan.get("time_limit_minutes")
                or 40.0
            ),
            "per_leg_minutes": float(
                source_manifest.get("per_leg_minutes")
                or plan.get("per_leg_minutes")
                or option("--max-minutes", 40.0)
            ),
            "max_steps": int(
                source_manifest.get("max_steps")
                or plan.get("max_steps")
                or option("--max-steps", 150)
            ),
            "arm": str(source_manifest.get("arm") or plan.get("arm") or "graph"),
            "map_dir": plan.get("map_dir") or option("--output-dir", None),
            "leg_retries": int(plan.get("leg_retries") or option("--leg-retries", 1)),
            "timeout_grace": float(plan.get("timeout_grace_seconds") or 120.0),
            "sandbox_startup_timeout": max(
                1.0, float(plan.get("sandbox_startup_timeout_seconds") or 30.0)
            ),
            "capture_interval": float(
                source_manifest.get("capture_interval_seconds")
                if source_manifest.get("capture_interval_seconds") is not None
                else plan.get("capture_interval_seconds", 2.0)
            ),
        }

    def _stop_logical_try(self, battery: Path, prompt_id: str, attempt: int) -> None:
        canonical = battery / prompt_id / f"try{attempt:02d}"
        manifest_path = canonical / scan.ATTEMPT_MANIFEST
        manifest = scan._read_json(manifest_path)
        pid = manifest.get("pid")
        was_live = manifest.get("state") in {"starting", "running"}
        was_orphaned = bool(was_live and pid and not scan._pid_alive(pid))
        if was_live:
            _stamp(manifest_path, {
                "retry_requested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "killed_by": "watcher_retry",
            })
        if was_orphaned:
            return

        deadline = time.monotonic() + 45.0
        signalled = False
        while was_live and time.monotonic() < deadline:
            manifest = scan._read_json(manifest_path)
            pid = manifest.get("pid")
            if pid and scan._pid_alive(pid):
                if not signalled:
                    try:
                        os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    signalled = True
                time.sleep(0.1)
                continue
            if manifest.get("finalized_at") or manifest.get("state") == "finished":
                return
            time.sleep(0.1)
        if was_live and pid and scan._pid_alive(pid):
            raise RuntimeError("agent did not stop within 45 seconds")

    def _clear_prompt_winner(self, battery: Path, prompt_id: str) -> None:
        battery_path = battery / scan.BATTERY_MANIFEST
        winner_key = ""
        with edit_json_locked(battery_path) as plan:
            winners = plan.get("human_verified_winners")
            if isinstance(winners, dict):
                winner = winners.pop(prompt_id, None)
                if isinstance(winner, dict):
                    winner_key = str(winner.get("winning_attempt_key") or "")
                plan["human_verified_winners"] = winners
        if winner_key:
            winner_dir = _safe_run_dir(battery, winner_key)
            if winner_dir is not None:
                manifest = scan._read_json(winner_dir / scan.ATTEMPT_MANIFEST)
                for field in ("verified_success", "verified_at", "verified_by", "verified_note"):
                    manifest.pop(field, None)
                _write_json(winner_dir / scan.ATTEMPT_MANIFEST, manifest)
        # Older batteries may have only the per-attempt verdict and no battery-level winner map.
        prompt_dir = battery / prompt_id
        if prompt_dir.is_dir():
            for run_dir in prompt_dir.iterdir():
                if not run_dir.is_dir() or ".requeue" in run_dir.name:
                    continue
                manifest_path = run_dir / scan.ATTEMPT_MANIFEST
                manifest = scan._read_json(manifest_path)
                if manifest.get("verified_success") is not True:
                    continue
                for field in ("verified_success", "verified_at", "verified_by", "verified_note"):
                    manifest.pop(field, None)
                _write_json(manifest_path, manifest)

    @staticmethod
    def _delete_logical_try(battery: Path, prompt_id: str, attempt: int) -> None:
        prompt_dir = battery / prompt_id
        if not prompt_dir.is_dir():
            return
        base = f"try{attempt:02d}"
        for child in list(prompt_dir.iterdir()):
            suffix = child.name[len(base + ".requeue"):] if child.name.startswith(base + ".requeue") else ""
            if child.name == base or (suffix.isdigit() and suffix):
                if child.is_dir():
                    shutil.rmtree(child)

    def verdict(self, key: str, success: bool, *, note: str = "", by: str = "") -> dict[str, Any]:
        """Records a human's pass/fail for one finished attempt.

        The verdict is stamped BESIDE `success`, never over it: a measured pass with a verified fail
        is exactly the discrepancy worth collecting, and overwriting `success` would erase it.

        The eligibility check is repeated here rather than trusted from the UI: the button is only
        rendered on a finished card, but the route is reachable without the page.
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
                return {
                    "ok": False,
                    "error": f"not reviewable (state={state or '?'}); only finished attempts can be judged",
                }
            fields = {
                "verified_success": bool(success),
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "verified_by": by or os.environ.get("USER") or "watcher",
                "verified_note": note,
            }
            _stamp(manifest_path, fields)
            cancellations = {"stopped": 0, "skipped": 0}
            if success:
                cancellations = self._cancel_successful_siblings(
                    battery, key, manifest, fields
                )
            # The cached snapshot predates the stamp, so the next poll would show the old badge for
            # up to `min_interval`. Drop it and let the reviewer see their own click land.
            self._cached_at = 0.0

        _log(f"verdict on {key}: {'SUCCESS' if success else 'FAIL'} by {fields['verified_by']}"
             f"{' (predicate disagreed)' if bool(manifest.get('success')) != bool(success) else ''}")
        return {
            "ok": True,
            **fields,
            "siblings_stopped": cancellations["stopped"],
            "siblings_skipped": cancellations["skipped"],
            "sibling_cancellations": cancellations,
        }

    def _cancel_successful_siblings(
        self,
        battery: Path,
        winner_key: str,
        winner_manifest: dict[str, Any],
        verdict_fields: dict[str, Any],
    ) -> dict[str, int]:
        """Durably cancels only tries of the winner's prompt ID, then signals published PIDs."""
        prompt_id = str(winner_manifest.get("prompt_id") or winner_key.split("/", 1)[0])
        cancellation = {
            "stop_reason": ALREADY_SUCCESSFUL,
            "stop_requested_at": verdict_fields["verified_at"],
            "stop_requested_by": verdict_fields["verified_by"],
            "winning_attempt_key": winner_key,
        }

        # This battery-level entry covers queued and waiting-for-lease work that has no run
        # manifest yet. It is intentionally never removed by clear_verdict(): administrative stops
        # are irreversible and must not be silently requeued later.
        battery_path = battery / scan.BATTERY_MANIFEST
        with edit_json_locked(battery_path) as battery_manifest:
            winners = battery_manifest.get("human_verified_winners")
            if not isinstance(winners, dict):
                winners = {}
            winners[prompt_id] = {
                "winning_attempt_key": winner_key,
                "stop_requested_at": verdict_fields["verified_at"],
                "stop_requested_by": verdict_fields["verified_by"],
            }
            battery_manifest["human_verified_winners"] = winners

        stopped = 0
        known_cancellable = 0
        known_attempts: set[int] = set()
        for run_dir in scan.run_dirs_of(battery):
            if run_dir.parent.name != prompt_id or ".requeue" in run_dir.name:
                continue
            sibling_key = f"{prompt_id}/{run_dir.name}"
            sibling = scan._read_json(run_dir / scan.ATTEMPT_MANIFEST)
            try:
                known_attempts.add(int(sibling.get("attempt") or run_dir.name[3:]))
            except (TypeError, ValueError):
                pass
            if sibling_key == winner_key or sibling.get("state") in {"finished", "requeued"}:
                continue

            known_cancellable += 1
            sibling_path = run_dir / scan.ATTEMPT_MANIFEST
            _stamp(sibling_path, cancellation)
            pid = sibling.get("pid")
            if not pid:
                continue
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                # The runner's post-PID check sees the durable stop request even if publication and
                # process exit raced this signal.
                continue
            _stamp(
                sibling_path,
                {
                    "killed_by": ALREADY_SUCCESSFUL,
                    "killed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            stopped += 1

        tries = int(battery_manifest.get("tries") or 0)
        queued = max(0, tries - len({n for n in known_attempts if n > 0}))
        skipped = max(0, known_cancellable - stopped) + queued
        _log(
            f"{winner_key} verified successful: requested stop for "
            f"{stopped} running and {skipped} unstarted sibling(s)"
        )
        return {"stopped": stopped, "skipped": skipped}

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
        if scan._read_json(run_dir / scan.ATTEMPT_MANIFEST).get("end_reason") == ALREADY_SUCCESSFUL:
            return replay_mod.UNAVAILABLE, None
        status = self.replay.request(key, run_dir)
        clip = run_dir / video.REPLAY_NAME
        return status, (clip if status == replay_mod.READY and clip.is_file() else None)

    def log_tail(
        self,
        key: str,
        lines: int = LOG_TAIL_LINES,
        since: int | None = None,
        full: bool = False,
    ) -> dict[str, Any]:
        """Reads agent.log as a full log, a tail, or the delta since a byte offset.

        The dashboard keeps one terminal per attempt open at all times, so re-sending the same tail
        every two seconds would be both wasteful and impossible to append to without duplicating
        lines. With `since` it gets exactly the bytes written after its cursor, which lets it append
        and so keep the reader's scroll position untouched.

        A trailing line the writer has not terminated yet comes back as `partial` rather than in
        `lines`, and does not advance the cursor: the reader sees it immediately, and it arrives once
        more - whole this time - when its newline lands.
        """
        empty = {"lines": [], "offset": 0, "size": 0, "partial": "", "reset": False}
        battery = self.resolve_battery()
        if battery is None:
            return empty
        run_dir = _safe_run_dir(battery, key)
        if run_dir is None:
            return empty
        path = run_dir / "agent.log"
        if not path.exists():
            return empty
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                # A cursor past the end means the file was truncated or rotated under us; the only
                # honest answer is to start over and tell the client to drop what it has.
                reset = since is not None and since > size
                bootstrap = since is None or reset
                start = (
                    0
                    if bootstrap and full
                    else max(0, size - LOG_BOOTSTRAP_BYTES)
                    if bootstrap
                    else since
                )
                handle.seek(start)
                raw = handle.read()
        except OSError as error:
            return {**empty, "lines": [f"<unreadable: {error!r}>"]}

        if bootstrap and start > 0:
            # The bootstrap window lands mid-line; drop that fragment rather than show half of it.
            head = raw.find(b"\n")
            start += len(raw) if head < 0 else head + 1
            raw = b"" if head < 0 else raw[head + 1:]

        partial = b""
        if raw and not raw.endswith(b"\n"):
            tail = raw.rfind(b"\n")
            partial, raw = (raw, b"") if tail < 0 else (raw[tail + 1:], raw[:tail + 1])

        text = raw.decode("utf-8", errors="replace")
        return {
            "lines": text.splitlines() if bootstrap and full else text.splitlines()[-lines:],
            "offset": start + len(raw),
            "size": size,
            "partial": partial.decode("utf-8", errors="replace"),
            "reset": reset,
        }

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


def _int_param(
    query: dict[str, list[str]], name: str, default: int | None, low: int, high: int | None
) -> int | None:
    """Reads one clamped integer out of a query string, falling back on anything unparseable."""
    values = query.get(name)
    if not values:
        return default
    try:
        value = int(values[0])
    except ValueError:
        return default
    value = max(low, value)
    return value if high is None else min(high, value)


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

    def _write_body(self, body: bytes) -> None:
        """Writes a response body without reporting an ordinary client disconnect."""
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browsers routinely abandon an in-flight video response when seeking or replacing the
            # source. There is no response left to recover and socketserver would otherwise print a
            # misleading request-handler traceback.
            pass

    def _send(
        self, code: int, body: bytes, content_type: str, extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self._write_body(body)

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
        self._write_body(chunk)

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

    def _report_csv(self) -> None:
        """The same attempts.csv `python -m sari_bench report` writes, served as a download.

        `report.collect` is the one implementation of this flattening and it does no output I/O, so
        the button and the CLI cannot drift. It re-scans every run dir, which costs a second or two
        on a large battery - fine for a click, which is why this is not on the 2s poll path.
        """
        battery = self.state.resolve_battery()
        if battery is None:
            self._send(404, b"no battery found", "text/plain")
            return

        # Imported here rather than at module scope: the watcher starts without paying for it, and
        # report.py only pulls in scan, which this module already has.
        import csv
        import io

        from sari_bench import report

        rows, _legs = report.collect(battery)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=report.ATTEMPT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        self._send(
            200,
            buffer.getvalue().encode("utf-8"),
            "text/csv; charset=utf-8",
            {"Content-Disposition": f'attachment; filename="{battery.name}-attempts.csv"'},
        )

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

        if path == "/api/report.csv":
            self._report_csv()
            return

        if path.startswith("/api/attempt/"):
            rest = path[len("/api/attempt/"):]
            key, _, action = rest.rpartition("/")
            if action == "frame.png":
                frame = self.state.frame_path(key)
                if frame is None or not frame.exists():
                    self._send(404, b"no frame yet", "text/plain")
                    return
                content_type = (
                    "image/jpeg"
                    if frame.suffix.lower() in {".jpg", ".jpeg"}
                    else "image/png"
                )
                self._send(200, frame.read_bytes(), content_type)
                return
            if action == "log":
                query = parse_qs(parsed.query)
                self._json(self.state.log_tail(
                    key,
                    lines=_int_param(query, "lines", LOG_TAIL_LINES, 1, LOG_MAX_LINES),
                    since=_int_param(query, "since", None, 0, None),
                    full=query.get("full", ["0"])[0] == "1",
                ))
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
                                "reason": "no frames or ffmpeg is unavailable"},
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
            if rest.endswith("/retry"):
                result = self.state.retry(rest[:-len("/retry")])
                self._json(result, code=202 if result.get("ok") else 400)
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
                        help="Bind address. 0.0.0.0 exposes the dashboard - including its destructive "
                             "kill and retry endpoints - to the network.")
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
        coordinator_url=args.coordinator,
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
