"""Runs a prompt battery across a fleet of leased sandboxes.

Work is the cross product of prompts and attempts. Each unit of work leases a sandbox, runs one
agent subprocess against it, and hands it back - the coordinator resets it before it is used again,
so no attempt inherits state from the one before it. By default the worker pool grows with the
coordinator's sandbox fleet; ``--concurrency`` remains available as an explicit upper bound.

Failures are contained per attempt: a crashed agent, an attempt that blows its time limit, or a
sandbox that dies mid-run all record an outcome and move on. Only a sandbox death requeues the
attempt, because that one is the harness's fault rather than the agent's.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from sari_bench import capture
from sari_bench.client import CoordinatorClient, Lease, SandboxLost
from sari_bench.protocol import DEFAULT_COORDINATOR_PORT, STATE_READY
from sari_bench.watch import scan   # per-role token block parsing; pure filesystem/dict helpers
from sari_bench.storage import (
    RUNNER_LOCK,
    battery_dir_name,
    canonical_attempt_rows,
    edit_json_locked,
    file_lock,
    purge_attempt_rows,
    upsert_attempt_row,
    write_json_atomic,
)
from overhaul.vision.ocr_client import OcrUnavailable, check_ocr_health, resolve_ocr_url
from sari_runconfig import RunConfigError, load_run_config
from overhaul.agent_core.context_policy import CONTEXT_POLICY_NAMES, resolve_context_policy

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERHAUL_DIR = REPO_ROOT / "overhaul"
ORCHESTRATOR_ENTRY = "orchestrator/subtask_agents.py"

# Grace on top of the agent's own --max-minutes before the harness kills it. The agent's cap is
# per leg, so a multi-leg task legitimately runs longer than one cap; this is the outer bound.
DEFAULT_TIMEOUT_GRACE_SECONDS = 120.0

# Seconds between SIGTERM and SIGKILL for an attempt that overran.
TERMINATE_GRACE_SECONDS = 20.0

# An attempt whose sandbox died is retried this many times before being recorded as failed. Guards
# against a permanently sick machine turning into an infinite requeue loop.
MAX_SANDBOX_LOST_REQUEUES = 3
ALREADY_SUCCESSFUL = "already_successful"
DEFAULT_CAPACITY_POLL_SECONDS = 1.0


class SandboxStartupError(RuntimeError):
    """The configured coordinator has no usable fleet at runner startup."""


class ResumeError(RuntimeError):
    """The requested output-directory operation is unsafe or incompatible."""


class OcrPreflightError(RuntimeError):
    """The required central OCR service was unavailable before sandbox leasing."""


# Per-attempt manifest, written INTO the run dir before the agent is spawned. attempts.jsonl only
# gains a row when an attempt finishes, so this is the only record that an attempt is in flight -
# it is what `sari_bench watch` discovers runs by, and it survives the runner dying.
ATTEMPT_MANIFEST = "attempt.json"
# Battery-level manifest at the output-dir root.
BATTERY_MANIFEST = "battery.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Writes JSON via a temp file + rename, so a reader polling this path never sees half a file."""
    write_json_atomic(path, payload)


def _patch_json(path: Path, fields: dict[str, Any]) -> None:
    """Merges `fields` into an existing JSON object. Best-effort: manifest bookkeeping must never
    take down an attempt that is otherwise fine."""
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(current, dict):
            current = {}
        current.update(fields)
        _write_json_atomic(path, current)
    except (OSError, ValueError) as error:  # noqa: BLE001
        _log(f"could not patch {path}: {error!r}")


def _manifest_field(path: Path, key: str) -> Any:
    """Reads one field back out of a manifest, or None if it is missing/unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload.get(key) if isinstance(payload, dict) else None


def purge_attempt_records(output_dir: Path, prompt_id: str, attempt: int) -> None:
    """Removes one logical try from the canonical result index."""
    purge_attempt_rows(output_dir, prompt_id, attempt)


@dataclass
class Prompt:
    id: str
    prompt: str
    family: str = ""
    looking_for: str = ""


@dataclass
class AttemptResult:
    prompt_id: str
    attempt: int
    prompt: str
    family: str
    outcome: str
    context_policy: str = "baseline"
    success: bool = False
    end_reason: str = ""
    sandbox_id: str = ""
    commands_uri: str = ""
    ocr_url: str = ""
    exit_code: int | None = None
    wall_seconds: float = 0.0
    run_dir: str = ""
    requeues: int = 0
    error: str = ""
    winning_attempt_key: str = ""
    legs: dict[str, Any] = field(default_factory=dict)
    # Token cost of the attempt: prompt tokens in, completion tokens out, across every reasoner the
    # agent ran (agent_core.token_meter). Zero means "the agent recorded none", which for a crashed
    # attempt can also mean it died before its first tokens.json write.
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    # role -> {tokens_in, tokens_out, calls}, from the same source as the totals above. Empty for an
    # attempt whose agent predates per-role accounting; see scan.normalize_by_role on why that is
    # not zero-filled.
    tokens_by_role: dict[str, Any] = field(default_factory=dict)


def load_prompts(path: Path) -> list[Prompt]:
    """Reads a prompt battery.

    Accepts the shape already used by ``overhaul/tests/decompose_battery.json`` - either a bare
    list or an object with a ``prompts`` key - so existing batteries work unchanged.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("prompts") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} contains no prompts")

    prompts: list[Prompt] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            prompts.append(Prompt(id=f"prompt_{index:02d}", prompt=entry))
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {index} is neither a string nor an object")

        text = entry.get("prompt") or entry.get("task")
        if not text:
            raise ValueError(f"{path}: entry {index} has no 'prompt'")
        prompts.append(
            Prompt(
                id=str(entry.get("id") or f"prompt_{index:02d}"),
                prompt=str(text),
                family=str(entry.get("family") or ""),
                looking_for=str(entry.get("looking_for") or ""),
            )
        )

    duplicates = {p.id for p in prompts if sum(1 for q in prompts if q.id == p.id) > 1}
    if duplicates:
        raise ValueError(f"{path}: duplicate prompt ids {sorted(duplicates)}")
    return prompts


class BenchmarkRunner:
    def __init__(
        self,
        *,
        prompts: list[Prompt],
        coordinator_url: str,
        output_dir: Path,
        tries: int,
        time_limit_minutes: float,
        concurrency: int | None,
        max_steps: int,
        arm: str,
        map_dir: str | None,
        leg_retries: int,
        completion_guard: str = "deterministic",
        context_policy: str = "baseline",
        per_leg_minutes: float | None = None,
        timeout_grace: float = DEFAULT_TIMEOUT_GRACE_SECONDS,
        sandbox_startup_timeout: float = 0.0,
        capture_interval: float = capture.DEFAULT_INTERVAL_SECONDS,
        capacity_poll_interval: float = DEFAULT_CAPACITY_POLL_SECONDS,
        python_executable: str | None = None,
        agent_entry: str = ORCHESTRATOR_ENTRY,
        agent_cwd: Path = OVERHAUL_DIR,
        work_items: list[tuple[str, int]] | None = None,
        initialize_battery: bool = True,
        resume: bool = False,
        ocr_url: str | None = None,
        ocr_health_check: Callable[[str], dict[str, Any]] = check_ocr_health,
    ) -> None:
        self.prompts = {prompt.id: prompt for prompt in prompts}
        self.coordinator_url = coordinator_url
        # The agent subprocess runs with cwd=overhaul/, not the runner's cwd. Keep the attempt path
        # absolute so --run-dir and the harness manifests always name the same directory.
        self.output_dir = output_dir.resolve()
        self.tries = tries
        self.time_limit_minutes = time_limit_minutes
        # The agent's --max-minutes is a PER-LEG cap; time_limit_minutes bounds the whole attempt.
        # Defaulting one to the other preserves the old single-knob behaviour, but they are not the
        # same number: a 120-minute attempt limit handed to a 5-leg task as a per-leg cap means the
        # agent's own time_cap can never fire, so every overrun lands as a SIGKILLed
        # `harness_timeout` with no summary.json and no per-leg detail.
        self.per_leg_minutes = time_limit_minutes if per_leg_minutes is None else per_leg_minutes
        self.concurrency = concurrency
        self.max_steps = max_steps
        self.arm = arm
        resolve_context_policy(context_policy)
        self.context_policy = context_policy
        self.map_dir = map_dir
        self.leg_retries = leg_retries
        if completion_guard not in {"deterministic", "vlm"}:
            raise ValueError(f"unsupported completion guard: {completion_guard!r}")
        self.completion_guard = completion_guard
        self.timeout_grace = timeout_grace
        self.sandbox_startup_timeout = sandbox_startup_timeout
        self.capture_interval = capture_interval
        self.capacity_poll_interval = capacity_poll_interval
        self.python_executable = python_executable or sys.executable
        # Overridable so tests can drive the whole lease/spawn/release cycle against a stub agent
        # instead of the real orchestrator (which pulls the entire model stack on import).
        self.agent_entry = agent_entry
        self.agent_cwd = agent_cwd
        self.work_items = work_items
        self.initialize_battery = initialize_battery
        self.resume = resume
        self.ocr_url = resolve_ocr_url(ocr_url)
        self.ocr_health_check = ocr_health_check

        self._queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
        self._results: list[AttemptResult] = []
        self._results_lock = asyncio.Lock()
        self._started_at = 0.0
        self._peak_workers = 0
        self._prior_wall_seconds = 0.0

    async def run(self) -> dict[str, Any]:
        if self.initialize_battery:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            try:
                with file_lock(self.output_dir / RUNNER_LOCK, blocking=False):
                    return await self._run_locked()
            except BlockingIOError as error:
                raise ResumeError(
                    f"another full battery runner is already using {self.output_dir}"
                ) from error
        return await self._run_locked()

    async def _run_locked(self) -> dict[str, Any]:
        """Validate local state, derive pending work, then touch the coordinator if needed."""
        self._started_at = time.monotonic()

        if self.initialize_battery:
            work_items = await self._prepare_battery()
        elif self.work_items is None:
            work_items = [
                (prompt.id, attempt)
                for prompt in self.prompts.values()
                for attempt in range(1, self.tries + 1)
            ]
        else:
            work_items = list(self.work_items)
        for prompt_id, attempt in work_items:
            if prompt_id not in self.prompts:
                raise ValueError(f"unknown work-item prompt: {prompt_id}")
            self._queue.put_nowait((prompt_id, attempt, 0))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        total = self._queue.qsize()
        planned = len(self.prompts) * self.tries
        _log(
            f"{len(self.prompts)} prompt(s) x {self.tries} attempt(s) = "
            f"{planned} planned, {total} pending"
        )
        if total == 0:
            _log("battery already complete; no sandbox required")
            return self._write_summary()

        # Fail before coordinator contact/acquire: a dead shared OCR daemon must not consume a
        # simulator lease merely to make every agent subprocess fail.
        try:
            health = self.ocr_health_check(self.ocr_url)
        except OcrUnavailable as error:
            raise OcrPreflightError(str(error)) from error
        _log(f"OCR preflight: {health['model']} at {self.ocr_url}")

        await self._wait_for_registered_sandbox()
        if self.initialize_battery and not self.resume:
            self._write_battery_manifest(planned)

        workers: list[asyncio.Task[None]] = []
        scaler: asyncio.Task[None] | None = None
        if self.concurrency is None:
            scaler = asyncio.create_task(self._scale_workers_with_fleet(workers, total))
        else:
            worker_count = min(self.concurrency, total)
            workers.extend(asyncio.create_task(self._worker(i)) for i in range(worker_count))
            self._record_worker_count(worker_count)
        try:
            await self._queue.join()
        finally:
            if scaler is not None:
                scaler.cancel()
                await asyncio.gather(scaler, return_exceptions=True)
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        return self._write_summary()

    @staticmethod
    def _payload_entries(path: Path) -> list[Path]:
        control = {RUNNER_LOCK, ".attempts.lock", ".battery.lock"}
        return [entry for entry in path.iterdir() if entry.name not in control]

    async def _prepare_battery(self) -> list[tuple[str, int]]:
        """Apply the fresh/resume directory contract and return only unfinished work."""
        entries = self._payload_entries(self.output_dir)
        if not self.resume:
            if entries:
                raise ResumeError(
                    f"{self.output_dir} is not empty; pass --resume to continue its battery "
                    "or choose a new --output-dir"
                )
            _log(f"new output dir {self.output_dir}")
            return [
                (prompt.id, attempt)
                for prompt in self.prompts.values()
                for attempt in range(1, self.tries + 1)
            ]

        battery_manifest_path = self.output_dir / BATTERY_MANIFEST
        if not battery_manifest_path.is_file():
            raise ResumeError(
                f"--resume requires an existing battery.json in {self.output_dir}"
            )
        try:
            battery = json.loads(battery_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ResumeError(f"cannot read {battery_manifest_path}: {error}") from error
        if not isinstance(battery, dict):
            raise ResumeError(f"{battery_manifest_path} is not a JSON object")
        self._validate_resume_config(battery)
        self._peak_workers = int(battery.get("peak_workers") or 0)
        previous_summary = self._read_manifest(self.output_dir / "summary.json")
        self._prior_wall_seconds = float(previous_summary.get("wall_seconds") or 0.0)

        try:
            rows = canonical_attempt_rows(self.output_dir, strict=True, rewrite=True)
        except ValueError as error:
            raise ResumeError(str(error)) from error

        planned_keys = {
            (prompt.id, attempt)
            for prompt in self.prompts.values()
            for attempt in range(1, self.tries + 1)
        }
        completed: set[tuple[str, int]] = set()
        rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["prompt_id"]), int(row["attempt"]))
            if key not in planned_keys:
                raise ResumeError(
                    f"attempts.jsonl contains out-of-plan result {key[0]} try {key[1]}"
                )
            completed.add(key)
            rows_by_key[key] = row

        for prompt_id, attempt in sorted(planned_keys):
            run_dir = self.output_dir / prompt_id / f"try{attempt:02d}"
            attempt_manifest_path = run_dir / ATTEMPT_MANIFEST
            manifest = self._read_manifest(attempt_manifest_path)
            key = (prompt_id, attempt)
            if key in completed:
                # The aggregate row is published only after the runner closes the manifest. Repair
                # a stale copy rather than rerunning a result that is already durably recorded.
                if manifest and manifest.get("state") != "finished":
                    row = rows_by_key[key]
                    _patch_json(
                        attempt_manifest_path,
                        {
                            "state": "finished",
                            "outcome": row.get("outcome", ""),
                            "success": bool(row.get("success")),
                            "end_reason": row.get("end_reason", ""),
                            "exit_code": row.get("exit_code"),
                            "wall_seconds": row.get("wall_seconds", 0.0),
                            "tokens_in": row.get("tokens_in", 0),
                            "tokens_out": row.get("tokens_out", 0),
                            "finalized_at": datetime.now().isoformat(timespec="seconds"),
                        },
                    )
                    _log(f"repaired stale finished manifest for {prompt_id} try {attempt}")
                continue
            if manifest.get("state") == "finished":
                upsert_attempt_row(
                    self.output_dir,
                    asdict(self._result_from_finished_manifest(prompt_id, attempt, run_dir, manifest)),
                )
                completed.add(key)
                _log(f"recovered missing result row for {prompt_id} try {attempt}")
                continue
            if run_dir.exists():
                await self._discard_interrupted_run(run_dir, manifest)

        now = datetime.now().isoformat(timespec="seconds")
        with edit_json_locked(battery_manifest_path) as current:
            current["resume_count"] = int(current.get("resume_count") or 0) + 1
            current["last_resumed_at"] = now
            current["coordinator"] = self.coordinator_url
            current["concurrency"] = self.concurrency if self.concurrency is not None else "auto"
            current["concurrency_mode"] = "fixed" if self.concurrency is not None else "auto"
            current["concurrency_limit"] = self.concurrency

        pending = [key for key in planned_keys if key not in completed]
        _log(
            f"resuming {self.output_dir}: {len(completed)} finished, {len(pending)} pending"
        )
        return sorted(pending)

    def _semantic_config(self) -> dict[str, Any]:
        return {
            "tries": self.tries,
            "time_limit_minutes": self.time_limit_minutes,
            "per_leg_minutes": self.per_leg_minutes,
            "max_steps": self.max_steps,
            "arm": self.arm,
            "context_policy": self.context_policy,
            "capture_interval_seconds": self.capture_interval,
            "map_dir": str(Path(self.map_dir).resolve()) if self.map_dir else None,
            "leg_retries": self.leg_retries,
            "completion_guard": self.completion_guard,
            "ocr_url": self.ocr_url,
            "timeout_grace_seconds": self.timeout_grace,
            "agent_entry": self.agent_entry,
            "agent_cwd": str(Path(self.agent_cwd).resolve()),
            "planned_attempts": len(self.prompts) * self.tries,
            "prompts": [asdict(prompt) for prompt in self.prompts.values()],
        }

    def _validate_resume_config(self, battery: dict[str, Any]) -> None:
        # Batteries created before completion guards were configurable used the deterministic
        # backend. Treat an absent legacy field as that default, while still refusing a resume that
        # would silently switch an old deterministic battery to VLM.
        def existing_value(key: str) -> Any:
            if key == "completion_guard" and key not in battery:
                return "deterministic"
            if key == "context_policy" and key not in battery:
                return "baseline"
            if key == "ocr_url" and key not in battery:
                return resolve_ocr_url()
            return battery.get(key)

        mismatches = [
            f"{key}: existing={existing_value(key)!r}, requested={requested!r}"
            for key, requested in self._semantic_config().items()
            if existing_value(key) != requested
        ]
        if mismatches:
            raise ResumeError(
                "resume configuration does not match battery.json:\n  "
                + "\n  ".join(mismatches)
            )

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _result_from_finished_manifest(
        self,
        prompt_id: str,
        attempt: int,
        run_dir: Path,
        manifest: dict[str, Any],
    ) -> AttemptResult:
        prompt = self.prompts[prompt_id]
        result = self._result_from_run_dir(
            prompt,
            attempt,
            run_dir,
            exit_code=manifest.get("exit_code"),
            outcome=str(manifest.get("outcome") or "finished"),
        )
        # The process outcome is authoritative over a stale/partial agent summary. In particular,
        # an agent can write `success: true` and then crash during teardown; a non-zero exit is still
        # an agent_error and must never become a successful benchmark attempt.
        result.success = (
            bool(manifest.get("success", result.success))
            if result.outcome == "completed" else False
        )
        result.end_reason = str(manifest.get("end_reason") or result.end_reason)
        result.sandbox_id = str(manifest.get("sandbox_id") or "")
        result.commands_uri = str(manifest.get("commands_uri") or "")
        result.ocr_url = str(manifest.get("ocr_url") or self.ocr_url)
        result.wall_seconds = float(manifest.get("wall_seconds") or 0.0)
        result.run_dir = str(run_dir)
        result.requeues = int(manifest.get("requeues") or 0)
        result.winning_attempt_key = str(manifest.get("winning_attempt_key") or "")
        return result

    async def _discard_interrupted_run(
        self,
        run_dir: Path,
        manifest: dict[str, Any],
    ) -> None:
        """Stop a verifiable orphan, release its old lease, then remove partial artifacts."""
        recorded_host = str(manifest.get("runner_host") or "")
        if recorded_host and recorded_host != socket.gethostname():
            raise ResumeError(
                f"{run_dir} was started on host {recorded_host}; cannot safely verify or stop "
                "its orphan process from this host"
            )

        recorded_boot = str(manifest.get("runner_boot_id") or "")
        same_boot = not recorded_boot or recorded_boot == self._boot_id()
        pid = int(manifest.get("pid") or 0)
        live_pids: list[int] = []
        if same_boot and pid and self._pid_alive(pid):
            live_pids = [pid]
        elif same_boot and not pid:
            live_pids = self._matching_run_pids(run_dir)

        for live_pid in live_pids:
            if not self._pid_matches_manifest(live_pid, run_dir, manifest):
                raise ResumeError(
                    f"{run_dir} records live pid {live_pid}, but its process identity does not match; "
                    "refusing to delete or rerun it"
                )
            await self._kill_process_group(live_pid)

        lease_id = str(manifest.get("lease_id") or "")
        if lease_id:
            try:
                async with CoordinatorClient(self.coordinator_url) as client:
                    known = await client.release_lease_id(lease_id, outcome="runner_resume")
                _log(
                    f"resume released stale lease {lease_id} for {run_dir.name} "
                    f"({'known' if known else 'already reaped'})"
                )
            except Exception as error:  # coordinator disconnect reaping and TTL remain authoritative
                _log(f"could not defensively release stale lease {lease_id}: {error!r}")

        shutil.rmtree(run_dir)
        _log(f"removed interrupted run dir before clean rerun: {run_dir}")

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _boot_id() -> str:
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _matching_run_pids(self, run_dir: Path) -> list[int]:
        matches: list[int] = []
        proc = Path("/proc")
        try:
            entries = list(proc.iterdir())
        except OSError:
            return matches
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                command = (entry / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if str(run_dir).encode() in command and self.agent_entry.encode() in command:
                matches.append(pid)
        return matches

    def _pid_matches_manifest(
        self,
        pid: int,
        run_dir: Path,
        manifest: dict[str, Any],
    ) -> bool:
        try:
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            command_text = [part.decode(errors="replace") for part in command if part]
        except OSError:
            return False
        if str(run_dir) not in command_text or self.agent_entry not in command_text:
            return False
        recorded_start = str(manifest.get("process_start_ticks") or "")
        return not recorded_start or recorded_start == self._process_start_ticks(pid)

    @staticmethod
    def _process_start_ticks(pid: int) -> str:
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
            return stat[stat.rfind(")") + 2 :].split()[19]
        except (OSError, IndexError):
            return ""

    @staticmethod
    async def _kill_process_group(pid: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
        while BenchmarkRunner._pid_alive(pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if not BenchmarkRunner._pid_alive(pid):
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
        while BenchmarkRunner._pid_alive(pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if BenchmarkRunner._pid_alive(pid):
            raise ResumeError(f"orphan process group {pid} did not stop")

    def _record_worker_count(self, count: int) -> None:
        if count <= self._peak_workers:
            return
        self._peak_workers = count
        if self.initialize_battery:
            with edit_json_locked(self.output_dir / BATTERY_MANIFEST) as battery:
                battery["peak_workers"] = max(int(battery.get("peak_workers") or 0), count)

    async def _scale_workers_with_fleet(
        self,
        workers: list[asyncio.Task[None]],
        planned_attempts: int,
    ) -> None:
        """Grow the default worker pool as store-loaded sandboxes join the coordinator.

        Workers are deliberately not removed when the observed fleet shrinks. Cancelling one may
        terminate an attempt that is still healthy, while an idle worker already blocks harmlessly
        in either queue.get() or bench.acquire. The high-water mark also means a replacement
        sandbox can take queued work immediately.
        """
        connection_error: str | None = None
        while True:
            try:
                async with CoordinatorClient(self.coordinator_url) as client:
                    while True:
                        pool = await client.pool()
                        if connection_error is not None:
                            _log("automatic concurrency: coordinator connection restored")
                            connection_error = None

                        capacity = sum(
                            1
                            for sandbox in pool
                            if bool(sandbox.get("store_loaded", True))
                        )
                        desired = min(planned_attempts, capacity)
                        while len(workers) < desired:
                            index = len(workers)
                            workers.append(asyncio.create_task(self._worker(index)))
                            _log(
                                f"automatic concurrency: started worker {index} "
                                f"for {capacity} registered sandbox(es)"
                            )
                        self._record_worker_count(len(workers))
                        await asyncio.sleep(self.capacity_poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - keep watching through coordinator restarts
                current_error = f"{type(error).__name__}: {error}"
                if current_error != connection_error:
                    _log(f"automatic concurrency: waiting for coordinator ({current_error})")
                    connection_error = current_error
                await asyncio.sleep(self.capacity_poll_interval)

    async def _wait_for_registered_sandbox(self) -> None:
        """Fail clearly at startup instead of parking every worker against an empty pool."""
        if self.sandbox_startup_timeout <= 0:
            return

        deadline = time.monotonic() + self.sandbox_startup_timeout
        last_pool: list[dict[str, Any]] = []
        reached_coordinator = False

        async def fetch_pool() -> list[dict[str, Any]]:
            async with CoordinatorClient(self.coordinator_url) as client:
                return await client.pool()

        def pool_problem() -> str:
            if not last_pool:
                return "none registered"
            if not any(bool(sandbox.get("store_loaded", True)) for sandbox in last_pool):
                return "registered sandbox(es) have no store loaded"
            states: dict[str, int] = {}
            for sandbox in last_pool:
                state = str(sandbox.get("state") or "unknown")
                states[state] = states.get(state, 0) + 1
            state_summary = ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
            return f"none ready or coordinator-leased; sim states: {state_summary}"

        while True:
            remaining = deadline - time.monotonic()
            try:
                # websockets' own opening-handshake timeout is normally 10 seconds. Keep the
                # operator-facing timeout honest even when it is configured lower than that.
                last_pool = await asyncio.wait_for(fetch_pool(), timeout=max(0.001, remaining))
            except Exception as error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if reached_coordinator:
                        raise SandboxStartupError(
                            f"No usable sandbox registered with {self.coordinator_url} after "
                            f"{self.sandbox_startup_timeout:g}s ({pool_problem()}). Start the Distributed "
                            "Sari Bench Unity player with SARI_BENCH_COORDINATOR pointing to "
                            f"{self.coordinator_url.rstrip('/')}/sandbox, then rerun dbench."
                        ) from error
                    reason = str(error) or type(error).__name__
                    raise SandboxStartupError(
                        f"Coordinator {self.coordinator_url} was not reachable within "
                        f"{self.sandbox_startup_timeout:g}s: {reason}. Start it with "
                        "`uv run poe coordinator`, then start the Unity sandbox fleet."
                    ) from error
            else:
                reached_coordinator = True
                # A leased member is healthy but busy; a Ready member can be acquired immediately.
                # Booting/Resetting members get the full startup window to become Ready.
                if any(
                    bool(sandbox.get("store_loaded", True))
                    and (bool(sandbox.get("lease_id")) or sandbox.get("state") == STATE_READY)
                    for sandbox in last_pool
                ):
                    _log(f"sandbox preflight: {len(last_pool)} registered")
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SandboxStartupError(
                        f"No usable sandbox registered with {self.coordinator_url} after "
                        f"{self.sandbox_startup_timeout:g}s ({pool_problem()}). Start the Distributed "
                        "Sari Bench Unity player with SARI_BENCH_COORDINATOR pointing to "
                        f"{self.coordinator_url.rstrip('/')}/sandbox, then rerun dbench."
                    )

            await asyncio.sleep(min(1.0, max(0.0, remaining)))

    def _write_battery_manifest(self, planned_attempts: int) -> None:
        """Battery-level facts the watcher cannot infer from run dirs alone.

        Without this the dashboard has no denominator: it can count the attempts that have started
        but not the ones still queued, so it cannot show progress.
        """
        with edit_json_locked(self.output_dir / BATTERY_MANIFEST) as battery:
            battery.clear()
            battery.update(
                {
                "battery_id": self.output_dir.name,
                "output_dir": str(self.output_dir),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "coordinator": self.coordinator_url,
                "tries": self.tries,
                "concurrency": self.concurrency if self.concurrency is not None else "auto",
                "concurrency_mode": "fixed" if self.concurrency is not None else "auto",
                "concurrency_limit": self.concurrency,
                "peak_workers": self._peak_workers,
                "time_limit_minutes": self.time_limit_minutes,
                "per_leg_minutes": self.per_leg_minutes,
                "max_steps": self.max_steps,
                "arm": self.arm,
                "context_policy": self.context_policy,
                "capture_interval_seconds": self.capture_interval,
                "map_dir": str(Path(self.map_dir).resolve()) if self.map_dir else None,
                "leg_retries": self.leg_retries,
                "completion_guard": self.completion_guard,
                "ocr_url": self.ocr_url,
                "timeout_grace_seconds": self.timeout_grace,
                "sandbox_startup_timeout_seconds": self.sandbox_startup_timeout,
                "agent_entry": self.agent_entry,
                "agent_cwd": str(Path(self.agent_cwd).resolve()),
                "planned_attempts": planned_attempts,
                "prompts": [asdict(prompt) for prompt in self.prompts.values()],
                "resume_count": 0,
                }
            )

    @staticmethod
    def _rotate_run_dir(run_dir: Path) -> None:
        """Moves a non-empty run dir aside so a fresh attempt never writes into it.

        A requeued attempt (sandbox death) reuses the same ``<prompt_id>/try<NN>`` path, and the
        orchestrator opens ``legNN.jsonl`` in APPEND mode - so without this the dead attempt's steps
        and the new attempt's steps interleave in one file while ``stepNN.png`` frames overwrite
        each other. The merged log then misreports timesteps for both.
        """
        if not run_dir.exists() or not any(run_dir.iterdir()):
            return
        index = 0
        while (aside := run_dir.with_name(f"{run_dir.name}.requeue{index:02d}")).exists():
            index += 1
        run_dir.rename(aside)
        # The manifest inside still says "running"; correct it so the watcher shows an abandoned
        # attempt rather than a live one that will never advance.
        _patch_json(aside / ATTEMPT_MANIFEST, {"state": "requeued", "outcome": "requeued"})
        _log(f"rotated a previous run dir aside: {aside.name}")

    async def _worker(self, index: int) -> None:
        """One worker owns one coordinator connection, and therefore one lease at a time."""
        while True:
            prompt_id, attempt, requeues = await self._queue.get()
            try:
                winner = self._human_verified_winner(prompt_id)
                if winner is not None:
                    await self._record_skipped(prompt_id, attempt, requeues, winner)
                    _log(
                        f"[w{index}] {prompt_id} try {attempt}: skipped; "
                        f"{winner['winning_attempt_key']} is human-verified successful"
                    )
                    continue
                await self._run_attempt(index, prompt_id, attempt, requeues)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - one bad attempt must not stop the battery
                await self._record(
                    AttemptResult(
                        prompt_id=prompt_id,
                        attempt=attempt,
                        prompt=self.prompts[prompt_id].prompt,
                        family=self.prompts[prompt_id].family,
                        outcome="harness_error",
                        requeues=requeues,
                        error=repr(error),
                    )
                )
                _log(f"[w{index}] {prompt_id} try {attempt}: harness error {error!r}")
            finally:
                self._queue.task_done()

    async def _run_attempt(self, index: int, prompt_id: str, attempt: int, requeues: int) -> None:
        prompt = self.prompts[prompt_id]
        run_dir = self.output_dir / prompt_id / f"try{attempt:02d}"

        async with CoordinatorClient(self.coordinator_url) as client:
            _log(f"[w{index}] {prompt_id} try {attempt}: waiting for a sandbox")
            lease = await client.acquire()
            _log(f"[w{index}] {prompt_id} try {attempt}: leased {lease.sandbox_id} ({lease.commands_uri})")

            # A winner can land while this worker is blocked in acquire(). Do not start an agent
            # merely because the work item was dequeued before the reviewer clicked success.
            winner = self._human_verified_winner(prompt_id)
            if winner is not None:
                with contextlib.suppress(Exception):
                    await client.release(lease, outcome="done")
                await self._record_skipped(prompt_id, attempt, requeues, winner)
                _log(
                    f"[w{index}] {prompt_id} try {attempt}: skipped after lease; "
                    f"{winner['winning_attempt_key']} is human-verified successful"
                )
                return

            started = time.monotonic()
            try:
                result = await self._spawn_agent(client, lease, prompt, attempt, run_dir)
            except SandboxLost as lost:
                # The machine went away, not the agent. Put the attempt back rather than scoring it.
                if requeues < MAX_SANDBOX_LOST_REQUEUES:
                    _log(f"[w{index}] {prompt_id} try {attempt}: {lost}; requeueing")
                    self._queue.put_nowait((prompt_id, attempt, requeues + 1))
                    return
                result = AttemptResult(
                    prompt_id=prompt_id,
                    attempt=attempt,
                    prompt=prompt.prompt,
                    family=prompt.family,
                    outcome="sandbox_lost",
                    error=str(lost),
                )
            finally:
                # Always release: the sandbox has to be reset and re-pooled even when the agent
                # crashed, timed out, or was cancelled.
                with contextlib.suppress(Exception):
                    await client.release(lease, outcome="done")

            result.sandbox_id = lease.sandbox_id
            result.commands_uri = lease.commands_uri
            result.ocr_url = self.ocr_url
            result.wall_seconds = round(time.monotonic() - started, 1)
            result.requeues = requeues
            result.run_dir = str(run_dir)
            if _manifest_field(run_dir / ATTEMPT_MANIFEST, "stop_reason") == ALREADY_SUCCESSFUL:
                result.end_reason = ALREADY_SUCCESSFUL
                result.winning_attempt_key = str(
                    _manifest_field(run_dir / ATTEMPT_MANIFEST, "winning_attempt_key") or ""
                )
            # Close out the manifest so the watcher stops showing this attempt as live. Every exit
            # path lands here, including SandboxLost after its requeues are exhausted.
            _patch_json(
                run_dir / ATTEMPT_MANIFEST,
                {
                    "state": "finished",
                    "outcome": result.outcome,
                    "success": result.success,
                    "end_reason": result.end_reason,
                    "exit_code": result.exit_code,
                    "wall_seconds": result.wall_seconds,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "tokens_by_role": result.tokens_by_role,
                    "ended_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            await self._record(result)
            _patch_json(
                run_dir / ATTEMPT_MANIFEST,
                {"finalized_at": datetime.now().isoformat(timespec="seconds")},
            )
            _log(
                f"[w{index}] {prompt_id} try {attempt}: {result.outcome} "
                f"(success={result.success}, {result.wall_seconds}s, "
                f"tokens {result.tokens_in}in/{result.tokens_out}out)"
            )

    async def _spawn_agent(
        self,
        client: CoordinatorClient,
        lease: Lease,
        prompt: Prompt,
        attempt: int,
        run_dir: Path,
    ) -> AttemptResult:
        self._rotate_run_dir(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        command = self._agent_command(prompt, lease, run_dir)

        env = dict(os.environ)
        # agent.log is read live by the watch dashboard mid-attempt. Without this, stdout
        # redirected to a file (not a tty) is fully block-buffered, so the watcher's "log" tab
        # shows nothing for minutes at a time even while the agent is actively working.
        env["PYTHONUNBUFFERED"] = "1"
        # How the agent finds its sandbox. sim/env.py reads this for every command's default URI.
        env["SARI_WS_URI"] = lease.commands_uri
        env["SARI_OCR_URL"] = self.ocr_url
        if self.map_dir:
            # Belt to --output-dir's braces. The flag only reaches the call sites the orchestrator
            # explicitly threads it through; this reaches every StoreMap() in the attempt's process
            # (nav/store_map.default_output_dir reads it), so no helper can silently fall back to
            # the frozen slamtest/output - which in this checkout has no topology_final_shelf.json
            # and used to take every attempt down in ~2s as a bare `agent_error`.
            # Absolute because the agent runs with cwd=overhaul/.
            env["SARI_MAP_DIR"] = str(Path(self.map_dir).resolve())

        timeout = self.time_limit_minutes * 60.0 + self.timeout_grace
        manifest_path = run_dir / ATTEMPT_MANIFEST
        started_wall = time.time()
        _write_json_atomic(
            manifest_path,
            {
                "run_id": uuid.uuid4().hex,
                "prompt_id": prompt.id,
                "prompt": prompt.prompt,
                "family": prompt.family,
                "looking_for": prompt.looking_for,
                "attempt": attempt,
                "arm": self.arm,
                "context_policy": self.context_policy,
                "completion_guard": self.completion_guard,
                "ocr_url": self.ocr_url,
                "sandbox_id": lease.sandbox_id,
                "commands_uri": lease.commands_uri,
                "lease_id": getattr(lease, "lease_id", ""),
                "run_dir": str(run_dir),
                "state": "starting",
                "pid": None,
                "runner_host": socket.gethostname(),
                "runner_boot_id": self._boot_id(),
                "started_at": datetime.fromtimestamp(started_wall).isoformat(timespec="seconds"),
                "started_epoch": round(started_wall, 3),
                "deadline_epoch": round(started_wall + timeout, 3),
                "time_limit_minutes": self.time_limit_minutes,
                "per_leg_minutes": self.per_leg_minutes,
                "max_steps": self.max_steps,
                "capture_interval_seconds": self.capture_interval,
                "command": command,
            },
        )

        with (run_dir / "agent.log").open("wb") as log_file:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.agent_cwd),
                env=env,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )

            # The pid is what makes an attempt killable from the dashboard: the watcher signals the
            # process group and the runner's ordinary non-zero-exit path releases the lease.
            _patch_json(
                manifest_path,
                {
                    "pid": process.pid,
                    "state": "running",
                    "runner_host": socket.gethostname(),
                    "process_start_ticks": self._process_start_ticks(process.pid),
                },
            )

            # Close the last publication race: success may have been reviewed after the pre-spawn
            # check, or the watcher may have stamped this starting manifest before it had a PID to
            # signal. Publishing the PID and checking immediately ensures either side stops it.
            winner = self._human_verified_winner(prompt.id)
            stop_requested = _manifest_field(manifest_path, "stop_reason") == ALREADY_SUCCESSFUL
            if winner is not None or stop_requested:
                if winner is not None:
                    _patch_json(manifest_path, self._cancellation_fields(winner))
                _patch_json(manifest_path, {"killed_by": ALREADY_SUCCESSFUL})
                await self._kill(process)

            wait_task = asyncio.create_task(process.wait())
            lost_task = asyncio.create_task(client.wait_for_sandbox_lost(lease))
            capture_task = (
                asyncio.create_task(
                    self._capture_until_exit(process, run_dir, lease.commands_uri, manifest_path)
                )
                if self.capture_interval > 0
                else None
            )

            try:
                done, pending = await asyncio.wait(
                    {wait_task, lost_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                # Agent attempts run in their own sessions, so the terminal's Ctrl+C does not
                # reach them when it interrupts the runner. Explicitly stop the whole agent
                # process group before propagating cancellation; _run_attempt's finally block
                # can then release and reset the sandbox without an orphan still commanding it.
                await self._kill(process)
                _patch_json(
                    manifest_path,
                    {
                        "state": "finished",
                        "outcome": "interrupted",
                        "exit_code": process.returncode,
                        "ended_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                if capture_task is not None:
                    await capture_task
                raise
            finally:
                for task in (wait_task, lost_task):
                    if not task.done():
                        task.cancel()

            if lost_task in done:
                await self._kill(process)
                if capture_task is not None:
                    await capture_task
                raise lost_task.result()

            if wait_task not in done:
                _log(f"{prompt.id} try {attempt}: exceeded {timeout:.0f}s; terminating")
                await self._kill(process)
                if capture_task is not None:
                    await capture_task
                return self._result_from_run_dir(
                    prompt, attempt, run_dir, exit_code=None, outcome="harness_timeout"
                )

            exit_code = wait_task.result()
            if capture_task is not None:
                await capture_task

        stopped_for_success = (
            _manifest_field(run_dir / ATTEMPT_MANIFEST, "stop_reason") == ALREADY_SUCCESSFUL
        )
        outcome = "completed" if exit_code == 0 else "agent_error"
        if stopped_for_success or (
            outcome == "agent_error" and _manifest_field(run_dir / ATTEMPT_MANIFEST, "killed_by")
        ):
            # An operator killing a collapsed attempt from the dashboard reaches the agent as a
            # signal, which looks exactly like a crash from here. Don't score it as one.
            outcome = "operator_kill"
        result = self._result_from_run_dir(
            prompt, attempt, run_dir, exit_code=exit_code, outcome=outcome
        )
        if stopped_for_success:
            result.end_reason = ALREADY_SUCCESSFUL
            result.winning_attempt_key = str(
                _manifest_field(run_dir / ATTEMPT_MANIFEST, "winning_attempt_key") or ""
            )
        return result

    def _human_verified_winner(self, prompt_id: str) -> dict[str, Any] | None:
        """Returns durable cancellation metadata for this exact prompt ID, if one exists."""
        try:
            battery = json.loads(
                (self.output_dir / BATTERY_MANIFEST).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            battery = {}
        durable = (
            (battery.get("human_verified_winners") or {}).get(prompt_id)
            if isinstance(battery, dict)
            else None
        )
        if isinstance(durable, dict) and durable.get("winning_attempt_key"):
            return {
                "winning_attempt_key": str(durable["winning_attempt_key"]),
                "stop_requested_at": str(durable.get("stop_requested_at") or ""),
                "stop_requested_by": str(durable.get("stop_requested_by") or ""),
            }

        prompt_dir = self.output_dir / prompt_id
        if not prompt_dir.is_dir():
            return None
        for run_dir in sorted(prompt_dir.iterdir()):
            if not run_dir.is_dir() or ".requeue" in run_dir.name:
                continue
            manifest = {}
            try:
                manifest = json.loads(
                    (run_dir / ATTEMPT_MANIFEST).read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if (
                isinstance(manifest, dict)
                and manifest.get("state") == "finished"
                and manifest.get("verified_success") is True
            ):
                return {
                    "winning_attempt_key": f"{prompt_id}/{run_dir.name}",
                    "stop_requested_at": str(
                        manifest.get("verified_at")
                        or datetime.now().isoformat(timespec="seconds")
                    ),
                    "stop_requested_by": str(manifest.get("verified_by") or "watcher"),
                }
        return None

    @staticmethod
    def _cancellation_fields(winner: dict[str, Any]) -> dict[str, Any]:
        return {
            "stop_reason": ALREADY_SUCCESSFUL,
            "stop_requested_at": winner.get("stop_requested_at", ""),
            "stop_requested_by": winner.get("stop_requested_by", ""),
            "winning_attempt_key": winner.get("winning_attempt_key", ""),
        }

    async def _record_skipped(
        self,
        prompt_id: str,
        attempt: int,
        requeues: int,
        winner: dict[str, Any],
    ) -> None:
        """Materialises a queued/non-spawned try as a real terminal attempt."""
        prompt = self.prompts[prompt_id]
        run_dir = self.output_dir / prompt_id / f"try{attempt:02d}"
        self._rotate_run_dir(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        ended_at = datetime.now().isoformat(timespec="seconds")
        fields = self._cancellation_fields(winner)
        _write_json_atomic(
            run_dir / ATTEMPT_MANIFEST,
            {
                "run_id": uuid.uuid4().hex,
                "prompt_id": prompt.id,
                "prompt": prompt.prompt,
                "family": prompt.family,
                "looking_for": prompt.looking_for,
                "attempt": attempt,
                "arm": self.arm,
                "context_policy": self.context_policy,
                "ocr_url": self.ocr_url,
                "run_dir": str(run_dir),
                "state": "finished",
                "outcome": "skipped",
                "success": False,
                "end_reason": ALREADY_SUCCESSFUL,
                "pid": None,
                "wall_seconds": 0.0,
                "started_at": ended_at,
                "ended_at": ended_at,
                **fields,
            },
        )
        await self._record(
            AttemptResult(
                prompt_id=prompt.id,
                attempt=attempt,
                prompt=prompt.prompt,
                family=prompt.family,
                outcome="skipped",
                end_reason=ALREADY_SUCCESSFUL,
                wall_seconds=0.0,
                run_dir=str(run_dir),
                requeues=requeues,
                ocr_url=self.ocr_url,
                winning_attempt_key=str(winner.get("winning_attempt_key") or ""),
            )
        )
        _patch_json(run_dir / ATTEMPT_MANIFEST, {"finalized_at": ended_at})

    async def _capture_until_exit(
        self,
        process: asyncio.subprocess.Process,
        run_dir: Path,
        commands_uri: str,
        manifest_path: Path,
    ) -> None:
        """Own the recorder for exactly the lifetime of its agent process."""
        stats = capture.CaptureStats()
        recorder = asyncio.create_task(
            capture.record_previews(
                run_dir,
                commands_uri,
                self.capture_interval,
                stats=stats,
                log=_log,
            )
        )
        try:
            await process.wait()
        finally:
            recorder.cancel()
            await asyncio.gather(recorder, return_exceptions=True)
            _patch_json(
                manifest_path,
                {
                    "capture_frames": stats.frames,
                    "capture_failures": stats.failures,
                },
            )

    def _agent_command(self, prompt: Prompt, lease: Lease, run_dir: Path) -> list[str]:
        command = [
            self.python_executable,
            self.agent_entry,
            "--task",
            prompt.prompt,
            "--arm",
            self.arm,
            "--context-policy",
            self.context_policy,
            "--run-dir",
            str(run_dir),
            "--max-steps",
            str(self.max_steps),
            "--max-minutes",
            str(self.per_leg_minutes),
            "--leg-retries",
            str(self.leg_retries),
            "--completion-guard",
            self.completion_guard,
            "--ws-uri",
            lease.commands_uri,
            "--ocr-url",
            self.ocr_url,
        ]
        if self.map_dir:
            # Resolved, for the same reason SARI_MAP_DIR is (see _spawn_agent): --map-dir is given
            # relative to the repo root, but the agent runs with cwd=overhaul/, so handing the flag
            # over verbatim points it at overhaul/<map-dir> and StoreMap dies on its first load.
            command += ["--output-dir", str(Path(self.map_dir).resolve())]
        return command

    @staticmethod
    async def _kill(process: asyncio.subprocess.Process) -> None:
        """Stops an attempt, escalating to SIGKILL. The agent is started in its own session so the
        signal reaches any helper processes it spawned rather than just the launcher."""
        if process.returncode is not None:
            return

        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        try:
            await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)
            return
        except asyncio.TimeoutError:
            pass

        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)

    def _result_from_run_dir(
        self,
        prompt: Prompt,
        attempt: int,
        run_dir: Path,
        *,
        exit_code: int | None,
        outcome: str,
    ) -> AttemptResult:
        """Folds the orchestrator's own summary.json into the attempt row.

        A missing summary is normal for a killed or crashed attempt - the row still records the
        outcome, it just has no per-leg detail.
        """
        result = AttemptResult(
            prompt_id=prompt.id,
            attempt=attempt,
            prompt=prompt.prompt,
            family=prompt.family,
            outcome=outcome,
            context_policy=self.context_policy,
            exit_code=exit_code,
        )

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            # No summary means a killed or crashed attempt - but it still spent tokens, and the
            # agent's periodic tokens.json is the only place they survive.
            self._apply_tokens(result, run_dir, summary=None)
            return result

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            result.error = f"unreadable summary.json: {error}"
            self._apply_tokens(result, run_dir, summary=None)
            return result

        # Only a clean process exit may carry the agent's task-success predicate. A summary can be
        # written before a later teardown crash, so copying it onto `agent_error` would produce the
        # contradictory and score-corrupting `agent_error (success=True)`.
        result.success = bool(summary.get("success")) if outcome == "completed" else False
        result.llm_calls = int(summary.get("llm_calls") or 0)
        self._apply_tokens(result, run_dir, summary=summary)
        legs = summary.get("legs") or []
        result.legs = {
            "planned": summary.get("legs_planned"),
            "completed": summary.get("legs_completed"),
            "end_reasons": [leg.get("end_reason") for leg in legs if isinstance(leg, dict)],
        }
        # The last leg's end_reason is what actually stopped the task.
        if result.legs["end_reasons"]:
            result.end_reason = str(result.legs["end_reasons"][-1] or "")
        return result

    @staticmethod
    def _apply_tokens(result: AttemptResult, run_dir: Path, summary: dict[str, Any] | None) -> None:
        """Fills in the attempt's token cost from whichever record the agent got as far as writing.

        summary.json is authoritative but only exists for an attempt that exited cleanly; tokens.json
        is rewritten every few seconds while the agent runs, so a timed-out or operator-killed attempt
        - exactly the expensive ones worth accounting for - still reports what it burned.
        """
        source: dict[str, Any] | None = None
        if isinstance(summary, dict) and isinstance(summary.get("tokens"), dict):
            source = summary["tokens"]
        elif isinstance(summary, dict) and "tokens_in" in summary:
            source = summary

        if source is None:
            try:
                payload = json.loads((run_dir / "tokens.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            if not isinstance(payload, dict):
                return
            source = payload

        result.tokens_in = int(source.get("tokens_in") or 0)
        result.tokens_out = int(source.get("tokens_out") or 0)
        result.tokens_by_role = scan.normalize_by_role(source.get("by_role"))
        if not result.llm_calls:
            result.llm_calls = int(source.get("calls") or 0)

    async def _record(self, result: AttemptResult) -> None:
        async with self._results_lock:
            result.context_policy = self.context_policy
            self._results.append(result)
            # Written incrementally and atomically so an interrupted battery remains usable, while
            # one logical prompt/attempt slot always has exactly one canonical row.
            upsert_attempt_row(self.output_dir, asdict(result))

    def _write_summary(self) -> dict[str, Any]:
        # The result spine is shared by the original runner and any watcher-launched retry
        # runners. Its latest row for a logical try is canonical; an in-memory list belongs only to
        # one of those processes and would make a retry overwrite the battery summary with one row.
        latest: dict[tuple[str, int], AttemptResult] = {}
        rows = canonical_attempt_rows(self.output_dir)
        allowed = set(AttemptResult.__dataclass_fields__)
        for row in rows:
            try:
                result = AttemptResult(**{key: value for key, value in row.items() if key in allowed})
            except (TypeError, ValueError):
                continue
            # Normalize historical rows too, so resuming an affected battery repairs its aggregate
            # summary even if the old attempts.jsonl row recorded agent_error as successful.
            if result.outcome != "completed":
                result.success = False
            latest[(result.prompt_id, result.attempt)] = result
        results = list(latest.values())

        by_prompt: dict[str, dict[str, Any]] = {}
        for result in results:
            row = by_prompt.setdefault(
                result.prompt_id,
                {
                    "prompt_id": result.prompt_id,
                    "prompt": result.prompt,
                    "family": result.family,
                    "attempts": 0,
                    "successes": 0,
                    "outcomes": {},
                    "end_reasons": {},
                    "sandboxes": [],
                    "tokens_in": 0,
                    "tokens_out": 0,
                },
            )
            row["attempts"] += 1
            row["successes"] += int(result.success)
            row["tokens_in"] += result.tokens_in
            row["tokens_out"] += result.tokens_out
            row["outcomes"][result.outcome] = row["outcomes"].get(result.outcome, 0) + 1
            if result.end_reason:
                row["end_reasons"][result.end_reason] = row["end_reasons"].get(result.end_reason, 0) + 1
            if result.sandbox_id and result.sandbox_id not in row["sandboxes"]:
                row["sandboxes"].append(result.sandbox_id)

        for row in by_prompt.values():
            row["success_rate"] = round(row["successes"] / row["attempts"], 3) if row["attempts"] else 0.0
            # Per-attempt averages, because prompts run a different number of attempts once
            # sandbox-lost requeues and --only are in play - the totals alone don't compare.
            row["tokens_in_avg"] = round(row["tokens_in"] / row["attempts"]) if row["attempts"] else 0
            row["tokens_out_avg"] = round(row["tokens_out"] / row["attempts"]) if row["attempts"] else 0

        total_in = sum(result.tokens_in for result in results)
        total_out = sum(result.tokens_out for result in results)
        # Battery-wide cost per reasoner: the number an ablation compares across arms. Attempts that
        # recorded no roles simply contribute nothing here, so this can total less than `tokens_in`
        # above - which is why the two are kept as separate lines rather than one being derived.
        tokens_by_role: dict[str, dict[str, int]] = {}
        for result in results:
            for name, row in (result.tokens_by_role or {}).items():
                into = tokens_by_role.setdefault(
                    name, {"tokens_in": 0, "tokens_out": 0, "calls": 0})
                for field_name in into:
                    into[field_name] += int(row.get(field_name) or 0)
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": round(
                self._prior_wall_seconds + time.monotonic() - self._started_at,
                1,
            ),
            "coordinator": self.coordinator_url,
            "tries": self.tries,
            "concurrency": self.concurrency if self.concurrency is not None else "auto",
            "concurrency_mode": "fixed" if self.concurrency is not None else "auto",
            "concurrency_limit": self.concurrency,
            "peak_workers": self._peak_workers,
            "time_limit_minutes": self.time_limit_minutes,
            "max_steps": self.max_steps,
            "arm": self.arm,
            "context_policy": self.context_policy,
            "total_attempts": len(results),
            "total_successes": sum(1 for r in results if r.success),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "tokens_total": total_in + total_out,
            "tokens_by_role": {name: tokens_by_role[name]
                               for name in scan.sorted_roles(tokens_by_role)},
            "llm_calls": sum(result.llm_calls for result in results),
            "prompts": sorted(by_prompt.values(), key=lambda row: row["prompt_id"]),
            "attempts": [asdict(result) for result in results],
        }

        summary_path = self.output_dir / "summary.json"
        _write_json_atomic(summary_path, summary)
        _log(
            f"{summary['total_successes']}/{summary['total_attempts']} attempt(s) succeeded "
            f"in {summary['wall_seconds']}s, tokens in/out {total_in}/{total_out} -> {summary_path}"
        )
        return summary


def _log(message: str) -> None:
    print(f"[sari-bench] {message}", flush=True)


async def async_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(argv)
    config = None
    if config_args.config:
        try:
            config = load_run_config(config_args.config)
        except RunConfigError as error:
            config_parser.error(str(error))

    def configured(key: str, fallback: Any = None) -> Any:
        return config.get("bench", key, fallback) if config else fallback

    parser = argparse.ArgumentParser(description="Run a prompt battery across a sandbox fleet.")
    parser.add_argument(
        "--config",
        help="TOML run configuration. Explicit command-line flags override configured values.",
    )
    parser.add_argument("--prompts", required=not configured("prompts"), type=Path,
                        default=Path(configured("prompts")) if configured("prompts") else None,
                        help="Prompt battery JSON.")
    parser.add_argument("--tries", type=int, default=configured("tries", 3),
                        help="Attempts per prompt.")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=configured("time_limit", 40.0),
        help="Minutes for the WHOLE attempt, after which the harness kills it (+grace).",
    )
    parser.add_argument(
        "--per-leg-minutes",
        type=float,
        default=configured("per_leg_minutes"),
        help="The agent's own per-leg cap (--max-minutes). Defaults to --time-limit, which is only "
             "sensible for single-leg tasks: set it lower for a long attempt limit so the agent "
             "can hit its own time_cap and write a summary.json instead of being SIGKILLed.",
    )
    parser.add_argument(
        "--coordinator",
        default=configured("coordinator", f"ws://localhost:{DEFAULT_COORDINATOR_PORT}"),
        help="Coordinator URL. /bench is appended if missing.",
    )
    parser.add_argument(
        "--sandbox-startup-timeout",
        type=float,
        default=configured("sandbox_startup_timeout", 0.0),
        help="Seconds to wait for a usable registered sandbox before failing (0 waits indefinitely).",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path(configured("output_dir"))
                        if configured("output_dir") else None,
                        help="Defaults to bench_runs/<timestamp>.")
    parser.add_argument(
        "--name",
        default=configured("name", ""),
        help="Name this battery: results land in bench_runs/<timestamp>_<name> instead of "
             "bench_runs/<timestamp>. The timestamp stays so runs still sort and never collide. "
             "Anything outside letters, digits, dot, dash and underscore becomes a dash. Cannot be "
             "combined with --output-dir, which already says where the results go.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=configured("resume", False),
        help="Resume a compatible existing --output-dir, skipping finished attempts.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=configured("concurrency"),
        help="Maximum attempts to run in parallel. Defaults to all registered sandboxes and grows "
             "automatically as new sandboxes join.",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=configured("capture_interval", capture.DEFAULT_INTERVAL_SECONDS),
        help="Seconds between saved observations; recent agent step frames suppress redundant "
             "captures (default: 0.25, or 4 frames/second; 0 disables supplementary capture).",
    )
    parser.add_argument("--only", default=configured("only"),
                        help="Comma-separated prompt ids to run.")
    parser.add_argument("--max-steps", type=int, default=configured("max_steps", 150),
                        help="Per-leg step cap for the agent.")
    parser.add_argument("--arm", choices=["vlm", "graph", "graph-advised"],
                        default=configured("arm", "graph"))
    parser.add_argument(
        "--context-policy",
        choices=CONTEXT_POLICY_NAMES,
        default=configured("context_policy", "baseline"),
        help="named context-window policy passed to every agent attempt",
    )
    parser.add_argument("--map-dir", default=configured("map_dir"),
                        help="slamtest output dir the agent loads its map from.")
    parser.add_argument(
        "--ocr-url",
        default=configured("ocr_url"),
        help="OCR service base URL. Resolution: this flag, $SARI_OCR_URL, then "
             "http://127.0.0.1:9100.",
    )
    parser.add_argument("--leg-retries", type=int, default=configured("leg_retries", 1))
    parser.add_argument(
        "--completion-guard",
        choices=["deterministic", "vlm"],
        default=configured("completion_guard", "deterministic"),
        help="Pickup completion guard passed to the agent (default deterministic).",
    )
    args = parser.parse_args(argv)

    prompts = load_prompts(args.prompts)
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        unknown = wanted - {prompt.id for prompt in prompts}
        if unknown:
            parser.error(f"--only names unknown prompt ids: {sorted(unknown)}")
        prompts = [prompt for prompt in prompts if prompt.id in wanted]

    if args.tries < 1:
        parser.error("--tries must be at least 1")
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.sandbox_startup_timeout < 0:
        parser.error("--sandbox-startup-timeout cannot be negative")
    if args.capture_interval < 0:
        parser.error("--capture-interval cannot be negative")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires an explicit --output-dir")
    if args.map_dir:
        # Check the map BEFORE leasing anything. A map dir missing its topology kills the agent on
        # its first StoreMap load, which the harness can only see as a generic agent_error - a
        # whole battery of them, one sandbox lease each, before anyone thinks to read a log.
        missing = [
            name for name in ("topology_final_shelf.json", "annotations_final_shelf.json")
            if not (Path(args.map_dir) / name).is_file()
        ]
        if missing:
            parser.error(f"--map-dir {args.map_dir} is not a usable map: missing {', '.join(missing)}")

    # A name is a suffix on the timestamp, never a replacement for it: two batteries named the same
    # thing on the same day have to be two directories, and the dashboard's list still sorts by when.
    name = battery_dir_name(args.name)
    if args.name and not name:
        parser.error(f"--name {args.name!r} has no usable characters for a directory name")
    if name and args.output_dir is not None:
        parser.error("--name and --output-dir both name the results directory; pass one")
    output_dir = args.output_dir or (
        REPO_ROOT / "bench_runs"
        / (datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{name}" if name else ""))
    )

    runner = BenchmarkRunner(
        prompts=prompts,
        coordinator_url=args.coordinator,
        output_dir=output_dir,
        tries=args.tries,
        time_limit_minutes=args.time_limit,
        per_leg_minutes=args.per_leg_minutes,
        concurrency=args.concurrency,
        max_steps=args.max_steps,
        arm=args.arm,
        context_policy=args.context_policy,
        map_dir=args.map_dir,
        leg_retries=max(0, args.leg_retries),
        completion_guard=args.completion_guard,
        ocr_url=args.ocr_url,
        sandbox_startup_timeout=args.sandbox_startup_timeout,
        capture_interval=args.capture_interval,
        resume=args.resume,
    )
    await runner.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except SandboxStartupError as error:
        _log(f"error: {error}")
        return 1
    except ResumeError as error:
        _log(f"error: {error}")
        return 2
    except OcrPreflightError as error:
        _log(f"error: {error}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
