"""Runs a prompt battery across a fleet of leased sandboxes.

Work is the cross product of prompts and attempts. Each unit of work leases a sandbox, runs one
agent subprocess against it, and hands it back - the coordinator resets it before it is used again,
so no attempt inherits state from the one before it. Concurrency is bounded by ``--concurrency``
workers and, above that, by how many sandboxes the pool actually has.

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
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sari_bench.client import CoordinatorClient, Lease, SandboxLost
from sari_bench.protocol import DEFAULT_COORDINATOR_PORT, STATE_READY

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


class SandboxStartupError(RuntimeError):
    """The configured coordinator has no usable fleet at runner startup."""

# Per-attempt manifest, written INTO the run dir before the agent is spawned. attempts.jsonl only
# gains a row when an attempt finishes, so this is the only record that an attempt is in flight -
# it is what `sari_bench watch` discovers runs by, and it survives the runner dying.
ATTEMPT_MANIFEST = "attempt.json"
# Battery-level manifest at the output-dir root.
BATTERY_MANIFEST = "battery.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Writes JSON via a temp file + rename, so a reader polling this path never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temp, path)


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
    success: bool = False
    end_reason: str = ""
    sandbox_id: str = ""
    commands_uri: str = ""
    exit_code: int | None = None
    wall_seconds: float = 0.0
    run_dir: str = ""
    requeues: int = 0
    error: str = ""
    legs: dict[str, Any] = field(default_factory=dict)
    # Token cost of the attempt: prompt tokens in, completion tokens out, across every reasoner the
    # agent ran (agent_core.token_meter). Zero means "the agent recorded none", which for a crashed
    # attempt can also mean it died before its first tokens.json write.
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0


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
        concurrency: int,
        max_steps: int,
        arm: str,
        map_dir: str | None,
        leg_retries: int,
        per_leg_minutes: float | None = None,
        timeout_grace: float = DEFAULT_TIMEOUT_GRACE_SECONDS,
        sandbox_startup_timeout: float = 0.0,
        python_executable: str | None = None,
        agent_entry: str = ORCHESTRATOR_ENTRY,
        agent_cwd: Path = OVERHAUL_DIR,
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
        self.map_dir = map_dir
        self.leg_retries = leg_retries
        self.timeout_grace = timeout_grace
        self.sandbox_startup_timeout = sandbox_startup_timeout
        self.python_executable = python_executable or sys.executable
        # Overridable so tests can drive the whole lease/spawn/release cycle against a stub agent
        # instead of the real orchestrator (which pulls the entire model stack on import).
        self.agent_entry = agent_entry
        self.agent_cwd = agent_cwd

        self._queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
        self._results: list[AttemptResult] = []
        self._results_lock = asyncio.Lock()
        self._started_at = 0.0

    async def run(self) -> dict[str, Any]:
        await self._wait_for_registered_sandbox()

        for prompt in self.prompts.values():
            for attempt in range(1, self.tries + 1):
                self._queue.put_nowait((prompt.id, attempt, 0))

        # Say plainly whether this battery is starting clean or landing in a directory that already
        # holds results - a reused dir mixes old attempts into attempts.jsonl and the watcher's view.
        reused = self.output_dir.exists() and any(self.output_dir.iterdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if reused:
            _log(f"REUSING existing output dir {self.output_dir} (it already holds results; new "
                 f"attempts append to attempts.jsonl and prior run dirs are rotated aside)")
        else:
            _log(f"new output dir {self.output_dir}")

        self._started_at = time.monotonic()
        total = self._queue.qsize()
        _log(f"{len(self.prompts)} prompt(s) x {self.tries} attempt(s) = {total} run(s)")
        self._write_battery_manifest(total)

        workers = [asyncio.create_task(self._worker(i)) for i in range(self.concurrency)]
        try:
            await self._queue.join()
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        return self._write_summary()

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
        _write_json_atomic(
            self.output_dir / "battery.json",
            {
                "battery_id": self.output_dir.name,
                "output_dir": str(self.output_dir),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "coordinator": self.coordinator_url,
                "tries": self.tries,
                "concurrency": self.concurrency,
                "time_limit_minutes": self.time_limit_minutes,
                "per_leg_minutes": self.per_leg_minutes,
                "max_steps": self.max_steps,
                "arm": self.arm,
                "planned_attempts": planned_attempts,
                "prompts": [asdict(prompt) for prompt in self.prompts.values()],
            },
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
            result.wall_seconds = round(time.monotonic() - started, 1)
            result.requeues = requeues
            result.run_dir = str(run_dir)
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
                    "ended_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            await self._record(result)
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
                "prompt_id": prompt.id,
                "prompt": prompt.prompt,
                "family": prompt.family,
                "looking_for": prompt.looking_for,
                "attempt": attempt,
                "arm": self.arm,
                "sandbox_id": lease.sandbox_id,
                "commands_uri": lease.commands_uri,
                "lease_id": getattr(lease, "lease_id", ""),
                "run_dir": str(run_dir),
                "state": "starting",
                "pid": None,
                "started_at": datetime.fromtimestamp(started_wall).isoformat(timespec="seconds"),
                "started_epoch": round(started_wall, 3),
                "deadline_epoch": round(started_wall + timeout, 3),
                "time_limit_minutes": self.time_limit_minutes,
                "per_leg_minutes": self.per_leg_minutes,
                "max_steps": self.max_steps,
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
            _patch_json(manifest_path, {"pid": process.pid, "state": "running"})

            wait_task = asyncio.create_task(process.wait())
            lost_task = asyncio.create_task(client.wait_for_sandbox_lost(lease))

            try:
                done, pending = await asyncio.wait(
                    {wait_task, lost_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in (wait_task, lost_task):
                    if not task.done():
                        task.cancel()

            if lost_task in done:
                await self._kill(process)
                raise lost_task.result()

            if wait_task not in done:
                _log(f"{prompt.id} try {attempt}: exceeded {timeout:.0f}s; terminating")
                await self._kill(process)
                return self._result_from_run_dir(
                    prompt, attempt, run_dir, exit_code=None, outcome="harness_timeout"
                )

            exit_code = wait_task.result()

        outcome = "completed" if exit_code == 0 else "agent_error"
        if outcome == "agent_error" and _manifest_field(run_dir / ATTEMPT_MANIFEST, "killed_by"):
            # An operator killing a collapsed attempt from the dashboard reaches the agent as a
            # signal, which looks exactly like a crash from here. Don't score it as one.
            outcome = "operator_kill"
        return self._result_from_run_dir(prompt, attempt, run_dir, exit_code=exit_code, outcome=outcome)

    def _agent_command(self, prompt: Prompt, lease: Lease, run_dir: Path) -> list[str]:
        command = [
            self.python_executable,
            self.agent_entry,
            "--task",
            prompt.prompt,
            "--arm",
            self.arm,
            "--run-dir",
            str(run_dir),
            "--max-steps",
            str(self.max_steps),
            "--max-minutes",
            str(self.per_leg_minutes),
            "--leg-retries",
            str(self.leg_retries),
            "--ws-uri",
            lease.commands_uri,
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

        result.success = bool(summary.get("success"))
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
        if not result.llm_calls:
            result.llm_calls = int(source.get("calls") or 0)

    async def _record(self, result: AttemptResult) -> None:
        async with self._results_lock:
            self._results.append(result)
            # Written incrementally so a battery interrupted after six hours still leaves usable
            # results behind.
            with (self.output_dir / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def _write_summary(self) -> dict[str, Any]:
        by_prompt: dict[str, dict[str, Any]] = {}
        for result in self._results:
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

        total_in = sum(result.tokens_in for result in self._results)
        total_out = sum(result.tokens_out for result in self._results)
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "wall_seconds": round(time.monotonic() - self._started_at, 1),
            "coordinator": self.coordinator_url,
            "tries": self.tries,
            "concurrency": self.concurrency,
            "time_limit_minutes": self.time_limit_minutes,
            "max_steps": self.max_steps,
            "arm": self.arm,
            "total_attempts": len(self._results),
            "total_successes": sum(1 for r in self._results if r.success),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "tokens_total": total_in + total_out,
            "llm_calls": sum(result.llm_calls for result in self._results),
            "prompts": sorted(by_prompt.values(), key=lambda row: row["prompt_id"]),
            "attempts": [asdict(result) for result in self._results],
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(
            f"{summary['total_successes']}/{summary['total_attempts']} attempt(s) succeeded "
            f"in {summary['wall_seconds']}s, tokens in/out {total_in}/{total_out} -> {summary_path}"
        )
        return summary


def _log(message: str) -> None:
    print(f"[sari-bench] {message}", flush=True)


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a prompt battery across a sandbox fleet.")
    parser.add_argument("--prompts", required=True, type=Path, help="Prompt battery JSON.")
    parser.add_argument("--tries", type=int, default=3, help="Attempts per prompt.")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=40.0,
        help="Minutes for the WHOLE attempt, after which the harness kills it (+grace).",
    )
    parser.add_argument(
        "--per-leg-minutes",
        type=float,
        default=None,
        help="The agent's own per-leg cap (--max-minutes). Defaults to --time-limit, which is only "
             "sensible for single-leg tasks: set it lower for a long attempt limit so the agent "
             "can hit its own time_cap and write a summary.json instead of being SIGKILLed.",
    )
    parser.add_argument(
        "--coordinator",
        default=f"ws://localhost:{DEFAULT_COORDINATOR_PORT}",
        help="Coordinator URL. /bench is appended if missing.",
    )
    parser.add_argument(
        "--sandbox-startup-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for a usable registered sandbox before failing (0 waits indefinitely).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to bench_runs/<timestamp>.")
    parser.add_argument("--concurrency", type=int, default=2, help="Attempts to run in parallel.")
    parser.add_argument("--only", default=None, help="Comma-separated prompt ids to run.")
    parser.add_argument("--max-steps", type=int, default=150, help="Per-leg step cap for the agent.")
    parser.add_argument("--arm", choices=["vlm", "graph", "graph-advised"], default="graph")
    parser.add_argument("--map-dir", default=None, help="slamtest output dir the agent loads its map from.")
    parser.add_argument("--leg-retries", type=int, default=1)
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
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.sandbox_startup_timeout < 0:
        parser.error("--sandbox-startup-timeout cannot be negative")
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

    output_dir = args.output_dir or (
        REPO_ROOT / "bench_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
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
        map_dir=args.map_dir,
        leg_retries=max(0, args.leg_retries),
        sandbox_startup_timeout=args.sandbox_startup_timeout,
    )
    await runner.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except SandboxStartupError as error:
        _log(f"error: {error}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
