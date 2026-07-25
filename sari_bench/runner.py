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
from sari_bench.protocol import DEFAULT_COORDINATOR_PORT

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
        timeout_grace: float = DEFAULT_TIMEOUT_GRACE_SECONDS,
        python_executable: str | None = None,
        agent_entry: str = ORCHESTRATOR_ENTRY,
        agent_cwd: Path = OVERHAUL_DIR,
    ) -> None:
        self.prompts = {prompt.id: prompt for prompt in prompts}
        self.coordinator_url = coordinator_url
        self.output_dir = output_dir
        self.tries = tries
        self.time_limit_minutes = time_limit_minutes
        self.concurrency = concurrency
        self.max_steps = max_steps
        self.arm = arm
        self.map_dir = map_dir
        self.leg_retries = leg_retries
        self.timeout_grace = timeout_grace
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
        for prompt in self.prompts.values():
            for attempt in range(1, self.tries + 1):
                self._queue.put_nowait((prompt.id, attempt, 0))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.monotonic()
        total = self._queue.qsize()
        _log(f"{len(self.prompts)} prompt(s) x {self.tries} attempt(s) = {total} run(s)")

        workers = [asyncio.create_task(self._worker(i)) for i in range(self.concurrency)]
        try:
            await self._queue.join()
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        return self._write_summary()

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
            await self._record(result)
            _log(
                f"[w{index}] {prompt_id} try {attempt}: {result.outcome} "
                f"(success={result.success}, {result.wall_seconds}s)"
            )

    async def _spawn_agent(
        self,
        client: CoordinatorClient,
        lease: Lease,
        prompt: Prompt,
        attempt: int,
        run_dir: Path,
    ) -> AttemptResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        command = self._agent_command(prompt, lease, run_dir)

        env = dict(os.environ)
        # How the agent finds its sandbox. sim/env.py reads this for every command's default URI.
        env["SARI_WS_URI"] = lease.commands_uri

        with (run_dir / "agent.log").open("wb") as log_file:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.agent_cwd),
                env=env,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )

            wait_task = asyncio.create_task(process.wait())
            lost_task = asyncio.create_task(client.wait_for_sandbox_lost(lease))
            timeout = self.time_limit_minutes * 60.0 + self.timeout_grace

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
            str(self.time_limit_minutes),
            "--leg-retries",
            str(self.leg_retries),
            "--ws-uri",
            lease.commands_uri,
        ]
        if self.map_dir:
            command += ["--output-dir", self.map_dir]
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
            return result

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            result.error = f"unreadable summary.json: {error}"
            return result

        result.success = bool(summary.get("success"))
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
                },
            )
            row["attempts"] += 1
            row["successes"] += int(result.success)
            row["outcomes"][result.outcome] = row["outcomes"].get(result.outcome, 0) + 1
            if result.end_reason:
                row["end_reasons"][result.end_reason] = row["end_reasons"].get(result.end_reason, 0) + 1
            if result.sandbox_id and result.sandbox_id not in row["sandboxes"]:
                row["sandboxes"].append(result.sandbox_id)

        for row in by_prompt.values():
            row["success_rate"] = round(row["successes"] / row["attempts"], 3) if row["attempts"] else 0.0

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
            "prompts": sorted(by_prompt.values(), key=lambda row: row["prompt_id"]),
            "attempts": [asdict(result) for result in self._results],
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(
            f"{summary['total_successes']}/{summary['total_attempts']} attempt(s) succeeded "
            f"in {summary['wall_seconds']}s -> {summary_path}"
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
        help="Minutes per attempt; also passed to the agent as its per-leg cap.",
    )
    parser.add_argument(
        "--coordinator",
        default=f"ws://localhost:{DEFAULT_COORDINATOR_PORT}",
        help="Coordinator URL. /bench is appended if missing.",
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

    output_dir = args.output_dir or (
        REPO_ROOT / "bench_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    runner = BenchmarkRunner(
        prompts=prompts,
        coordinator_url=args.coordinator,
        output_dir=output_dir,
        tries=args.tries,
        time_limit_minutes=args.time_limit,
        concurrency=args.concurrency,
        max_steps=args.max_steps,
        arm=args.arm,
        map_dir=args.map_dir,
        leg_retries=max(0, args.leg_retries),
    )
    await runner.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
