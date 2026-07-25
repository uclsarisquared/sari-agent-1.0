"""Runner tests: the full lease -> spawn -> release cycle against a stub agent.

The stub stands in for orchestrator/subtask_agents.py (importing the real one pulls the whole model
stack). It writes the same summary.json the orchestrator does, so the result-folding path is
exercised for real. What is being pinned down:

  1. prompts x tries all run, and each attempt lands in its own run dir;
  2. concurrency is bounded by the POOL, not just by --concurrency - two sandboxes means at most
     two agents alive at once;
  3. an attempt that overruns its time limit is killed and recorded, not left hanging;
  4. a crashed agent still releases its sandbox, so the battery finishes.

    python sari_bench/tests/test_runner.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sari_bench.coordinator import Coordinator
from sari_bench.runner import BenchmarkRunner, Prompt, load_prompts
from sari_bench.tests.test_coordinator import FakeSandbox

# Records its own liveness window so the test can prove two agents overlapped (or did not), and
# honours the run-dir contract the runner reads results back out of.
STUB_AGENT = '''
import json, os, sys, time

run_dir = sys.argv[sys.argv.index("--run-dir") + 1]
task = sys.argv[sys.argv.index("--task") + 1]
os.makedirs(run_dir, exist_ok=True)

with open(os.path.join(run_dir, "liveness.json"), "w") as f:
    json.dump({"start": time.time(), "uri": os.environ.get("SARI_WS_URI", "")}, f)

mode = os.environ.get("STUB_MODE", "ok")
if mode == "crash":
    sys.exit(3)
if mode == "hang":
    time.sleep(600)

time.sleep(float(os.environ.get("STUB_SLEEP", "0.3")))
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump({"task": task, "success": True, "legs_planned": 1, "legs_completed": 1,
               "legs": [{"end_reason": "halt_granted", "success": True}]}, f)
with open(os.path.join(run_dir, "liveness.json"), "r+") as f:
    data = json.load(f); data["end"] = time.time()
    f.seek(0); json.dump(data, f); f.truncate()
'''


async def _start_coordinator() -> tuple[Coordinator, str]:
    coordinator = Coordinator(host="127.0.0.1", port=0, log=lambda _m: None)
    await coordinator.start()
    return coordinator, f"ws://127.0.0.1:{coordinator.bound_port}"


def _runner(url: str, prompts: list[Prompt], workspace: Path, **kwargs) -> BenchmarkRunner:
    stub = workspace / "stub_agent.py"
    stub.write_text(STUB_AGENT, encoding="utf-8")
    options = dict(
        tries=1,
        time_limit_minutes=1.0,
        concurrency=2,
        max_steps=10,
        arm="graph",
        map_dir=None,
        leg_retries=0,
        timeout_grace=1.0,
    )
    options.update(kwargs)
    return BenchmarkRunner(
        prompts=prompts,
        coordinator_url=url,
        output_dir=workspace / "runs",
        agent_entry=str(stub),
        agent_cwd=workspace,
        **options,
    )


def test_load_prompts_accepts_the_battery_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "battery.json"
        path.write_text(
            json.dumps(
                {
                    "prompts": [
                        {"id": "p1", "family": "pickup", "prompt": "Pick up soy sauce",
                         "looking_for": "a bottle"},
                        {"id": "p2", "prompt": "Go to the checkout"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        prompts = load_prompts(path)
        assert [p.id for p in prompts] == ["p1", "p2"]
        assert prompts[0].family == "pickup"

        # A bare list, and bare strings, are both accepted.
        path.write_text(json.dumps(["do a thing", {"prompt": "do another"}]), encoding="utf-8")
        assert [p.id for p in load_prompts(path)] == ["prompt_00", "prompt_01"]

        path.write_text(json.dumps([{"id": "dupe", "prompt": "a"}, {"id": "dupe", "prompt": "b"}]),
                        encoding="utf-8")
        try:
            load_prompts(path)
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("duplicate prompt ids were accepted")
    print("ok  prompt batteries load in every documented shape")


async def test_battery_runs_every_prompt_and_attempt() -> None:
    coordinator, url = await _start_coordinator()
    sandboxes = [FakeSandbox("sandbox-a", 51001), FakeSandbox("sandbox-b", 51002)]
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            for sandbox in sandboxes:
                await sandbox.connect(url)

            prompts = [Prompt(id="p1", prompt="task one"), Prompt(id="p2", prompt="task two")]
            summary = await _runner(url, prompts, workspace, tries=2).run()

            assert summary["total_attempts"] == 4, summary["total_attempts"]
            assert summary["total_successes"] == 4, summary["total_successes"]
            assert {row["prompt_id"] for row in summary["prompts"]} == {"p1", "p2"}
            assert all(row["success_rate"] == 1.0 for row in summary["prompts"])
            assert all(row["end_reason"] == "halt_granted" for row in summary["attempts"])

            # Every attempt got its own directory, and the agent was pointed at a real sandbox.
            for prompt_id in ("p1", "p2"):
                for attempt in (1, 2):
                    liveness = workspace / "runs" / prompt_id / f"try{attempt:02d}" / "liveness.json"
                    assert liveness.exists(), liveness
                    assert json.loads(liveness.read_text())["uri"].endswith("/commands")

            # Reset-per-attempt: four attempts over two sandboxes means four resets.
            assert sum(s.reset_count for s in sandboxes) == 4

            # The incremental log is a complete record on its own.
            lines = (workspace / "runs" / "attempts.jsonl").read_text().strip().splitlines()
            assert len(lines) == 4
        finally:
            for sandbox in sandboxes:
                await sandbox.close()
            await coordinator.stop()
    print("ok  every prompt x attempt runs, in its own dir, with a reset between each")


async def test_pool_size_bounds_concurrency() -> None:
    """One sandbox must mean one agent at a time, however many workers are configured."""
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            prompts = [Prompt(id=f"p{i}", prompt=f"task {i}") for i in range(3)]
            os.environ["STUB_SLEEP"] = "0.5"
            try:
                await _runner(url, prompts, workspace, concurrency=3).run()
            finally:
                os.environ.pop("STUB_SLEEP", None)

            windows = sorted(
                (json.loads(path.read_text())["start"], json.loads(path.read_text())["end"])
                for path in (workspace / "runs").rglob("liveness.json")
            )
            assert len(windows) == 3
            for (_, earlier_end), (later_start, _) in zip(windows, windows[1:]):
                assert later_start >= earlier_end, "two agents shared one sandbox"
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  a one-sandbox pool serialises attempts even at concurrency 3")


async def test_overrunning_attempt_is_killed_and_recorded() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "hang"
            try:
                runner = _runner(
                    url,
                    [Prompt(id="p1", prompt="never finishes")],
                    workspace,
                    time_limit_minutes=1.0 / 60.0,  # one second
                    timeout_grace=0.5,
                )
                summary = await asyncio.wait_for(runner.run(), timeout=60)
            finally:
                os.environ.pop("STUB_MODE", None)

            assert summary["total_attempts"] == 1
            assert summary["attempts"][0]["outcome"] == "harness_timeout"
            assert summary["total_successes"] == 0
            # Killed or not, the sandbox has to come back to the pool clean.
            assert sandbox.reset_count == 1
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  an attempt past its time limit is killed, recorded, and its sandbox reset")


async def test_crashed_agent_still_releases_its_sandbox() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51001)
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            await sandbox.connect(url)
            os.environ["STUB_MODE"] = "crash"
            try:
                prompts = [Prompt(id="p1", prompt="crashes"), Prompt(id="p2", prompt="also crashes")]
                summary = await asyncio.wait_for(
                    _runner(url, prompts, workspace, concurrency=1).run(), timeout=60
                )
            finally:
                os.environ.pop("STUB_MODE", None)

            assert summary["total_attempts"] == 2
            assert all(row["outcome"] == "agent_error" for row in summary["attempts"])
            assert all(row["exit_code"] == 3 for row in summary["attempts"])
            # The second attempt only ran because the first released despite crashing.
            assert sandbox.reset_count == 2
        finally:
            await sandbox.close()
            await coordinator.stop()
    print("ok  a crashed agent still releases, so the battery finishes")


async def main() -> int:
    test_load_prompts_accepts_the_battery_schema()
    for test in (
        test_battery_runs_every_prompt_and_attempt,
        test_pool_size_bounds_concurrency,
        test_overrunning_attempt_is_killed_and_recorded,
        test_crashed_agent_still_releases_its_sandbox,
    ):
        await test()
    print("\nAll runner tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
