from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sari_bench.runner import BenchmarkRunner, async_main
from sari_runconfig import RunConfigError, load_run_config


def test_loader_resolves_paths_from_the_config_and_rejects_typos(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "run.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
[agent]
arm = "graph-advised"

[environment]
map_dir = "../maps/frozen"

[bench]
prompts = "../prompts/battery.json"
tries = 2
""",
        encoding="utf-8",
    )

    config = load_run_config(config_path)
    assert config.get("agent", "arm") == "graph-advised"
    assert config.get("environment", "map_dir") == str(tmp_path / "maps" / "frozen")
    assert config.get("bench", "prompts") == str(tmp_path / "prompts" / "battery.json")

    config_path.write_text("[agent]\ncompletion_gard = \"vlm\"\n", encoding="utf-8")
    with pytest.raises(RunConfigError, match=r"agent\.completion_gard"):
        load_run_config(config_path)


@pytest.mark.parametrize(
    "body, message",
    [
        ('[agent]\narm = "magic"\n', "agent.arm must be one of"),
        ("[limits]\nmax_steps = true\n", "limits.max_steps must be an integer"),
        ("[bench]\ntries = 0\n", "bench.tries must be at least 1"),
        ("[mystery]\nvalue = 1\n", r"unknown section\(s\)"),
    ],
)
def test_loader_rejects_invalid_values(tmp_path: Path, body: str, message: str) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(body, encoding="utf-8")
    with pytest.raises(RunConfigError, match=message):
        load_run_config(config_path)


def test_bench_uses_config_and_explicit_cli_flags_win(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"id": "p1", "prompt": "Pick it up"}]), encoding="utf-8")
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[bench]
prompts = "prompts.json"
tries = 2
time_limit = 90.0
per_leg_minutes = 12.0
max_steps = 77
arm = "vlm"
completion_guard = "vlm"
name = "from-config"
""",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    async def fake_run(runner: BenchmarkRunner) -> dict:
        seen.update(
            tries=runner.tries,
            time_limit=runner.time_limit_minutes,
            per_leg_minutes=runner.per_leg_minutes,
            max_steps=runner.max_steps,
            arm=runner.arm,
            completion_guard=runner.completion_guard,
        )
        return {}

    with patch.object(BenchmarkRunner, "run", fake_run):
        result = asyncio.run(
            async_main(
                [
                    "--config",
                    str(config_path),
                    "--tries",
                    "4",
                    "--arm",
                    "graph",
                ]
            )
        )

    assert result == 0
    assert seen == {
        "tries": 4,
        "time_limit": 90.0,
        "per_leg_minutes": 12.0,
        "max_steps": 77,
        "arm": "graph",
        "completion_guard": "vlm",
    }


def test_standalone_agent_uses_config_and_explicit_cli_flags_win(tmp_path: Path) -> None:
    from overhaul.orchestrator import subtask_agents

    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[agent]
arm = "vlm"
resolver_backend = "claude-cli"
completion_guard = "vlm"
leg_retries = 3

[limits]
max_steps = 21
max_minutes = 6.5

[environment]
map_dir = "map"
reset_start = true

[output]
run_dir = "runs/one"
summary = "runs/one/result.json"
""",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    def fake_orchestrate(task, **kwargs):
        seen["task"] = task
        seen.update(kwargs)

    with patch.object(subtask_agents, "orchestrate", fake_orchestrate):
        subtask_agents.main(
            [
                "--config",
                str(config_path),
                "--task",
                "configured run",
                "--max-steps",
                "42",
                "--no-reset-start",
            ]
        )

    assert seen["task"] == "configured run"
    assert seen["arm"] == "vlm"
    assert seen["caps"] == (42, 6.5)
    assert seen["resolver_backend"] == "claude-cli"
    assert seen["completion_guard"] == "vlm"
    assert seen["leg_retries"] == 3
    assert seen["output_dir"] == str(tmp_path / "map")
    assert seen["run_dir"] == str(tmp_path / "runs" / "one")
    assert seen["out"] == str(tmp_path / "runs" / "one" / "result.json")
    assert seen["reset_start"] is False
