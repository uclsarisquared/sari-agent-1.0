import json

import pytest

from orchestrator import leg_artifacts


def test_leg_artifacts_write_timestamped_json_without_mutating_input(tmp_path):
    log_path = tmp_path / "leg.jsonl"
    record = {"event": "step", "step": 1}

    with leg_artifacts.LegArtifacts(log_path, started_at=0) as artifacts:
        artifacts.log(record)

    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["event"] == "step"
    assert "wall" in row
    assert record == {"event": "step", "step": 1}


def test_leg_artifacts_close_the_log_when_execution_raises(tmp_path):
    artifacts = leg_artifacts.LegArtifacts(tmp_path / "leg.jsonl")

    with pytest.raises(RuntimeError, match="boom"):
        with artifacts:
            raise RuntimeError("boom")

    assert artifacts._log_fh is None


def test_event_logger_assembles_common_leg_and_guard_fields(tmp_path):
    log_path = tmp_path / "leg.jsonl"
    with leg_artifacts.LegArtifacts(log_path) as artifacts:
        events = artifacts.event_logger(4)
        events.guard(
            2,
            "vlm",
            {"match": True, "reason": "visible", "conclusive": True},
            guard="inspect",
            query="expiration date",
        )

    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["event"] == "completion_guard"
    assert row["leg"] == 4
    assert row["step"] == 2
    assert row["guard"] == "inspect"
    assert row["match"] is True
    assert row["query"] == "expiration date"


def test_event_logger_rejects_reserved_field_overrides():
    events = leg_artifacts.LegEventLogger(lambda _row: None, 1)

    with pytest.raises(ValueError, match="reserved keys"):
        events.emit("step", leg=99)


def test_write_step_output_preserves_all_model_channels(tmp_path):
    leg_artifacts.write_step_output(
        tmp_path,
        3,
        {
            "agent_mode": "perception",
            "halt": False,
            "nav_note": "next shelf",
            "semantic": "semantic decision",
            "text": "actor response",
            "episodic": "reflection",
        },
        stamp="_stamp",
    )

    text = (tmp_path / "step03_stamp.txt").read_text(encoding="utf-8")
    assert "next shelf" in text
    assert "semantic decision" in text
    assert "actor response" in text
    assert "reflection" in text
