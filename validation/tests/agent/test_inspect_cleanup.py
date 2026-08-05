"""Offline coverage for inspect-leg cleanup on every runner exit path."""
import os
import sys
import json

import pytest

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from orchestrator import leg_runner as SA
from orchestrator.held_item_inspection import _inspection_macro_summary


class FakeAgent:
    def __init__(self, cleanup=None, error=None):
        self._hand_pose = None
        self.cleanup = cleanup or {"restored": True, "hands": {}}
        self.error = error
        self.calls = 0

    def _restore_hands_after_inspection(self):
        self.calls += 1
        if self.error:
            raise self.error
        self._hand_pose = "rest" if self.cleanup["restored"] else None
        return self.cleanup


@pytest.fixture
def runner_stubs(monkeypatch):
    monkeypatch.setattr(
        SA, "_fresh_agent_state",
        lambda: {
            "leftTranslation": ("rest", "left"),
            "rightTranslation": ("rest", "right"),
            "leftRotation": (0, 0, 0),
            "rightRotation": (0, 0, 0),
            "leftGrippedState": True,
            "rightGrippedState": False,
        })


@pytest.mark.parametrize("end_reason", [
    "halt_granted", "completed_no_stop", "step_cap", "time_cap", "errors", "halt_forced",
])
def test_all_normal_inspect_exits_restore_and_refresh_final_state(
        monkeypatch, runner_stubs, end_reason):
    result = {
        "end_reason": end_reason,
        "final_state": {
            "leftTranslation": ("displaced",),
            "leftRotation": (0, 90, 0),
            "sticky_metric": "preserved",
        },
    }
    monkeypatch.setattr(SA, "_run_leg_impl", lambda *args, **kwargs: result)
    agent = FakeAgent()

    returned = SA.run_leg(agent, {"type": "inspect"}, None, (1, 1))

    assert returned is result
    assert agent.calls == 1
    assert agent._hand_pose == "rest"
    assert returned["inspection_cleanup"]["restored"] is True
    assert returned["final_state"]["leftTranslation"] == ("rest", "left")
    assert returned["final_state"]["rightTranslation"] == ("rest", "right")
    assert returned["final_state"]["leftRotation"] == (0, 0, 0)
    assert returned["final_state"]["rightRotation"] == (0, 0, 0)
    assert returned["final_state"]["leftGrippedState"] is True
    assert returned["final_state"]["sticky_metric"] == "preserved"


def test_thrown_exception_attempts_cleanup_and_preserves_original_error(monkeypatch, runner_stubs):
    def fail(*args, **kwargs):
        raise RuntimeError("original leg failure")

    monkeypatch.setattr(SA, "_run_leg_impl", fail)
    agent = FakeAgent()

    with pytest.raises(RuntimeError, match="original leg failure"):
        SA.run_leg(agent, {"type": "inspect"}, None, (1, 1))
    assert agent.calls == 1
    assert agent._hand_pose == "rest"


def test_cleanup_failure_does_not_mask_result_and_leaves_pose_unknown(monkeypatch, runner_stubs):
    result = {"end_reason": "step_cap", "final_state": {"leftRotation": (0, 90, 0)}}
    monkeypatch.setattr(SA, "_run_leg_impl", lambda *args, **kwargs: result)
    agent = FakeAgent(error=ValueError("transform stalled"))
    agent._hand_pose = "inspection"

    returned = SA.run_leg(agent, {"type": "inspect"}, None, (1, 1))

    assert returned is result
    assert returned["end_reason"] == "step_cap"
    assert returned["inspection_cleanup"]["restored"] is False
    assert "transform stalled" in returned["inspection_cleanup"]["error"]
    assert agent._hand_pose is None


def test_non_inspect_leg_does_not_run_cleanup(monkeypatch, runner_stubs):
    result = {"end_reason": "halt_granted", "final_state": {}}
    monkeypatch.setattr(SA, "_run_leg_impl", lambda *args, **kwargs: result)
    agent = FakeAgent()

    assert SA.run_leg(agent, {"type": "pickup"}, None, (1, 1)) is result
    assert agent.calls == 0


def test_cleanup_logs_one_summary_without_inverse_rotation_events(
        monkeypatch, runner_stubs, tmp_path):
    result = {"end_reason": "halt_granted", "final_state": {}}
    monkeypatch.setattr(SA, "_run_leg_impl", lambda *args, **kwargs: result)
    cleanup = {
        "restored": True,
        "hands": {
            "left": {"rotation": (0, 0, 0)},
            "right": {"rotation": (0, 0, 0)},
        },
    }
    log_path = tmp_path / "agent.jsonl"

    SA.run_leg(
        FakeAgent(cleanup=cleanup),
        {"type": "inspect"},
        None,
        (1, 1),
        log_path=str(log_path),
        leg_idx=4,
    )

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    rotations = [row for row in rows if row["event"] == "inspect_cleanup_rotation"]
    assert rotations == []
    assert rows[-1]["event"] == "inspect_cleanup"
    assert rows[-1]["restored"] is True


def test_inspection_evidence_frames_never_reach_a_log_row_or_the_prompt():
    """`frame_b64` is a full-resolution screenshot the completion guard consumes IN CODE.

    Two independent sinks must drop it: the macro's own `inspection_macro_end` log row, and the
    model-facing state view (the actor already receives that frame as its image input, so shipping
    it again as base64 text would cost thousands of tokens per step for nothing).
    """
    result = {
        "blocked": False,
        "hand": "left",
        "label_visible": True,
        "label_legible": True,
        "frame_b64": "AAAABBBBCCCC",
        "steps": [{"check_index": 1}],
    }
    summary = _inspection_macro_summary(result)
    assert "frame_b64" not in summary and "steps" not in summary
    assert summary["label_legible"] is True

    view = SA._model_facing_state({
        "last_inspection": dict(result),
        "inspection_evidence": [{"hand": "left", "sku": "COKE", "step": 4}],
        "visited_checkpoints": {1, 2},
    })
    assert "frame_b64" not in view["last_inspection"]
    assert "steps" not in view["last_inspection"]
    assert "visited_checkpoints" not in view
    # The frame-free ledger IS shown: it tells the actor which held item it has already read.
    assert view["inspection_evidence"] == [{"hand": "left", "sku": "COKE", "step": 4}]
