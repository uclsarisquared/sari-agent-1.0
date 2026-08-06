import json
from types import SimpleNamespace

from orchestrator import leg_artifacts, leg_runner


class _StoreMap:
    by_id = {1: object()}

    @staticmethod
    def nearest_checkpoint(_position):
        return 1

    @staticmethod
    def hops(_source, _target):
        return 0


class _Agent:
    nav_mode = "graph"
    vlm_agent = SimpleNamespace(
        semantic_log=SimpleNamespace(since=lambda _before: ["new memory"]),
        extractable_json_structured_output=r"(?s)(.*)",
    )

    def __init__(self, responses=None):
        self.requests = []
        self.responses = list(responses or [])

    @staticmethod
    def begin_leg(_candidates, _target_name, _leg_idx):
        return 4

    def execute_lean(self, request, step):
        self.requests.append((request, step))
        if self.responses:
            return self.responses.pop(0)
        return {
            "halt": True,
            "agent_mode": "perception",
            "text": "STOP",
        }


def test_one_step_stop_returns_the_stable_metrics_contract(monkeypatch):
    state = {
        "translation": (0, 0, 0),
        "leftGrippedState": False,
        "rightGrippedState": False,
        "leftHoveredObject": "None",
        "rightHoveredObject": "None",
    }
    monkeypatch.setattr(leg_runner, "_fresh_agent_state", lambda: dict(state))
    monkeypatch.setattr(
        leg_runner, "_REQUEST_SCREENSHOT_", lambda: {"image": b"frame"}
    )
    agent = _Agent()

    result = leg_runner._run_leg_impl(
        agent,
        {"type": "goto", "text": "Go to the shelf", "candidates": [1]},
        _StoreMap(),
        (3, 1.0),
        completion_guard="none",
        visited=set(),
        leg_idx=1,
    )

    assert result["success"] is True
    assert result["end_reason"] == "halt_granted"
    assert result["timesteps"] == 1
    assert result["llm_calls"] == 3
    assert result["new_semantic_entries"] == ["new memory"]
    assert result["final_state"]["nearest_checkpoint"] == 1
    request, step = agent.requests[0]
    assert step == 1
    assert request["nav_goal"] == "Go to the shelf"
    assert "visited_checkpoints" not in request["state"]


def test_action_step_uses_session_state_and_event_assembler(monkeypatch, tmp_path):
    state = {
        "translation": (0, 0, 0),
        "leftGrippedState": False,
        "rightGrippedState": False,
        "leftHoveredObject": "None",
        "rightHoveredObject": "None",
    }
    monkeypatch.setattr(leg_runner, "_fresh_agent_state", lambda: dict(state))
    monkeypatch.setattr(
        leg_runner, "_REQUEST_SCREENSHOT_", lambda: {"image": b"frame"}
    )
    monkeypatch.setattr(leg_artifacts, "downscale_for_storage", lambda data: data)
    agent = _Agent(
        [
            {
                "halt": False,
                "agent_mode": "perception",
                "text": '{"actions": [], "times": [], "notes": {"status": "looking"}}',
            },
            {"halt": True, "agent_mode": "perception", "text": "STOP"},
        ]
    )
    log_path = tmp_path / "leg.jsonl"

    result = leg_runner._run_leg_impl(
        agent,
        {"type": "goto", "text": "Go to the shelf", "candidates": [1]},
        _StoreMap(),
        (3, 1.0),
        log_path=log_path,
        completion_guard="none",
        visited=set(),
        leg_idx=3,
    )

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    step_row = next(row for row in rows if row["event"] == "step")
    assert result["timesteps"] == 2
    assert step_row["leg"] == 3
    assert step_row["step"] == 1
    assert step_row["actions"] == []
    assert step_row["status"] == "looking"
