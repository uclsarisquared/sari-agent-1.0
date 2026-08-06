from types import SimpleNamespace

from orchestrator import leg_completion
from orchestrator.leg_runtime import GripTracker, InspectionEvidence


def _metrics():
    return {
        "llm_calls": 0,
        "success": False,
        "end_reason": None,
        "completion_evidence": None,
        "reported_answer": None,
        "halts_refused": 0,
        "halt_forced": False,
        "corrective_release": None,
    }


def test_disabled_guard_clears_stale_implicit_completion():
    metrics = _metrics()
    controller = leg_completion.CompletionController(
        SimpleNamespace(), {"type": "goto"}, "none", metrics, lambda _row: None, 1
    )
    state = {"goal_check": "stale"}

    guards = controller.prepare(
        state, "frame", 1, InspectionEvidence(), last_actor_text=""
    )
    observation = controller.observe_after_action(state, "", guards, 1)

    assert state["goal_check"] is None
    assert observation.met is False
    assert observation.terminal is False


def test_granted_stop_updates_the_public_result_contract(monkeypatch):
    monkeypatch.setattr(
        leg_completion, "reported_completion_answer", lambda _response: "verified answer"
    )
    monkeypatch.setattr(
        leg_completion, "completion_predicate", lambda *_args, **_kwargs: (True, "verified")
    )
    metrics = _metrics()
    controller = leg_completion.CompletionController(
        SimpleNamespace(), {"type": "inspect"}, "deterministic", metrics,
        lambda _row: None, 2,
    )
    tracker = GripTracker(names={"left": None, "right": None}, start_grips=set())

    status = controller.handle_stop(
        {"halt": True}, {}, "", leg_completion.StepGuards(), 3, tracker
    )

    assert status == "granted"
    assert metrics["success"] is True
    assert metrics["end_reason"] == "halt_granted"
    assert metrics["completion_evidence"] == "verified"
    assert metrics["reported_answer"] == "verified answer"

