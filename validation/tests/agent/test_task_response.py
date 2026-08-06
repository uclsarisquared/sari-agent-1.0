"""Offline tests for the task-level response journal and responder fallback."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from orchestrator.task_response import (
    attach_findings,
    bounded_response_view,
    deterministic_fallback,
    finalize_response_memory,
    new_response_memory,
    record_attempt,
    save_response_memory,
    set_planned_subtasks,
    synthesize_response,
    write_response_artifact,
)


def _state(**extra):
    state = {
        "nearest_checkpoint": 17,
        "translation": [1.0, 0.0, 3.0],
        "leftGrippedState": True,
        "rightGrippedState": False,
        "gripped_names": {"left": "PIATTOS_ROADHOUSE_BARBECUE", "right": None},
    }
    state.update(extra)
    return state


def test_multi_attempt_journal_preserves_evidence_answers_and_learned_memory(tmp_path):
    memory = new_response_memory("Pick up the green chips, then tell me their expiration date.")
    plan = [
        {"type": "pickup", "text": "Pick up the green chips", "target": "green chips",
         "candidates": [17]},
        {"type": "inspect", "text": "Read their expiration date", "query": "expiration date"},
    ]
    set_planned_subtasks(memory, plan)
    save_response_memory(tmp_path, memory)

    failed = {
        "success": False,
        "end_reason": "halt_forced",
        "completion_evidence": "",
        "reported_answer": "",
        "new_semantic_entries": "@ leg 1: green chips are at checkpoint 17",
        "final_state": _state(
            leftGrippedState=False,
            gripped_names={"left": None, "right": None},
            last_halt_refused="the held item did not match the requested chips",
        ),
    }
    record_attempt(
        memory, leg_number=1, attempt_number=1, subtask=plan[0], metrics=failed,
        episodic_reflection="The first grab selected the neighbouring bag.",
    )

    pickup = {
        "success": True,
        "end_reason": "halt_granted",
        "completion_evidence": "target chips are held in the left hand",
        "reported_answer": "",
        "new_semantic_entries": "@ leg 1: approach the shelf from checkpoint 17",
        "final_state": _state(),
    }
    record_attempt(
        memory, leg_number=1, attempt_number=2, subtask=plan[0], metrics=pickup,
        episodic_reflection="Approaching straight-on made the correct bag reachable.",
    )
    attach_findings(memory, 1, 2, "The green chips are now held in the left hand.")

    inspection = {
        "success": True,
        "end_reason": "halt_granted",
        "completion_evidence": "inspection complete: the answer was visually verified",
        "reported_answer": "The expiration date is 17/09/26.",
        "new_semantic_entries": "@ leg 2: the date is printed along the upper seam",
        "final_state": _state(last_inspection={
            "hand": "left",
            "label_visible": True,
            "label_legible": True,
            "reason": "date is front-facing",
            "frame_b64": "very bulky screenshot",
            "steps": {"rotate": 4},
        }),
    }
    record_attempt(
        memory, leg_number=2, attempt_number=1, subtask=plan[1], metrics=inspection,
        episodic_reflection="The upper seam became legible after rotating the held bag.",
    )
    finalize_response_memory(memory, success=True, planned_subtasks=plan)
    save_response_memory(tmp_path, memory)

    persisted = json.loads((tmp_path / "response_memory.json").read_text(encoding="utf-8"))
    assert persisted["prompt"].startswith("Pick up")
    assert len(persisted["planned_subtasks"]) == 2
    assert [row["success"] for row in persisted["attempts"]] == [False, True, True]
    assert persisted["attempts"][0]["failure_reason"].startswith("the held item")
    assert persisted["attempts"][1]["completion_evidence"].startswith("target chips")
    assert persisted["attempts"][2]["verified_reported_answer"] == (
        "The expiration date is 17/09/26."
    )
    assert persisted["attempts"][1]["findings"].startswith("The green chips")
    assert persisted["attempts"][2]["semantic_memory_delta"].startswith("@ leg 2")
    assert persisted["latest_episodic_reflection"].startswith("The upper seam")
    assert persisted["final"]["location"] == 17
    assert persisted["final"]["held_items"]["left"] == "PIATTOS_ROADHOUSE_BARBECUE"
    assert persisted["final"]["inspection"]["label_legible"] is True
    assert "frame_b64" not in persisted["final"]["inspection"]
    assert "steps" not in persisted["final"]["inspection"]
    assert "bulky screenshot" not in json.dumps(persisted)


def test_bounded_view_drops_old_narrative_before_required_facts():
    memory = new_response_memory("What date is printed on the held item?")
    plan = [{"type": "inspect", "text": "Read the printed date"}]
    set_planned_subtasks(memory, plan)
    metrics = {
        "success": True,
        "end_reason": "halt_granted",
        "completion_evidence": "visual verifier accepted the answer",
        "reported_answer": "It expires on 02/11/26.",
        "new_semantic_entries": "old semantic narrative " * 1000,
        "final_state": _state(),
    }
    row = record_attempt(
        memory, leg_number=1, attempt_number=1, subtask=plan[0], metrics=metrics,
        episodic_reflection="episodic narrative " * 1000,
    )
    row["findings"] = "findings narrative " * 1000
    finalize_response_memory(memory, success=True)

    view = bounded_response_view(memory, max_chars=1400)
    encoded = json.dumps(view)
    assert view["prompt"] == memory["prompt"]
    assert view["attempts"][0]["success"] is True
    assert view["attempts"][0]["end_reason"] == "halt_granted"
    assert view["attempts"][0]["verified_reported_answer"] == "It expires on 02/11/26."
    assert view["attempts"][0]["failure_reason"] == ""
    assert "old semantic narrative" not in encoded
    assert "episodic narrative" not in encoded
    assert "findings narrative" not in encoded


def test_responder_uses_bounded_journal_and_model_result():
    memory = new_response_memory("Bring the noodles to checkout.")
    finalize_response_memory(memory, success=True)
    captured = {}

    def complete(system, user):
        captured["system"] = system
        captured["user"] = user
        return "Done — the noodles were checked out."

    response, source = synthesize_response(memory, complete)
    assert response == "Done — the noodles were checked out."
    assert source == "model"
    assert "1-3 concise" in captured["system"]
    assert "Bring the noodles to checkout." in captured["user"]


def test_responder_exception_falls_back_to_verified_answer_and_writes_artifact(tmp_path):
    memory = new_response_memory("What is the sugar content?")
    plan = [{"type": "inspect", "text": "Read the sugar content"}]
    set_planned_subtasks(memory, plan)
    record_attempt(
        memory,
        leg_number=1,
        attempt_number=1,
        subtask=plan[0],
        metrics={
            "success": True,
            "end_reason": "halt_granted",
            "completion_evidence": "answer verified from the label",
            "reported_answer": "The label lists 9 g of sugar.",
            "new_semantic_entries": "",
            "final_state": _state(),
        },
    )
    finalize_response_memory(memory, success=True)

    def fail(_system, _user):
        raise RuntimeError("provider unavailable")

    response, source = synthesize_response(memory, fail)
    path = write_response_artifact(tmp_path, response)
    memory["response"] = response
    memory["response_source"] = source
    save_response_memory(tmp_path, memory)

    assert source == "fallback"
    assert response == "The label lists 9 g of sugar."
    assert path.read_text(encoding="utf-8").strip() == response
    persisted = json.loads((tmp_path / "response_memory.json").read_text())
    assert persisted["response"] == path.read_text().strip()
    assert persisted["response_source"] == "fallback"


def test_partial_failure_fallback_never_claims_full_success():
    memory = new_response_memory("Pick up the cereal and check it out.")
    plan = [
        {"type": "pickup", "text": "pick up the cereal"},
        {"type": "checkout", "text": "check out the cereal"},
    ]
    set_planned_subtasks(memory, plan)
    record_attempt(
        memory,
        leg_number=1,
        attempt_number=1,
        subtask=plan[0],
        metrics={
            "success": True,
            "end_reason": "halt_granted",
            "completion_evidence": "the cereal is held",
            "final_state": _state(),
        },
    )
    record_attempt(
        memory,
        leg_number=2,
        attempt_number=1,
        subtask=plan[1],
        metrics={
            "success": False,
            "end_reason": "time_cap",
            "final_state": _state(last_halt_refused=None),
        },
    )
    finalize_response_memory(memory, success=False)

    response = deterministic_fallback(memory)
    assert "completed pick up the cereal" in response
    assert "could not complete check out the cereal" in response
    assert "available time ran out" in response
    assert "completed the requested task" not in response


def test_orchestrate_returns_and_persists_the_same_final_response(tmp_path, monkeypatch, capsys):
    """A simulator-free integration pin for the summary/artifact/CLI contract."""
    from orchestrator import orchestration as orchestrator
    from sim import env as sim_env
    from vision import ocr_client

    leg = {
        "type": "pickup",
        "text": "Pick up the green chips",
        "target": "green chips",
        "candidates": [17],
        "feasible": True,
    }

    class FakeVLM:
        episodic_memory = "The correct bag was reachable from checkpoint 17."

    class FakeAgent:
        vlm_agent = FakeVLM()
        _graph_nav = None

    metrics = {
        "type": "pickup",
        "text": leg["text"],
        "t_manip": 0.1,
        "t_grip": 0.2,
        "t_checkout": None,
        "success": True,
        "timesteps": 1,
        "llm_calls": 3,
        "errors": 0,
        "completion_guard": "deterministic",
        "halts_refused": 0,
        "halt_forced": False,
        "corrective_release": None,
        "end_reason": "halt_granted",
        "completion_evidence": "the requested green chips are held",
        "reported_answer": None,
        "wall_s": 0.2,
        "final_state": _state(),
        "new_semantic_entries": "@ leg 1: green chips are at checkpoint 17",
    }

    monkeypatch.setattr(ocr_client, "resolve_ocr_url", lambda _url=None: "http://ocr.test")
    monkeypatch.setattr(
        ocr_client, "check_ocr_health",
        lambda _url: {"model": "fake-ocr"},
    )
    monkeypatch.setattr(sim_env, "wait_for_ready", lambda: True)
    monkeypatch.setattr(orchestrator, "_llm_client", lambda: object())
    monkeypatch.setattr(orchestrator, "init_logger", lambda **_kwargs: None)
    monkeypatch.setattr(orchestrator, "EmbodiedAgent", lambda **_kwargs: FakeAgent())
    monkeypatch.setattr(orchestrator, "decompose_task", lambda _client, _task: [dict(leg)])
    monkeypatch.setattr(orchestrator, "_load_store_map", lambda _output=None: object())
    monkeypatch.setattr(orchestrator, "make_resolve_call", lambda _backend: object())
    monkeypatch.setattr(
        orchestrator, "plan_legs",
        lambda _sm, _resolve, _subtasks: ([dict(leg)], 0),
    )
    monkeypatch.setattr(orchestrator, "_current_nearest_cp", lambda _sm: 1)
    monkeypatch.setattr(orchestrator, "order_legs", lambda _sm, legs, _start: legs)
    monkeypatch.setattr(orchestrator, "run_leg", lambda *_args, **_kwargs: dict(metrics))
    monkeypatch.setattr(orchestrator.chime, "beep", lambda: None)

    responder_roles = []

    def fake_llm(_client, _system, _user, role):
        responder_roles.append(role)
        return "Done — I picked up the green chips."

    monkeypatch.setattr(orchestrator, "_llm_call", fake_llm)
    monkeypatch.setattr(orchestrator.token_meter, "install", lambda _run_dir=None: None)
    monkeypatch.setattr(orchestrator.token_meter, "dump", lambda _run_dir=None: None)
    monkeypatch.setattr(orchestrator.token_meter, "snapshot", lambda: {})
    monkeypatch.setattr(
        orchestrator.token_meter,
        "delta",
        lambda _before: {"tokens_in": 0, "tokens_out": 0, "calls": 0, "by_role": {}},
    )
    monkeypatch.setattr(
        orchestrator.token_meter,
        "totals",
        lambda: {
            "tokens_in": 10,
            "tokens_out": 5,
            "tokens_total": 15,
            "calls": 1,
            "untracked_calls": 0,
            "by_model": {},
            "by_role": {
                "responder": {"tokens_in": 10, "tokens_out": 5, "calls": 1},
            },
        },
    )

    summary = orchestrator.orchestrate(orchestrator.OrchestrationConfig(
        task="Pick up the green chips",
        run_dir=tmp_path,
        caps=(1, 1.0),
    ))

    expected = "Done — I picked up the green chips."
    persisted_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    memory = json.loads((tmp_path / "response_memory.json").read_text(encoding="utf-8"))
    assert summary["response"] == persisted_summary["response"] == memory["response"] == expected
    assert (tmp_path / "response.txt").read_text(encoding="utf-8").strip() == expected
    assert summary["response_source"] == persisted_summary["response_source"] == "model"
    assert summary["context_policy"] == {
        "semantic_dedupe": None,
        "semantic_dedupe_window": 8,
        "semantic_keep_last": None,
        "findings_max_chars": None,
        "findings_enabled": True,
        "episodic_in_actor": True,
        "actor_image_history": None,
    }
    assert summary["run_config"]["context_policy"] == "baseline"
    assert responder_roles == [orchestrator.token_meter.ROLE_RESPONDER]
    assert summary["llm_calls"] == 5  # decomposer + three per-step reasoners + responder
    assert summary["tokens"]["by_role"]["responder"]["calls"] == 1
    output = capsys.readouterr().out
    assert output.rfind("[RESPONSE]") > output.rfind("[ORCHESTRATOR] task success=")
    assert Path(tmp_path / "response.txt").is_file()
