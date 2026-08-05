from types import SimpleNamespace

import pytest

from orchestrator import orchestration
from orchestrator.task_response import new_response_memory


def _config(tmp_path, *, retries=1):
    return orchestration._OrchestrationConfig(
        task="Pick up the chips",
        arm="graph",
        caps=(10, 1.0),
        out=None,
        requested_run_dir=str(tmp_path),
        resolver_backend="endpoint",
        reset_start=False,
        restart_env=False,
        leg_retries=retries,
        output_dir=None,
        completion_guard="deterministic",
        ocr_url=None,
        runs_dir=None,
        context_policy="baseline",
    )


def _state(tmp_path, *, retries=1):
    agent = SimpleNamespace(vlm_agent=SimpleNamespace(episodic_memory="remembered"))
    return orchestration._RunState(
        config=_config(tmp_path, retries=retries),
        policy=SimpleNamespace(findings_enabled=False),
        run_dir=str(tmp_path),
        response_memory=new_response_memory("Pick up the chips"),
        agent=agent,
        store_map=object(),
    )


def _metrics(*, success, reason, held=None):
    return {
        "success": success,
        "end_reason": reason,
        "llm_calls": 1,
        "t_grip": None,
        "t_checkout": None,
        "timesteps": 1,
        "halts_refused": 0,
        "wall_s": 0.1,
        "final_state": {
            "last_halt_refused": None if success else "wrong item",
            "gripped_names": held or {},
        },
        "new_semantic_entries": "",
    }


def test_leg_retry_uses_failure_context_and_separate_log(monkeypatch, tmp_path):
    state = _state(tmp_path)
    leg = {"type": "pickup", "text": "Pick up the chips"}
    state.legs = [leg]
    calls = []
    results = [
        _metrics(success=False, reason="halt_refused"),
        _metrics(success=True, reason="halt_granted", held={"left": "chips"}),
    ]

    def fake_run_leg(*_args, **kwargs):
        calls.append(kwargs)
        return results.pop(0)

    monkeypatch.setattr(orchestration, "run_leg", fake_run_leg)
    monkeypatch.setattr(orchestration.token_meter, "snapshot", lambda: {})
    monkeypatch.setattr(
        orchestration.token_meter,
        "delta",
        lambda _before: {"tokens_in": 2, "tokens_out": 1, "by_role": {}},
    )
    monkeypatch.setattr(orchestration.token_meter, "dump", lambda *_args: None)

    metrics, attempt = orchestration._run_leg_with_retries(state, leg, 0)

    assert metrics["success"] is True
    assert attempt == 2
    assert calls[0]["context"] == ""
    assert "wrong item" in calls[1]["context"]
    assert calls[0]["log_path"].endswith("leg00.jsonl")
    assert calls[1]["log_path"].endswith("leg00_retry1.jsonl")
    assert [row["attempt"] for row in state.leg_rows] == [1, 2]
    assert state.carried_names == {"left": "chips"}
    assert state.llm_calls == 2


def test_execute_plan_aborts_remaining_legs(monkeypatch, tmp_path):
    state = _state(tmp_path, retries=0)
    state.legs = [
        {"type": "pickup", "text": "Pick up the chips"},
        {"type": "checkout", "text": "Check out"},
    ]
    attempted = []

    def fail_first(_state, _leg, leg_index):
        attempted.append(leg_index)
        return _metrics(success=False, reason="time_cap"), 1

    monkeypatch.setattr(orchestration, "_run_leg_with_retries", fail_first)

    orchestration._execute_plan(state)

    assert attempted == [0]
    assert state.success is False


def test_orchestrate_preserves_run_error_and_always_closes(monkeypatch, tmp_path):
    state = _state(tmp_path)
    closed = []
    state.agent.close = lambda: closed.append(True)

    monkeypatch.setattr(orchestration, "_new_run_state", lambda _config: state)

    def initialize(current):
        current.ready_to_finalize = True

    monkeypatch.setattr(orchestration, "_initialize_runtime", initialize)
    monkeypatch.setattr(
        orchestration,
        "_plan_task",
        lambda _state: (_ for _ in ()).throw(RuntimeError("planning failed")),
    )
    monkeypatch.setattr(
        orchestration,
        "_finalize_run",
        lambda _state: (_ for _ in ()).throw(ValueError("finalization failed")),
    )
    monkeypatch.setattr(orchestration.chime, "beep", lambda: None)

    with pytest.raises(RuntimeError, match="planning failed"):
        orchestration.orchestrate("Pick up the chips", run_dir=tmp_path)

    assert closed == [True]
    assert isinstance(state.error, RuntimeError)
    assert state.success is False
