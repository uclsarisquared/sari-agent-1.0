from types import SimpleNamespace

from orchestrator import leg_runtime as runtime


def test_build_augmented_task_keeps_current_context_and_future_goals_separate():
    prompt = runtime.build_augmented_task(
        {"type": "pickup", "text": "Pick up crackers"},
        context="Milk was already checked out.",
        future_legs=[{"text": "Go to checkout"}, "Report completion"],
    )

    assert "CURRENT GOAL: Pick up crackers" in prompt
    assert "CONTEXT FROM PREVIOUS SUBTASKS:\nMilk was already checked out." in prompt
    assert "  1. Go to checkout\n  2. Report completion" in prompt
    assert "do NOT pursue these yet" in prompt


def test_inspect_prompt_preserves_the_restricted_action_contract():
    prompt = runtime.build_augmented_task(
        {"type": "inspect", "text": "Read the expiration date"}
    )

    assert "Never grab, release, or check out during this inspect leg" in prompt
    assert "Reporting a definite absence is a valid answer" in prompt


def test_model_facing_state_removes_code_only_and_large_nested_fields():
    state = {
        "visited_checkpoints": {1, 2},
        "last_checkout": {"scanned": True, "steps": {"scan": "large"}},
        "last_inspection": None,
        "translation": (1, 0, 2),
    }

    view = runtime.model_facing_state(state)

    assert "visited_checkpoints" not in view
    assert view["last_checkout"] == {"scanned": True}
    assert view["translation"] == (1, 0, 2)


def test_grip_tracker_preserves_only_live_carried_names():
    state = {
        "leftGrippedState": True,
        "rightGrippedState": False,
    }
    tracker = runtime.GripTracker.from_state(
        state, {"left": "Piattos", "right": "Stale value"}
    )

    assert tracker.names == {"left": "Piattos", "right": None}
    assert tracker.start_grips == {"left"}
    assert state["gripped_name"] is None

    state["leftGrippedState"] = False
    state["rightGrippedState"] = True
    tracker.record_grab({"gripped": True, "hovered": "Ritz", "hand": "right"})
    tracker.reconcile(state)

    assert tracker.names == {"left": None, "right": "Ritz"}
    assert state["released_grip_this_leg"] is True
    assert state["new_grip_this_leg"] is True


def test_reconcile_after_actions_merges_durable_state_and_visit_tracking():
    live_state = {
        "translation": (4, 0, 6),
        "leftGrippedState": True,
        "rightGrippedState": False,
    }
    grip_tracker = runtime.GripTracker(
        names={"left": "Ritz", "right": None}, start_grips=set()
    )
    evidence = runtime.InspectionEvidence()
    metrics = {"t_checkout": None, "t_grip": None}
    visited = {2}
    outcome = SimpleNamespace(
        blocked_reason=False,
        center_message="centered",
        last_reach={"reachable": True},
        grab_failed=False,
        checkout_result={"scanned": True},
    )
    store_map = SimpleNamespace(nearest_checkpoint=lambda _position: 7)

    state, near = runtime.reconcile_after_actions(
        outcome=outcome,
        mode="manipulation",
        last_inspection_result=None,
        store_map=store_map,
        visited=visited,
        grip_tracker=grip_tracker,
        inspection_evidence=evidence,
        metrics=metrics,
        started_at=0,
        read_state=lambda: dict(live_state),
    )

    assert near == 7
    assert visited == {2, 7}
    assert state["visited_checkpoints"] == {2, 7}
    assert state["last_checkout"] == {"scanned": True}
    assert state["gripped_name"] == "Ritz"
