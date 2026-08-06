"""Offline unit tests for the Phase 6.1 (step 2) agent mode-router hand-pose state machine:
EmbodiedAgent._set_hands / _set_hand_pose / _invalidate_hand_pose.

Verifies the properties the router relies on:
  - hands are kept ACTIVE; _set_hand_pose drives REST only on a real change (no per-step websocket spam);
  - a manipulation step (_invalidate_hand_pose) marks the pose UNKNOWN so the NEXT nav/perception step
    re-asserts REST - this is what recovers from a manual hand poke having displaced the hand;
  - a between-task hard stow (_set_hands(False)) invalidates the tracker so the next task re-activates
    AND re-drives REST.

No sim: env.SetHandsActive and manipulation.set_hand_pose are monkeypatched with spies; the agent is
built with object.__new__ so no VLM/model clients are constructed.

    uv run pytest validation/tests/agent/test_agent_hand_pose_router.py   # or: pytest validation/tests/agent/test_agent_hand_pose_router.py
"""
import os
import base64
import re
import sys
import tempfile
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim import env
from manip import manipulation as M
from agent_core import agent as A
from agent_core.context import SemanticLog
from agent_core.sys_inst import SYS_INST_ASSOCIATIVE_SEMANTIC
from agent_core.context_policy import ContextPolicy

_ORIG = {"SetHandsActive": env.SetHandsActive, "ResetHands": env.ResetHands,
         "ToggleLeftGrip": env.ToggleLeftGrip, "ToggleRightGrip": env.ToggleRightGrip,
         "TransformHands_env": env.TransformHands,
         "set_hand_pose": M.set_hand_pose,
         "set_hand_transform": M.set_hand_transform, "TransformHands": M.TransformHands}

# Spies replaces globals on `env` and `manipulation`; restore after each test so a shared-process
# runner (pytest) can't leak a spy into another test file.
try:
    import pytest

    @pytest.fixture(autouse=True)
    def _restore_globals():
        yield
        env.SetHandsActive = _ORIG["SetHandsActive"]
        env.ResetHands = _ORIG["ResetHands"]
        env.ToggleLeftGrip = _ORIG["ToggleLeftGrip"]
        env.ToggleRightGrip = _ORIG["ToggleRightGrip"]
        env.TransformHands = _ORIG["TransformHands_env"]
        M.set_hand_pose = _ORIG["set_hand_pose"]
        M.set_hand_transform = _ORIG["set_hand_transform"]
        M.TransformHands = _ORIG["TransformHands"]
except ImportError:
    pass


class Spies:
    def __init__(self):
        self.active = []      # SetHandsActive(active) history
        self.poses = []       # set_hand_pose(pose) history (one entry per hand driven)
        self.hands = []       # which hand each drive targeted (dual-hand: left+right per pose set)
        self.arrived = True   # flip to test the non-convergence warning path
        self.reset_calls = 0
        env.SetHandsActive = lambda active, *a, **k: self.active.append(active)

        def _reset(*_args, **_kwargs):
            self.reset_calls += 1
            return {
                "leftTranslation": M.pose_for_hand("rest", "left"),
                "rightTranslation": M.pose_for_hand("rest", "right"),
                "leftRotation": (0, 0, 0),
                "rightRotation": (0, 0, 0),
                "leftGrippedState": True,
                "rightGrippedState": False,
            }
        env.ResetHands = _reset

        def _spy_pose(pose, *a, hand="left", **k):
            self.poses.append(pose)
            self.hands.append(hand)
            return (self.arrived, (0.0, 0.0, 0.0), 0.0)
        M.set_hand_pose = _spy_pose


def _agent():
    a = object.__new__(A.EmbodiedAgent)   # skip __init__ (no clients)
    a._hands_active = None
    a._hand_pose = None
    a.context_policy = ContextPolicy()
    return a


def test_first_set_activates_and_drives_rest():
    s = Spies()
    a = _agent()
    a._set_hand_pose("rest")
    assert s.active == [True], "first call must activate the hands"
    assert s.poses == ["rest", "rest"] and s.hands == ["left", "right"], \
        "first call must drive REST on BOTH hands (dual-hand)"
    assert a._hands_active is True and a._hand_pose == "rest"


def test_repeat_is_a_noop_no_spam():
    s = Spies()
    a = _agent()
    a._set_hand_pose("rest")
    a._set_hand_pose("rest")
    a._set_hand_pose("rest")
    assert s.active == [True], "hands already active -> no repeat SetHandsActive"
    assert s.poses == ["rest", "rest"], "pose unchanged -> no repeat drive (fire-on-change)"


def test_invalidate_forces_next_rest_to_redrive():
    s = Spies()
    a = _agent()
    a._set_hand_pose("rest")            # poses: [rest]
    a._invalidate_hand_pose()           # manipulation step: pose now UNKNOWN, hands stay active
    assert a._hand_pose is None and a._hands_active is True
    assert s.active == [True], "invalidate must NOT toggle hands off"
    a._set_hand_pose("rest")            # poses: [rest, rest]  <- re-driven after manipulation
    assert s.poses == ["rest"] * 4, "REST must be re-asserted (both hands) after a manipulation step"


def test_hard_stow_then_next_task_reactivates_and_redrives():
    s = Spies()
    a = _agent()
    a._set_hand_pose("rest")            # active:[True]  poses:[rest]
    a._set_hands(False)                 # between-task stow
    assert a._hands_active is False and a._hand_pose is None
    assert s.active == [True, False]
    a._set_hand_pose("rest")            # new task
    assert s.active == [True, False, True], "new task must re-activate the hands"
    assert s.poses == ["rest"] * 4, "new task must re-drive REST (both hands)"


def test_grab_pose_is_never_set_by_router():
    s = Spies()
    a = _agent()
    a._set_hand_pose("rest")
    a._invalidate_hand_pose()
    a._set_hand_pose("rest")
    assert "grab" not in s.poses, "the router must never set GRAB (tool-internal only)"


def test_non_convergence_still_tracks_and_does_not_raise():
    s = Spies()
    s.arrived = False                   # set_hand_pose reports it did NOT reach the target
    a = _agent()
    a._set_hand_pose("rest")            # must log a warning, not crash
    assert a._hand_pose == "rest", "tracker still advances (best-effort) after a warned non-convergence"


def test_inspection_restoration_uses_one_atomic_unity_reset():
    s = Spies()
    a = _agent()
    result = a._restore_hands_after_inspection()
    assert result["restored"] is True
    assert s.reset_calls == 1
    assert s.poses == [] and s.hands == []
    assert result["hands"]["left"]["rotation"] == (0, 0, 0)
    assert result["hands"]["left"]["gripped"] is True
    assert a._hand_pose == "rest"
    assert s.active == [True]


def test_inspection_restoration_keeps_pose_unknown_when_reset_fails():
    Spies()
    env.ResetHands = lambda: (_ for _ in ()).throw(RuntimeError("reset failed"))
    a = _agent()
    a._hand_pose = "inspection"
    try:
        a._restore_hands_after_inspection()
    except RuntimeError as exc:
        assert str(exc) == "reset failed"
    else:
        raise AssertionError("ResetHands failure should propagate to run_leg cleanup")
    assert a._hand_pose is None


def test_inspection_restoration_opens_closed_empty_hand_without_releasing_carried_item():
    s = Spies()
    toggled = []
    env.ResetHands = lambda: {
        "leftTranslation": (-0.14, -0.08, 0.25),
        "rightTranslation": (0.18, -0.09, 0.27),
        "leftRotation": (0, 0, 90),
        "rightRotation": (0, 0, 0),
        "leftGrippedState": False,
        "leftHoldingItem": False,
        "leftGripClosedState": True,
        "rightGrippedState": True,
        "rightHoldingItem": True,
        "rightGripClosedState": True,
    }
    env.ToggleLeftGrip = lambda: toggled.append("left") or {"gripped": False}
    env.ToggleRightGrip = lambda: toggled.append("right") or {"gripped": False}
    env.TransformHands = lambda *_args: {
        "leftTranslation": (-0.14, -0.08, 0.25),
        "rightTranslation": (0.18, -0.09, 0.27),
        "leftRotation": (0, 0, 90),
        "rightRotation": (0, 0, 0),
        "leftGrippedState": False,
        "leftHoldingItem": False,
        "leftGripClosedState": False,
        "rightGrippedState": True,
        "rightHoldingItem": True,
        "rightGripClosedState": True,
    }
    a = _agent()

    result = a._restore_hands_after_inspection()

    assert toggled == ["left"]
    assert result["restored"] is True
    assert result["recovered_ghost_grips"] == ["left"]
    assert result["hands"]["left"]["gripped"] is False
    assert result["hands"]["right"]["holding_item"] is True
    assert a._hand_pose == "rest"
    assert s.active == [True]


def test_semantic_response_preserves_structured_reported_answer():
    parsed = A._parse_semantic_response(
        re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL),
        "{'new_semantic_memory': 'counted', 'recall': 'done', 'next_action': 'stop', "
        "'reported_answer': '14 unique products', 'mode': 'STOP'}",
    )
    assert parsed["reported_answer"] == "14 unique products"


def test_semantic_response_defaults_missing_reported_answer_to_empty():
    parsed = A._parse_semantic_response(
        re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL),
        "{'new_semantic_memory': '', 'recall': '', 'next_action': 'stop', 'mode': 'STOP'}",
    )
    assert parsed["reported_answer"] == ""


def test_stop_response_carries_reported_answer_separately_from_placeholder():
    response = A._stop_response(
        {"mode": "STOP", "reported_answer": "14 unique products"},
        "{'mode': 'STOP', 'reported_answer': '14 unique products'}",
    )
    assert response["halt"] is True and response["agent_mode"] == "STOP"
    assert response["reported_answer"] == "14 unique products"
    assert "14 unique products" not in response["text"]


def test_mode_overrides_preserve_stop_navigation_and_unforced_routing():
    resolve = A._resolve_agent_mode
    assert resolve("perception", force_manipulate=True) == "manipulation"
    assert resolve("STOP", force_manipulate=True) == "STOP"
    assert resolve("navigation", force_manipulate=True) == "navigation"
    assert resolve("perception") == "perception"
    assert resolve("manipulation") == "manipulation"
    assert resolve("navigation", force_navigate=True, inspect_mode="held") == "manipulation"
    assert resolve("navigation", force_navigate=True, inspect_mode="visual") == "perception"
    assert resolve("STOP", force_navigate=True, inspect_mode="held") == "STOP"


def test_force_navigate_takes_precedence_over_force_manipulate():
    resolve = A._resolve_agent_mode
    assert resolve("perception", force_navigate=True, force_manipulate=True) == "navigation"
    assert resolve("manipulation", force_navigate=True, force_manipulate=True) == "navigation"
    assert resolve("STOP", force_navigate=True, force_manipulate=True) == "STOP"


def test_held_item_inspection_vocabulary_exposes_only_restricted_macro():
    actions = A._available_actions("manipulation", held_item_inspection=True)
    names = {line.split(":", 1)[0] for line in actions.splitlines() if line.strip()}
    assert names == {
        "inspect_held_item",
        "inspect_held_item_left",
        "inspect_held_item_right",
    }
    for forbidden in (
        "grip_left", "extend_arm_until_grabbed",
        "checkout_held_item", "center_object_on_screen", "move_forward",
        "present_left_item_for_inspection", "extend_left_hand_forward",
        "reset_left_hand_after_inspection", "rotate_left_clockwise",
        "rotate_left_to_next_inspection_face",
    ):
        assert forbidden not in actions


class _FakeVLM:
    def __init__(self):
        self.semantic_log = SemanticLog("", ContextPolicy())
        self.episodic_memory = ""
        self.actor_prompts = []

    def send_message(self, content):
        self.actor_prompts.append(content)
        return ("{'actions': ['inspect_held_item'], "
                "'times': [1], 'notes': {}}")

    def get_history_text(self, n=8):
        return ""


class _FakeAssociative:
    extractable_json_structured_output = re.compile(
        r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)


def _png_b64():
    buf = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_held_inspection_stop_defers_rest_until_guard_in_both_branches():
    semantic = (
        "{'new_semantic_memory': '', 'recall': 'date is visible', "
        "'next_action': 'stop', 'reported_answer': '31/12/26', 'mode': 'STOP'}"
    )
    for timestep in (1, 2):
        agent = _agent()
        agent.nav_mode = "vlm"
        agent._mem_leg = None
        agent.vlm_agent = _FakeVLM()
        agent.associative_learner = _FakeAssociative()
        agent._call_associative = lambda system, image, text: semantic
        pose_calls = []
        agent._set_hand_pose = lambda pose: pose_calls.append(pose)
        agent._invalidate_hand_pose = lambda: pose_calls.append("invalidated")

        response = agent.execute_lean(
            {
                "task": "Read the held expiration date.",
                "state": {"leftGrippedState": True, "rightGrippedState": False},
                "image": _png_b64(),
                "force_navigate": False,
                "force_manipulate": True,
                "inspect_mode": "held",
            },
            timestep,
        )

        assert response["halt"] is True
        assert response["reported_answer"] == "31/12/26"
        assert pose_calls == [], "held evidence pose must survive until the STOP guard verdict"


def test_noninspection_stop_keeps_existing_rest_behavior_in_both_branches():
    semantic = (
        "{'new_semantic_memory': '', 'recall': 'done', 'next_action': 'stop', "
        "'reported_answer': '', 'mode': 'STOP'}"
    )
    for timestep in (1, 2):
        agent = _agent()
        agent.nav_mode = "vlm"
        agent._mem_leg = None
        agent.vlm_agent = _FakeVLM()
        agent.associative_learner = _FakeAssociative()
        agent._call_associative = lambda system, image, text: semantic
        pose_calls = []
        agent._set_hand_pose = lambda pose: pose_calls.append(pose)
        agent._invalidate_hand_pose = lambda: pose_calls.append("invalidated")

        response = agent.execute_lean(
            {
                "task": "Finish the ordinary leg.",
                "state": {"leftGrippedState": False, "rightGrippedState": False},
                "image": _png_b64(),
                "force_navigate": False,
                "force_manipulate": False,
            },
            timestep,
        )

        assert response["halt"] is True
        assert pose_calls == ["rest"]


def test_execute_lean_returns_manipulation_and_restricted_inspection_prompt_in_both_branches():
    semantic = (
        "{'new_semantic_memory': '', 'recall': 'inspect the held item', "
        "'next_action': 'reorient it', 'reported_answer': '', 'mode': 'perception'}"
    )
    expected = {
        "inspect_held_item",
        "inspect_held_item_left",
        "inspect_held_item_right",
    }
    with tempfile.TemporaryDirectory() as run_dir:
        for timestep in (1, 2):
            agent = _agent()
            agent.nav_mode = "vlm"
            agent._run_dir = run_dir
            agent._mem_leg = None
            agent.vlm_agent = _FakeVLM()
            agent.associative_learner = _FakeAssociative()
            agent._call_associative = lambda system, image, text: semantic
            agent._call_episodic = lambda history: (
                "{'dense_summary': '', 'what_worked': '', 'what_to_avoid': ''}")
            agent._set_hand_pose = lambda pose: None
            agent._invalidate_hand_pose = lambda: None

            response = agent.execute_lean(
                {
                    "task": "Read the held label.",
                    "state": {"leftGrippedState": True, "rightGrippedState": False},
                    "image": _png_b64(),
                    "force_navigate": False,
                    "force_manipulate": True,
                    "inspect_mode": "held",
                },
                timestep,
            )

            assert response["agent_mode"] == "manipulation"
            text_parts = [
                item["text"] for item in agent.vlm_agent.actor_prompts[-1]
                if item.get("type") == "text"
            ]
            prompt = "\n".join(text_parts)
            action_block = prompt.split("## AVAILABLE ACTIONS:\n", 1)[1]
            names = {line.split(":", 1)[0]
                     for line in action_block.splitlines() if ":" in line}
            assert names == expected
            assert "grip_left" not in action_block
            assert "extend_arm_until_grabbed" not in action_block
            assert "checkout_held_item" not in action_block
            assert "fresh isolated VLM visibility check" in action_block
            assert "eight X turns of exactly 45 degrees" in action_block
            assert "Y turns that check the top and bottom" in action_block
            assert "actor cannot choose or sequence rotations" in action_block
            assert "last_inspection" in action_block


def test_unheld_inspection_cannot_enter_graph_navigation_dispatch():
    semantic = (
        "{'new_semantic_memory': '', 'recall': 'look around', "
        "'next_action': 'navigate', 'reported_answer': '', 'mode': 'navigation'}"
    )
    with tempfile.TemporaryDirectory() as run_dir:
        agent = _agent()
        agent.nav_mode = "graph"
        agent._run_dir = run_dir
        agent._mem_leg = None
        agent.vlm_agent = _FakeVLM()
        agent.associative_learner = _FakeAssociative()
        agent._call_associative = lambda system, image, text: semantic
        agent._call_episodic = lambda history: (
            "{'dense_summary': '', 'what_worked': '', 'what_to_avoid': ''}")
        agent._set_hand_pose = lambda pose: None
        agent._invalidate_hand_pose = lambda: None
        agent._graph_navigate = lambda *args: (_ for _ in ()).throw(
            AssertionError("inspect leg entered graph navigation"))

        response = agent.execute_lean(
            {
                "task": "Visually inspect the shelf.",
                "state": {"leftGrippedState": False, "rightGrippedState": False},
                "image": _png_b64(),
                "force_navigate": True,
                "force_manipulate": False,
                "inspect_mode": "visual",
            },
            1,
        )

    assert response["agent_mode"] == "perception"
    prompt = "\n".join(
        item["text"] for item in agent.vlm_agent.actor_prompts[-1]
        if item.get("type") == "text")
    action_block = prompt.split("## AVAILABLE ACTIONS:\n", 1)[1]
    assert "center_object_on_screen" in action_block
    assert "move_forward" not in action_block


def test_agent_selected_rotation_action_is_blocked_during_held_inspection():
    from orchestrator import action_dispatch as SA

    action = "rotate_left_to_next_inspection_face"
    original = SA.MANIPULATION_ACTIONS_REF[action]
    calls = []
    SA.MANIPULATION_ACTIONS_REF[action] = lambda times: calls.append(times)
    try:
        result = SA.dispatch_action(
            action, 99, {}, mode="manipulation", leg_type="inspect",
            state={"leftGrippedState": True, "rightGrippedState": False})
    finally:
        SA.MANIPULATION_ACTIONS_REF[action] = original
    assert calls == []
    assert result["blocked"] and result["inspect_scope_violation"]
    assert "restricted inspect_held_item macro" in result["reason"]


def test_restricted_inspection_macro_must_be_the_only_action_in_its_timestep():
    from orchestrator import held_item_inspection as SA

    assert SA._inspection_action_batch(
        ["inspect_held_item", "inspect_held_item"], [1, 1]
    ) == [("inspect_held_item", 1)]
    assert SA._inspection_action_batch(
        ["inspect_held_item", "lower_right_hand"], [1, 2]
    ) == [("inspect_held_item", 1)]
    assert SA._inspection_action_batch(
        ["lower_left_hand", "inspect_held_item"], [1, 1]
    ) == [("inspect_held_item", 1)]
    assert SA._inspection_action_batch(
        ["inspect_held_item_right", "inspect_held_item_left"], [1, 1]
    ) == [("inspect_held_item_right", 1)]


def test_restricted_inspection_macro_can_force_right_hand():
    from orchestrator import held_item_inspection as SA

    presented = []
    png_bytes = base64.b64decode(_png_b64())
    resets = []
    agent = SimpleNamespace(
        vlm_agent=SimpleNamespace(
            client=object(),
            config=SimpleNamespace(model_id="m", max_tokens=256),
        ),
        _restore_hands_after_inspection=lambda: (
            resets.append(True) or {"restored": True, "hands": {}}),
    )
    originals = (
        SA.MANIPULATION_ACTIONS_REF["present_right_item_for_inspection"],
        SA._REQUEST_SCREENSHOT_,
        SA.classify_inspection_visibility,
    )
    SA.MANIPULATION_ACTIONS_REF["present_right_item_for_inspection"] = (
        lambda _times: presented.append("right") or {
            "arrived": True, "blocked": False,
        })
    SA._REQUEST_SCREENSHOT_ = lambda: {"image": png_bytes}
    SA.classify_inspection_visibility = lambda *_args, **_kwargs: {
        "match": True,
        "reason": "requested label is legible",
        "conclusive": True,
        "latency_ms": 1.0,
    }
    try:
        result = SA._run_held_item_inspection_macro(
            agent,
            "Read the expiration date.",
            {"leftGrippedState": True, "rightGrippedState": True},
            hand="right",
        )
    finally:
        (SA.MANIPULATION_ACTIONS_REF["present_right_item_for_inspection"],
         SA._REQUEST_SCREENSHOT_,
         SA.classify_inspection_visibility) = originals

    assert presented == ["right"]
    assert resets == [True]
    assert result["hand"] == "right"
    assert result["label_visible"] is True


def test_right_hand_inspection_swaps_local_xy_axes_to_preserve_x_then_y_sweep():
    from orchestrator import held_item_inspection as SA

    assert SA._inspection_rotation_delta("left", (45.0, 0.0, 0.0)) == (
        45.0, 0.0, 0.0)
    assert SA._inspection_rotation_delta("left", (0.0, -90.0, 0.0)) == (
        0.0, -90.0, 0.0)
    assert SA._inspection_rotation_delta("right", (45.0, 0.0, 0.0)) == (
        0.0, 45.0, 0.0)
    assert SA._inspection_rotation_delta("right", (0.0, -90.0, 0.0)) == (
        -90.0, 0.0, 0.0)
    assert SA._inspection_rotation_delta(
        "right", SA._INSPECTION_PASS_RESET_DELTA
    ) == (90.0, 0.0, 0.0)


def test_right_hand_macro_commands_side_axis_before_top_bottom_axis():
    from orchestrator import held_item_inspection as SA

    turns = []
    visibility_checks = 0
    png_bytes = base64.b64decode(_png_b64())
    agent = SimpleNamespace(
        vlm_agent=SimpleNamespace(
            client=object(),
            config=SimpleNamespace(model_id="m", max_tokens=256),
        ),
        _restore_hands_after_inspection=lambda: {"restored": True, "hands": {}},
    )
    originals = (
        SA.MANIPULATION_ACTIONS_REF["present_right_item_for_inspection"],
        SA.TransformHands,
        SA._REQUEST_SCREENSHOT_,
        SA.classify_inspection_visibility,
        SA.classify_inspection_label_presence,
    )

    def classify(*_args, **_kwargs):
        nonlocal visibility_checks
        visibility_checks += 1
        return {
            "match": visibility_checks == 10,
            "reason": "visible on first top/bottom check",
            "conclusive": True,
            "latency_ms": 1.0,
        }

    SA.MANIPULATION_ACTIONS_REF["present_right_item_for_inspection"] = (
        lambda _times: {"arrived": True, "blocked": False})
    SA.TransformHands = lambda _lt, _lr, _rt, rr: (
        turns.append(tuple(rr)) or {"rightRotation": (0, 0, 0)})
    SA._REQUEST_SCREENSHOT_ = lambda: {"image": png_bytes}
    SA.classify_inspection_visibility = classify
    SA.classify_inspection_label_presence = lambda *_args: {
        "match": False,
        "reason": "not this side",
        "conclusive": True,
        "latency_ms": 1.0,
    }
    try:
        result = SA._run_held_item_inspection_macro(
            agent,
            "Read the label.",
            {"leftGrippedState": False, "rightGrippedState": True},
            hand="right",
        )
    finally:
        (SA.MANIPULATION_ACTIONS_REF["present_right_item_for_inspection"],
         SA.TransformHands,
         SA._REQUEST_SCREENSHOT_,
         SA.classify_inspection_visibility,
         SA.classify_inspection_label_presence) = originals

    assert result["label_visible"] is True
    assert result["visible_phase"] == "y_top"
    assert turns == [(0.0, 45.0, 0.0)] * 8 + [(90.0, 0.0, 0.0)]


def test_restricted_inspection_macro_refuses_forced_empty_hand():
    from orchestrator import held_item_inspection as SA

    result = SA._run_held_item_inspection_macro(
        object(),
        "Read the label.",
        {"leftGrippedState": True, "rightGrippedState": False},
        hand="right",
    )

    assert result["blocked"] is True
    assert result["executed"] is False
    assert result["hand"] == "right"
    assert "right hand" in result["reason"]


def test_inspect_dispatch_blocks_actor_selected_presentation():
    from orchestrator import action_dispatch as SA

    action = "present_left_item_for_inspection"
    original = SA.MANIPULATION_ACTIONS_REF[action]
    calls = []
    SA.MANIPULATION_ACTIONS_REF[action] = lambda times: calls.append(times) or {"arrived": True}
    try:
        result = SA.dispatch_action(
            action, 1, {}, mode="manipulation", leg_type="inspect",
            state={
                "leftGrippedState": True,
                "rightGrippedState": False,
                "leftRotation": (0, 90, 37),
            })
    finally:
        SA.MANIPULATION_ACTIONS_REF[action] = original
    assert result["blocked"] and result["inspect_scope_violation"]
    assert calls == []


def test_restricted_inspection_macro_runs_fixed_x_then_y_loop_and_logs_each_check():
    from orchestrator import held_item_inspection as SA

    turns = []
    closer_poses = []
    cleanup_calls = []
    events = []
    checks = []
    png_bytes = base64.b64decode(_png_b64())
    agent = SimpleNamespace(vlm_agent=SimpleNamespace(
        client=object(),
        config=SimpleNamespace(model_id="visibility-model", max_tokens=256),
    ), _restore_hands_after_inspection=lambda: (
        cleanup_calls.append(True) or {"restored": True, "hands": {}}))
    originals = (
        SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"],
        SA.TransformHands,
        SA._REQUEST_SCREENSHOT_,
        SA.classify_inspection_visibility,
        SA.classify_inspection_label_presence,
        M.set_hand_pose,
    )

    def transform(_lt, lr, _rt, rr):
        delta = lr if any(lr) else rr
        if any(delta):
            turns.append(tuple(delta))
        return {
            "leftRotation": (0, 0, 37),
            "rightRotation": (0, 0, 0),
            "leftGrippedState": True,
            "rightGrippedState": False,
        }

    def classify(_client, model, _config, _image, query):
        checks.append((model, query))
        return {
            "match": False,
            "reason": "target label is not legible",
            "conclusive": True,
            "latency_ms": 1.0,
        }

    SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"] = (
        lambda _times: {
            "arrived": True,
            "blocked": False,
        })
    SA.TransformHands = transform
    M.set_hand_pose = lambda pose, hand, max_iters: (
        closer_poses.append((tuple(pose), hand, max_iters))
        or (True, tuple(pose), 0.0)
    )
    SA._REQUEST_SCREENSHOT_ = lambda: {"image": png_bytes}
    SA.classify_inspection_visibility = classify
    SA.classify_inspection_label_presence = lambda *_args: {
        "match": False,
        "reason": "requested label is not recognizable on this side",
        "conclusive": True,
        "latency_ms": 1.0,
    }
    try:
        result = SA._run_held_item_inspection_macro(
            agent,
            "What nutritional facts are printed on the box?",
            {"leftGrippedState": True, "rightGrippedState": False},
            log_event=events.append,
        )
    finally:
        (SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"],
         SA.TransformHands,
         SA._REQUEST_SCREENSHOT_,
         SA.classify_inspection_visibility,
         SA.classify_inspection_label_presence,
         M.set_hand_pose) = originals

    assert result["sweep_exhausted"] is True and result["label_visible"] is False
    assert result["failure_cleanup"]["restored"] is True
    assert cleanup_calls == [True, True]
    assert result["checks"] == 60
    assert result["vlm_calls"] == 120
    assert result["passes_completed"] == 5
    assert turns == (
        (
            [(45.0, 0.0, 0.0)] * 8
            + [(0.0, 90.0, 0.0), (0.0, -90.0, 0.0), (0.0, -90.0, 0.0)]
            + [(0.0, 90.0, 0.0)]
        ) * 4
        + [(45.0, 0.0, 0.0)] * 8
        + [(0.0, 90.0, 0.0), (0.0, -90.0, 0.0), (0.0, -90.0, 0.0)]
    )
    assert [(round(pose[2], 2), hand, max_iters)
            for pose, hand, max_iters in closer_poses] == [
        (0.28, "left", 5),
        (0.23, "left", 5),
        (0.18, "left", 5),
        (0.13, "left", 5),
    ]
    assert all(
        pose[:2] == M.INSPECTION_POSE[:2]
        for pose, _hand, _max_iters in closer_poses
    )
    assert len(checks) == 60
    assert all(model == "visibility-model" for model, _ in checks)
    assert all("nutritional facts" in query for _, query in checks)
    assert len([event for event in events
                if event["event"] == "inspection_visibility_check"]) == 60
    assert len([event for event in events
                if event["event"] == "inspection_label_presence_check"]) == 60
    assert len([event for event in events
                if event["event"] == "inspection_rotation"]) == 55
    assert [event["closer_by_m"] for event in events
            if event["event"] == "inspection_reposition"] == [0.05, 0.1, 0.15, 0.2]
    assert len([event for event in events
                if event["event"] == "inspection_failure_cleanup"]) == 1
    assert events[0]["event"] == "inspection_macro_start"
    assert events[1]["event"] == "inspection_pre_reset"
    assert events[-1]["event"] == "inspection_macro_end"


def test_restricted_inspection_macro_returns_control_as_soon_as_label_is_visible():
    from orchestrator import held_item_inspection as SA

    turns = []
    cleanup_calls = []
    outcomes = iter((False, False, True))
    png_bytes = base64.b64decode(_png_b64())
    agent = SimpleNamespace(vlm_agent=SimpleNamespace(
        client=object(),
        config=SimpleNamespace(model_id="m", max_tokens=256),
    ), _restore_hands_after_inspection=lambda: (
        cleanup_calls.append(True) or {"restored": True, "hands": {}}))
    originals = (
        SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"],
        SA.TransformHands,
        SA._REQUEST_SCREENSHOT_,
        SA.classify_inspection_visibility,
        SA.classify_inspection_label_presence,
    )
    SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"] = (
        lambda _times: {"arrived": True, "blocked": False})
    SA.TransformHands = lambda _lt, lr, _rt, _rr: (
        turns.append(tuple(lr)) or {"leftRotation": (0, 0, 0)})
    SA._REQUEST_SCREENSHOT_ = lambda: {"image": png_bytes}
    SA.classify_inspection_visibility = lambda *_args: {
        "match": next(outcomes),
        "reason": "visibility verdict",
        "conclusive": True,
        "latency_ms": 1.0,
    }
    SA.classify_inspection_label_presence = lambda *_args: {
        "match": False,
        "reason": "no requested label on this side",
        "conclusive": True,
        "latency_ms": 1.0,
    }
    try:
        result = SA._run_held_item_inspection_macro(
            agent,
            "What expiration date is printed?",
            {"leftGrippedState": True, "rightGrippedState": False},
        )
    finally:
        (SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"],
         SA.TransformHands,
         SA._REQUEST_SCREENSHOT_,
         SA.classify_inspection_visibility,
         SA.classify_inspection_label_presence) = originals

    assert result["label_visible"] is True
    assert result["visible_phase"] == "x"
    assert result["checks"] == 3 and result["vlm_calls"] == 5
    assert turns == [(45.0, 0.0, 0.0), (45.0, 0.0, 0.0)]
    assert cleanup_calls == [True], (
        "inspection must reset before presentation, then preserve its evidence pose until STOP")


def test_inspection_locks_unreadable_label_side_and_only_moves_closer():
    from orchestrator import held_item_inspection as SA
    from vision import ocr_client

    turns = []
    closer_poses = []
    cleanup_calls = []
    ocr_crops = []
    presence = iter((False, True))
    png_bytes = base64.b64decode(_png_b64())
    agent = SimpleNamespace(vlm_agent=SimpleNamespace(
        client=object(),
        config=SimpleNamespace(model_id="m", max_tokens=256),
    ), _restore_hands_after_inspection=lambda: (
        cleanup_calls.append(True) or {"restored": True, "hands": {}}))
    originals = (
        SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"],
        SA.TransformHands,
        SA._REQUEST_SCREENSHOT_,
        SA.classify_inspection_visibility,
        SA.classify_inspection_label_presence,
        M.set_hand_pose,
        ocr_client.ocr_lines,
    )
    SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"] = (
        lambda _times: {"arrived": True, "blocked": False})
    SA.TransformHands = lambda _lt, lr, _rt, _rr: (
        turns.append(tuple(lr)) or {"leftRotation": (45, 0, 0)})
    SA._REQUEST_SCREENSHOT_ = lambda: {"image": png_bytes}
    SA.classify_inspection_visibility = lambda *_args, **_kwargs: {
        "match": False,
        "reason": "label text is too small to read",
        "conclusive": True,
        "latency_ms": 1.0,
    }
    SA.classify_inspection_label_presence = lambda *_args: {
        "match": next(presence),
        "reason": "Nutrition Facts panel is recognizable",
        "conclusive": True,
        "latency_ms": 1.0,
    }
    M.set_hand_pose = lambda pose, hand, max_iters: (
        closer_poses.append(tuple(pose)) or (True, tuple(pose), 0.0))
    ocr_client.ocr_lines = lambda crop: (
        ocr_crops.append(crop.size) or ["Nutrition Facts", "Calories 140"])
    try:
        result = SA._run_held_item_inspection_macro(
            agent,
            "Read the nutritional facts.",
            {"leftGrippedState": True, "rightGrippedState": False},
        )
    finally:
        (SA.MANIPULATION_ACTIONS_REF["present_left_item_for_inspection"],
         SA.TransformHands,
         SA._REQUEST_SCREENSHOT_,
         SA.classify_inspection_visibility,
         SA.classify_inspection_label_presence,
         M.set_hand_pose,
         ocr_client.ocr_lines) = originals

    assert turns == [(45.0, 0.0, 0.0)], "rotation must stop on the detected label side"
    assert [round(pose[2], 2) for pose in closer_poses] == [0.28, 0.23, 0.18, 0.13]
    assert result["label_visible"] is True
    assert result["label_legible"] is False
    assert result["label_locked"] is True
    assert result["best_effort_read"] is True
    assert result["checks"] == 6 and result["vlm_calls"] == 9
    assert result["ocr_lines"] == ["Nutrition Facts", "Calories 140"]
    assert ocr_crops == [(1, 1)] * 5
    assert "best-effort read" in result["reason"]
    assert cleanup_calls == [True], (
        "inspection must reset before presentation, then preserve the closest evidence frame")


def test_inspect_dispatch_hard_blocks_mutators_and_body_motion_before_execution():
    from orchestrator import action_dispatch as SA

    held = {"leftGrippedState": True, "rightGrippedState": False}
    called = []
    replacements = {
        "grip_left": lambda n: called.append("grip"),
        "extend_arm_until_grabbed": lambda n: called.append("grab"),
        "rotate_left_clockwise": lambda n: called.append("roll"),
        "move_forward": lambda n: called.append("move"),
    }
    originals = {}
    for name, replacement in replacements.items():
        table = (SA.NAVIGATION_ACTIONS_REF if name == "move_forward"
                 else SA.MANIPULATION_ACTIONS_REF)
        originals[name] = (table, table[name])
        table[name] = replacement
    try:
        for action in (
            "grip_left", "extend_arm_until_grabbed", "rotate_left_clockwise",
            "checkout_held_item", "move_forward",
        ):
            result = SA.dispatch_action(
                action, 1, {}, mode="manipulation", leg_type="inspect", state=held)
            assert result["blocked"] is True
            assert result["executed"] is False
            assert result["inspect_scope_violation"] is True
    finally:
        for name, (table, original) in originals.items():
            table[name] = original
    assert called == []


def test_inspect_dispatch_allows_visual_camera_only_when_no_item_held():
    from orchestrator import action_dispatch as SA

    original = SA.NAVIGATION_ACTIONS_REF["pan_left"]
    SA.NAVIGATION_ACTIONS_REF["pan_left"] = lambda n: {"panned": n}
    try:
        allowed = SA.dispatch_action(
            "pan_left", 2, {}, mode="perception", leg_type="inspect",
            state={"leftGrippedState": False, "rightGrippedState": False})
        blocked = SA.dispatch_action(
            "move_left", 2, {}, mode="perception", leg_type="inspect",
            state={"leftGrippedState": False, "rightGrippedState": False})
    finally:
        SA.NAVIGATION_ACTIONS_REF["pan_left"] = original
    assert allowed == {"panned": 2}
    assert blocked["blocked"] and blocked["executed"] is False


def test_inspect_dispatch_spends_a_metered_approach_budget_when_no_item_is_held():
    """An unheld inspect leg may step closer, clamped to the allowance it was given.

    The decomposer contract has always defined inspect as covering "stepping closer"; before
    2026-07-30 the scope gate allowed camera actions only, so an inspect leg that started away from
    its target could only pan forever (every other exit is sealed - see _INSPECT_MOVE_BUDGET_STEPS).
    """
    from orchestrator import action_dispatch as SA

    calls = []
    original = SA.NAVIGATION_ACTIONS_REF["move_forward"]
    SA.NAVIGATION_ACTIONS_REF["move_forward"] = lambda n: calls.append(n) or {"moved": n}
    try:
        ok = SA.dispatch_action(
            "move_forward", 5, {}, mode="perception", leg_type="inspect",
            state={"leftGrippedState": False, "rightGrippedState": False},
            inspect_move_allowance=20)
        clamped = SA.dispatch_action(
            "move_forward", 10, {}, mode="perception", leg_type="inspect",
            state={"leftGrippedState": False, "rightGrippedState": False},
            inspect_move_allowance=3)
    finally:
        SA.NAVIGATION_ACTIONS_REF["move_forward"] = original
    assert calls == [5, 3], "an over-large request must be clamped to what is left, not refused"
    assert ok["inspect_move_steps"] == 5 and clamped["inspect_move_steps"] == 3
    assert not ok.get("blocked") and not clamped.get("blocked")


def test_inspect_dispatch_blocks_approach_once_the_budget_is_spent_and_points_at_stop():
    from orchestrator import action_dispatch as SA

    calls = []
    original = SA.NAVIGATION_ACTIONS_REF["move_forward"]
    SA.NAVIGATION_ACTIONS_REF["move_forward"] = lambda n: calls.append(n)
    try:
        spent = SA.dispatch_action(
            "move_forward", 4, {}, mode="perception", leg_type="inspect",
            state={"leftGrippedState": False, "rightGrippedState": False},
            inspect_move_allowance=0)
    finally:
        SA.NAVIGATION_ACTIONS_REF["move_forward"] = original
    assert calls == []
    assert spent["blocked"] and spent["executed"] is False
    assert spent["inspect_scope_violation"] is True
    # The block must name the exit, or the agent is back in the pan-forever loop it came from.
    assert "not at this location" in spent["reason"] and "STOP" in spent["reason"]


def test_inspect_approach_budget_never_applies_while_an_item_is_held():
    """A held-item inspect reads a label already in hand - walking cannot help, and the restricted
    macro owns that path. The budget must not reopen body motion there."""
    from orchestrator import action_dispatch as SA

    calls = []
    original = SA.NAVIGATION_ACTIONS_REF["move_forward"]
    SA.NAVIGATION_ACTIONS_REF["move_forward"] = lambda n: calls.append(n)
    try:
        result = SA.dispatch_action(
            "move_forward", 2, {}, mode="perception", leg_type="inspect",
            state={"leftGrippedState": True, "rightGrippedState": False},
            inspect_move_allowance=20)
    finally:
        SA.NAVIGATION_ACTIONS_REF["move_forward"] = original
    assert calls == []
    assert result["blocked"] and result["inspect_scope_violation"] is True


def test_inspect_guard_accepts_a_definite_absence_but_still_refuses_inability():
    from orchestrator.pickup_vlm_guard import INSPECT_GUARD_SYSTEM as G

    assert "DEFINITE ABSENCE is a real answer" in G     # the new exit
    assert "not on this shelf" in G
    # ... without reopening the visibility excuse the guard exists to refuse.
    assert "needs rotation" in G and "match=false" in G


def test_inspect_dispatch_refuses_restricted_macro_when_no_item_is_held():
    from orchestrator import action_dispatch as SA

    result = SA.dispatch_action(
        "inspect_held_item", 1, {}, mode="manipulation", leg_type="inspect",
        state={"leftGrippedState": False, "rightGrippedState": False})
    assert result["blocked"] and result["executed"] is False
    assert "unheld inspect leg" in result["reason"]


def test_semantic_prompt_requires_exact_reported_answer_on_stop():
    prompt = SYS_INST_ASSOCIATIVE_SEMANTIC
    assert "'reported_answer':" in prompt
    assert "exact concise answer" in prompt
    assert "Never put 'STOP'" in prompt


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for t in tests:
            t()
            print(f"  PASS {t.__name__}")
        print(f"OK: {len(tests)} agent hand-pose router tests passed")
    finally:
        env.SetHandsActive = _ORIG["SetHandsActive"]
        M.set_hand_pose = _ORIG["set_hand_pose"]


if __name__ == "__main__":
    _run()
