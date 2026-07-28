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

    python tests/test_agent_hand_pose_router.py   # or: pytest tests/test_agent_hand_pose_router.py
"""
import os
import base64
import re
import sys
import tempfile
from io import BytesIO

from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # overhaul/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim import env
from manip import manipulation as M
from agent_core import agent as A
from agent_core.sys_inst import SYS_INST_ASSOCIATIVE_SEMANTIC

_ORIG = {"SetHandsActive": env.SetHandsActive, "set_hand_pose": M.set_hand_pose,
         "set_hand_transform": M.set_hand_transform}

# Spies replaces globals on `env` and `manipulation`; restore after each test so a shared-process
# runner (pytest) can't leak a spy into another test file.
try:
    import pytest

    @pytest.fixture(autouse=True)
    def _restore_globals():
        yield
        env.SetHandsActive = _ORIG["SetHandsActive"]
        M.set_hand_pose = _ORIG["set_hand_pose"]
        M.set_hand_transform = _ORIG["set_hand_transform"]
except ImportError:
    pass


class Spies:
    def __init__(self):
        self.active = []      # SetHandsActive(active) history
        self.poses = []       # set_hand_pose(pose) history (one entry per hand driven)
        self.hands = []       # which hand each drive targeted (dual-hand: left+right per pose set)
        self.arrived = True   # flip to test the non-convergence warning path
        env.SetHandsActive = lambda active, *a, **k: self.active.append(active)

        def _spy_pose(pose, *a, hand="left", **k):
            self.poses.append(pose)
            self.hands.append(hand)
            return (self.arrived, (0.0, 0.0, 0.0), 0.0)
        M.set_hand_pose = _spy_pose


def _agent():
    a = object.__new__(A.EmbodiedAgent)   # skip __init__ (no clients)
    a._hands_active = None
    a._hand_pose = None
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


def test_inspection_restoration_resets_both_transforms_and_tracks_only_on_success():
    s = Spies()
    calls = []

    def transform(pose, rotation, hand):
        calls.append((pose, rotation, hand))
        state = {
            f"{hand}Translation": M.pose_for_hand("rest", hand),
            f"{hand}Rotation": (0, 0, 0),
        }
        return True, state, 0.0, 0.0

    M.set_hand_transform = transform
    a = _agent()
    result = a._restore_hands_after_inspection()
    assert result["restored"] is True
    assert calls == [
        ("rest", (0, 0, 0), "left"),
        ("rest", (0, 0, 0), "right"),
    ]
    assert a._hand_pose == "rest"
    assert s.active == [True]


def test_inspection_restoration_keeps_pose_unknown_when_one_hand_stalls():
    Spies()

    def transform(pose, rotation, hand):
        state = {f"{hand}Translation": (0, 0, 0), f"{hand}Rotation": (0, 0, 0)}
        return hand == "left", state, 1.0 if hand == "right" else 0.0, 0.0

    M.set_hand_transform = transform
    a = _agent()
    a._hand_pose = "inspection"
    result = a._restore_hands_after_inspection()
    assert result["restored"] is False
    assert a._hand_pose is None


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


def test_held_item_inspection_vocabulary_is_exactly_safe_presentation_controls():
    actions = A._available_actions("manipulation", held_item_inspection=True)
    names = {line.split(":", 1)[0] for line in actions.splitlines() if line.strip()}
    assert names == {
        "present_left_item_for_inspection",
        "present_right_item_for_inspection",
        "reset_left_hand_after_inspection",
        "reset_right_hand_after_inspection",
        "extend_left_hand_forward",
        "extend_right_hand_forward",
        "pull_left_hand_backward",
        "pull_right_hand_backward",
        "raise_left_hand",
        "raise_right_hand",
        "lower_left_hand",
        "lower_right_hand",
        "rotate_left_clockwise",
        "rotate_left_counterclockwise",
        "rotate_right_clockwise",
        "rotate_right_counterclockwise",
    }
    for forbidden in (
        "grip_left", "extend_arm_until_grabbed",
        "checkout_held_item", "center_object_on_screen", "move_forward",
    ):
        assert forbidden not in actions


class _FakeVLM:
    def __init__(self):
        self.base_semantic_memory = ""
        self.episodic_memory = ""
        self.actor_prompts = []

    def send_message(self, content):
        self.actor_prompts.append(content)
        return "{'actions': ['rotate_left_clockwise'], 'times': [2], 'notes': {}}"

    def get_history_text(self, n=8):
        return ""


class _FakeAssociative:
    extractable_json_structured_output = re.compile(
        r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)


def _png_b64():
    buf = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_execute_lean_returns_manipulation_and_rotation_prompt_in_both_branches():
    semantic = (
        "{'new_semantic_memory': '', 'recall': 'inspect the held item', "
        "'next_action': 'reorient it', 'reported_answer': '', 'mode': 'perception'}"
    )
    expected = {
        "present_left_item_for_inspection",
        "present_right_item_for_inspection",
        "reset_left_hand_after_inspection",
        "reset_right_hand_after_inspection",
        "extend_left_hand_forward",
        "extend_right_hand_forward",
        "pull_left_hand_backward",
        "pull_right_hand_backward",
        "raise_left_hand",
        "raise_right_hand",
        "lower_left_hand",
        "lower_right_hand",
        "rotate_left_clockwise",
        "rotate_left_counterclockwise",
        "rotate_right_clockwise",
        "rotate_right_counterclockwise",
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


def test_inspection_rotation_passes_existing_manipulation_dispatch_gate():
    from orchestrator import subtask_agents as SA

    original = SA.MANIPULATION_ACTIONS_REF["rotate_left_clockwise"]
    SA.MANIPULATION_ACTIONS_REF["rotate_left_clockwise"] = (
        lambda steps: {"rotated_steps": steps})
    try:
        result = SA.dispatch_action(
            "rotate_left_clockwise",
            3,
            {},
            mode="manipulation",
            leg_type="inspect",
            state={"leftGrippedState": True, "rightGrippedState": False},
        )
    finally:
        SA.MANIPULATION_ACTIONS_REF["rotate_left_clockwise"] = original
    assert result == {"rotated_steps": 3}


def test_inspect_dispatch_hard_blocks_mutators_and_body_motion_before_execution():
    from orchestrator import subtask_agents as SA

    held = {"leftGrippedState": True, "rightGrippedState": False}
    called = []
    replacements = {
        "grip_left": lambda n: called.append("grip"),
        "extend_arm_until_grabbed": lambda n: called.append("grab"),
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
            "grip_left", "extend_arm_until_grabbed", "checkout_held_item", "move_forward",
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
    from orchestrator import subtask_agents as SA

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


def test_inspect_dispatch_refuses_action_for_empty_selected_hand():
    from orchestrator import subtask_agents as SA

    result = SA.dispatch_action(
        "rotate_right_clockwise", 1, {}, mode="manipulation", leg_type="inspect",
        state={"leftGrippedState": True, "rightGrippedState": False})
    assert result["blocked"] and result["executed"] is False
    assert "right hand is empty" in result["reason"]


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
