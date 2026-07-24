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
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # overhaul/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim import env
from manip import manipulation as M
from agent_core import agent as A

_ORIG = {"SetHandsActive": env.SetHandsActive, "set_hand_pose": M.set_hand_pose}

# Spies replaces globals on `env` and `manipulation`; restore after each test so a shared-process
# runner (pytest) can't leak a spy into another test file.
try:
    import pytest

    @pytest.fixture(autouse=True)
    def _restore_globals():
        yield
        env.SetHandsActive = _ORIG["SetHandsActive"]
        M.set_hand_pose = _ORIG["set_hand_pose"]
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
