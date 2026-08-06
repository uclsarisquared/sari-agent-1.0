"""Offline unit tests for manipulation.extend_arm_until_grabbed's guard/pose contract:

  1. full-hand guard   - an explicitly-named hand that is already gripping is REFUSED (blocked), NOT
     force-opened, and NO pose is touched (the carried item stays exactly where it is). With
     hand='auto' (the default since dual-hand, 2026-07-23) a full LEFT hand resolves to the FREE
     RIGHT hand instead; only BOTH hands full refuses.
  2. tool owns poses   - GRAB is set at entry, REST is restored on EVERY exit: success, miss, AND
     exception (the finally). That REST-restore is what carries a gripped item back to the rest pose.

No sim: the env primitives extend_arm_until_grabbed calls (TransformHands, the extend/grip/pull
lambdas for BOTH hands, TransformAgent) and manipulation.set_hand_pose are monkeypatched.

    uv run pytest validation/tests/agent/test_extend_arm_guard.py    # or: pytest validation/tests/agent/test_extend_arm_guard.py
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from manip import manipulation as M

# Rig replaces module globals on `manipulation`; restore them after each test so a shared-process
# runner (pytest) can't leak a mock into another test file. Snapshot at import = the real functions.
_SNAP = {k: getattr(M, k) for k in
         ("set_hand_pose", "TransformHands", "TransformAgent",
          "_XTNFWD_LEFT_", "_GRIP_LEFT_", "_PLLBCK_LEFT_",
          "_XTNFWD_RIGHT_", "_GRIP_RIGHT_", "_PLLBCK_RIGHT_")}

try:
    import pytest

    @pytest.fixture(autouse=True)
    def _restore_manipulation():
        yield
        for k, v in _SNAP.items():
            setattr(M, k, v)
except ImportError:
    pass


class Rig:
    """Installs side-aware mocks on the manipulation module and records set_hand_pose calls."""
    def __init__(self, left_gripped=False, right_gripped=False, hover_after=None,
                 raise_on_extend=False):
        self.pose_calls = []          # list of (pose_tuple, hand)
        self.grip_toggled = 0
        self.extended_sides = []      # which hand's extend fn actually ran
        self._gripped = {"left": left_gripped, "right": right_gripped}
        self._hover_after = hover_after      # step index at which an item appears under the hand
        self._raise = raise_on_extend
        self._z = {"left": 0.0, "right": 0.0}
        self._extends = 0

        M.set_hand_pose = self._set_hand_pose
        M.TransformHands = self._transform_hands
        M.TransformAgent = lambda *a, **k: {"rotation": (5.0, 0.0, 0.0)}  # shallow pitch -> "too far"
        M._XTNFWD_LEFT_ = lambda: self._extend("left")
        M._GRIP_LEFT_ = self._grip
        M._PLLBCK_LEFT_ = self._pull
        M._XTNFWD_RIGHT_ = lambda: self._extend("right")
        M._GRIP_RIGHT_ = self._grip
        M._PLLBCK_RIGHT_ = self._pull

    def _set_hand_pose(self, pose, hand="left", **k):
        self.pose_calls.append((tuple(pose), hand))
        return True, tuple(pose), 0.0

    def _transform_hands(self, lt, lr, rt, rr):
        # Used for the grip-state read and _reach_once's start read (deltas are all zero here).
        return {"leftTranslation": (0.0, 0.0, self._z["left"]),
                "rightTranslation": (0.0, 0.0, self._z["right"]),
                "leftGrippedState": self._gripped["left"],
                "rightGrippedState": self._gripped["right"],
                "leftHoveredObject": "null", "rightHoveredObject": "null"}

    def _extend(self, side):
        if self._raise:
            raise RuntimeError("simulated sim transport error mid-reach")
        self._z[side] += 0.025
        self._extends += 1
        self.extended_sides.append(side)
        hovered = (self._hover_after is not None and self._extends >= self._hover_after)
        key = f"{side}Translation"
        return {key: (0.0, 0.0, self._z[side]),
                f"{side}HoveredObject": "item42" if hovered else "null",
                f"{side}GrippedState": False}

    def _grip(self):
        self.grip_toggled += 1
        return {"gripped": True}

    def _pull(self):
        return {}

    # convenience assertions on the recorded pose sequence
    def poses(self):
        return [p for (p, _h) in self.pose_calls]


def test_guard_refuses_explicit_hand_already_holding():
    rig = Rig(left_gripped=True)
    res = M.extend_arm_until_grabbed(hand="left")
    assert res.get("blocked") is True
    assert res.get("reason") == "the left hand is already holding an item"
    # Crucial: refusing must NOT touch the hand pose (no GRAB, no REST) - the carried item stays put.
    assert rig.pose_calls == [], f"guard must not move the hand, saw {rig.pose_calls}"
    assert rig.grip_toggled == 0, "guard must never toggle the grip (that would drop the item)"


def test_guard_refuses_when_both_hands_full():
    rig = Rig(left_gripped=True, right_gripped=True)
    res = M.extend_arm_until_grabbed()          # hand='auto'
    assert res.get("blocked") is True
    assert "both hands" in res.get("reason", "")
    assert rig.pose_calls == [] and rig.grip_toggled == 0


def test_auto_falls_back_to_right_when_left_full():
    rig = Rig(left_gripped=True, hover_after=3)
    res = M.extend_arm_until_grabbed()          # hand='auto' -> right (left is carrying)
    assert res.get("gripped") is True and res.get("hand") == "right"
    assert set(rig.extended_sides) == {"right"}, "must reach with the RIGHT hand only"
    assert [h for (_p, h) in rig.pose_calls] == ["right", "right"], \
        f"GRAB/REST must target the right hand, saw {rig.pose_calls}"


def test_success_sets_grab_then_restores_rest():
    rig = Rig(hover_after=3)
    res = M.extend_arm_until_grabbed()          # hand='auto' -> left (both free, left preferred)
    assert res.get("gripped") is True and res.get("hovered") == "item42"
    assert res.get("hand") == "left"
    assert rig.poses() == [M.GRAB_POSE, M.REST_POSE], \
        f"expected GRAB then REST, saw {rig.poses()}"


def test_miss_still_restores_rest():
    rig = Rig(hover_after=None)                 # item never appears -> miss
    res = M.extend_arm_until_grabbed()
    assert res.get("gripped") is False and "reason" in res
    assert rig.poses() == [M.GRAB_POSE, M.REST_POSE], \
        f"a miss must still end at REST, saw {rig.poses()}"


def test_exception_still_restores_rest():
    rig = Rig(raise_on_extend=True)
    raised = False
    try:
        M.extend_arm_until_grabbed()
    except RuntimeError:
        raised = True
    assert raised, "the mid-reach error should propagate"
    # ...but the finally must have restored REST first (GRAB set, then REST on the way out).
    assert rig.poses() == [M.GRAB_POSE, M.REST_POSE], \
        f"an exception must still leave the hand at REST, saw {rig.poses()}"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"OK: {len(tests)} extend_arm_until_grabbed guard/restore tests passed")


if __name__ == "__main__":
    _run()
