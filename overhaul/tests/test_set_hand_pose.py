"""Offline unit tests for manipulation.set_hand_pose (Phase 6.1, step 1).

No sim: manipulation.TransformHands is monkeypatched with a stateful mock that mimics the sim's
per-component clamp (handMoveRange = 0.5). Confirms the closed loop converges, splits an
over-clamp move across iterations, resolves named poses, drives the correct hand, and reports a
frame mismatch as arrived=False instead of a silent wrong pose.

    python tests/test_set_hand_pose.py      # or: pytest tests/test_set_hand_pose.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # overhaul/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from manip import manipulation as M

# This test replaces module globals on `manipulation`; restore them after each test so a shared-process
# runner (pytest) can't leak a mock into another test file. Snapshot at import = the real functions.
_SNAP = {"TransformHands": M.TransformHands}

try:
    import pytest

    @pytest.fixture(autouse=True)
    def _restore_manipulation():
        yield
        for k, v in _SNAP.items():
            setattr(M, k, v)
except ImportError:
    pass


class FakeHands:
    """Mimics TransformHands: applies each per-component delta clamped to +/-0.5 (Unity's
    handMoveRange), tracks both hands, and reports their translations. `frozen=True` ignores deltas
    to simulate a pose given in the wrong coordinate frame (the hand never moves toward target)."""
    def __init__(self, left=(0.0, 0.0, 0.0), right=(0.0, 0.0, 0.0), frozen=False,
                 left_rotation=(0.0, 0.0, 0.0), right_rotation=(0.0, 0.0, 0.0),
                 left_gripped=False, right_gripped=False):
        self.left, self.right, self.frozen = list(left), list(right), frozen
        self.left_rotation, self.right_rotation = list(left_rotation), list(right_rotation)
        self.left_gripped, self.right_gripped = left_gripped, right_gripped
        self.rotation_deltas = []
        self.calls = 0

    def __call__(self, lt, lr, rt, rr):
        self.calls += 1
        if not self.frozen:
            for k in range(3):
                self.left[k] += max(-0.5, min(0.5, lt[k]))
                self.right[k] += max(-0.5, min(0.5, rt[k]))
                self.left_rotation[k] = (self.left_rotation[k] + lr[k]) % 360
                self.right_rotation[k] = (self.right_rotation[k] + rr[k]) % 360
        self.rotation_deltas.append((tuple(lr), tuple(rr)))
        return {"leftTranslation": tuple(self.left), "rightTranslation": tuple(self.right),
                "leftRotation": tuple(self.left_rotation),
                "rightRotation": tuple(self.right_rotation),
                "leftGrippedState": self.left_gripped, "rightGrippedState": self.right_gripped,
                "leftHoveredObject": "null", "rightHoveredObject": "null"}


def _install(fake):
    M.TransformHands = fake
    return fake


def test_converges_to_named_rest():
    _install(FakeHands())
    arrived, reported, resid = M.set_hand_pose("rest")
    assert arrived and resid <= M._POSE_TOL
    assert all(abs(reported[k] - M.REST_POSE[k]) <= M._POSE_TOL for k in range(3))


def test_converges_to_named_grab():
    _install(FakeHands())
    arrived, reported, resid = M.set_hand_pose("grab")
    assert arrived
    assert all(abs(reported[k] - M.GRAB_POSE[k]) <= M._POSE_TOL for k in range(3))


def test_accepts_raw_xyz_tuple():
    _install(FakeHands())
    target = (0.1, -0.2, 0.15)
    arrived, reported, _ = M.set_hand_pose(target)
    assert arrived and all(abs(reported[k] - target[k]) <= M._POSE_TOL for k in range(3))


def test_large_move_splits_across_iterations():
    # x error to REST is ~1.11 m (> the 0.5 clamp), so it MUST take multiple TransformHands calls.
    fake = _install(FakeHands(left=(0.9, 0.0, 0.0)))
    arrived, _, resid = M.set_hand_pose("rest")
    assert arrived and resid <= M._POSE_TOL
    assert fake.calls >= 3, f"expected an over-clamp move to iterate, took {fake.calls} calls"


def test_drives_right_hand_only():
    fake = _install(FakeHands(left=(1.0, 1.0, 1.0)))
    arrived, reported, _ = M.set_hand_pose("grab", hand="right")
    assert arrived
    # left must be untouched (only the right delta slot was used)
    assert tuple(fake.left) == (1.0, 1.0, 1.0)
    # the named poses are LEFT-calibrated; the right hand gets the x-mirror (pose_for_hand)
    mirrored = M.pose_for_hand("grab", "right")
    assert all(abs(reported[k] - mirrored[k]) <= M._POSE_TOL for k in range(3))


def test_pose_for_hand_mirrors_x_for_right():
    assert M.pose_for_hand("rest", "left") == M.REST_POSE
    rx, ry, rz = M.pose_for_hand("rest", "right")
    assert (rx, ry, rz) == (-M.REST_POSE[0], M.REST_POSE[1], M.REST_POSE[2])
    # explicit xyz poses are treated as LEFT-frame and mirrored the same way
    assert M.pose_for_hand((0.2, 0.1, 0.3), "right") == (-0.2, 0.1, 0.3)
    assert M.INSPECTION_POSE == M.GRAB_POSE
    assert M.pose_for_hand("inspection", "right") == (
        -M.INSPECTION_POSE[0], M.INSPECTION_POSE[1], M.INSPECTION_POSE[2])


def test_frame_mismatch_reports_not_arrived():
    # Hand never moves toward the target -> residual can't shrink -> honest arrived=False.
    _install(FakeHands(left=(5.0, 5.0, 5.0), frozen=True))
    arrived, _, resid = M.set_hand_pose("rest")
    assert not arrived and resid > M._POSE_TOL


def test_transform_normalizes_positive_negative_and_wrapped_rotations_shortest_path():
    for rotation, expected_first_y_sign in (
            ((20, 30, 40), -1), ((-20, -30, -40), 1), ((0, 350, 0), 1)):
        fake = _install(FakeHands(left_rotation=rotation))
        arrived, state, tresid, rresid = M.set_hand_transform("rest", hand="left")
        assert arrived and tresid <= M._POSE_TOL and rresid <= M._ROT_TOL_DEG
        nonzero = [lr for lr, _ in fake.rotation_deltas if any(lr)]
        assert nonzero
        assert (nonzero[0][1] > 0) == (expected_first_y_sign > 0)
        assert all(abs(component) <= 15 for delta in nonzero for component in delta)
        assert max(abs(M._shortest_angle_delta(0, v))
                   for v in state["leftRotation"]) <= M._ROT_TOL_DEG


def test_presentation_mirrors_right_preserves_grip_and_refuses_empty_hand():
    fake = _install(FakeHands(left=(1, 1, 1), right=(-1, 1, 1), right_gripped=True,
                              right_rotation=(0, 350, 0)))
    result = M.present_right_item_for_inspection(times=99)
    assert result["arrived"] and result["hand"] == "right"
    assert tuple(fake.right) == M.pose_for_hand("inspection", "right")
    assert fake.right_gripped is True
    blocked = M.present_left_item_for_inspection()
    assert blocked["blocked"] and blocked["executed"] is False
    assert fake.left_gripped is False


def test_reset_transform_restores_both_hands_without_opening_grips():
    fake = _install(FakeHands(
        left=(0.2, 0.3, 0.4), right=(-0.2, 0.3, 0.4),
        left_rotation=(45, 350, 180), right_rotation=(315, 10, 180),
        left_gripped=True, right_gripped=True))
    for side in ("left", "right"):
        arrived, state, _, _ = M.set_hand_transform("rest", hand=side)
        assert arrived
        assert tuple(getattr(fake, side)) == M.pose_for_hand("rest", side)
        assert max(abs(M._shortest_angle_delta(0, v))
                   for v in state[f"{side}Rotation"]) <= M._ROT_TOL_DEG
    assert fake.left_gripped and fake.right_gripped


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"OK: {len(tests)} set_hand_pose tests passed")


if __name__ == "__main__":
    _run()
