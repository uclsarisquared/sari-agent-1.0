"""Offline unit tests for manipulation.set_hand_pose (Phase 6.1, step 1).

No sim: manipulation.TransformHands is monkeypatched with a stateful mock that mimics the sim's
per-component clamp (handMoveRange = 0.5). Confirms the closed loop converges, splits an
over-clamp move across iterations, resolves named poses, drives the correct hand, and reports a
frame mismatch as arrived=False instead of a silent wrong pose.

    python plan6/test_files/test_set_hand_pose.py      # or: pytest plan6/test_files/test_set_hand_pose.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # overhaul/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import manipulation as M

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
    def __init__(self, left=(0.0, 0.0, 0.0), right=(0.0, 0.0, 0.0), frozen=False):
        self.left, self.right, self.frozen = list(left), list(right), frozen
        self.calls = 0

    def __call__(self, lt, lr, rt, rr):
        self.calls += 1
        if not self.frozen:
            for k in range(3):
                self.left[k] += max(-0.5, min(0.5, lt[k]))
                self.right[k] += max(-0.5, min(0.5, rt[k]))
        return {"leftTranslation": tuple(self.left), "rightTranslation": tuple(self.right),
                "leftGrippedState": False, "rightGrippedState": False,
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
    assert all(abs(reported[k] - M.GRAB_POSE[k]) <= M._POSE_TOL for k in range(3))


def test_frame_mismatch_reports_not_arrived():
    # Hand never moves toward the target -> residual can't shrink -> honest arrived=False.
    _install(FakeHands(left=(5.0, 5.0, 5.0), frozen=True))
    arrived, _, resid = M.set_hand_pose("rest")
    assert not arrived and resid > M._POSE_TOL


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"OK: {len(tests)} set_hand_pose tests passed")


if __name__ == "__main__":
    _run()
