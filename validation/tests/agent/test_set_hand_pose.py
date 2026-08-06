"""Offline unit tests for manipulation.set_hand_pose (Phase 6.1, step 1).

No sim: manipulation.TransformHands is monkeypatched with a stateful mock that mimics the sim's
per-component clamp (handMoveRange = 0.5). Confirms the closed loop converges, splits an
over-clamp move across iterations, resolves named poses, drives the correct hand, and reports a
frame mismatch as arrived=False instead of a silent wrong pose.

    uv run pytest validation/tests/agent/test_set_hand_pose.py      # or: pytest validation/tests/agent/test_set_hand_pose.py
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from manip import manipulation as M

# This test replaces module globals on `manipulation`; restore them after each test so a shared-process
# runner (pytest) can't leak a mock into another test file. Snapshot at import = the real functions.
_SNAP = {"TransformHands": M.TransformHands, "ResetHands": M.ResetHands}

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
        self.reset_calls = 0
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


class CoupledEulerHands(FakeHands):
    """Mimic Unity reporting another Euler component after a single-axis quaternion turn."""

    def __call__(self, lt, lr, rt, rr):
        result = super().__call__(lt, lr, rt, rr)
        if abs(lr[0]) > 0 or abs(lr[1]) > 0:
            self.left_rotation[0] = (self.left_rotation[0] + 23) % 360
            result["leftRotation"] = tuple(self.left_rotation)
        if abs(rr[0]) > 0 or abs(rr[1]) > 0:
            self.right_rotation[0] = (self.right_rotation[0] + 23) % 360
            result["rightRotation"] = tuple(self.right_rotation)
        return result


def _install(fake):
    M.TransformHands = fake
    def reset():
        fake.reset_calls += 1
        fake.left[:] = M.pose_for_hand("rest", "left")
        fake.right[:] = M.pose_for_hand("rest", "right")
        fake.left_rotation[:] = (0.0, 0.0, 0.0)
        fake.right_rotation[:] = (0.0, 0.0, 0.0)
        return fake((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    M.ResetHands = reset
    M._reset_inspection_sweep("left")
    M._reset_inspection_sweep("right")
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
    for rotation, expected_y_sign in (
            ((20, 30, 40), -1), ((-20, -30, -40), 1), ((0, 350, 0), 1)):
        fake = _install(FakeHands(left_rotation=rotation))
        arrived, state, tresid, rresid = M.set_hand_transform("rest", hand="left")
        assert arrived and tresid <= M._POSE_TOL and rresid <= M._ROT_TOL_DEG
        nonzero = [lr for lr, _ in fake.rotation_deltas if any(lr)]
        assert nonzero
        y_deltas = [delta[1] for delta in nonzero if abs(delta[1]) > 1e-9]
        assert y_deltas
        assert (y_deltas[0] > 0) == (expected_y_sign > 0)
        assert all(abs(component) <= 45 for delta in nonzero for component in delta)
        assert all(sum(abs(component) > 1e-9 for component in delta) == 1
                   for delta in nonzero)
        assert max(abs(M._shortest_angle_delta(0, v))
                   for v in state["leftRotation"]) <= M._ROT_TOL_DEG


def test_presentation_mirrors_right_without_rest_flash_and_preserves_grip():
    fake = _install(FakeHands(left=(1, 1, 1), right=(-1, 1, 1), right_gripped=True,
                              right_rotation=(45, 350, 37)))
    result = M.present_right_item_for_inspection(times=99)
    assert result["arrived"] and result["hand"] == "right"
    target = M.pose_for_hand("inspection", "right")
    assert all(abs(fake.right[k] - target[k]) <= M._POSE_TOL for k in range(3))
    assert tuple(round(v) for v in fake.right_rotation) == (45, 350, 37)
    assert fake.reset_calls == 0, "presentation must not visibly teleport the item through REST"
    assert fake.right_gripped is True
    blocked = M.present_left_item_for_inspection()
    assert blocked["blocked"] and blocked["executed"] is False
    assert fake.left_gripped is False


def test_inspection_face_sweep_emits_one_discrete_xy_turn_per_tool_call():
    fake = _install(FakeHands(left_rotation=(0, 0, 0), left_gripped=True))
    expected = list(M.INSPECTION_ROTATION_DELTAS)
    reported = []
    commanded = []
    for target_delta in expected:
        before = len(fake.rotation_deltas)
        result = M.rotate_left_to_next_inspection_face(times=999)
        assert result["arrived"]
        assert result["commanded_rotation_delta"] == target_delta
        new_nonzero = [
            left_delta for left_delta, _ in fake.rotation_deltas[before:]
            if any(left_delta)
        ]
        assert new_nonzero == [target_delta]
        commanded.append(result["commanded_rotation_delta"])
        reported.append(tuple(round(v) for v in result["rotation"]))
    assert commanded == expected
    assert reported == [
        (0, 90, 0), (0, 180, 0), (0, 270, 0), (0, 0, 0),
        (90, 0, 0), (0, 0, 0), (270, 0, 0),
    ]
    assert all(abs(rotation[2]) == 0 for rotation in reported)
    assert all(
        sum(abs(component) > 1e-9 for component in delta) <= 1
        for pair in fake.rotation_deltas for delta in pair
    ), "every TransformHands rotation delta must touch at most one axis"
    exhausted = M.rotate_left_to_next_inspection_face()
    assert exhausted["blocked"] and exhausted["faces_exhausted"] is True
    assert fake.left_gripped is True


def test_inspection_presentation_preserves_existing_roll_without_commanding_z():
    fake = _install(FakeHands(left_rotation=(0, 0, -90), left_gripped=True))
    presented = M.present_left_item_for_inspection()
    assert presented["arrived"]
    assert tuple(round(v) for v in presented["rotation"]) == (0, 0, 270)
    assert fake.reset_calls == 0

    results = [M.rotate_left_to_next_inspection_face()
               for _ in M.INSPECTION_ROTATION_DELTAS]
    assert all(result["arrived"] and result["executed"] is True for result in results)
    assert [result["commanded_rotation_delta"] for result in results] == \
        list(M.INSPECTION_ROTATION_DELTAS)
    assert all(round(result["rotation"][2]) == 270 for result in results)
    assert all(left_delta[2] == 0 and right_delta[2] == 0
               for left_delta, right_delta in fake.rotation_deltas)

    restored = M.reset_left_hand_after_inspection()
    assert restored["arrived"]
    assert fake.reset_calls == 1
    assert tuple(round(v) for v in restored["rotation"]) == (0, 0, 0)


def test_inspection_face_sweep_does_not_infer_progress_from_coupled_euler_readback():
    fake = _install(FakeHands(left_rotation=(45, 45, -90), left_gripped=True))
    result = M.rotate_left_to_next_inspection_face()
    assert result["arrived"] and result["executed"] is True
    assert result["commanded_rotation_delta"] == (0.0, 90.0, 0.0)
    nonzero = [left for left, _ in fake.rotation_deltas if any(left)]
    assert nonzero == [(0.0, 90.0, 0.0)]


def test_inspection_face_sweep_does_not_servo_or_oscillate_on_coupled_euler_reports():
    fake = _install(CoupledEulerHands(left_gripped=True, left_rotation=(0, 0, 37)))
    results = [M.rotate_left_to_next_inspection_face() for _ in range(3)]
    assert all(result["arrived"] and not result["blocked"] for result in results)
    nonzero = [left for left, _ in fake.rotation_deltas if any(left)]
    assert nonzero == [(0.0, 90.0, 0.0)] * 3
    assert all(
        sum(abs(value) > 1e-9 for value in delta) == 1
        and abs(next(value for value in delta if abs(value) > 1e-9)) == 90
        for delta in nonzero
    )


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
