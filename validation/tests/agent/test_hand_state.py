"""Offline coverage for grip-toggle versus physical held-item state."""
import os
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim import env


def _hand_reply(*, left_gripping, right_gripping, holding_lines=True):
    lines = [
        "Current left hand position: (-0.14, -0.08, 0.25)",
        "Current left hand rotation: (0.0, 0.0, 89.57)",
        "Left hand hovering: null",
        f"Left hand gripping: {left_gripping}",
        "Current right hand position: ",
        "Current right hand position: (0.18, -0.09, 0.27)",
        "Current right hand rotation: (0.0, 0.0, 0.0)",
        "Right hand hovering: null",
        f"Right hand gripping: {right_gripping}",
    ]
    if holding_lines:
        lines.extend([
            "Left hand holding item: False",
            "Right hand holding item: True",
        ])
    return "\n".join(lines)


@pytest.mark.parametrize("left_gripping,right_gripping", [(True, True), (False, True)])
def test_transform_hands_uses_physical_attachment_as_public_gripped_state(
        monkeypatch, left_gripping, right_gripping):
    async def send_command(*_args, **_kwargs):
        return _hand_reply(
            left_gripping=left_gripping,
            right_gripping=right_gripping,
            holding_lines=True,
        )

    monkeypatch.setattr(env, "SendCommand", send_command)
    state = env.TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    assert state["leftGripClosedState"] is left_gripping
    assert state["leftHoldingItem"] is False
    assert state["leftGrippedState"] is False
    assert state["rightGripClosedState"] is right_gripping
    assert state["rightHoldingItem"] is True
    assert state["rightGrippedState"] is True


def test_transform_hands_falls_back_to_grip_toggle_with_older_sim_reply(monkeypatch):
    async def send_command(*_args, **_kwargs):
        return _hand_reply(left_gripping=True, right_gripping=False, holding_lines=False)

    monkeypatch.setattr(env, "SendCommand", send_command)
    state = env.TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    assert state["leftHoldingItem"] is True
    assert state["leftGrippedState"] is True
    assert state["rightHoldingItem"] is False
    assert state["rightGrippedState"] is False
