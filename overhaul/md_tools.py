from env import (
    _GRIP_LEFT_,
    _GRIP_RIGHT_,
    extend_left_hand_forward,
    extend_right_hand_forward,
    pull_left_hand_backward,
    pull_right_hand_backward,
)
from hand_reset import reset_hands_in_front2


def reach_and_grip_right(step_units: int = 3, max_attempts: int = 8) -> dict:
    for _ in range(max_attempts):
        extend_right_hand_forward(step_units)
        result = _GRIP_RIGHT_()
        if result.get("gripped"):
            reset_hands_in_front2(extra_elevation=-0.1, hand="right")
            return {"gripped": True}
    pull_right_hand_backward(step_units * max_attempts)
    return {"gripped": False}


def reach_and_grip_left(step_units: int = 3, max_attempts: int = 8) -> dict:
    for _ in range(max_attempts):
        extend_left_hand_forward(step_units)
        result = _GRIP_LEFT_()
        if result.get("gripped"):
            reset_hands_in_front2(extra_elevation=-0.1, hand="left")
            return {"gripped": True}
    pull_left_hand_backward(step_units * max_attempts)
    return {"gripped": False}
