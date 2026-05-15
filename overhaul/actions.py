from env import (
    move_forward,
    move_backward,
    move_left,
    move_right,
    pan_left,
    pan_right,
    tilt_up,
    tilt_down,

    extend_left_hand_forward,
    extend_right_hand_forward,
    pull_left_hand_backward,
    pull_right_hand_backward,
    raise_left_hand,
    raise_right_hand,
    lower_left_hand,
    lower_right_hand,
    rotate_left_clockwise,
    rotate_left_counterclockwise,
    rotate_right_clockwise,
    rotate_right_counterclockwise,

    _GRIP_LEFT_,
    _GRIP_RIGHT_,
)

from Tests.perception import (
    center_object_on_screen,
    retrieve_item
)


NAVIGATION_ACTIONS_REF = {
    "move_forward": move_forward,
    "pan_left": pan_left,
    "pan_right": pan_right,
    "tilt_up": tilt_up,
    "tilt_down": tilt_down,
}

PERCEPTION_ACTIONS_REF = {
    'center_object_on_screen': retrieve_item,
    'approach_object': retrieve_item,
}