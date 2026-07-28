from sim.env import (
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

def grip_left(units=1): _GRIP_LEFT_()
def grip_right(units=1): _GRIP_RIGHT_()

from vision.perception import (
    center_object_on_screen,
    retrieve_item
)

# grab_item_in_view_* (md_tools) drove the sim's ReachAtPixel command, which is NOT implemented in
# SariSandboxMY - it hits the `default: Unknown command` branch. Temporarily de-wired in favour of
# extend_arm_until_grabbed (manipulation.py), which grabs using only commands the sim handles.
from manip.manipulation import (
    extend_arm_until_grabbed,
    present_left_item_for_inspection,
    present_right_item_for_inspection,
    reset_left_hand_after_inspection,
    reset_right_hand_after_inspection,
)


# Dual-hand variants (2026-07-23): the bare action auto-selects a FREE hand (left preferred, right
# when the left is carrying - manipulation.resolve_grab_hand); the _left/_right variants pin the side
# for when the VLM wants a specific hand (e.g. keep an item in one hand while grabbing another).
def extend_arm_until_grabbed_left(times=1):
    return extend_arm_until_grabbed(times, hand="left")


def extend_arm_until_grabbed_right(times=1):
    return extend_arm_until_grabbed(times, hand="right")


NAVIGATION_ACTIONS_REF = {
    "move_forward": move_forward,
    "move_backward": move_backward,
    "move_left": move_left,
    "move_right": move_right,
    "pan_left": pan_left,
    "pan_right": pan_right,
    "tilt_up": tilt_up,
    "tilt_down": tilt_down,
}

PERCEPTION_ACTIONS_REF = {
    'center_object_on_screen': center_object_on_screen,
    # 'retrieve_item': retrieve_item,
    # 'approach_object': retrieve_item,
}

MANIPULATION_ACTIONS_REF = {
    'present_left_item_for_inspection': present_left_item_for_inspection,
    'present_right_item_for_inspection': present_right_item_for_inspection,
    'reset_left_hand_after_inspection': reset_left_hand_after_inspection,
    'reset_right_hand_after_inspection': reset_right_hand_after_inspection,
    'extend_left_hand_forward': extend_left_hand_forward,
    'extend_right_hand_forward': extend_right_hand_forward,
    'pull_left_hand_backward': pull_left_hand_backward,
    'pull_right_hand_backward': pull_right_hand_backward,
    'raise_left_hand': raise_left_hand,
    'raise_right_hand': raise_right_hand,
    'lower_left_hand': lower_left_hand,
    'lower_right_hand': lower_right_hand,
    'rotate_left_clockwise': rotate_left_clockwise,
    'rotate_left_counterclockwise': rotate_left_counterclockwise,
    'rotate_right_clockwise': rotate_right_clockwise,
    'rotate_right_counterclockwise': rotate_right_counterclockwise,
    'grip_left': grip_left,
    'grip_right': grip_right,
    'extend_arm_until_grabbed': extend_arm_until_grabbed,
    'extend_arm_until_grabbed_left': extend_arm_until_grabbed_left,
    'extend_arm_until_grabbed_right': extend_arm_until_grabbed_right,
}
