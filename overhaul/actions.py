from env import (
    move_forward,
    move_backward,
    move_left,
    move_right,
    tilt_up,
    tilt_down,
    turn_left_90,
    turn_right_90,
    snap_to_cardinal,
    get_heading,

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

from perception import (
    center_object_on_screen,
    retrieve_item,
    scan_shelf_left,
    scan_shelf_right,
)

from md_tools import reach_and_grip_right, reach_and_grip_left


NAVIGATION_ACTIONS_REF = {
    "move_forward": move_forward,
    "move_backward": move_backward,
    "move_left": move_left,
    "move_right": move_right,
    "tilt_up": tilt_up,
    "tilt_down": tilt_down,
    "turn_left_90":    lambda units=1: turn_left_90(),
    "turn_right_90":   lambda units=1: turn_right_90(),
    "snap_to_cardinal":lambda units=1: snap_to_cardinal(),
    "get_heading":     lambda units=1: get_heading(),
}

SCAN_ACTIONS_REF = {
    "move_left": move_left,
    "move_right": move_right,
    "tilt_up": tilt_up,
    "tilt_down": tilt_down,
    "scan_shelf_left": scan_shelf_left,
    "scan_shelf_right": scan_shelf_right,
}

PERCEPTION_ACTIONS_REF = {
    'center_object_on_screen': center_object_on_screen,
    # 'retrieve_item': retrieve_item,
    # 'approach_object': retrieve_item,
}

MANIPULATION_ACTIONS_REF = {
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
    'reach_and_grip_right': reach_and_grip_right,
    'reach_and_grip_left': reach_and_grip_left,
}