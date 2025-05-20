from env import (
    _MOVE_FWD_,
    _MOVE_BCK_,
    _MOVE_LEFT_,
    _MOVE_RIGHT_,
    _PAN_LEFT_,
    _PAN_RIGHT_,
    _TILT_UP_,
    _TILT_DOWN_,
    _XTNFWD_LEFT_,
    _PLLBCK_LEFT_,
    _XTNFWD_RIGHT_,
    _PLLBCK_RIGHT_,
    _GRIP_LEFT_,
    _GRIP_RIGHT_,

    move_forward,
    move_backward,
    move_left,
    move_right,
    pan_left,
    pan_right,
    pan_up,
    pan_down,

    extend_left_hand_forward,
    extend_right_hand_forward,
    pull_left_hand_backward,
    pull_right_hand_backward,
    raise_left_hand,
    raise_right_hand,
    lower_left_hand,
    lower_right_hand,
)
from perception import (
    center_item_on_screen,
    detect_object_in_frame_gemini
)

navigation_actions_ref = {
    "move_forward": move_forward,
    "move_backward": move_backward,
    "move_left": move_left,
    "move_right": move_right,
    "pan_left": pan_left,
    "pan_right": pan_right,
    "pan_up": pan_up,
    "pan_down": pan_down,
}
perception_actions_ref = {
    'center_object_on_screen': detect_object_in_frame_gemini,
}
manipulation_actions_ref = {
    'extend_left_hand_forward': extend_left_hand_forward,
    'pull_left_hand_backward': pull_left_hand_backward,
    'extend_right_hand_forward': extend_right_hand_forward,
    'pull_right_hand_backward': pull_right_hand_backward,
    'raise_left_hand': raise_left_hand,
    'raise_right_hand': raise_right_hand,
    'lower_left_hand': lower_left_hand,
    'lower_right_hand': lower_right_hand,
    'toggle_left_grip': _GRIP_LEFT_,
    'toggle_right_grip': _GRIP_RIGHT_,
}
actions_ref = {
    'MOVE_FWD': _MOVE_FWD_,
    'MOVE_BACK': _MOVE_BCK_,
    'MOVE_LEFT': _MOVE_LEFT_,
    'MOVE_RIGHT': _MOVE_RIGHT_,
    'PAN_LEFT': _PAN_LEFT_,
    'PAN_RIGHT': _PAN_RIGHT_,
    'TILT_UP': _TILT_UP_,
    'TILT_DOWN': _TILT_DOWN_,
    'EXTEND_LEFT': _XTNFWD_LEFT_,
    'PULL_LEFT': _PLLBCK_LEFT_,
    'EXTEND_RIGHT': _XTNFWD_RIGHT_,
    'PULL_RIGHT': _PLLBCK_RIGHT_,
    'GRIP_LEFT': _GRIP_LEFT_,
    'GRIP_RIGHT': _GRIP_RIGHT_,
    
}