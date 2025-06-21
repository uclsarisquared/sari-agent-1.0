from env import (
    _XTNFWD_LEFT_,
    _XTNFWD_RIGHT_,
    _GRIP_LEFT_,
    _GRIP_RIGHT_,
    _PLLBCK_LEFT_,
    _PLLBCK_RIGHT_,
    _RSE_LEFT_,
    _RSE_RIGHT_,
    _REQUEST_SCREENSHOT_,
    TransformAgent,
    rotate_right_clockwise,
    rotate_left_clockwise,
    move_backward
)
from hand_reset import reset_hands_in_front2, reset_hands_in_front

def reset_agent_cam_to_forward():
    pitch = TransformAgent((0,0,0), (0,0,0))['rotation'][0]
    return TransformAgent((0,0,0), (0-pitch,0,0))

def reach_and_grasp(hand='left', max_attempts=20):
    grasped = False
    attempts = 0

    while not grasped and attempts < max_attempts:
        if hand == 'left':
            _XTNFWD_LEFT_()  # e.g., moves hand slightly forward
            grasped = _GRIP_LEFT_()['gripped']
        elif hand == 'right':
            _XTNFWD_RIGHT_()
            grasped = _GRIP_RIGHT_()['gripped']
        attempts += 1

    return grasped, attempts



def pull_back(hand='left', max_frames=10):
    frames_captured = 0

    while frames_captured < max_frames:
        if hand == 'left':
            _PLLBCK_LEFT_()
        elif hand == 'right':
            _PLLBCK_RIGHT_()
        
        frames_captured += 1


def rotate_and_read(hand="left", max_frames=10, retract_steps=5, text_read_fn=None):
    # Step 2: Rotate and OCR
    texts = []
    rotate_fn = rotate_left_clockwise if hand == 'left' else rotate_right_clockwise

    for i in range(4):  # Full 360° sweep
        _REQUEST_SCREENSHOT_()
        if text_read_fn:
            texts.append(text_read_fn())
        rotate_fn(units=6)
    return texts


def raise_hand_to_eye_level(hand="left", raise_steps=5):
    if hand == "left":
        for i in range(raise_steps):
            _RSE_LEFT_()
    elif hand == "right":
        for i in range(raise_steps):
            _RSE_RIGHT_()


def grab_and_read_item(hand="left", max_attempts=30, text_read_fn=None):
    print('[REACH AND GRAB]')
    accessed, attempts = reach_and_grasp(hand=hand, max_attempts=max_attempts)
    
    if accessed:
        move_backward(units=5)
        reset_agent_cam_to_forward()
        reset_hands_in_front2(extra_elevation=-0.1, hand="left")
        raise_hand_to_eye_level(hand=hand)
        return rotate_and_read(hand="left", text_read_fn=text_read_fn)
    return ["No object grabbed"]
