# ACTIONS = ("MOVE_FWD: Move forward 0.1 units.\n"
#            "MOVE_BACK: Move backward 0.1 units.\n"
#            "MOVE_LEFT: Move left 0.1 units.\n"
#            "MOVE_RIGHT: Move right 0.1 units.\n"
#            "PAN_LEFT: Pan left 2.5 degrees.\n"
#            "PAN_RIGHT: Pan right 2.5 degrees.\n"
#            "TILT_UP: Tilt up 2.5 degrees.\n"
#            "TILT_DOWN: Tilt down 2.5 degrees.\n"
#            "EXTEND_LEFT: Extend left hand 0.025 units.\n"
#            "PULL_LEFT: Pull left hand 0.025 units.\n"
#            "EXTEND_RIGHT: Extend right hand 0.025 units.\n"
#            "PULL_RIGHT: Pull right hand 0.025 units.\n"
#            "GRIP_LEFT: Toggle left grip.\n"
#            "GRIP_RIGHT: Toggle right grip.\n"
#            "STOP: Stop all actions. Use this when you think you have completed the task.\n")


NAVIGATION_ACTIONS = ("move_forward: Move forward 0.1 meters. This will move in the Z-axis. Maximum 10 steps per action.\n"
                      "move_backward: Move backward 0.1 meters. Maximum 10 steps per action.\n"
                      "move_left: Move left 0.1 meters. Maximum 10 steps per action.\n"
                      "move_right: Move right 0.1 meters. Maximum 10 steps per action.\n"
                      "pan_left: Pan left 2.5 degrees. Maximum 15 steps per action.\n"
                      "pan_right: Pan right 2.5 degrees. Maximum 15 steps per action.\n"
                      "tilt_up: Tilt up 2.5 degrees. Maximum 10 steps per action.\n"
                      "tilt_down: Tilt down 2.5 degrees. Maximum 10 steps per action.\n")

PERCEPTION_ACTIONS = (
    "pan_left: Pan left 2.5 degrees. Maximum 15 steps per action.\n"
    "pan_right: Pan right 2.5 degrees. Maximum 15 steps per action.\n"
    "tilt_up: Tilt up 2.5 degrees. Maximum 10 steps per action.\n"
    "tilt_down: Tilt down 2.5 degrees. Maximum 10 steps per action.\n"
    "center_object_on_screen: Rotate the camera in a closed loop until the target object is centred in the frame (it detects the target and verifies the result). Use this to centre the target before grabbing - do not rely on eyeballed pan_left/pan_right for the final centring.\n"
    # "retrieve_item: Approach the target object, grab it with the agent's hand, and read it.\n"
)

MANIPULATION_ACTIONS = (
    "extend_left_hand_forward: Extend left hand forward 0.025 meters per step.\n"
    "extend_right_hand_forward: Extend right hand forward 0.025 meters per step.\n"
    "pull_left_hand_backward: Pull left hand backward 0.025 meters per step.\n"
    "pull_right_hand_backward: Pull right hand backward 0.025 meters per step.\n"
    "raise_left_hand: Raise left hand 0.025 meters per step.\n"
    "raise_right_hand: Raise right hand 0.025 meters per step.\n"
    "lower_left_hand: Lower left hand 0.025 meters per step.\n"
    "lower_right_hand: Lower right hand 0.025 meters per step.\n"
    "rotate_left_clockwise: Rotate left hand clockwise 15 degrees per step.\n"
    "rotate_left_counterclockwise: Rotate left hand counterclockwise 15 degrees per step.\n"
    "rotate_right_clockwise: Rotate right hand clockwise 15 degrees per step.\n"
    "rotate_right_counterclockwise: Rotate right hand counterclockwise 15 degrees per step.\n"
    "grip_left: Toggle left grip (times value is ignored).\n"
    "grip_right: Toggle right grip (times value is ignored).\n"
    "extend_arm_until_grabbed: Extend the LEFT hand straight forward until a grabbable item is under it, grip it, then retract to the starting pose (times value is ignored). Works ONLY in *manipulation* mode - the hands are inactive in perception/navigation mode, so calling it there does nothing. It does NOT aim, so centre the item in the frame first; if it reports gripped=False the item was out of reach, so move the body closer and retry.\n"
)

ATOMIC_ACTIONS = NAVIGATION_ACTIONS + PERCEPTION_ACTIONS + MANIPULATION_ACTIONS