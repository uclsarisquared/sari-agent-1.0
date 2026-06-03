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


NAVIGATION_ACTIONS = ("move_forward: Move forward 0.1 meters. This will move in the Z-axis.\n"
                      "move_backward: Move backward 0.1 meters.\n"
                      "move_left: Move left 0.1 meters.\n"
                      "move_right: Move right 0.1 meters.\n"
                      "tilt_up: Tilt up 2.5 degrees.\n"
                      "tilt_down: Tilt down 2.5 degrees.\n"
                      "turn_left_90: Snap body yaw 90 degrees counter-clockwise to the nearest cardinal direction.\n"
                      "turn_right_90: Snap body yaw 90 degrees clockwise to the nearest cardinal direction.\n"
                      "snap_to_cardinal: Snap body yaw to the nearest cardinal direction (0°, 90°, 180°, 270°). Use this to recover from yaw drift.\n"
                      "get_heading: Print the current cardinal heading as a human-readable label (e.g. North, East, South, West). Use this to confirm orientation before planning movement.\n")

SCAN_ACTIONS = ("move_left: Move left 0.1 meters.\n"
                "move_right: Move right 0.1 meters.\n"
                "tilt_up: Tilt up 2.5 degrees.\n"
                "tilt_down: Tilt down 2.5 degrees.\n"
                "scan_shelf_left: Strafe left along the shelf face 0.1 meters per step. Use while facing the shelf to sweep across products.\n"
                "scan_shelf_right: Strafe right along the shelf face 0.1 meters per step. Use while facing the shelf to sweep across products.\n")

PERCEPTION_ACTIONS = (
    "center_object_on_screen: Center the agent's body on the target object in the frame.\n"
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
    "reach_and_grip_right: Extend the right hand forward in steps, attempting a grip at each step until the item is gripped or max extension is reached. Hand resets to default position on success. Use once vertically and horizontally aligned with the target item.\n"
    "reach_and_grip_left: Extend the left hand forward in steps, attempting a grip at each step until the item is gripped or max extension is reached. Hand resets to default position on success. Use once vertically and horizontally aligned with the target item.\n"
)

ATOMIC_ACTIONS = NAVIGATION_ACTIONS + SCAN_ACTIONS + PERCEPTION_ACTIONS + MANIPULATION_ACTIONS