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


NAVIGATION_ACTIONS = ("move_forward: Move forward 0.1 units. This will move in the Z-axis\n"
                      "move_backward: Move backward 0.1 units. This will move in the Z-axis\n"
                      "move_left: Move left 0.1 units. This will move in the X-axis\n"
                      "move_right: Move right 0.1 units. This will move in the X-axis\n"
                      "pan_left: Pan left 2.5 degrees.\n"
                      "pan_right: Pan right 2.5 degrees.\n"
                      "tilt_up: Tilt up 2.5 degrees.\n"
                      "tilt_down: Tilt down 2.5 degrees.\n")

PERCEPTION_ACTIONS = ("center_object_on_screen: Center the agent's body on the target object in the frame.\n")

ATOMIC_ACTIONS = NAVIGATION_ACTIONS + PERCEPTION_ACTIONS