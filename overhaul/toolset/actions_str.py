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
    "extend_arm_until_grabbed: Extend a FREE hand straight forward until a grabbable item is under it, grip it, then retract to the starting pose (times value is ignored). The hand is chosen automatically: LEFT if free, otherwise RIGHT (so you can carry one item and grab a second); if BOTH hands are already holding it refuses - check out or drop something first. The result's `hand` field says which hand grabbed. Works ONLY in *manipulation* mode - the hands are inactive in perception/navigation mode, so calling it there does nothing. It does NOT aim and does NOT move the body, so centre the item first; if it reports gripped=False the item was out of reach - move the body closer, then RE-CENTER (center_object_on_screen) before retrying.\n"
    "extend_arm_until_grabbed_left: Same as extend_arm_until_grabbed but forces the LEFT hand; refused if the left hand is already holding an item.\n"
    "extend_arm_until_grabbed_right: Same as extend_arm_until_grabbed but forces the RIGHT hand; refused if the right hand is already holding an item.\n"
    "checkout_held_item: Emit THIS to check out the item(s) you are holding in ONE deterministic step - it drives to the self-checkout counter, squares onto the scan pad, scans, and bags in the tray. When you are carrying items in BOTH hands, this ONE action checks out both in a single pass: it scans the left item, scans the right item, then bags them both (you do NOT call it twice). When only one hand holds, it checks out that item. Do NOT hand-sequence the driving / aligning / scanning / placing yourself - this single action does all of it (times value is ignored). Works ONLY in *manipulation* mode, and ONLY when the CURRENT GOAL is a checkout/scan-and-bag subtask - if your goal is merely to pick something up, do NOT check out; finish by emitting STOP and a later agent handles the checkout. Afterwards, read `last_checkout` in the state: STOP only once it shows scanned AND placed; if it did not scan or bag, re-emit checkout_held_item or reposition per its reason. Use checkout_held_item_left / checkout_held_item_right only if you need to check out ONE specific hand's item on its own.\n"
    "checkout_held_item_left: Same as checkout_held_item but forces the LEFT hand's item; refused if the left hand is empty.\n"
    "checkout_held_item_right: Same as checkout_held_item but forces the RIGHT hand's item; refused if the right hand is empty.\n"
)

# Held-item inspection is a restricted macro like the grab tool. The actor cannot choose rotations,
# presentation, grip changes, checkout, camera actions, or body motion while an item is held.
INSPECTION_ACTIONS = (
    "inspect_held_item: Run the complete restricted held-item inspection loop (times is ignored). The tool auto-selects a held hand, preferring LEFT when both hands hold items. Use a hand-specific variant when the inspection request targets one of two held items. It presents the item and uses a fresh isolated VLM visibility check for legibility plus a separate fresh label-presence check. Each search pass performs eight X turns of exactly 45 degrees followed by deterministic Y turns that check the top and bottom. Rotation stops only when the requested Nutrition Facts/expiration label is directly front-facing, not merely readable at an angle. That side then stays facing the camera while the hand moves 5 cm closer at each remaining stage, up to five total distance stages; PaddleOCR reads a centered item crop at every locked stage and its lines are supplied as auxiliary evidence. The actor cannot choose or sequence rotations. Read last_inspection: label_legible=true means read the exact requested values from the CURRENT screen, using ocr_lines as supporting text, and STOP. best_effort_read=true means the label is locked at the closest position but remains formally illegible; nevertheless use both the CURRENT screen and ocr_lines to transcribe the best answer possible and STOP. Do not call the macro again in either case. sweep_exhausted=true means no requested directly front-facing label was found after all five search sweeps.\n"
    "inspect_held_item_left: Run the same restricted inspection macro on the item in the LEFT hand; refused if the left hand is empty.\n"
    "inspect_held_item_right: Run the same restricted inspection macro on the item in the RIGHT hand; refused if the right hand is empty.\n"
)

ATOMIC_ACTIONS = NAVIGATION_ACTIONS + PERCEPTION_ACTIONS + MANIPULATION_ACTIONS
