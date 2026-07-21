import math

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
    TransformHands,
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


def extend_arm_until_grabbed(times=1, hand="left", max_extend=25,
                             creep_steps=3, creep_len=0.1, max_pitch_deg=20.0):
    """Extend one hand straight forward until a grabbable item comes under it, grip it, then
    retract the hand to its starting pose. If the hand reaches its limit empty-handed, creep the
    BODY forward a little and try again. LEFT hand by default.

    This REPLACES grab_item_in_view_* (md_tools.py), whose ReachAtPixel command does NOT exist in
    SariSandboxMY - it lands in the sim's `default: Unknown command` branch. This tool uses only
    commands the sim actually implements: TransformHands (extend/pull the hand), ToggleGrip, and
    TransformAgent (creep the body).

    GRAB mechanism, verified against SariSandboxV2/Assets/Scripts/AgentControllerBase.cs:
      * Each extend step returns the hand state; `<side>HoveredObject` is the id of the item under
        the hand's collision detector, or "null" when nothing is there.
      * ToggleGrip reads that SAME detector: a toggle from the open hand runs InstantiateItemFromBBox
        iff an item is detected (ToggleGrip, ~line 809). So hovered-non-null <=> a grip toggle will
        actually pick the item up. We therefore extend until hovered, then toggle exactly once.
      * The hand's local offset is clamped at handMoveRange = 0.5 (TranslateHand, ~line 656), so
        from the default pose the hand only reaches a little further before it stalls. We detect the
        stall (world position stops changing) and stop reaching from this spot.

    FORWARD-CREEP and the ANGLE caveat (why it is handled the way it is):
      * The reach is short, so when a sweep stalls without a hover we nudge the body forward and
        re-reach, up to `creep_steps` creeps of `creep_len` m (a small, bounded total).
      * HORIZONTAL angle: the creep must go along the agent's HEADING, not raw world +Z.
        TransformAgent translation is WORLD-space (explore.py:462 - sending (0,0,step) moves
        world-north), so we build the delta from the yaw: (sin(yaw), 0, cos(yaw)) * len. Y is held
        at 0, so the creep is purely HORIZONTAL even when the camera is pitched down at a low item -
        it closes distance to the shelf without diving into the floor. Each creep is small and
        followed by a fresh reach (the hover flag is ground truth), so a small mis-aim self-corrects
        on the next reach instead of compounding.
      * VERTICAL angle: forward motion CANNOT close a vertical gap. If the camera is pitched steeply
        (|pitch| > max_pitch_deg) the target is a low/high row; creeping would just drive the body
        into the shelf before the angled hand reaches the item. So we do NOT creep there - we stop
        and report, leaving it to the caller to crouch (low rows) or raise/lower the hand. The agent
        has no crouch action wired yet, so bottom-shelf grabs remain an open gap.

    The caller is expected to have CENTRED the target in view first (perception). `times` is accepted
    so the mode-machine's `action_ref(time_units)` dispatch works, but it is IGNORED.

    NOT yet verified in a live Play-mode run; the mechanism is read off the C#, and creep_len /
    creep_steps / max_pitch_deg are first-guess defaults that want a sim check.

    Returns {'gripped': bool, 'hovered': <id|None>, 'creeps_used': int[, 'reason': str]}.
    """
    hand = str(hand).lower()
    assert hand in ("left", "right"), "hand must be 'left' or 'right'"
    if hand == "left":
        extend_fn, pull_fn, grip_fn = _XTNFWD_LEFT_, _PLLBCK_LEFT_, _GRIP_LEFT_
        trans_key, hover_key, grip_key = "leftTranslation", "leftHoveredObject", "leftGrippedState"
    else:
        extend_fn, pull_fn, grip_fn = _XTNFWD_RIGHT_, _PLLBCK_RIGHT_, _GRIP_RIGHT_
        trans_key, hover_key, grip_key = "rightTranslation", "rightHoveredObject", "rightGrippedState"

    _EMPTY = {None, "null", "None", ""}

    def _has_item(state):
        return state.get(hover_key) not in _EMPTY

    def _dist(a, b):
        return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5

    def _reach_once():
        """One extend-until-hover-or-stall sweep from the current rest pose, then a slow retract to
        that same pose - one pull-back per extend that actually MOVED the hand, so a clamped extend
        can't over-retract past the origin. A gripped item is parented to the hand, so it rides back
        with it. Returns (gripped, hovered)."""
        start = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        if start.get(grip_key):
            # A toggle on an already-closed hand would RELEASE what it holds - never re-toggle.
            return True, start.get(hover_key)

        prev = start[trans_key]
        moved_steps = 0
        stalled = 0
        hovered = None
        gripped = False
        for _ in range(max_extend):
            state = extend_fn()                  # extend 0.025 along local forward; returns hand state
            cur = state[trans_key]
            moved = _dist(cur, prev) > 1e-4
            prev = cur
            if _has_item(state):
                hovered = state.get(hover_key)
                gripped = bool(grip_fn().get("gripped"))   # ToggleGrip picks up the detected item
                if moved:
                    moved_steps += 1
                break
            if moved:
                moved_steps += 1
                stalled = 0
            else:
                stalled += 1
                if stalled >= 2:                 # hand clamped at its reach limit
                    break
        for _ in range(moved_steps):
            pull_fn()
        return gripped, hovered

    def _creep_forward(length):
        """Nudge the body forward along its HEADING (horizontal, pitch-independent). See the ANGLE
        note in the docstring for why the delta is built from the yaw and Y is pinned to 0."""
        yaw = math.radians(TransformAgent((0, 0, 0), (0, 0, 0))["rotation"][1])
        TransformAgent((math.sin(yaw) * length, 0.0, math.cos(yaw) * length), (0, 0, 0))

    gripped, hovered = _reach_once()
    creeps_used = 0
    reason = None
    while not gripped and creeps_used < creep_steps:
        pitch = TransformAgent((0, 0, 0), (0, 0, 0))["rotation"][0]
        pitch = ((pitch + 180) % 360) - 180      # eulerAngles wraps to [0,360); fold to [-180,180]
        if abs(pitch) > max_pitch_deg:
            reason = (f"steep pitch {pitch:.0f}deg (low/high row) - creep can't close a vertical "
                      f"gap; crouch or raise/lower the hand instead")
            break
        _creep_forward(creep_len)
        creeps_used += 1
        gripped, hovered = _reach_once()

    print(f"[extend_arm_until_grabbed] hand={hand} creeps={creeps_used} "
          f"hovered={hovered!r} gripped={gripped}" + (f" | {reason}" if reason else ""))
    result = {"gripped": gripped,
              "hovered": None if hovered in _EMPTY else hovered,
              "creeps_used": creeps_used}
    if reason:
        result["reason"] = reason
    return result
