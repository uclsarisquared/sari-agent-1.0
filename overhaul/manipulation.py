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


def extend_arm_until_grabbed(times=1, hand="left", max_extend=25, max_pitch_deg=20.0):
    """Extend one hand straight forward until a grabbable item comes under it, grip it, then
    retract the hand to its starting pose. If the hand reaches its limit empty-handed, report
    out-of-reach and let the CALLER reposition - this tool no longer moves the body. LEFT hand by
    default.

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

    NO FORWARD-CREEP (removed 2026-07-21, user directive): the reach is short, so it often stalls
    just shy of the item. This tool used to nudge the BODY forward and re-reach - but that motion
    deviated the agent off the centred target (a small lateral drift, amplified as it closed in, so
    the hand ended up over the NEIGHBOURING item). It now does ONE reach from where it stands and,
    if it can't grab, REPORTS out-of-reach for the caller to fix: move the body closer, then
    RE-CENTER (center_object_on_screen, since moving de-centres the target) and retry.
      * VERTICAL vs HORIZONTAL miss (reported in `reason` to guide that adjustment): if the camera
        is pitched steeply (|pitch| > max_pitch_deg) the target is a low/high row and moving forward
        can't close a vertical gap - raise/lower the hand (or crouch, once wired) instead. Otherwise
        the item is simply too far - close the distance.

    The caller is expected to have CENTRED the target in view first (perception). `times` is accepted
    so the mode-machine's `action_ref(time_units)` dispatch works, but it is IGNORED.

    NOT yet verified in a live Play-mode run; the mechanism is read off the C#.

    Returns {'gripped': bool, 'hovered': <id|None>[, 'reason': str]}.
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
            # Hand starts CLOSED. A reach needs an OPEN hand to have something to grip onto; leaving it
            # shut used to short-circuit to a phantom "gripped=True" over empty air. That never bit
            # while the grip command was a silent no-op, but once ToggleGrip actually fires a stray
            # close (keyboard Q, a prior missed grab's cleanup, an interrupted run) leaves the hand shut
            # and the next call would falsely claim a grab. So OPEN it (a toggle from closed releases),
            # re-read, then reach normally. If it happened to be holding an item this drops it - but
            # calling a GRAB with a full hand is a caller error, and a clean open beats a false grab.
            grip_fn()
            start = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

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

    gripped, hovered = _reach_once()

    # No body creep - one reach from where we stand. If it missed, tell the caller HOW to adjust:
    # a steep pitch means a low/high row (vertical gap, forward motion won't help), otherwise the
    # item is too far. Either way the caller must move, then RE-CENTER before retrying.
    reason = None
    if not gripped:
        pitch = TransformAgent((0, 0, 0), (0, 0, 0))["rotation"][0]
        pitch = ((pitch + 180) % 360) - 180      # eulerAngles wraps to [0,360); fold to [-180,180]
        if abs(pitch) > max_pitch_deg:
            reason = (f"out of reach at steep pitch {pitch:.0f}deg (low/high row): moving forward "
                      f"can't close a vertical gap - raise/lower the hand (or crouch), then re-center "
                      f"and retry")
        else:
            reason = ("out of reach: hand stalled with nothing under it - move the body closer, then "
                      "re-center on the target (center_object_on_screen) and retry")

    print(f"[extend_arm_until_grabbed] hand={hand} hovered={hovered!r} gripped={gripped}"
          + (f" | {reason}" if reason else ""))
    result = {"gripped": gripped, "hovered": None if hovered in _EMPTY else hovered}
    if reason:
        result["reason"] = reason
    return result


# ===== Phase D: depth-gated reach planning ======================================================
# plan_reach turns ONE RequestLidarCenter sample into a reach verdict. The tool still never moves the
# body (the 2026-07-21 no-creep directive); this only DECIDES, and the router acts on the verdict.
#
# REACH_ENVELOPE - MEASURED 2026-07-22 from 24 reach_probe grabs. The hand extends along the CAMERA
# GAZE, so a grab lands iff the SLANT distance to the centred target is within reach - it is NOT a
# vertical band + horizontal standoff (that model, hand_drop/r_eff/standoff, was REFUTED by the data:
# same-height items grabbed or missed purely on distance; see slamtest/plans/phaseD_reach_and_grab.md).
# Every grab was <=0.885 m, every miss >=0.900 m - a clean split, and `reachable iff distance<=0.89`
# predicted all 24 rows. Re-fit from a fresh CSV with `python fit_envelope.py`.
REACH_ENVELOPE = {
    "max_reach": 0.85,   # grab if the slant distance <= this (m). The ~0.89 m boundary minus a margin,
                         # so you grab INSIDE the edge, not at it. fit_envelope sets this from the CSV.
    "move_unit": 0.10,   # metres per move_forward unit (env.py _move_relative)
    "move_cap":  10,     # move_forward clamps a single call to 10 units
}


def _reach_result(verdict, sample=None, move_steps=0, target_height=None,
                  horizontal_gap=None, reason=""):
    s = sample or {}
    return {
        "verdict": verdict,            # reachable | move | crouch | bail | recenter | unavailable
        "move_steps": int(move_steps),
        "distance": s.get("distance"),
        "pitch_deg": s.get("pitch_deg"),
        "camera_height": s.get("camera_height"),
        "target_height": target_height,
        "horizontal_gap": horizontal_gap,
        "reason": reason,
    }


def plan_reach(sample, envelope=REACH_ENVELOPE):
    """Turn ONE RequestLidarCenter sample into a reach verdict - the deterministic geometry brain of
    Phase D. PURE function (no sim, no I/O), unit-tested in test_plan_reach.py.

    MEASURED MODEL (2026-07-22, 24 reach_probe grabs): the hand extends along the CAMERA GAZE, so
    reachability is decided by ONE quantity - the SLANT distance to the centred target - not by a
    vertical band + horizontal standoff. Every grab had distance <= 0.885 m, every miss >= 0.900 m
    (clean split, 24/24), and same-HEIGHT items with opposite outcomes were separated only by distance.
    The earlier hand_drop/r_eff/standoff model was refuted; see slamtest/plans/phaseD_reach_and_grab.md.

    Geometry (slant range d, gaze pitch theta [+ = down], camera world-Y cam_h):
        vertical_offset = d * sin(theta)     # + = target BELOW the camera; equals cam_h - target_height
        horizontal_gap  = d * cos(theta)     # forward distance (a body move is horizontal)
        target_height   = cam_h - vertical_offset
    Moving forward shrinks horizontal_gap (so the slant distance) but NOT vertical_offset (the item does
    not move), and the smallest slant distance reachable by moving alone is |vertical_offset|. So if
    |vertical_offset| already exceeds max_reach, moving can't help - you must change posture.

    Verdicts:
      reachable   - distance <= max_reach: grab now.
      move        - too far but |vertical_offset| <= max_reach: move `move_steps` forward, re-centre, retry.
      crouch      - too far AND target more than max_reach BELOW the camera: crouch to get closer, re-measure.
      bail        - too far AND target more than max_reach ABOVE the camera: moving/crouching can't reach it.
      recenter    - miss (nothing solid at centre): don't plan off a meaningless distance.
      unavailable - sample lacks pose (old sim build, pre-Phase-D recompile): caller falls back.
    """
    if not sample or sample.get("error"):
        return _reach_result("recenter", sample,
                             reason=f"lidar error: {(sample or {}).get('error', 'no sample')}")
    if sample.get("pitch_deg") is None or sample.get("camera_height") is None:
        return _reach_result("unavailable", sample,
                             reason="sample has no pose (pitch_deg/camera_height) - the sim needs the "
                                    "Phase-D recompile; falling back to a blind reach")
    if not sample.get("hit", False):
        return _reach_result("recenter", sample,
                             reason="no surface at frame centre (gap / occlusion / beyond range) - "
                                    "re-center on the target and retry")

    d = float(sample["distance"])
    theta = math.radians(float(sample["pitch_deg"]))
    cam_h = float(sample["camera_height"])
    vertical_offset = d * math.sin(theta)        # + = target below the camera
    horizontal_gap = d * math.cos(theta)
    target_height = cam_h - vertical_offset
    max_reach = envelope["max_reach"]

    if d <= max_reach:
        return _reach_result("reachable", sample, target_height=target_height,
                             horizontal_gap=horizontal_gap,
                             reason=f"in reach: slant distance {d:.2f} m <= max_reach {max_reach:.2f} m")

    if abs(vertical_offset) > max_reach:
        # The vertical gap alone exceeds the reach; moving forward cannot close it.
        if vertical_offset > 0:      # target below the camera -> crouching gets you closer to it
            return _reach_result("crouch", sample, target_height=target_height,
                                 horizontal_gap=horizontal_gap,
                                 reason=f"too low: target is {vertical_offset:.2f} m below the camera "
                                        f"(> reach {max_reach:.2f} m) - crouch to get closer, then "
                                        f"re-center and re-measure")
        return _reach_result("bail", sample, target_height=target_height,
                             horizontal_gap=horizontal_gap,
                             reason=f"too high: target is {-vertical_offset:.2f} m above the camera "
                                    f"(> reach {max_reach:.2f} m) - moving or crouching can't reach it")

    # Reachable once close enough: move forward to bring the slant distance down to max_reach.
    needed_gap = math.sqrt(max(0.0, max_reach ** 2 - vertical_offset ** 2))
    move_steps = max(1, min(envelope["move_cap"],
                            math.ceil((horizontal_gap - needed_gap) / envelope["move_unit"])))
    return _reach_result("move", sample, move_steps=move_steps, target_height=target_height,
                         horizontal_gap=horizontal_gap,
                         reason=f"move_forward {move_steps} (~{move_steps * envelope['move_unit']:.1f} m): "
                                f"slant distance {d:.2f} m > reach {max_reach:.2f} m; close the horizontal "
                                f"gap from {horizontal_gap:.2f} to ~{needed_gap:.2f} m")
