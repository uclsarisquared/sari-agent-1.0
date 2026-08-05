import math

from sim.env import (
    _XTNFWD_LEFT_,
    _XTNFWD_RIGHT_,
    _GRIP_LEFT_,
    _GRIP_RIGHT_,
    _PLLBCK_LEFT_,
    _PLLBCK_RIGHT_,
    _RSE_LEFT_,
    _RSE_RIGHT_,
    _REQUEST_SCREENSHOT_,
    ResetHands,
    TransformAgent,
    TransformHands,
    rotate_right_clockwise,
    rotate_left_clockwise,
    move_backward
)
from sim.hand_reset import reset_hands_in_front2, reset_hands_in_front

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
        reset_hands_in_front2(extra_elevation=-0.1, hand=hand)
        raise_hand_to_eye_level(hand=hand)
        return rotate_and_read(hand=hand, text_read_fn=text_read_fn)
    return ["No object grabbed"]


# ===== Phase 6.1: hand-pose state machine (REST / GRAB) =========================================
# Hands stay ACTIVE throughout a task now (no longer disabled for nav/perception), parked at a named
# REST pose so a carried item is never dropped by a mode change - it rides at REST. The two poses are
# the user's manual calibration (2026-07-22), LEFT-hand agent-local xyz, validated live by the step-0
# calibration (validation/calibration/step0_hand_pose_probe.py, user-confirmed working 2026-07-22).
# HAND FRAME DEFINITION (dual-hand, 2026-07-23). All poses here are AGENT-LOCAL xyz, the frame
# TransformHands reports in: +x = the AGENT'S RIGHT, +y = up, +z = forward. "Left hand" = the hand
# resting on the agent's LEFT side (negative x); "right hand" = its counterpart at positive x. The
# calibration below is the LEFT hand's (user-measured 2026-07-22); the RIGHT hand is defined as that
# same pose MIRRORED across the sagittal plane x = 0 - i.e. x negated, y and z unchanged (user
# directive 2026-07-23: the right hand's rest/active positions are the left's mirror along x).
REST_POSE = (-0.213, -0.09, 0.26)  # LEFT-hand out-of-frame resting pose: navigate, perceive, CARRY
                                   # (z revised 0.2 -> 0.26, user-validated 2026-07-22).
                                   # Right hand rests at (+0.213, -0.09, 0.26) via pose_for_hand.
GRAB_POSE = (-0.01, 0.006, 0.33)   # LEFT-hand centred forward "ready to grab" pose - TOOL-INTERNAL
                                   # (see below). Right hand: (+0.01, 0.006, 0.33) via pose_for_hand.
INSPECTION_POSE = (-0.01, 0.006, 0.33)  # Separately tunable held-item presentation calibration.
_NAMED_POSES = {"rest": REST_POSE, "grab": GRAB_POSE, "inspection": INSPECTION_POSE}


def pose_for_hand(pose, hand):
    """Resolve a pose (name or LEFT-frame xyz) to the given hand's agent-local coordinates.
    left -> unchanged; right -> x negated (the sagittal mirror defined above). Callers therefore
    always author poses in the LEFT hand's frame, and this is the ONE place the mirror lives."""
    target = _NAMED_POSES[pose] if isinstance(pose, str) else tuple(pose)
    if str(hand).lower() == "right":
        return (-target[0], target[1], target[2])
    return target
_HAND_MOVE_RANGE = 0.5   # Unity clamps each TransformHands delta component here (TranslateHand, ~656)
_POSE_TOL = 0.012        # m; "arrived" once the reported translation is within this of the target
_ROT_STEP_DEG = 45.0
_ROT_TOL_DEG = 1.0


def _vlen(v):
    return math.sqrt(sum(c * c for c in v))


def set_hand_pose(pose, hand="left", max_iters=5):
    """Closed-loop drive one hand to `pose` (a name 'rest'/'grab', or an agent-local xyz), reading the
    reported translation and nudging by the residual each iteration. Returns (arrived, reported, resid).

    Closed-loop on purpose (validated by the step-0 probe, user-confirmed 2026-07-22):
      - it self-corrects Unity's per-call clamp (handMoveRange = 0.5 on each delta component): a move
        larger than the clamp just takes another iteration, so one logical set_hand_pose still arrives
        ("split into two calls", as the phase6 plan flags).
      - if the reported translation does NOT march toward the target, the residual won't shrink and this
        returns arrived=False - the honest signal "the pose is in a different frame than assumed",
        rather than a silent wrong pose.

    Only the mode router (agent._set_hand_pose -> 'rest') and the grab/place tools (transient 'grab')
    call this. TransformHands takes agent-local DELTAS, so this drives by (target - reported) each step.
    Named poses are LEFT-calibrated; for the RIGHT hand they are mirrored across x (pose_for_hand).
    An explicit xyz pose is mirrored the same way, so callers always pass LEFT-frame coordinates.
    """
    target = pose_for_hand(pose, hand)
    tkey = "leftTranslation" if hand == "left" else "rightTranslation"
    cur = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))[tkey]
    for _ in range(max_iters):
        err = [target[k] - cur[k] for k in range(3)]
        if _vlen(err) <= _POSE_TOL:
            return True, cur, _vlen(err)
        clamp = _HAND_MOVE_RANGE - 1e-3
        d = tuple(max(-clamp, min(clamp, err[k])) for k in range(3))
        if hand == "left":
            cur = TransformHands(d, (0, 0, 0), (0, 0, 0), (0, 0, 0))[tkey]
        else:
            cur = TransformHands((0, 0, 0), (0, 0, 0), d, (0, 0, 0))[tkey]
    resid = _vlen([target[k] - cur[k] for k in range(3)])
    return resid <= _POSE_TOL, cur, resid


def _shortest_angle_delta(target, current):
    """Signed target-current delta in [-180, 180), including wrapped simulator Euler values."""
    return (float(target) - float(current) + 180.0) % 360.0 - 180.0


def set_hand_transform(pose, rotation=(0, 0, 0), hand="left", max_iters=60):
    """Closed-loop translation + Euler drive for inspection presentation/restoration.

    Rotation residuals follow the shortest wrapped path. Each TransformHands call changes exactly
    one Euler axis by at most 45 degrees. Axis-wise correction is important when returning an
    inspected item to REST: simultaneous Euler corrections can couple in Unity and leave a residual
    Z/roll even when the reported X/Y appear settled. This helper never toggles either grip.
    Returns ``(arrived, state, translation_residual, rotation_residual)``.
    """
    hand = str(hand).lower()
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    target_translation = pose_for_hand(pose, hand)
    target_rotation = tuple(float(v) for v in rotation)
    tkey = f"{hand}Translation"
    rkey = f"{hand}Rotation"
    state = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    for _ in range(max_iters):
        cur_t = state[tkey]
        cur_r = state[rkey]
        terr = tuple(target_translation[k] - cur_t[k] for k in range(3))
        rerr = tuple(_shortest_angle_delta(target_rotation[k], cur_r[k]) for k in range(3))
        tresid, rresid = _vlen(terr), max(abs(v) for v in rerr)
        if tresid <= _POSE_TOL and rresid <= _ROT_TOL_DEG:
            return True, state, tresid, rresid

        clamp = _HAND_MOVE_RANGE - 1e-3
        td = tuple(max(-clamp, min(clamp, v)) for v in terr)
        rotation_axis = max(range(3), key=lambda index: abs(rerr[index]))
        rd_values = [0.0, 0.0, 0.0]
        rd_values[rotation_axis] = max(
            -_ROT_STEP_DEG, min(_ROT_STEP_DEG, rerr[rotation_axis]))
        rd = tuple(rd_values)
        zero = (0, 0, 0)
        if hand == "left":
            state = TransformHands(td, rd, zero, zero)
        else:
            state = TransformHands(zero, zero, td, rd)

    cur_t, cur_r = state[tkey], state[rkey]
    tresid = _vlen(tuple(target_translation[k] - cur_t[k] for k in range(3)))
    rresid = max(abs(_shortest_angle_delta(target_rotation[k], cur_r[k])) for k in range(3))
    return tresid <= _POSE_TOL and rresid <= _ROT_TOL_DEG, state, tresid, rresid


def _reset_inspection_sweep(hand):
    _INSPECTION_SWEEP_STEP[str(hand).lower()] = 0


def _inspection_transform(hand, pose):
    hand = str(hand).lower()
    grips = hand_grip_states()
    if hand not in grips:
        return {"blocked": True, "executed": False, "reason": "hand must be 'left' or 'right'"}
    if not grips[hand]:
        return {"blocked": True, "executed": False,
                "reason": f"the {hand} hand is empty; inspection actions require a held item"}

    if pose == "inspection":
        # The restricted inspection macro resets both hands immediately before calling this action.
        # Presentation itself changes translation only, avoiding any additional REST flash between
        # that deliberate reset and the first inspection evidence frame.
        arrived, _, tresid = set_hand_pose(pose, hand=hand, max_iters=5)
        zero = (0, 0, 0)
        state = TransformHands(zero, zero, zero, zero)
    else:
        # End-of-inspection restoration has no following evidence frame to preserve, so the native
        # atomic reset is the correct path here.
        state = ResetHands()
        target = pose_for_hand(pose, hand)
        reported = state[f"{hand}Translation"]
        tresid = _vlen(tuple(target[k] - reported[k] for k in range(3)))
        arrived = tresid <= _POSE_TOL
    if not state.get(f"{hand}GrippedState"):
        _reset_inspection_sweep(hand)
        return {
            "blocked": True,
            "executed": True,
            "hand": hand,
            "arrived": False,
            "reason": f"inspection transform did not preserve the {hand} grip",
        }
    rresid = max(
        abs(_shortest_angle_delta(0.0, value))
        for value in state[f"{hand}Rotation"]
    )
    if pose != "inspection":
        arrived = arrived and rresid <= _ROT_TOL_DEG
    _reset_inspection_sweep(hand)
    return {
        "blocked": not arrived,
        "executed": True,
        "hand": hand,
        "arrived": arrived,
        "translation": state.get(f"{hand}Translation"),
        "rotation": state.get(f"{hand}Rotation"),
        "translation_residual": tresid,
        "rotation_residual": rresid,
        **({} if arrived else {"reason": f"{hand} hand transform did not converge"}),
    }


def present_left_item_for_inspection(times=1):
    return _inspection_transform("left", "inspection")


def present_right_item_for_inspection(times=1):
    return _inspection_transform("right", "inspection")


def reset_left_hand_after_inspection(times=1):
    return _inspection_transform("left", "rest")


def reset_right_hand_after_inspection(times=1):
    return _inspection_transform("right", "rest")


# One relative turn per agent tool call. Four yaw turns visit every side and return to the default
# face. Three pitch turns then visit top, default, and bottom. The duplicate default frames are
# intentional: reaching the opposite pitch face without a 180-degree command requires two 90-degree
# calls. Explicit progress replaces inference from Unity's coupled/wrapped Euler readback.
INSPECTION_ROTATION_DELTAS = (
    (0.0, 90.0, 0.0),
    (0.0, 90.0, 0.0),
    (0.0, 90.0, 0.0),
    (0.0, 90.0, 0.0),
    (90.0, 0.0, 0.0),
    (-90.0, 0.0, 0.0),
    (-90.0, 0.0, 0.0),
)
_INSPECTION_SWEEP_STEP = {"left": 0, "right": 0}


def _next_inspection_face(hand):
    hand = str(hand).lower()
    grips = hand_grip_states()
    if hand not in grips:
        return {"blocked": True, "executed": False, "reason": "hand must be 'left' or 'right'"}
    if not grips[hand]:
        return {"blocked": True, "executed": False,
                "reason": f"the {hand} hand is empty; inspection actions require a held item"}

    step = _INSPECTION_SWEEP_STEP[hand]
    if step >= len(INSPECTION_ROTATION_DELTAS):
        return {
            "blocked": True,
            "executed": False,
            "hand": hand,
            "faces_exhausted": True,
            "reason": "all side, top, and bottom inspection faces have already been shown",
        }

    rotation_delta = INSPECTION_ROTATION_DELTAS[step]
    nonzero_axes = [
        axis for axis, value in enumerate(rotation_delta)
        if abs(value) > _ROT_TOL_DEG
    ]
    if (len(nonzero_axes) != 1 or nonzero_axes[0] not in (0, 1)
            or abs(rotation_delta[nonzero_axes[0]]) not in (45.0, 90.0)):
        return {
            "blocked": True,
            "executed": False,
            "hand": hand,
            "reason": "invalid inspection turn: exactly one 45/90-degree X or Y delta is allowed",
        }

    zero = (0, 0, 0)
    if hand == "left":
        state = TransformHands(zero, rotation_delta, zero, zero)
    else:
        state = TransformHands(zero, zero, zero, rotation_delta)
    _INSPECTION_SWEEP_STEP[hand] = step + 1
    face_index = step + 1
    return {
        "blocked": False,
        "executed": True,
        "hand": hand,
        "arrived": True,
        "face_index": face_index,
        "face_phase": "side" if face_index <= 4 else "top_bottom",
        "commanded_rotation_delta": rotation_delta,
        "translation": state.get(f"{hand}Translation"),
        "rotation": state.get(f"{hand}Rotation"),
    }


def rotate_left_to_next_inspection_face(times=1):
    return _next_inspection_face("left")


def rotate_right_to_next_inspection_face(times=1):
    return _next_inspection_face("right")


# ===== Dual-hand selection policy (2026-07-23) ==================================================
# Every grab-side tool takes hand='left'|'right'|'auto'. 'auto' resolves DETERMINISTICALLY from the
# live grip state (one TransformHands no-op read) - the VLM picks a side only when it wants to:
#   grabbing   -> a FREE hand, LEFT preferred (the left is the measured calibration; the right is its
#                 x-mirror). Both full -> None: refuse, don't force-open (that drops a carried item).
#   releasing  -> the hand that IS holding (scan/place/drop act on a held item). Both holding ->
#                 LEFT first (stable order, the caller can name 'right' to pick the other item).
#                 Neither holding -> None: refuse.

def hand_grip_states():
    hs = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    return {"left": bool(hs.get("leftGrippedState")), "right": bool(hs.get("rightGrippedState"))}


def resolve_grab_hand(hand="auto"):
    """Return ('left'|'right', None) or (None, reason) for a tool that wants an EMPTY hand."""
    hand = str(hand).lower()
    grips = hand_grip_states()
    if hand in ("left", "right"):
        if grips[hand]:
            return None, f"the {hand} hand is already holding an item"
        return hand, None
    assert hand == "auto", "hand must be 'left', 'right' or 'auto'"
    for side in ("left", "right"):
        if not grips[side]:
            return side, None
    return None, "both hands are already holding an item - check out or drop one first"


def resolve_release_hand(hand="auto"):
    """Return ('left'|'right', None) or (None, reason) for a tool that acts on a HELD item."""
    hand = str(hand).lower()
    grips = hand_grip_states()
    if hand in ("left", "right"):
        if not grips[hand]:
            return None, f"the {hand} hand is not holding an item"
        return hand, None
    assert hand == "auto", "hand must be 'left', 'right' or 'auto'"
    for side in ("left", "right"):
        if grips[side]:
            return side, None
    return None, "no hand is holding an item"


def extend_arm_until_grabbed(times=1, hand="auto", max_extend=25, max_pitch_deg=20.0):
    """Set the GRAB pose, extend the hand straight forward until a grabbable item comes under it, grip
    it, then retract to REST (Phase 6.1 - the tool owns both poses). If the hand reaches its limit
    empty-handed, report out-of-reach and let the CALLER reposition - this tool no longer moves the
    body. hand='auto' (default) picks a FREE hand via resolve_grab_hand (left preferred, right when
    the left is carrying); 'left'/'right' forces a side, refused if that hand is full.

    Phase 6.1 hand-pose ownership: this tool sets GRAB_POSE at entry and restores REST_POSE on EVERY
    exit path (success, miss, exception), so a gripped item rides back to the carry pose. It also
    REFUSES (returns {'blocked': True, ...}) if the hand is already gripping at entry, rather than
    force-opening it - force-opening would drop a carried item mid-task.

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

    Returns {'gripped': bool, 'hovered': <id|None>[, 'reason': str]}, or
    {'blocked': True, 'reason': 'hand already holding an item'} if it refused (full-hand guard).
    """
    # Resolve the side FIRST (auto -> a free hand; explicit -> validated free). This replaces the old
    # entry-time full-hand guard: a full/unavailable hand is refused here, never force-opened.
    hand, refuse_reason = resolve_grab_hand(hand)
    if hand is None:
        print(f"[extend_arm_until_grabbed] REFUSED: {refuse_reason}")
        return {"blocked": True, "reason": refuse_reason}
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
        """One extend-until-hover-or-stall sweep from the GRAB pose the caller just set, then a slow
        retract - one pull-back per extend that actually MOVED the hand, so a clamped extend can't
        over-retract past the origin. A gripped item is parented to the hand, so it rides back with it.
        Returns (gripped, hovered). The hand is guaranteed OPEN here: the full-hand guard in
        extend_arm_until_grabbed refuses a closed hand before this runs, so there is no stray-close to
        recover from (that recovery lived here pre-6.1; it force-opened the hand, which would drop a
        carried item mid-task, so it moved to the guard - which refuses instead of opening)."""
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

    # Phase 6.1 full-hand guard now lives in resolve_grab_hand above: a closed hand is refused, never
    # force-opened (that would drop the carried item); recovery from a stray close is task-start
    # (harness) business, never mid-task.

    # The tool OWNS the GRAB pose: set it now, and restore REST on EVERY exit (success, miss, and
    # exception - via the finally) so a gripped item rides back to the carry pose and the mode
    # router's REST tracker stays valid. The router never sets GRAB; only this tool (and 6.2's place
    # tool) do. NOTE: REACH_ENVELOPE (0.85 m) was fit BEFORE GRAB was set at entry - if the live gate
    # shows the reach boundary shifted, re-run reach_probe/fit_envelope from the GRAB start pose.
    set_hand_pose(GRAB_POSE, hand=hand)
    try:
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
                          f"can't close a vertical gap - raise/lower the hand (or crouch), then "
                          f"re-center and retry")
            else:
                reason = ("out of reach: hand stalled with nothing under it - move the body closer, "
                          "then re-center on the target (center_object_on_screen) and retry")

        print(f"[extend_arm_until_grabbed] hand={hand} hovered={hovered!r} gripped={gripped}"
              + (f" | {reason}" if reason else ""))
        result = {"gripped": gripped, "hovered": None if hovered in _EMPTY else hovered, "hand": hand}
        if reason:
            result["reason"] = reason
        return result
    finally:
        set_hand_pose(REST_POSE, hand=hand)


# ===== Phase D: depth-gated reach planning ======================================================
# plan_reach turns ONE RequestLidarCenter sample into a reach verdict. The tool still never moves the
# body (the 2026-07-21 no-creep directive); this only DECIDES, and the router acts on the verdict.
#
# REACH_ENVELOPE - MEASURED 2026-07-22 from 24 reach_probe grabs. The hand extends along the CAMERA
# GAZE, so a grab lands iff the SLANT distance to the centred target is within reach - it is NOT a
# vertical band + horizontal standoff (that model, hand_drop/r_eff/standoff, was REFUTED by the data:
# same-height items grabbed or missed purely on distance; see mapping/plans/phaseD_reach_and_grab.md).
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
    The earlier hand_drop/r_eff/standoff model was refuted; see mapping/plans/phaseD_reach_and_grab.md.

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


# ===== Phase 6.2: depth-gated PLACE planning ====================================================
# plan_place is plan_reach's sibling for RELEASING onto the bagging tray. Same measured model - the
# held item hangs off the hand which extends along the CAMERA GAZE, so whether a release lands on the
# surface is decided by the SLANT distance to the centred surface - but a place can have BOTH a far
# bound (too far -> the item falls SHORT of the tray) and a near bound (too close -> the hand can't
# clear the tray's front lip). See fit_place_envelope.py, which fits the range and FLAGS a near bound.
#
# PLACE_ENVELOPE - PROVENANCE (honest, 2026-07-23): the boundary is bracketed from two probe sessions,
# NOT a single clean fit, because the landing rows and the miss rows live in different runs:
#   - LANDS at slant <= 1.29 m  (first place_probe run, finding 2, 2026-07-22)
#   - MISSES at 1.36 / 1.44 / 1.48 m (M7 run, place_envelope.csv) and 1.64 m (first run)
# So the far edge sits in (1.29, 1.36): midpoint 1.325 - MARGIN 0.04 = 1.28 -> place_max = 1.28.
# NO near bound was observed (every miss was FARTHER than every land), so place_min = None (the
# too-close verdict stays dormant). CAVEAT: the current place_envelope.csv holds ONLY the M7 misses;
# the <=1.29 land row is from the earlier session and is not in the file, so `fit_place_envelope.py`
# cannot re-derive place_max until a landing row is logged again. Re-fit from a CSV that has BOTH
# sides before trusting a tighter number. There is also an unresolved aim confound (Option B, deferred
# by user 2026-07-23): the misses could be "fell short" OR "rolled off the front edge" (center_to_counter's
# un-dialed aim) - if the gate shows drops rolling off, dial COUNTER_AIM_NORM deeper and re-measure;
# the envelope may then widen.
PLACE_ENVELOPE = {
    "place_max": 1.28,   # release lands on the tray iff slant distance <= this (m)
    "place_min": None,   # no near bound observed; None disables the too-close ('back') verdict
    "move_unit": 0.10,   # metres per move_forward unit (env.py _move_relative)
    "move_cap":  10,     # move_forward clamps a single call to 10 units
}


def _place_result(verdict, sample=None, move_steps=0, surface_height=None,
                  horizontal_gap=None, reason=""):
    s = sample or {}
    return {
        "verdict": verdict,            # placeable | move | back | crouch | bail | recenter | unavailable
        "move_steps": int(move_steps),
        "distance": s.get("distance"),
        "pitch_deg": s.get("pitch_deg"),
        "camera_height": s.get("camera_height"),
        "surface_height": surface_height,
        "horizontal_gap": horizontal_gap,
        "reason": reason,
    }


def plan_place(sample, envelope=PLACE_ENVELOPE):
    """Turn ONE RequestLidarCenter sample of the bagging tray into a place verdict - the deterministic
    geometry brain of the bag step. PURE (no sim, no I/O), unit-tested in test_plan_place.py.

    Same slant-distance model as plan_reach (the item extends along the gaze), with a place-specific
    range: place iff place_min <= slant <= place_max (place_min None => no near bound).

    Geometry (slant range d, gaze pitch theta [+ = down], camera world-Y cam_h):
        vertical_offset = d * sin(theta)     # + = surface BELOW the camera
        horizontal_gap  = d * cos(theta)     # forward distance (a body move is horizontal)
        surface_height  = cam_h - vertical_offset
    Moving forward shrinks horizontal_gap (so slant) but not vertical_offset, so the smallest slant a
    move can reach is |vertical_offset|; if that already exceeds place_max, moving can't help.

    Verdicts:
      placeable   - place_min <= slant <= place_max: release now.
      move        - too far (slant > place_max) but |vertical_offset| <= place_max: move forward, re-center, retry.
      back        - too close (slant < place_min): move backward, re-center, retry. Only if place_min is set.
      crouch      - too far AND surface more than place_max BELOW the camera: crouch to close it, re-measure.
      bail        - too far AND surface more than place_max ABOVE the camera (not a countertop): can't place.
      recenter    - miss (nothing solid at centre): don't plan off a meaningless distance.
      unavailable - sample lacks pose (old sim build): caller falls back.
    """
    if not sample or sample.get("error"):
        return _place_result("recenter", sample,
                             reason=f"lidar error: {(sample or {}).get('error', 'no sample')}")
    if sample.get("pitch_deg") is None or sample.get("camera_height") is None:
        return _place_result("unavailable", sample,
                             reason="sample has no pose (pitch_deg/camera_height) - the sim needs the "
                                    "Phase-D recompile; falling back to a blind place")
    if not sample.get("hit", False):
        return _place_result("recenter", sample,
                             reason="no surface at frame centre (aimed past the tray / at the floor) - "
                                    "re-center on the tray and retry")

    d = float(sample["distance"])
    theta = math.radians(float(sample["pitch_deg"]))
    cam_h = float(sample["camera_height"])
    vertical_offset = d * math.sin(theta)        # + = surface below the camera
    horizontal_gap = d * math.cos(theta)
    surface_height = cam_h - vertical_offset
    place_max = envelope["place_max"]
    place_min = envelope.get("place_min")

    # Near bound (only if measured): too close, the hand can't clear the tray lip - back off.
    if place_min is not None and d < place_min:
        needed_gap = math.sqrt(max(0.0, place_min ** 2 - vertical_offset ** 2))
        move_steps = max(1, min(envelope["move_cap"],
                                math.ceil((needed_gap - horizontal_gap) / envelope["move_unit"])))
        return _place_result("back", sample, move_steps=move_steps, surface_height=surface_height,
                             horizontal_gap=horizontal_gap,
                             reason=f"too close: slant {d:.2f} m < place_min {place_min:.2f} m - "
                                    f"move_backward {move_steps} and re-center")

    if d <= place_max:
        return _place_result("placeable", sample, surface_height=surface_height,
                             horizontal_gap=horizontal_gap,
                             reason=f"placeable: slant distance {d:.2f} m <= place_max {place_max:.2f} m")

    if abs(vertical_offset) > place_max:
        # The vertical gap alone exceeds place_max; moving forward cannot bring the slant under it.
        if vertical_offset > 0:      # surface below the camera -> crouching lowers the camera toward it
            return _place_result("crouch", sample, surface_height=surface_height,
                                 horizontal_gap=horizontal_gap,
                                 reason=f"surface is {vertical_offset:.2f} m below the camera "
                                        f"(> place_max {place_max:.2f} m) - crouch, then re-center and re-measure")
        return _place_result("bail", sample, surface_height=surface_height,
                             horizontal_gap=horizontal_gap,
                             reason=f"surface is {-vertical_offset:.2f} m above the camera "
                                    f"(> place_max {place_max:.2f} m) - not a countertop; can't place here")

    # Placeable once close enough: move forward to bring the slant down to place_max.
    needed_gap = math.sqrt(max(0.0, place_max ** 2 - vertical_offset ** 2))
    move_steps = max(1, min(envelope["move_cap"],
                            math.ceil((horizontal_gap - needed_gap) / envelope["move_unit"])))
    return _place_result("move", sample, move_steps=move_steps, surface_height=surface_height,
                         horizontal_gap=horizontal_gap,
                         reason=f"move_forward {move_steps} (~{move_steps * envelope['move_unit']:.1f} m): "
                                f"slant distance {d:.2f} m > place_max {place_max:.2f} m; close the "
                                f"horizontal gap from {horizontal_gap:.2f} to ~{needed_gap:.2f} m")
