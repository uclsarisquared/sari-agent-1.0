import os
import asyncio
import websockets
import json
import re
from datetime import datetime

from typing import (
    Dict,
    Tuple,
    Any,
)

from loguru import logger


def init_logger(run_name: str, directory: str = "logs"):
    """
    Call this once before you send any commands.
    - `run_name`: if None, defaults to timestamp "run_YYYYmmdd_HHMMSS".
    - `directory`: where to store your log files
    Configures the global loguru (logger).
    """
    if not run_name:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{run_name}.log")

    logger.remove()
    logger.add(
        path,
        level='INFO',
        format='{time:YYYY-MM-DD HH:mm:ss} {level: <8} {message}',
        rotation=None,
        retention=None
    )
    logger.info(f"=== STARTING NEW RUN: {run_name} ===")
    return logger

async def SendCommand(command: Dict[str, Any], uri: str):
    async with websockets.connect(uri, max_size=None) as websocket:
        await websocket.send(json.dumps(command))
        if command["command"] == "RequestScreenshot" or command["command"] == "RequestAnnotation":
            image_bytes = await websocket.recv()
            save_image = command["save_image"]

            if save_image:
                folder_name = os.path.join(command["folder_name"])
                prefix = str(command["prefix"]) + "-" if str(command["prefix"]) else ""
                suffix = "-" + str(command["suffix"]) if str(command["suffix"]) else ""
                file_name = f"{prefix}ClientScreenshot{suffix}.png"

                if folder_name:
                    os.makedirs(folder_name, exist_ok=True)  # Creates the folder if it doesn't exist

                # Save the image file in the specified folder
                file_path = os.path.join(folder_name, file_name) if folder_name else file_name

                with open(file_path, "wb") as file:
                    file.write(image_bytes)

            return {'image': image_bytes}
        else:
            response = await websocket.recv()
            return response

def _v1_or_raise(result, cmd, n_tuples, n_lines):
    """env.py speaks the sim's V1 TEXT protocol: it parses `cmd` replies POSITIONALLY (n_tuples
    `(x,y,z)` groups over >= n_lines lines). If the sim's WebSocketHandler has
    `sariSandboxV1CompatibilityLayer` turned OFF it returns JSON instead (one line, no `(...)`
    groups), which this parser cannot read - and the grip/hover toggles break the same way. Detect
    that and raise something ACTIONABLE instead of an opaque IndexError/AssertionError."""
    text = result.decode(errors="replace") if isinstance(result, (bytes, bytearray)) else str(result)
    lines = text.split("\n")
    tuples = re.findall(r'\((.*?)\)', text, re.DOTALL)
    if len(tuples) != n_tuples or len(lines) < n_lines:
        raise RuntimeError(
            f"{cmd}: expected the V1 text reply ({n_tuples} (x,y,z) tuples over {n_lines}+ lines) but "
            f"got {len(tuples)} tuple(s) / {len(lines)} line(s). The sim is almost certainly running "
            f"with `sariSandboxV1CompatibilityLayer` OFF (it then returns JSON, which env.py's text "
            f"protocol can't read - TransformAgent/TransformHands and the grip toggles all break). "
            f"Turn it ON: select the WebSocketHandler in the scene, check 'Sari Sandbox V1 "
            f"Compatibility Layer', and re-enter Play. Raw reply: {text!r}")
    return lines, tuples


def TransformAgent(translation: Tuple[float],
                   rotation: Tuple[float],
                   uri: str = "ws://localhost:8080/commands") -> Dict[str, Tuple[float]]:
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "TransformAgent",
        "translation": translation,
        "rotation": rotation
    }, uri))

    lines, extracted_state = _v1_or_raise(result, "TransformAgent", 2, 3)
    is_colliding = lines[2].split(": ")[-1].strip() == "True"
    agent_state = {
        'translation': tuple(map(float, extracted_state[0].split(', '))),
        'rotation': tuple(map(float, extracted_state[1].split(', '))),
        'isColliding': is_colliding
    }
    return agent_state

def SetCrouch(active: bool, uri: str = "ws://localhost:8080/commands") -> str:
    """Crouch (True) or stand (False). Crouching halves the agent's view height, so a level camera
    looks straight at the shelf's lower rows instead of down at them from standing height.

    This is the command form of the LeftControl key in Unity's own agent controller - the two are
    OR'd together agent-side, so this never fights a human holding the key. Applied immediately, so
    a RequestScreenshot right after this call already sees the lowered viewpoint.
    """
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "SetCrouch", "active": active
    }, uri))
    return result

def SetHandsActive(active: bool, side: str = None, uri: str = "ws://localhost:8080/commands") -> str:
    """Enable/disable the hand prefab(s) so they can't clip shelf items while navigating.

    `side`: None/omitted disables both hands, or pass "Left"/"Right" for one hand only
    (e.g. keep a hand active mid-grab while the other is stowed for travel).
    """
    command = {"command": "SetHandsActive", "active": active}
    if side:
        command["side"] = side
    result = asyncio.get_event_loop().run_until_complete(SendCommand(command, uri))
    return result


def TransformHands(leftTranslation: Tuple[float],
                   leftRotation: Tuple[float],
                   rightTranslation: Tuple[float],
                   rightRotation: Tuple[float],
                   uri: str = "ws://localhost:8080/commands"):
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "TransformHands",
        "leftTranslation": leftTranslation,
        "leftRotation": leftRotation,
        "rightTranslation": rightTranslation,
        "rightRotation": rightRotation
    }, uri))
    
    lines, extracted_state = _v1_or_raise(result, "TransformHands", 4, 9)
    object_hovered_over_left = lines[2].split(": ")[-1]
    is_gripped_state_left = lines[3].split(": ")[-1].strip() == "True"
    object_hovered_over_right = lines[7].split(": ")[-1]
    is_gripped_state_right = lines[8].split(": ")[-1].strip() == "True"
    current_state = {
        'leftTranslation': tuple(map(float, extracted_state[0].split(', '))),
        'leftRotation': tuple(map(float, extracted_state[1].split(', '))),
        'rightTranslation': tuple(map(float, extracted_state[2].split(', '))),
        'rightRotation': tuple(map(float, extracted_state[3].split(', '))),
        'leftHoveredObject': object_hovered_over_left,
        'leftGrippedState': is_gripped_state_left,
        'rightHoveredObject': object_hovered_over_right,
        'rightGrippedState': is_gripped_state_right, 
    }
    return current_state

def ToggleLeftGrip(uri: str="ws://localhost:8080/commands"):
    # The single-agent /commands handler names this `ToggleLeftHandGrip`
    # (SariAgentCommandBehavior.cs). The bare `ToggleLeftGrip`/`ToggleRightGrip` names live only in
    # SariMultiplayerBehavior.cs (a different ws path); sending them here lands in the sim's
    # `default: Unknown command` branch, so the hand never grips (translation still works). Verified
    # live 2026-07-22: `ToggleLeftGrip` -> "Unknown command", `ToggleLeftHandGrip` -> "Left Grip: True".
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "ToggleLeftHandGrip"
    }, uri))

    if "True" in result:    return {"gripped": True}
    return {"gripped": False}

def ToggleRightGrip(uri: str="ws://localhost:8080/commands"):
    # See ToggleLeftGrip: the /commands handler uses `ToggleRightHandGrip`, not `ToggleRightGrip`.
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "ToggleRightHandGrip"
    }, uri))

    if "True" in result:    return {"gripped": True}
    return {"gripped": False}

def RequestScreenshot(prefix: str="", suffix: str="", folder_name: str="screenshots", save_image=False, uri: str="ws://localhost:8080/commands"):
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "RequestScreenshot",
        "prefix": prefix,
        "suffix": suffix,
        "folder_name": folder_name,
        "save_image": save_image,
    }, uri))
    return result

def Reset(uri: str="ws://localhost:8080/commands"):
    asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "ResetEnvironment"
    }, uri))

def RequestAnnotation(uri: str="ws://localhost:8080/commands"):
    result = asyncio.get_event_loop().run_until_complete(SendCommand(
        {
        "command": "RequestAnnotation"
    }, uri))
    return result

def RequestJson(uri: str="ws://localhost:8080/commands"):
    result = asyncio.get_event_loop().run_until_complete(SendCommand(
        {
        "command": "RequestJson"
    }, uri))
    return result

def RequestLidarCenter(uri: str="ws://localhost:8080/commands") -> Dict[str, Any]:
    """Depth (metres) along the agent's *actual pitched gaze* at the CENTRE pixel, plus the pose to
    decompose it. Phase D - the metric distance the manipulation router plans a reach from, replacing
    the removed monocular depth hint (see slamtest/plans/phaseD_reach_and_grab.md).

    Returns the sim's JSON reply parsed to a dict:
        {distance, hit, pitch_deg, camera_height, min_range, max_range}
      - distance     : slant range along the gaze ray (m). On a MISS it equals max_range.
      - hit          : False when nothing solid is under the centre within range (gap / occlusion /
                       too far). Callers must NOT plan a move off a miss - distance is meaningless.
      - pitch_deg    : gaze pitch, + = looking DOWN.  camera_height : ray-origin world Y (m).
                       Bundled by the sim (LidarSensor.CenterSample) so Python never assumes the
                       VR-vs-IK prefab pitch or eye height.

    Unlike TransformAgent/TransformHands (V1 regex-TEXT replies), this command returns JSON - parse
    it as JSON. The hands are LiDAR self-culled (LidarSensor culledBySelf), so an active hand does
    not occlude this read. A pre-Phase-D sim build omits pitch_deg/camera_height; plan_reach treats
    that as `unavailable` and the caller falls back to a blind reach.
    """
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "RequestLidarCenter"
    }, uri))
    try:
        return json.loads(result)
    except (ValueError, TypeError):
        # Sim sent a plain-text error (e.g. "Error: no camera found for agent") instead of JSON.
        return {"hit": False, "error": str(result)}


# main controls
_MOVE_FWD_ = lambda: TransformAgent((0, 0, 0.1), (0, 0, 0))
_MOVE_BCK_ = lambda: TransformAgent((0, 0, -0.1), (0, 0, 0))
_MOVE_LEFT_ = lambda: TransformAgent((-0.1, 0, 0), (0, 0, 0))
_MOVE_RIGHT_ = lambda: TransformAgent((0.1, 0, 0), (0, 0, 0))
_PAN_LEFT_ = lambda: TransformAgent((0, 0, 0), (0, -2.5, 0))
_PAN_RIGHT_ = lambda: TransformAgent((0, 0, 0), (0, 2.5, 0))
_TILT_UP_ = lambda: TransformAgent((0, 0, 0), (-2.5, 0, 0))
_TILT_DOWN_ = lambda: TransformAgent((0, 0, 0), (2.5, 0, 0))
_GRIP_LEFT_ = lambda: ToggleLeftGrip()
_GRIP_RIGHT_ = lambda: ToggleRightGrip()
_XTNFWD_LEFT_ = lambda: TransformHands((0, 0, 0.025), (0, 0, 0), (0, 0, 0), (0, 0, 0))
_PLLBCK_LEFT_ = lambda: TransformHands((0, 0, -0.025), (0, 0, 0), (0, 0, 0), (0, 0, 0))
_XTNFWD_RIGHT_ = lambda: TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0.025), (0, 0, 0))
_PLLBCK_RIGHT_ = lambda: TransformHands((0, 0, 0), (0, 0, 0), (0, 0, -0.025), (0, 0, 0))
_RSE_LEFT_ = lambda: TransformHands((0, 0.025, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
_LWR_LEFT_ = lambda: TransformHands((0, -0.025, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
_RSE_RIGHT_ = lambda: TransformHands((0, 0, 0), (0, 0, 0), (0, 0.025, 0), (0, 0, 0))
_LWR_RIGHT_ = lambda: TransformHands((0, 0, 0), (0, 0, 0), (0, -0.025, 0), (0, 0, 0))
_ROT_LEFT_CLOCK_ = lambda: TransformHands((0, 0, 0), (0, 15, 0), (0, 0, 0), (0, 0, 0))
_ROT_LEFT_CTRCLOCK_ = lambda: TransformHands((0, 0, 0), (0, -15, 0), (0, 0, 0), (0, 0, 0))
_ROT_RIGHT_CLOCK_ = lambda: TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 15, 0))
_ROT_RIGHT_CTRCLOCK_ = lambda: TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, -15, 0))
_REQUEST_SCREENSHOT_ = lambda prefix="", suffix="", folder_name="", save_image=False: RequestScreenshot(prefix, suffix, folder_name, save_image)

# NOTE: the _MOVE_*_ lambdas above are already the correct BODY-RELATIVE (egocentric) primitives:
# _MOVE_FWD_ = (0,0,0.1) means "0 right, 0 up, 0.1 forward". The sim rotates that into world space
# itself (see _move_relative). The public move_* functions below build the same body-relative
# vectors from a (forward, right) pair and are what the agent/actions.py use.
def _move_relative(forward, right, units):
    """Step the agent `units` x 0.1m along a BODY-RELATIVE (forward, right) direction.

    The sim's TranslateAgent treats the incoming translation as EGOCENTRIC - (x=right, y=up,
    z=forward) - and rotates it into world space itself via EgocentricToWorldTranslation using the
    agent's current facing. So move_* must send the RAW body-relative vector and must NOT pre-rotate
    it by yaw; pre-rotating double-rotates the step. Y is held at 0 so the body stays level even
    when the camera is pitched at a low/high shelf.

    MEASURED / DE-SYNC 2026-07-22: the sim flipped its TranslateAgent contract from world-space to
    egocentric (SariSandboxV2 commit 3940ce7, merged to main 2026-07-21 22:33 - the WebSocket
    dispatcher's `case "TransformAgent"` is commented out and FALLS THROUGH to `case
    "TranslateAgent"`, which now wraps the input in `agent.EgocentricToWorldTranslation(...)`). The
    previous world-space workaround here (a `_heading_world_delta` that rotated by yaw in Python,
    added 2026-07-21 daytime against the OLD world-space sim) became a SECOND rotation after that
    merge, so move_forward drifted off at an angle equal to the agent's yaw - the "forward isn't
    forward" bug. Fix: send the body-relative vector unrotated and let the sim own the one rotation.
    Re-verify with probe_translation.py if the sim's dispatcher contract changes again."""
    units = min(units, 10)
    if units <= 0:
        return
    for _ in range(units):
        TransformAgent((right, 0.0, forward), (0, 0, 0))

def move_forward(units):
    if units > 10:
        print("Warning: Moving forward more than 10 units at once may cause instability. Setting to 10 units.")
    _move_relative(0.1, 0.0, units)

def move_backward(units):
    if units > 10:
        print("Warning: Moving backward more than 10 units at once may cause instability. Setting to 10 units.")
    _move_relative(-0.1, 0.0, units)

def move_left(units):
    if units > 10:
        print("Warning: Moving left more than 10 units at once may cause instability. Setting to 10 units.")
    _move_relative(0.0, -0.1, units)

def move_right(units):
    if units > 10:
        print("Warning: Moving right more than 10 units at once may cause instability. Setting to 10 units.")
    _move_relative(0.0, 0.1, units)

def pan_left(units):
    if units > 15:
        print("Warning: Panning left more than 15 units at once may cause instability. Setting to 15 units.")
    for _ in range(min(units, 15)):
        _PAN_LEFT_()

def pan_right(units):
    if units > 15:
        print("Warning: Panning right more than 15 units at once may cause instability. Setting to 15 units.")
    for _ in range(min(units, 15)):
        _PAN_RIGHT_()

def tilt_up(units):
    if units > 10:
        print("Warning: Tilting up more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _TILT_UP_()

def tilt_down(units):
    if units > 10:
        print("Warning: Tilting down more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _TILT_DOWN_()

def extend_left_hand_forward(units):
    if units > 10:
        print("Warning: Extending left hand forward more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _XTNFWD_LEFT_()

def extend_right_hand_forward(units):
    if units > 10:
        print("Warning: Extending right hand forward more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _XTNFWD_RIGHT_()

def pull_left_hand_backward(units):
    if units > 10:
        print("Warning: Pulling left hand backward more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _PLLBCK_LEFT_()

def pull_right_hand_backward(units):
    if units > 10:
        print("Warning: Pulling right hand backward more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _PLLBCK_RIGHT_()

def raise_left_hand(units):
    if units > 10:
        print("Warning: Raising left hand more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _RSE_LEFT_()

def lower_left_hand(units):
    if units > 10:
        print("Warning: Lowering left hand more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _LWR_LEFT_()

def raise_right_hand(units):
    if units > 10:
        print("Warning: Raising right hand more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _RSE_RIGHT_()

def lower_right_hand(units):
    if units > 10:
        print("Warning: Lowering right hand more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _LWR_RIGHT_()

def rotate_left_clockwise(units):
    if units > 10:
        print("Warning: Rotating left hand clockwise more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _ROT_LEFT_CLOCK_()

def rotate_left_counterclockwise(units):
    if units > 10:
        print("Warning: Rotating left hand counterclockwise more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _ROT_LEFT_CTRCLOCK_()

def rotate_right_clockwise(units):
    if units > 10:
        print("Warning: Rotating right hand clockwise more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _ROT_RIGHT_CLOCK_()

def rotate_right_counterclockwise(units):
    if units > 10:
        print("Warning: Rotating right hand counterclockwise more than 10 units at once may cause instability. Setting to 10 units.")
    for _ in range(min(units, 10)):
        _ROT_RIGHT_CTRCLOCK_()
