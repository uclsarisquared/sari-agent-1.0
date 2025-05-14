import os
import asyncio
import websockets
import json
import re
from datetime import datetime

from typing import (
    Tuple,
    Dict,
    Any
)
from loguru import logger


def init_logger(run_name: str = None, directory: str = "logs"):
    """
    Call this once, before you send any commands.
    - run_name: if None, defaults to timestamp “run_YYYYmmdd_HHMMSS”
    - directory: folder to store your log files
    Configures the global loguru logger.
    """
    if not run_name:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{run_name}.log")

    # Remove default handlers (so we don't duplicate logs)
    logger.remove()
    # Add file sink with desired format
    logger.add(
        path,   
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} {level: <8} {message}",
        rotation=None,
        retention=None
    )
    # TODO: Optionally, also log to stdout (will decide later):
    # logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    logger.info(f"=== Starting new run: {run_name} ===")
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

def TransformAgent(translation: Tuple[float],
                   rotation: Tuple[float],
                   uri: str = "ws://localhost:8080/commands") -> Dict[str, Tuple[float]]:
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "TransformAgent",
        "translation": translation,
        "rotation": rotation
    }, uri))

    extracted_state = re.findall(r'\((.*?)\)', result, re.DOTALL)
    is_colliding = True if result.split("\n")[2].split(": ")[-1] == "True" else False
    assert len(extracted_state) == 2, "Expected 2 Tuple[float], got " + str(len(extracted_state))
    agent_state = {
        'translation': tuple(map(float, extracted_state[0].split(', '))),
        'rotation': tuple(map(float, extracted_state[1].split(', '))),
        'isColliding': is_colliding
    }
    return agent_state

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
    
    extracted_state = re.findall(r'\((.*?)\)', result, re.DOTALL)
    lines = result.split("\n")
    object_hovered_over_left = lines[2].split(": ")[-1]
    is_gripped_state_left = True if lines[3].split(": ")[-1] == "True" else False
    object_hovered_over_right = lines[7].split(": ")[-1]
    is_gripped_state_right = True if lines[8].split(": ")[-1] == "True" else False
    assert len(extracted_state) == 4, "Expected 4 Tuple[float], got " + str(len(extracted_state))
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
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "ToggleLeftGrip"
    }, uri))

    if "True" in result:    return {"gripped": True}
    return {"gripped": False}

def ToggleRightGrip(uri: str="ws://localhost:8080/commands"):
    result = asyncio.get_event_loop().run_until_complete(SendCommand({
        "command": "ToggleRightGrip"
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


## controls
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
_RESET_ = lambda: Reset()
_REQUEST_SCREENSHOT_ = lambda prefix="", suffix="", folder_name="", save_image=False: RequestScreenshot(prefix, suffix, folder_name, save_image)
_REQUEST_ANNOTATION_ = lambda: RequestAnnotation()
_REQUEST_JSON_ = lambda: RequestJson()

_PROPER_HAND_POSITIONING_ = lambda: TransformHands((-0.065, 0, 0.350), (0, 0, 0), (0.065, 0, 0.350), (0, 0, 0))