"""
Pantrypal MCP server — WebSocket edition.

Wraps the Sari Sandbox's WebSocket API (ws://localhost:8080/commands) as MCP
tools so that any MCP-capable agent (e.g. claude_mcp_agent.py) can drive the
sandbox without knowing the wire protocol.

Run (stdio transport, used by MCP clients):
    python mcp_server.py

Debug with the MCP inspector:
    npx @modelcontextprotocol/inspector python mcp_server.py
"""

import asyncio
import base64
import json
import re
from typing import List

import websockets
from mcp.server.fastmcp import FastMCP
from mcp import types

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(name="pantrypal-mcp")

WS_URI = "ws://localhost:8080/commands"
UNIT_CAP = 10  # matches env.py's movement allowance


# ---------------------------------------------------------------------------
# Low-level WebSocket helper
# ---------------------------------------------------------------------------

async def _send(command: dict, uri: str = WS_URI) -> bytes | str:
    """
    Send a JSON command over a fresh WebSocket connection and return the raw
    response.  Screenshot commands return raw bytes; all others return str.
    """
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps(command))
        if command["command"] in ("RequestScreenshot", "RequestAnnotation"):
            return await ws.recv()   # raw PNG bytes
        return await ws.recv()       # text string


# ---------------------------------------------------------------------------
# Response parsers (mirrors env.py logic)
# ---------------------------------------------------------------------------

def _parse_agent_state(text: str) -> dict:
    coords = re.findall(r'\((.*?)\)', text, re.DOTALL)
    lines = text.split("\n")
    is_colliding = lines[2].split(": ")[-1].strip() == "True" if len(lines) > 2 else False
    return {
        "translation": tuple(map(float, coords[0].split(", "))) if coords else (),
        "rotation":    tuple(map(float, coords[1].split(", "))) if len(coords) > 1 else (),
        "isColliding": is_colliding,
    }


def _parse_hand_state(text: str) -> dict:
    coords = re.findall(r'\((.*?)\)', text, re.DOTALL)
    lines  = text.split("\n")
    return {
        "leftTranslation":  tuple(map(float, coords[0].split(", "))) if len(coords) > 0 else (),
        "leftRotation":     tuple(map(float, coords[1].split(", "))) if len(coords) > 1 else (),
        "rightTranslation": tuple(map(float, coords[2].split(", "))) if len(coords) > 2 else (),
        "rightRotation":    tuple(map(float, coords[3].split(", "))) if len(coords) > 3 else (),
        "leftHoveredObject":  lines[2].split(": ")[-1].strip() if len(lines) > 2 else "",
        "leftGrippedState":   lines[3].split(": ")[-1].strip() == "True" if len(lines) > 3 else False,
        "rightHoveredObject": lines[7].split(": ")[-1].strip() if len(lines) > 7 else "",
        "rightGrippedState":  lines[8].split(": ")[-1].strip() == "True" if len(lines) > 8 else False,
    }


# ---------------------------------------------------------------------------
# Shared movement helper
# ---------------------------------------------------------------------------

async def _repeat_transform_agent(translation, rotation, units: int) -> dict:
    """Call TransformAgent `units` times (capped at UNIT_CAP) and return the
    last state."""
    state = {}
    for _ in range(min(units, UNIT_CAP)):
        raw = await _send({
            "command": "TransformAgent",
            "translation": translation,
            "rotation": rotation,
        })
        state = _parse_agent_state(raw)
    return state


async def _repeat_transform_hands(l_trans, l_rot, r_trans, r_rot, units: int) -> dict:
    """Call TransformHands `units` times (capped at UNIT_CAP)."""
    state = {}
    for _ in range(min(units, UNIT_CAP)):
        raw = await _send({
            "command": "TransformHands",
            "leftTranslation":  l_trans,
            "leftRotation":     l_rot,
            "rightTranslation": r_trans,
            "rightRotation":    r_rot,
        })
        state = _parse_hand_state(raw)
    return state


# ===========================================================================
# NAVIGATION TOOLS
# ===========================================================================

@mcp.tool()
async def move_forward(units: int) -> dict:
    """Move the agent forward. Each unit moves the agent 0.1 metres forward.
    Maximum 10 units per call.

    Args:
        units: Number of 0.1-metre steps to move forward (1–10).

    Returns:
        Agent state dict with translation, rotation, and isColliding.
    """
    return await _repeat_transform_agent((0, 0, 0.1), (0, 0, 0), units)


@mcp.tool()
async def move_backward(units: int) -> dict:
    """Move the agent backward. Each unit moves the agent 0.1 metres backward.
    Maximum 10 units per call.

    Args:
        units: Number of 0.1-metre steps to move backward (1–10).

    Returns:
        Agent state dict with translation, rotation, and isColliding.
    """
    return await _repeat_transform_agent((0, 0, -0.1), (0, 0, 0), units)


@mcp.tool()
async def move_left(units: int) -> dict:
    """Strafe the agent left. Each unit moves the agent 0.1 metres left.
    Maximum 10 units per call.

    Args:
        units: Number of 0.1-metre steps to strafe left (1–10).

    Returns:
        Agent state dict with translation, rotation, and isColliding.
    """
    return await _repeat_transform_agent((-0.1, 0, 0), (0, 0, 0), units)


@mcp.tool()
async def move_right(units: int) -> dict:
    """Strafe the agent right. Each unit moves the agent 0.1 metres right.
    Maximum 10 units per call.

    Args:
        units: Number of 0.1-metre steps to strafe right (1–10).

    Returns:
        Agent state dict with translation, rotation, and isColliding.
    """
    return await _repeat_transform_agent((0.1, 0, 0), (0, 0, 0), units)


# ===========================================================================
# CAMERA / LOOK TOOLS
# ===========================================================================

@mcp.tool()
async def pan_left(units: int) -> dict:
    """Rotate the camera left (counterclockwise). Each unit rotates 2.5 degrees.
    Maximum 10 units per call.

    Args:
        units: Number of 2.5-degree steps to pan left (1–10).

    Returns:
        Agent state dict.
    """
    return await _repeat_transform_agent((0, 0, 0), (0, -2.5, 0), units)


@mcp.tool()
async def pan_right(units: int) -> dict:
    """Rotate the camera right (clockwise). Each unit rotates 2.5 degrees.
    Maximum 10 units per call.

    Args:
        units: Number of 2.5-degree steps to pan right (1–10).

    Returns:
        Agent state dict.
    """
    return await _repeat_transform_agent((0, 0, 0), (0, 2.5, 0), units)


@mcp.tool()
async def tilt_up(units: int) -> dict:
    """Tilt the camera upward. Each unit tilts 2.5 degrees up.
    Maximum 10 units per call.

    Args:
        units: Number of 2.5-degree steps to tilt up (1–10).

    Returns:
        Agent state dict.
    """
    return await _repeat_transform_agent((0, 0, 0), (-2.5, 0, 0), units)


@mcp.tool()
async def tilt_down(units: int) -> dict:
    """Tilt the camera downward. Each unit tilts 2.5 degrees down.
    Maximum 10 units per call.

    Args:
        units: Number of 2.5-degree steps to tilt down (1–10).

    Returns:
        Agent state dict.
    """
    return await _repeat_transform_agent((0, 0, 0), (2.5, 0, 0), units)


# ===========================================================================
# HAND — EXTENSION / RETRACTION
# ===========================================================================

@mcp.tool()
async def extend_left_hand_forward(units: int) -> dict:
    """Extend the left hand forward. Each unit moves the hand 0.025 metres forward.
    Maximum 10 units per call.

    Args:
        units: Number of 0.025-metre steps (1–10).

    Returns:
        Hand state dict with translations, rotations, hovered objects, and grip states.
    """
    return await _repeat_transform_hands((0, 0, 0.025), (0, 0, 0), (0, 0, 0), (0, 0, 0), units)


@mcp.tool()
async def pull_left_hand_backward(units: int) -> dict:
    """Retract the left hand backward. Each unit moves the hand 0.025 metres backward.
    Maximum 10 units per call.

    Args:
        units: Number of 0.025-metre steps (1–10).

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, -0.025), (0, 0, 0), (0, 0, 0), (0, 0, 0), units)


@mcp.tool()
async def extend_right_hand_forward(units: int) -> dict:
    """Extend the right hand forward. Each unit moves the hand 0.025 metres forward.
    Maximum 10 units per call.

    Args:
        units: Number of 0.025-metre steps (1–10).

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0, 0.025), (0, 0, 0), units)


@mcp.tool()
async def pull_right_hand_backward(units: int) -> dict:
    """Retract the right hand backward. Each unit moves the hand 0.025 metres backward.
    Maximum 10 units per call.

    Args:
        units: Number of 0.025-metre steps (1–10).

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0, -0.025), (0, 0, 0), units)


# ===========================================================================
# HAND — RAISE / LOWER
# ===========================================================================

@mcp.tool()
async def raise_left_hand(units: int) -> dict:
    """Raise the left hand upward. Each unit raises the hand 0.025 metres.
    Maximum 10 units per call.

    Args:
        units: Number of 0.025-metre steps (1–10).

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0.025, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), units)


@mcp.tool()
async def lower_left_hand(units: int) -> dict:
    """Lower the left hand downward. Each unit lowers the hand 0.025 metres.
    Maximum 10 units per call.

    Args:
        units: Number of 0.025-metre steps (1–10).

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, -0.025, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), units)


@mcp.tool()
async def raise_right_hand(units: int) -> dict:
    """Raise the right hand upward. Each unit raises the hand 0.025 metres.
    Maximum 10 units per call.

    Args:
        units: Number of 0.025-metre steps (1–10).

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0.025, 0), (0, 0, 0), units)


@mcp.tool()
async def lower_right_hand(units: int) -> dict:
    """Lower the right hand downward. Each unit lowers the hand 0.025 metres.
    Maximum 10 units per call.

    Args:
        units: Number of 0.025-metre steps (1–10).

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, -0.025, 0), (0, 0, 0), units)


# ===========================================================================
# HAND — ROTATION
# ===========================================================================

@mcp.tool()
async def rotate_left_hand_clockwise(units: int) -> dict:
    """Rotate the left hand clockwise. Each unit rotates 15 degrees.

    Args:
        units: Number of 15-degree steps.

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, 0), (0, 15, 0), (0, 0, 0), (0, 0, 0), units)


@mcp.tool()
async def rotate_left_hand_counterclockwise(units: int) -> dict:
    """Rotate the left hand counterclockwise. Each unit rotates 15 degrees.

    Args:
        units: Number of 15-degree steps.

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, 0), (0, -15, 0), (0, 0, 0), (0, 0, 0), units)


@mcp.tool()
async def rotate_right_hand_clockwise(units: int) -> dict:
    """Rotate the right hand clockwise. Each unit rotates 15 degrees.

    Args:
        units: Number of 15-degree steps.

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 15, 0), units)


@mcp.tool()
async def rotate_right_hand_counterclockwise(units: int) -> dict:
    """Rotate the right hand counterclockwise. Each unit rotates 15 degrees.

    Args:
        units: Number of 15-degree steps.

    Returns:
        Hand state dict.
    """
    return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, -15, 0), units)


# ===========================================================================
# GRIP TOOLS
# ===========================================================================

@mcp.tool()
async def toggle_left_grip() -> dict:
    """Toggle the left hand grip open ↔ closed.

    Returns:
        dict with key "gripped" (bool).
    """
    raw = await _send({"command": "ToggleLeftGrip"})
    return {"gripped": "True" in raw}


@mcp.tool()
async def toggle_right_grip() -> dict:
    """Toggle the right hand grip open ↔ closed.

    Returns:
        dict with key "gripped" (bool).
    """
    raw = await _send({"command": "ToggleRightGrip"})
    return {"gripped": "True" in raw}


# ===========================================================================
# RAW TRANSFORM TOOLS  (precise, single-step control)
# ===========================================================================

@mcp.tool()
async def transform_agent(
    tx: float, ty: float, tz: float,
    rx: float, ry: float, rz: float,
) -> dict:
    """Apply an arbitrary single-step translation and rotation to the agent.
    Coordinates are in the agent's local frame.

    Args:
        tx: X translation (metres, left/right).
        ty: Y translation (metres, up/down).
        tz: Z translation (metres, forward/backward).
        rx: Pitch delta (degrees, tilt up/down).
        ry: Yaw delta (degrees, pan left/right).
        rz: Roll delta (degrees).

    Returns:
        Agent state dict with translation, rotation, and isColliding.
    """
    raw = await _send({
        "command": "TransformAgent",
        "translation": (tx, ty, tz),
        "rotation":    (rx, ry, rz),
    })
    return _parse_agent_state(raw)


@mcp.tool()
async def transform_hands(
    l_tx: float, l_ty: float, l_tz: float,
    l_rx: float, l_ry: float, l_rz: float,
    r_tx: float, r_ty: float, r_tz: float,
    r_rx: float, r_ry: float, r_rz: float,
) -> dict:
    """Apply arbitrary single-step translations and rotations to both hands.
    Coordinates are in the agent's local frame.

    Args:
        l_tx, l_ty, l_tz: Left hand translation delta (metres).
        l_rx, l_ry, l_rz: Left hand rotation delta (degrees).
        r_tx, r_ty, r_tz: Right hand translation delta (metres).
        r_rx, r_ry, r_rz: Right hand rotation delta (degrees).

    Returns:
        Hand state dict with translations, rotations, hovered objects, and grip states.
    """
    raw = await _send({
        "command": "TransformHands",
        "leftTranslation":  (l_tx, l_ty, l_tz),
        "leftRotation":     (l_rx, l_ry, l_rz),
        "rightTranslation": (r_tx, r_ty, r_tz),
        "rightRotation":    (r_rx, r_ry, r_rz),
    })
    return _parse_hand_state(raw)


# ===========================================================================
# PERCEPTION TOOLS
# ===========================================================================

@mcp.tool()
async def get_current_view() -> list:
    """Capture the agent's egocentric view as a JPEG image.

    Returns:
        MCP ImageContent containing the current first-person view (PNG, 1920×1080).
    """
    raw = await _send({
        "command": "RequestScreenshot",
        "prefix": "",
        "suffix": "",
        "folder_name": "screenshots",
        "save_image": False,
    })
    # raw is PNG bytes; encode to base64 for MCP image transport
    b64 = base64.b64encode(raw).decode("utf-8")
    return [types.ImageContent(type="image", data=b64, mimeType="image/png")]


@mcp.tool()
async def get_scene_json() -> str:
    """Request the scene's structured JSON state from the sandbox.

    Returns:
        Raw JSON string describing the current scene objects and their states.
    """
    return await _send({"command": "RequestJson"})


# ===========================================================================
# UTILITY TOOLS
# ===========================================================================

@mcp.tool()
async def get_agent_state() -> dict:
    """Query the agent's current position and rotation without moving.

    Returns:
        Agent state dict with translation, rotation, and isColliding.
    """
    raw = await _send({
        "command": "TransformAgent",
        "translation": (0, 0, 0),
        "rotation":    (0, 0, 0),
    })
    return _parse_agent_state(raw)


@mcp.tool()
async def get_hand_state() -> dict:
    """Query both hands' current positions and grip states without moving.

    Returns:
        Hand state dict with translations, rotations, hovered objects, and grip states.
    """
    raw = await _send({
        "command": "TransformHands",
        "leftTranslation":  (0, 0, 0),
        "leftRotation":     (0, 0, 0),
        "rightTranslation": (0, 0, 0),
        "rightRotation":    (0, 0, 0),
    })
    return _parse_hand_state(raw)


@mcp.tool()
async def reset_environment() -> bool:
    """Reset the sandbox environment to its initial state.

    Returns:
        True if the reset command was sent successfully.
    """
    await _send({"command": "ResetEnvironment"})
    return True


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
