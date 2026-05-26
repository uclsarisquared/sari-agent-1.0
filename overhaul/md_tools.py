import os
import asyncio
import io
import moondream as md
from PIL import Image

from env import SendCommand, RequestScreenshot, _GRIP_LEFT_, _GRIP_RIGHT_
from hand_reset import reset_hands_in_front2

_model = md.vl(api_key=os.environ.get("MDREAM_API_KEY"))
_URI = "ws://localhost:8080/commands"


def reach_item_in_view(name: str, use_right_hand: bool) -> dict:
    screenshot = RequestScreenshot()
    image = Image.open(io.BytesIO(screenshot["image"]))

    result = _model.point(image, name)
    points = result.get("points", [])

    if not points:
        return {"reached": False, "reason": f"No points found for '{name}'"}

    def dist_to_center(p):
        px, py = p["x"] * 100, p["y"] * 100
        return (px - 50) ** 2 + (py - 50) ** 2

    best = min(points, key=dist_to_center)
    x_pct = round(best["x"] * 100)
    y_pct = round(best["y"] * 100)

    command_name = "ReachRightAtPixel" if use_right_hand else "ReachLeftAtPixel"
    asyncio.get_event_loop().run_until_complete(
        SendCommand({"command": command_name, "x": x_pct, "y": y_pct}, _URI)
    )
    return {"reached": True}


def grab_item_in_view_right(name: str):
    result = reach_item_in_view(name, True)
    if not result["reached"]:
        return result
    grip = _GRIP_RIGHT_()
    if grip.get("gripped"):
        reset_hands_in_front2(extra_elevation=-0.1, hand="right")
        return {"gripped": True}
    return {"gripped": False}


def grab_item_in_view_left(name: str):
    result = reach_item_in_view(name, False)
    if not result["reached"]:
        return result
    grip = _GRIP_LEFT_()
    if grip.get("gripped"):
        reset_hands_in_front2(extra_elevation=-0.1, hand="left")
        return {"gripped": True}
    return {"gripped": False}
