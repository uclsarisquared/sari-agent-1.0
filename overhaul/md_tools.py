"""Grab-time item pointing. REEVALUATED 2026-07-19 (user request), verdict:

KEEP moondream as PRIMARY. Pointing ("point at <name>" -> pixel) is exactly the primitive
hand-aiming needs, moondream is purpose-built for it, and it is measured-working - swapping
the detector now would change grab quality mid-Phase-4.2, in machinery both A/B arms are
supposed to share frozen. The full replace-or-keep decision belongs to the manipulation
phase, made with a measured moondream-vs-qwen pointing comparison.

BUT it is now the runtime's ONLY non-self-hosted dependency (cloud API, $MDREAM_API_KEY) -
the same failure class that killed OpenRouter mid-run (402). So: on moondream ERROR (API
down, no key - NOT a legitimate "no points" answer), fall back to qwen-VL on the UCL server
(bbox -> center point). Resilience, not improvement; the fallback logs loudly so a run that
silently degraded to qwen pointing cannot masquerade as a moondream run."""
import os
import asyncio
import io
import json as _json
import re as _re
import moondream as md
from PIL import Image

from env import SendCommand, RequestScreenshot, _GRIP_LEFT_, _GRIP_RIGHT_
from hand_reset import reset_hands_in_front2

_model = md.vl(api_key=os.environ.get("MDREAM_API_KEY"))
_URI = "ws://localhost:8080/commands"


def _point_via_qwen(image: Image.Image, name: str):
    """Fallback pointer: qwen bbox -> center, normalized 0-1 like moondream's points."""
    from perception import CLIENT, MODEL_NAME, _encode_image
    prompt = (f"Detect the {name} in the image. Reply with ONLY a JSON object "
              '{"box_2d": [ymin, xmin, ymax, xmax]} normalized to 0-1000. '
              "If the item is not visible, reply {\"box_2d\": null}.")
    resp = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": [_encode_image(image),
                                               {"type": "text", "text": prompt}]}],
        temperature=0.0, max_tokens=200,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}})
    text = resp.choices[0].message.content
    mt = _re.search(r"\{[\s\S]*\}", text)
    box = _json.loads(mt.group(0)).get("box_2d") if mt else None
    if not box:
        return []
    ymin, xmin, ymax, xmax = box
    return [{"x": (xmin + xmax) / 2000.0, "y": (ymin + ymax) / 2000.0}]


def reach_item_in_view(name: str, use_right_hand: bool) -> dict:
    screenshot = RequestScreenshot()
    image = Image.open(io.BytesIO(screenshot["image"]))

    try:
        result = _model.point(image, name)
        points = result.get("points", [])
    except Exception as e:
        print(f"[md_tools] moondream FAILED ({type(e).__name__}: {e}) - "
              f"falling back to qwen pointing (logged, not silent)")
        points = _point_via_qwen(image, name)

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
