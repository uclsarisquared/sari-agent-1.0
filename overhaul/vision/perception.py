import ast
import json
import random
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import ast
import re
import math
from .ocr_client import OcrUnavailable, ocr_lines
# Repo-root config.env (overhaul/vision/ -> repo root is three parents up), resolved from __file__
# so it loads regardless of CWD.
load_dotenv(Path(__file__).resolve().parent.parent.parent / 'config.env')

GRAB_DISTANCE_THRESHOLD = 2.0  # units; beyond this, retrieve_item refuses to grab

# Agent runtime = UCL qwen (user directive 2026-07-19; OpenRouter retired on 402). This is
# the bounding-box/centering client - qwen-VL replaces Gemini here, identically in BOTH A/B
# arms; bbox quality vs Gemini is unmeasured and shared, so it cannot skew the arms.
from agent_core.agent import _ucl_creds, call_with_api_retries
# Every LLM call in this module is perception's cost - bbox, centering and OCR-bbox reasoning alike.
# The role blocks wrap call_with_api_retries, not the lambda inside it, so a flaky call's retries are
# billed to perception as well; the server charged for them.
from agent_core import token_meter
_UCL_HOST, _UCL_KEY = _ucl_creds()
MODEL_NAME = "Qwen/Qwen3.6-27B"
CLIENT = OpenAI(
    base_url=f"http://{_UCL_HOST}:8000/v1",
    api_key=_UCL_KEY,
    max_retries=0,
)
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080

# --- Camera projection: MEASURED, not assumed --------------------------------------
# The sim's agent camera is 60 deg VERTICAL FOV: SariSandboxV2/Assets/Prefabs/
# "IK Humanoid Agent.prefab" carries `field of view: 60` with no m_FOVAxisMode override,
# so Unity's default FOV axis (Vertical) applies. For a WxH frame with square pixels the
# pinhole focal length in PIXELS is identical on both axes: f = (H/2) / tan(vFOV/2) ~= 935.
# A bbox-centre offset (dx, dy) in pixels from the aim point maps to camera angles by
# yaw = atan(dx/f), pitch = atan(dy/f) (see bbox_to_rotation).
#
# This REPLACES the old single linear gain `px_offset / 19.2` (19.2 = 1920/100 deg, i.e. an
# assumed 100 deg *horizontal* FOV reused unchanged on the vertical axis - wrong on both
# counts). Measured against the real 60 deg vertical FOV the correct near-centre gain is
# f*pi/180 ~= 16.3 px/deg, so `/19.2` commanded only ~85% of the needed angle: a systematic
# UNDERSHOOT a single open-loop shot could never recover, worsening toward the frame edges
# where a straight line diverges from atan. Getting f right + closing the loop (see
# center_object_on_screen) is what makes centring actually converge.
CAMERA_VFOV_DEG = 60.0
FOCAL_PX = (ORIGINAL_HEIGHT / 2.0) / math.tan(math.radians(CAMERA_VFOV_DEG / 2.0))

# Measured pixels-per-degree of camera rotation (phase-correlation, live, 2026-07-21): pitch
# matches the f=935 model (16.4 vs 16.3), yaw is ~11% hotter (18.1, parallax over the oblique
# shelf). Used ONLY to PREDICT where a tracked item lands after a rotation so the loop can
# re-lock the SAME instance next look - the correction angle itself still comes from the atan
# model above, damped by `gain`. Good-enough-for-tracking, not a precision constant.
PPD_YAW = 18.1
PPD_PITCH = 16.4
# COORDINATE ORDER (measured 2026-07-21): the box is [xmin, ymin, xmax, ymax], NOT the
# [ymin, xmin, ...] this prompt used to state. These prompts were written for Gemini (ymin-first);
# the backend is now Qwen-VL, whose native grounding format is xmin-first ([x1,y1,x2,y2]). Qwen
# emits xmin-first regardless of what the prompt claims - verified: it boxed "Choco Crunchies" at
# [0,292,249,406], which is the real bottom-left product ONLY read as [xmin,ymin,xmax,ymax]; read
# ymin-first it lands on the ceiling. The parser (_detect_bbox_px/_detect_boxes_px) reads xmin-first
# to match, so keep the prompt and parser in agreement.
PERCEPTION_PROMPT = ("Detect the <target_object> from the provided info about it. The box_2d should be [xmin, ymin, xmax, ymax] in the image normalized to 0-1000. "
                     "The top-left corner of the image is the origin. The x- and y-axes go horizontally and vertically, respectively. "
                     "Return one JSON object with a label. If the target is genuinely not visible, return the empty JSON array []. "
                     "Never return masks or code fencing. Limit to one object only. "
                     "Example output:\n\n"
                     "```json\n"
                     "{'box_2d': box_2d, 'label': target_object}\n"
                     "```\n\n")
PERCEPTION_PROMPT_MULTI = ("Detect up to 12 instances of the <target_object> from the provided info about it - the ones CLOSEST to the centre of the image. "
                           "Each box_2d is [xmin, ymin, xmax, ymax] normalized to 0-1000, origin at the top-left, x horizontal and y vertical. "
                           "Return a JSON array with one entry per instance and nothing else; return [] only when no matching target is genuinely visible. "
                           "Never return masks or extra code fencing. "
                           "Example output:\n\n"
                           "```json\n"
                           "[{'box_2d': box_2d, 'label': target_object}, {'box_2d': box_2d, 'label': target_object}]\n"
                           "```\n\n")
FIND_MOST_SIMILAR_OCR_BBOX_PROMPT = ("An OCR tool was used to extract texts from the image. "
                                     "Find the most semantically similar bounding box to the <target_object>. "
                                     "An Embodied AI Agent will be using this bounding box to center the agent's perspective on the target. "
                                     "You will receive a list of bounding boxes and their labels along with the <target_object>. "
                                     "Return the bounding box that best matches the <target_object>. "
                                     "Example output:\n\n"
                                     "```json\n"
                                     "{'box_2d': box_2d, 'label': target_object}\n"
                                     "```\n\n")

EXTRACTABLE_JSON_PATTERN = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

from sim.env import *
from manip.manipulation import *
# `import *` skips underscore names, so pull the per-step hand primitives scan_held_item needs
# explicitly (the public extend_left_hand_forward loops N steps without returning per-step state;
# the sweep needs the state each 0.025 m to detect the stall). Same lambdas the probe used.
from sim.env import _XTNFWD_LEFT_, _PLLBCK_LEFT_, _XTNFWD_RIGHT_, _PLLBCK_RIGHT_


def _ocr_lines(source):
    """Send a path/PIL image/PNG body to the central OCR service and return validated lines."""
    return ocr_lines(source)


def _encode_image(image: Image.Image) -> dict:
    buf = BytesIO()
    image.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def read_text(image_path=None):
    image_path = image_path or os.path.join(screenshot_dir(), "ClientScreenshot.png")
    return "\n".join(_ocr_lines(image_path))

def extract_text_from_image(image_path):
    # The second element was historically raw Paddle output. It is now the service's parsed line
    # list so no Paddle result shape leaks into an agent process.
    lines = _ocr_lines(image_path)
    return "\n".join(lines), lines

def find_most_similar_bbox_to_target_name(target_name, ocr_result):
    bboxes = '\n'.join([f'* {box}' for box in ocr_result])
    with token_meter.role(token_meter.ROLE_PERCEPTION):
        resp = call_with_api_retries(
            lambda: CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"{FIND_MOST_SIMILAR_OCR_BBOX_PROMPT}\n\ntarget_name={target_name}\n\n{bboxes}"}],
                temperature=0.5,
                max_tokens=400,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}},
            )
        )
    annotated_bbox = resp.choices[0].message.content

    match = re.search(EXTRACTABLE_JSON_PATTERN, annotated_bbox)
    if match:
        extracted = match.group(1)
        box_2d = ast.literal_eval(extracted)
        box_2d = box_2d['box_2d']
        return box_2d
    return None

def transform_paddle_result_to_coco_label_format(paddle_result):
    return [(b[0][0][0],b[0][0][1], b[0][2][0], b[0][2][1], b[1][0]) for b in paddle_result[0]]


def annotate_target(ymin, xmin, ymax, xmax, file_path=None,
                    source_image=None):
    """Draw the detected box on the screenshot (debug eyeballing only). Inputs are PIXEL coords -
    every caller passes pixels (bbox already scaled by ORIGINAL_W/H). The previous body re-divided
    by 1000 and re-multiplied by ORIGINAL_*, treating pixels as if normalised, so it drew a box
    shrunk ~1.08x/1.92x near the top-left and CRASHED PIL whenever an inverted box made y1<y0.
    Now: sort so min<=max and clamp to the image, so a malformed detection can't kill the run.

    ``source_image`` lets live callers annotate the exact in-memory frame they detected against.
    This matters when parallel benchmark workers share the legacy ClientScreenshot.png path.
    Annotation is debug-only, so an I/O failure is reported but must never abort an agent run."""
    try:
        if source_image is None:
            file_path = file_path or os.path.join(screenshot_dir(), "ClientScreenshot.png")
            with Image.open(file_path) as opened:
                opened.load()
                image = opened.copy()
        else:
            source_image.load()
            image = source_image.copy()
        draw = ImageDraw.Draw(image)
        W, H = image.size
        # Inputs are in the fixed ORIGINAL_WIDTH x ORIGINAL_HEIGHT virtual frame (detections are
        # 0-1000 normalised * ORIGINAL_*), so scale to the ACTUAL frame before drawing. Without this
        # a higher-res screenshot paints the box/crosshair in the top-left quadrant.
        sx, sy = W / ORIGINAL_WIDTH, H / ORIGINAL_HEIGHT
        x0, x1 = sorted((int(xmin * sx), int(xmax * sx)))
        y0, y1 = sorted((int(ymin * sy), int(ymax * sy)))
        x0, x1 = max(0, min(x0, W - 1)), max(0, min(x1, W - 1))
        y0, y1 = max(0, min(y0, H - 1)), max(0, min(y1, H - 1))
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, max(0, y0 - 12)), "Target", fill="red")
        # Debug-only annotated frame: cap on disk at 1080p (drawn at capture res, which may be 4K).
        out_path = artifact_path("screenshots", "annotated_target.png",
                                 legacy_base="")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        downscale_pil_for_storage(image).save(out_path)
    except Exception as e:
        print(f"[perception] target annotation skipped: {type(e).__name__}: {e}")


def _draw_debug_frame(frame_source, boxes, chosen, aim_xy, out_path):
    """Debug-only: save `frame_source` with EVERY VLM candidate box (thin yellow), the chosen /
    tracked instance (thick red), and the aim crosshair (green). This is the "what did the
    detector actually return, and which one are we driving to centre" picture. Best-effort - a
    drawing error must never take down a centring run."""
    try:
        if isinstance(frame_source, Image.Image):
            frame_source.load()
            img = frame_source.convert("RGB")
        else:
            with Image.open(frame_source) as opened:
                opened.load()
                img = opened.convert("RGB")
        draw = ImageDraw.Draw(img)
        # boxes/aim are in the fixed ORIGINAL_WIDTH x ORIGINAL_HEIGHT virtual frame; scale to the
        # ACTUAL screenshot so the crosshair sits at true centre at any capture resolution (a 4K
        # frame would otherwise draw the green aim at 960,540 - its top-left quadrant, not centre).
        W, H = img.size
        sx, sy = W / ORIGINAL_WIDTH, H / ORIGINAL_HEIGHT
        for b in boxes:
            draw.rectangle([b['xmin'] * sx, b['ymin'] * sy, b['xmax'] * sx, b['ymax'] * sy],
                           outline="yellow", width=2)
        if chosen is not None:
            draw.rectangle([chosen['xmin'] * sx, chosen['ymin'] * sy,
                            chosen['xmax'] * sx, chosen['ymax'] * sy], outline="red", width=4)
            draw.text((chosen['xmin'] * sx, max(0, chosen['ymin'] * sy - 13)), "locked", fill="red")
        ax, ay = int(aim_xy[0] * sx), int(aim_xy[1] * sy)
        draw.line([(ax - 28, ay), (ax + 28, ay)], fill="lime", width=3)
        draw.line([(ax, ay - 28), (ax, ay + 28)], fill="lime", width=3)
        # Debug-only frame: cap on disk at 1080p (the detection ran on the full-res `image`, not this).
        downscale_pil_for_storage(img).save(out_path)
    except Exception as e:
        print(f"[CENTER] debug frame save failed: {type(e).__name__}: {e}")


DETECTION_PARSE_MAX_ATTEMPTS = 3


class BBoxResponseParseError(ValueError):
    """The VLM replied, but its bbox payload was malformed or truncated."""


def _bbox_payload(content):
    """Parse a complete JSON/Python-literal bbox payload.

    A valid ``[]`` is the only negative detection result. Invalid or truncated output raises so it
    cannot be confused with "target not visible".
    """
    if not isinstance(content, str) or not content.strip():
        raise BBoxResponseParseError("empty VLM response")
    text = content.strip()
    fenced = re.fullmatch(r"```\s*(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Qwen commonly emits single-quoted Python literals despite the JSON instruction. Keep
        # supporting those, while parsing actual JSON first (including its null/true/false values).
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as error:
            raise BBoxResponseParseError(
                f"bbox payload is not complete valid JSON/literal: {error}"
            ) from error
    if parsed == []:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, list) or not parsed:
        raise BBoxResponseParseError(
            f"bbox payload must be an object, non-empty list, or [] (got {type(parsed).__name__})"
        )
    return parsed


def _bbox_dict_px(box_2d):
    if not isinstance(box_2d, dict):
        raise BBoxResponseParseError(
            f"bbox entry must be an object (got {type(box_2d).__name__})"
        )
    coords = box_2d.get('box_2d') or box_2d.get('bbox_2d')
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        raise BBoxResponseParseError("bbox entry is missing a four-value box_2d")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in coords):
        raise BBoxResponseParseError("bbox coordinates must be finite numbers")
    xmin = coords[0] / 1000 * ORIGINAL_WIDTH
    ymin = coords[1] / 1000 * ORIGINAL_HEIGHT
    xmax = coords[2] / 1000 * ORIGINAL_WIDTH
    ymax = coords[3] / 1000 * ORIGINAL_HEIGHT
    if xmin >= xmax or ymin >= ymax:
        raise BBoxResponseParseError("bbox coordinates are empty or reversed")
    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'cx': (xmin + xmax) / 2.0, 'cy': (ymin + ymax) / 2.0,
            'label': box_2d.get('label', 'unknown')}


def _bbox_request(image, target_info, prompt, max_tokens, temperature):
    with token_meter.role(token_meter.ROLE_PERCEPTION):
        return call_with_api_retries(
            lambda: CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [
                    _encode_image(image),
                    {"type": "text", "text": f"{prompt}\n\ntarget_info={target_info}\n\n"},
                ]}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}},
            )
        ).choices[0].message.content


def _parsed_bbox_response(image, target_info, prompt, max_tokens, temperature):
    last_error = None
    for attempt in range(1, DETECTION_PARSE_MAX_ATTEMPTS + 1):
        content = _bbox_request(image, target_info, prompt, max_tokens, temperature)
        print(f"[DETECT OBJECT IN FRAME] Response: {content}")
        try:
            return [_bbox_dict_px(entry) for entry in _bbox_payload(content)]
        except BBoxResponseParseError as error:
            last_error = error
            print(
                f"[DETECT OBJECT IN FRAME] malformed bbox response "
                f"({attempt}/{DETECTION_PARSE_MAX_ATTEMPTS}): {error}"
            )
    raise BBoxResponseParseError(
        f"bbox response remained malformed after {DETECTION_PARSE_MAX_ATTEMPTS} attempts: "
        f"{last_error}"
    ) from last_error


def _detect_bbox_px(image, target_info, temperature=0.0):
    """Detect ONE target box and return it in PIXELS as
    {'xmin','ymin','xmax','ymax','cx','cy','label'}, or None only for a valid [] response.
    Malformed/truncated responses are retried, then raise BBoxResponseParseError.

    temperature defaults to 0.0 on purpose: this is a LOCALISATION call, not a creative
    one. At the old 0.5 the same frame returned different coordinates run to run, which a
    closed centring loop would chase as if the object itself were moving. Deterministic box
    in, deterministic correction out. Pulled out of center_object_on_screen so it (and the
    projection math) can be exercised offline on saved PNGs - see center_offline_check.py."""
    entries = _parsed_bbox_response(
        image, target_info, PERCEPTION_PROMPT, 400, temperature
    )
    return entries[0] if entries else None


def _detect_boxes_px(image, target_info, temperature=0.0):
    """ALL matching instances as a list of box dicts (same shape as _detect_bbox_px's return);
    [] only for a valid empty response. Malformed/truncated responses are retried, then raise
    BBoxResponseParseError. Lets the centring loop pick and TRACK one instance instead of taking whatever
    single box the model happens to return - the fix for the loop hopping between identical
    items on a dense shelf (measured 2026-07-21). temperature 0.0 for the same reason as
    _detect_bbox_px: a stable candidate set look to look."""
    entries = _parsed_bbox_response(
        image, target_info, PERCEPTION_PROMPT_MULTI, 1500, temperature
    )
    return entries


def bbox_to_rotation(cx, cy, aim_x, aim_y, focal_px=FOCAL_PX):
    """Camera (pitch, yaw) in DEGREES that swings the bbox centre (cx, cy) onto the aim
    point (aim_x, aim_y): pinhole/atan model, one focal length for both axes. Signs match
    the _PAN_RIGHT_/_TILT_DOWN_ primitives in env.py - object right-of-aim -> +yaw (pan
    right), object below-aim -> +pitch (tilt down) - so the pair drops straight into
    TransformAgent((0,0,0), (pitch, yaw, 0))."""
    yaw = math.degrees(math.atan2(cx - aim_x, focal_px))
    pitch = math.degrees(math.atan2(cy - aim_y, focal_px))
    return pitch, yaw


def _seed_front_instance(boxes, aim_x, aim_y, front_bias):
    """Pick which detected instance the centring loop LOCKS onto at look 1 - biased toward the
    FRONT-of-row item. Returns one box dict FROM `boxes` (must be non-empty). PURE (no sim/I/O)
    so it can be A/B'd offline on saved frames - see center_offline_check.py.

    Why this exists (measured failure mode): a row of near-identical products stacks in DEPTH, so
    several instances land almost on top of each other near the aim in the 2D frame. Pure
    nearest-to-aim (the prior seed) can't tell the item in FRONT from the one behind it and would
    sometimes lock the BACK one - which then also corrupts the reach, because RequestLidarCenter's
    centre ray hits whatever is actually in front along the gaze, not the boxed back item.

    The frontmost instance is CLOSEST to the camera, and a closer item projects a LARGER bbox (and,
    occluding the ones behind it, a more complete one). So area is a cheap depth proxy - but only
    WITHIN a same-size row; across different SKUs it is size, not depth. We therefore keep the seed
    NEAR the aim (so the loop can still centre it and doesn't lock a big off-target item) and let
    area only break the choice among the near cluster:

        score(b) = dist_to_aim^2 / diag^2  -  front_bias * area(b) / frame_area      (minimise)

    Both terms are normalised to [0,1] (frame diagonal, frame area), so front_bias is a single
    dimensionless, resolution-independent knob:
      * front_bias = 0.0  -> exact prior behaviour (pure nearest-to-aim) - use for A/B.
      * larger front_bias -> stronger pull to the bigger/nearer instance; too large hijacks the
        seed to the biggest box anywhere in frame. The default is UNVALIDATED - A/B on saved frames
        with center_offline_check.py before trusting it (project doctrine: measure, don't assume).

    Only the look-1 SEED uses this; after that the loop tracks the locked instance by predicted
    position (PPD_YAW/PPD_PITCH), so front/back is decided once, here."""
    if front_bias <= 0.0 or len(boxes) == 1:
        return min(boxes, key=lambda b: (b['cx'] - aim_x) ** 2 + (b['cy'] - aim_y) ** 2)
    diag2 = float(ORIGINAL_WIDTH ** 2 + ORIGINAL_HEIGHT ** 2)
    frame_area = float(ORIGINAL_WIDTH * ORIGINAL_HEIGHT)

    def _score(b):
        d2 = ((b['cx'] - aim_x) ** 2 + (b['cy'] - aim_y) ** 2) / diag2
        area = max(0.0, (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])) / frame_area
        return d2 - front_bias * area

    return min(boxes, key=_score)


def center_object_on_screen(target_info, aim_norm=(0.5, 0.5), max_iters=5, tol_px=20.0,
                            gain=0.8, max_step_deg=12.0, front_bias=0, debug_dir=None):
    """Rotate the camera until the target's bbox centre sits on the aim point - CLOSED-LOOP.

    Was a single open-loop shot with a wrong linear gain (see the FOCAL_PX note): detect
    once, rotate once by px/19.2, never look again. It reliably undershot and never
    recovered. Now:

        detect (temp 0) -> measure residual from aim -> within tol_px? stop
                        -> else rotate by the atan-model angle -> re-screenshot -> repeat

    up to max_iters corrective rotations, with one final measurement-only look. Because
    TransformAgent rotations are relative deltas, each pass shrinks the residual, so any
    leftover gain error self-corrects across iterations instead of standing as a permanent
    miss.

    gain (<1) DAMPS each correction - command only `gain` of the computed angle. This is a
    deliberate under-shoot so the loop converges monotonically instead of overshooting and
    ringing. It exists because the atan model is not perfect on both axes: MEASURED live
    (phase-correlation, 2026-07-21) the true rates are pitch ~16.4 px/deg (matches the f=935
    model) but yaw ~18.1 px/deg (~11% hotter than the model, so an undamped yaw over-rotates
    and sails the target past centre). Rather than hard-code a second focal - the yaw rate is
    depth-dependent (parallax over an oblique shelf) so no single constant is right - we damp:
    gain 0.8 keeps net per-step motion below 1.0 even on the hot yaw axis, and smaller steps
    also change the view less between looks, which curbs the detector re-locking onto a
    different identical item (see the caveat below).

    max_step_deg caps each rotation for that same reason: a large swing at a far-off target
    changes the view enough to shrink/shift the candidate set and break the instance lock (seen
    live - a 20 deg step dropped candidates 12 -> 3 and the lock jumped). Far targets are closed
    over several BOUNDED looks instead of one jump; near targets never hit the cap.

    TARGET LOCK (measured 2026-07-21): on a shelf of near-identical items the detector will
    otherwise hop between instances as the view shifts - a step that "moves" the target further
    than any camera rotation could is that hop, not a gain error, and it was the loop's actual
    failure to converge. So detection returns ALL instances (_detect_boxes_px) and the loop
    stays on ONE: the FRONT-of-row instance on look 1 (front_bias / _seed_front_instance), then
    nearest to where that instance is PREDICTED to land (PPD_YAW/PPD_PITCH) on every look after.

    front_bias: how strongly the look-1 SEED prefers the FRONT item of a row over merely the box
      nearest the aim. A row of identical products stacks in depth and clusters near the aim in 2D,
      so nearest-to-aim alone sometimes locked the BACK one (and, since the reach's RequestLidarCenter
      ray hits whatever is in front along the gaze, that mismatched the grab too). The frontmost
      instance is closest and projects the LARGEST bbox, so this biases the seed toward area among
      the near cluster (see _seed_front_instance). front_bias=0.0 restores the pure nearest-to-aim
      seed - use it to A/B this as one variable. The default is UNVALIDATED: A/B on saved frames
      (center_offline_check.py) before trusting it. It touches ONLY look-1 seeding; the tracking
      logic below is unchanged.

    aim_norm: normalised (x, y) aim point. (0.5, 0.5) is the geometric centre. The grab
      sweet-spot sits BELOW centre - the extended hand shows up around y~0.67 (see
      ../center_object.py) - so aiming a grab will pass aim_norm=(0.5, 0.67). That is the
      deliberate follow-up (Phase D); it is kept a parameter so the change is one argument,
      not a rewrite. Default stays dead-centre so THIS change can be A/B'd as pure aiming
      accuracy, one variable at a time.

    Vertical caveat: this pitches the CAMERA onto the target; it does not lower the HAND.
    For a low shelf the follow-on grab still needs crouch / hand-height, not just a centred
    view (see extend_arm_until_grabbed's vertical-gap note). Centring gets the target in
    front; it is not by itself a grab.

    debug_dir: if set, each look writes {debug_dir}/look<i>_bbox.png - the frame with every VLM
    candidate box (yellow), the locked instance (red) and the aim crosshair (green) - so a test
    can show what the detector returned and which instance was tracked. None in production (no
    image I/O beyond the annotate_target debug write).

    Returns the last TransformAgent state dict, augmented with 'centered' (bool), 'detected'
    (bool), 'residual_px' (dx, dy at the final look), 'iters' (looks taken), 'outcome'
    (success | not_detected | detection_error | stalled | incomplete), 'center_message' (a human-readable line
    the runner surfaces to the agent as `last_center`, so the actor and the episodic learner know
    whether centring worked instead of guessing), and 'box' - the final locked bbox dict
    (xmin/ymin/xmax/ymax/cx/cy in the ORIGINAL_WIDTH x ORIGINAL_HEIGHT virtual frame, plus label),
    or None if the target was never detected. 'box' lets a caller OCR/crop exactly what was
    centred (e.g. read the POS screen inside its own bbox) without re-detecting."""
    aim_x = aim_norm[0] * ORIGINAL_WIDTH
    aim_y = aim_norm[1] * ORIGINAL_HEIGHT
    state = TransformAgent((0, 0, 0), (0, 0, 0))  # read current pose (no-op move)
    residual = (None, None)
    centered = False
    detected = False        # did the detector ever return the target?
    detection_error = None  # malformed VLM output after detector-level parse retries
    stalled = False         # did the residual stop shrinking (target likely at the frame edge)?
    locked = None           # predicted (x, y) of the tracked instance; None until look 1 picks one
    box = None              # last locked box (xmin/ymin/xmax/ymax/cx/cy/label); None if never detected
    image = None            # exact final frame; private handoff for region OCR
    prev_mag = None
    no_improve = 0
    i = 0
    for i in range(1, max_iters + 2):   # max_iters rotations + one measurement-only look
        screenshot = RequestScreenshot(save_image=True)
        image = Image.open(BytesIO(screenshot["image"]))
        image.load()
        try:
            boxes = _detect_boxes_px(image, target_info)
        except BBoxResponseParseError as error:
            detection_error = str(error)
            print(f"[CENTER] look {i}: detector parse failure - {error}")
            break
        if not boxes:
            print(f"[CENTER] look {i}: target not detected - not rotating (let the caller search).")
            break
        detected = True
        # Stay on ONE instance. Look 1 SEEDS on the FRONT-of-row instance (front_bias), not merely
        # the box nearest the aim - a stacked row of identical items would otherwise sometimes lock
        # the one behind. Every look after tracks THAT instance: nearest to where it was predicted
        # to land. This is what stops the hop between identical items as they cross near the aim.
        if locked is None:
            box = _seed_front_instance(boxes, aim_x, aim_y, front_bias)
        else:
            box = min(boxes, key=lambda b: (b['cx'] - locked[0]) ** 2 + (b['cy'] - locked[1]) ** 2)
        annotate_target(box['ymin'], box['xmin'], box['ymax'], box['xmax'],
                        source_image=image)
        if debug_dir:
            _draw_debug_frame(image, boxes, box, (aim_x, aim_y),
                              os.path.join(debug_dir, f"look{i}_bbox.png"))
        dx, dy = box['cx'] - aim_x, box['cy'] - aim_y
        residual = (round(dx, 1), round(dy, 1))
        print(f"[CENTER] look {i}: '{box['label']}' center=({box['cx']:.0f},{box['cy']:.0f}) "
              f"residual=({dx:+.0f},{dy:+.0f})px  ({len(boxes)} candidate(s))")
        if abs(dx) <= tol_px and abs(dy) <= tol_px:
            centered = True
            print(f"[CENTER] within tolerance ({tol_px:.0f}px) in {i} look(s).")
            break
        # Stall guard: if the residual stops shrinking for two looks the target is likely at the
        # frame edge (detection jitter, not a gain error). Stop instead of grinding out the budget
        # - that wasted looping is what an onlooker/learner mislabels as "centring keeps failing".
        mag = (dx * dx + dy * dy) ** 0.5
        no_improve = no_improve + 1 if (prev_mag is not None and mag >= prev_mag - 3.0) else 0
        prev_mag = mag
        if no_improve >= 2:
            stalled = True
            print(f"[CENTER] residual stopped improving ({residual}px) - stopping (target likely at frame edge).")
            break
        if i > max_iters:
            print(f"[CENTER] out of rotation budget ({max_iters}); residual {residual}px remains.")
            break
        pitch, yaw = bbox_to_rotation(box['cx'], box['cy'], aim_x, aim_y)
        pitch, yaw = pitch * gain, yaw * gain
        # Cap the per-step swing: a big rotation changes the view so much the detector returns a
        # different (smaller) candidate set and the instance lock jumps to a neighbour. Bounded
        # steps keep enough of the scene stable for the nearest-to-prediction re-lock to hold;
        # a far target just takes a few more looks (measured 2026-07-21).
        pitch = max(-max_step_deg, min(max_step_deg, pitch))
        yaw = max(-max_step_deg, min(max_step_deg, yaw))
        # Predict where THIS instance lands so the next look re-locks it, not a neighbour
        # (pan right -> item moves left, tilt down -> item moves up; measured px/deg).
        locked = (box['cx'] - yaw * PPD_YAW, box['cy'] - pitch * PPD_PITCH)
        print(f"[CENTER] rotate pitch={pitch:+.2f} yaw={yaw:+.2f} deg (gain {gain})")
        state = TransformAgent((0, 0, 0), (pitch, yaw, 0))

    # Outcome surfaced to the agent as `last_center` (see the runner loops): the actor and the
    # episodic learner must KNOW whether centring worked. Without it a silent success gets guessed
    # at - the learner once concluded "avoid center_object_on_screen" from a centre that actually
    # succeeded. The failure strings are actionable so the lesson is "get the target in view",
    # never "stop using the tool".
    if centered:
        outcome = "success"
        message = f"SUCCESS - target centred (residual {residual}px, {i} look(s))"
    elif detection_error is not None:
        outcome = "detection_error"
        message = (
            "ERROR - target detection could not be parsed after retries; this is a VLM response "
            "failure, not evidence that the target is absent. Retry center_object_on_screen"
        )
    elif not detected:
        outcome = "not_detected"
        message = ("FAILED - target not detected in the frame; tilt/pan to bring it into view, "
                   "then center again (do not abandon center_object_on_screen)")
    elif stalled:
        outcome = "stalled"
        message = (f"STALLED - centring stopped improving at {residual}px; the target is likely near "
                   "the frame edge - bring it more into view, then center again")
    else:
        outcome = "incomplete"
        message = (f"INCOMPLETE - detected but not centred within tolerance (residual {residual}px "
                   f"after {i} looks); move a little closer or center again")
    print(f"[CENTER] result: {message}")
    state = dict(state)
    state.update({'centered': centered, 'detected': detected, 'residual_px': residual,
                  'iters': i, 'outcome': outcome, 'center_message': message, 'box': box,
                  'detection_error': detection_error,
                  '_source_image': image})
    return state


# ===== Phase 6.2: fixed-input centring on the checkout counter ===================================
# center_to_counter is center_object_on_screen with the target and aim PINNED, so the scripted place
# gate (and later the 6.3 actor) centre the counter with NO free-text target to misspell and no aim to
# re-guess per call - the same "deterministic where we can be" philosophy as the rest of the pipeline.
# The counter is a big, matte, unambiguous surface, so this SHOULD be the easy case for the centring
# loop. If instead the front-item bias (commit 7042d09) pulls the look-1 seed onto a product sitting
# ON the counter, that is a FINDING - fall back to the Phase-2 perpendicular yaw that NavSession.goto
# already faces landmarks with, and trust the yaw - not a blocker (phase6 plan, "Centring the counter").
#
# UNMEASURED (2026-07-22): both constants below are PROVISIONAL - place_probe.py sets them from live
# reads before this is trusted (project doctrine: measure, don't assume).
#   - COUNTER_AIM_NORM is a knob because the counter TOP sits BELOW camera height: the LiDAR sample
#     plan_place wants (a point ON the surface, not the front lip) may need the aim BELOW centre, the
#     way the grab sweet-spot aims at y~0.67. Default dead-centre so the probe can A/B lowering it as
#     one variable.
#   - front_bias defaults to 0.0 here (pure nearest-to-aim), NOT the centring tool's 0.25: a single
#     flat surface has no front-of-row depth stack to disambiguate, and 0.0 removes the one knob most
#     likely to hijack the seed onto an item on the counter. A/B against 0.25 only if centring misses.
COUNTER_TARGET_INFO = ("main_goal=the checkout counter surface - the flat countertop you set items "
                       "down on, not the items on it "
                       "The top surface of the counter, not the entire counter.")
COUNTER_AIM_NORM = (0.5, 0.5)   # provisional; place_probe measures whether the aim should sit lower


def center_to_counter(aim_norm=COUNTER_AIM_NORM, front_bias=0.0, **kwargs):
    """center_object_on_screen bound to the checkout counter: fixed target + aim, zero-argument for the
    caller. The scripted 6.2 gate and the 6.3 actor both centre the counter through THIS, so there is
    no target string to misspell and no aim to re-guess per call. Returns exactly what
    center_object_on_screen returns (the 'centered'/'outcome'/'center_message' contract is unchanged).

    Note for callers CARRYING an item: do NOT stow the hands around this call (that drops the item and
    undoes Phase 6.1). The REST carry pose is out of frame and LiDAR-culled, so centring reads a clean
    frame with the hand active. See the module note above for the provisional constants and the
    front-item-bias fallback."""
    return center_object_on_screen(COUNTER_TARGET_INFO, aim_norm=aim_norm,
                                   front_bias=front_bias, **kwargs)


# ===== Phase 6.2 (restructured): fixed-input centring on the barcode scanner =====================
# The scan sibling of center_to_counter: cp54 is a SELF-CHECKOUT (POS screen + barcode scanner +
# bagging tray, probe finding 2026-07-22), and the scan sequence needs the agent square on the
# SCANNER PAD specifically - the tray centring above aims at the wrong third of the furniture.
#
# UNMEASURED (2026-07-22): wording and aim are PROVISIONAL seeds - place_probe's `scanner` command
# (probe M2) measures whether the detector locks the pad, and the winning wording/aim get pinned
# here, same as the counter's did. Prompt changes A/B on identical frames per project doctrine.
#   - The pad is SMALL and dark (black scan pad under the scanner housing's red LED window) - the
#     opposite detection regime from the big matte countertop. The wording names the visually
#     loudest cue (the red LED window) and excludes the two look-alike neighbours (POS screen =
#     also a dark rectangle; tray = also recessed/dark).
#   - front_bias stays 0.0 for the same reason as the counter: no depth stack to disambiguate, and
#     the bias is the knob most likely to hijack the seed onto an item or the screen.
SCANNER_TARGET_INFO = ("main_goal=the barcode scanner on the checkout counter - the black square "
                       "scan pad below the scanner head with the red LED window. Not the POS "
                       "screen, not the bagging tray, and not any product.")
SCANNER_AIM_NORM = (0.5, 0.5)   # provisional; place_probe `scanner [y]` dials it like `center [y]` did


def center_to_scanner(aim_norm=SCANNER_AIM_NORM, front_bias=0.0, **kwargs):
    """center_object_on_screen bound to the checkout's barcode scan pad: fixed target + aim, so the
    scan sequence (align_to_scanner -> scan_held_item) centres the pad with no free-text target to
    misspell. Returns exactly what center_object_on_screen returns ('centered'/'outcome'/
    'center_message' contract unchanged).

    Callers are CARRYING by construction (you centre the scanner in order to scan the held item) -
    do NOT stow the hands around this call; the REST carry pose is out of frame and LiDAR-culled.
    See the module note above: wording/aim are provisional until probe M2 pins them."""
    return center_object_on_screen(SCANNER_TARGET_INFO, aim_norm=aim_norm,
                                   front_bias=front_bias, **kwargs)


# ===== Phase 6.2 (restructured): centre on the BAGGING TRAY ======================================
# The bag step's own target, split OFF center_to_counter after a live run (2026-07-23) dropped an item
# on the SCANNER PAD: center_to_counter targets "the counter surface", whose bbox up close is huge and
# ill-defined - its centre wobbles over the scan pad and STALLS, and even a clean centre lands on the
# countertop near the scanner, not the recessed bagging bin. So the tray gets a dedicated canned target
# the way the pad and screen did. The bin is a distinct recessed rectangle, so it SHOULD centre more
# stably than the whole counter - but that (and the aim) are UNMEASURED: probe/gate confirms, then pins.
# CONSEQUENCE (flagged, not hidden): PLACE_ENVELOPE's place_max=1.28 m was measured centring the COUNTER
# SURFACE, not the tray - it must be RE-MEASURED with this target before the bag numbers are trusted.
TRAY_TARGET_INFO = ("main_goal=the bagging tray - the empty recessed rectangular bin next to the "
                    "self-checkout where you drop bagged items. Not the flat counter top, not the "
                    "black barcode scan pad, not the POS screen, and not any product.")
TRAY_AIM_NORM = (0.5, 0.5)   # provisional; dial deeper (y>0.5) if releases catch the tray's near lip


def center_to_tray(aim_norm=TRAY_AIM_NORM, front_bias=0.0, **kwargs):
    """center_object_on_screen bound to the bagging tray: fixed target + aim, the bag step's centring.
    Returns the center_object_on_screen contract (incl. 'box'/'outcome'). Callers are CARRYING by
    construction (you centre the tray to drop into it) - do NOT stow the hands. Wording/aim are
    PROVISIONAL until a live probe/gate pins them, and the place envelope must be re-measured against
    THIS target (see the module note above)."""
    return center_object_on_screen(TRAY_TARGET_INFO, aim_norm=aim_norm,
                                   front_bias=front_bias, **kwargs)


# ===== Phase 6.2 (restructured): centre + region-OCR the POS screen ==============================
# The scan has NO websocket query (BarcodeScanner.cs), so the measured scan signal is the receipt on
# the POS screen. Reading the WHOLE frame drags in shelf text, product labels, the "Start" button -
# noise that swamps the receipt lines. So: centre the SCREEN as a target, then OCR only its bbox
# (read_text_in_box). Same canned-target pattern as the counter/scanner; wording is PROVISIONAL
# until probe M5 pins it (the receipt text is big/flat/high-contrast - the easy case for both the
# detector and paddleocr, but MEASURE it).
SCREEN_TARGET_INFO = ("main_goal=the self-checkout POS screen - the bright rectangular monitor "
                      "showing the receipt / item list and the green Start button. Not the barcode "
                      "scan pad, not the bagging tray, not any product.")
SCREEN_AIM_NORM = (0.5, 0.5)


def center_to_screen(aim_norm=SCREEN_AIM_NORM, front_bias=0.0, **kwargs):
    """center_object_on_screen bound to the POS screen: fixed target + aim. Its returned 'box' is
    the whole point here - pass it to read_text_in_box to OCR the receipt inside its own rectangle
    rather than the whole frame. Callers are CARRYING by construction (you read the screen to
    confirm the held item scanned); do NOT stow the hands. Wording is provisional until M5."""
    return center_object_on_screen(SCREEN_TARGET_INFO, aim_norm=aim_norm,
                                   front_bias=front_bias, **kwargs)


def read_text_in_box(box, pad_frac=0.04, image_path=None, source_image=None):
    """OCR only the region under `box` (a center_object_on_screen 'box' dict, whose
    xmin/ymin/xmax/ymax are in the ORIGINAL_WIDTH x ORIGINAL_HEIGHT virtual frame). Scales the box
    to the actual image size the same way annotate_target does, pads it by `pad_frac` of its size so
    a slightly-tight detection doesn't clip the edge glyphs, crops, and sends a PNG to the OCR API.

    Cropping first is the whole win over read_text(): the POS receipt is a small bright rectangle in
    a busy frame, and full-frame OCR mixes in shelf/label/button text. Live callers pass
    ``source_image`` from the centering result, so OCR cannot race another screenshot writer. The
    attempt-local saved frame is retained as a compatibility fallback for standalone callers.

    FAILURE CONTRACT — CHANGED, deliberately. This used to catch Exception, print
    ``[OCR] read_text_in_box failed (...)`` and return [], on the rule "OCR must never take down the
    caller". That rule was wrong here, because [] is not a neutral value for the callers: the
    checkout path (_fuzzy_new_lines, store_map's scan confirmation, gates/smoke_checkout) diffs
    receipt lines against a baseline, and an empty read is indistinguishable from "nothing new was
    scanned". Unavailable OCR therefore must fail the run rather than silently converting the
    MEASURED scan signal into a permanent "no scan detected", i.e. exactly the fake negative the
    measure-don't-assume rule exists to prevent. So OCR failure now propagates: an empty [] returned
    from here means paddle genuinely read no text, and nothing else.

    A genuinely absent box still returns [] — "no screen was centred" is a real, non-erroneous
    answer that the caller already branches on."""
    if not box:
        return []
    try:
        if source_image is not None:
            source_image.load()
            img = source_image.convert("RGB")
        else:
            image_path = image_path or os.path.join(screenshot_dir(), "ClientScreenshot.png")
            with Image.open(image_path) as opened:
                opened.load()
                img = opened.convert("RGB")
        W, H = img.size
        sx, sy = W / ORIGINAL_WIDTH, H / ORIGINAL_HEIGHT
        x0, x1 = sorted((box['xmin'] * sx, box['xmax'] * sx))
        y0, y1 = sorted((box['ymin'] * sy, box['ymax'] * sy))
        px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
        x0, x1 = max(0, int(x0 - px)), min(W, int(x1 + px))
        y0, y1 = max(0, int(y0 - py)), min(H, int(y1 + py))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return []
        crop = img.crop((x0, y0, x1, y1))
        # The crop stays in memory and is encoded as PNG by the API-only client. No scratch-file
        # handoff and no process-local Paddle model exist in an agent.
        return _ocr_lines(crop)
    except Exception as e:
        raise OcrUnavailable(
            f"read_text_in_box failed ({type(e).__name__}: {e}). Region OCR is REQUIRED by the "
            f"callers that diff receipt lines - returning no text here would silently read as "
            f"'nothing scanned'. Start the service with `uv run poe ocr-server`."
        ) from e


def _fuzzy_new_lines(before, after, ratio=0.8):
    """Multiset diff of OCR line lists tolerant of per-frame character jitter. A line in `after`
    counts as NEW iff no not-yet-consumed line in `before` matches it with a difflib ratio >= `ratio`.
    Consuming matches keeps multiset semantics (a second identical receipt line is genuinely new),
    while the fuzzy compare stops OCR noise (`JIN_RAKEN` one frame, `JIN_RAMEN` the next) from
    re-counting an already-present line as new - the exact-string bug the probe exposed. PURE, so it
    is unit-tested offline (tests/test_scan_diff.py)."""
    import difflib
    pool = list(before)
    new = []
    for a in after:
        hit = -1
        best = ratio
        for idx, b in enumerate(pool):
            r = difflib.SequenceMatcher(None, a, b).ratio()
            if r >= best:
                best, hit = r, idx
        if hit >= 0:
            pool.pop(hit)          # consumed - a real duplicate later still reads as new
        else:
            new.append(a)
    return new


def scan_held_item(hand="auto", baseline=None, max_extend=25, fuzzy_ratio=0.8, debug_dir=None):
    """Scan the item currently held in `hand` by sweeping it through the checkout scan zone WITHOUT
    releasing it, then confirm off the POS screen. Phase 6.2 - promoted from place_probe.sweep after
    the in-hand-sweep mechanic passed live (M1/M3/M5, 2 datapoints, 2026-07-22: item scanned and
    stayed held both runs at slant 0.84-0.89 m; OCR read real catalog IDs off the receipt).

    PRECONDITION: the agent is already ALIGNED on the scan pad (store_map.align_to_scanner) with the
    item in hand. This tool does NOT move the body or the camera onto the pad - it sweeps the HAND
    from where the agent stands, so a misaligned caller simply won't scan (an honest miss, not a
    crash). The full-hand REST/GRAB ownership mirrors extend_arm_until_grabbed.

    SEQUENCE: guard (must be holding; REFUSE otherwise) -> GRAB pose -> extend fully toward the pad
    (the hand's stall / Unity 0.5 clamp is the stop; the user-enlarged Easy trigger box covers the
    whole scan region, so a full extension transits it - there is NO stop-distance calibration) ->
    retract exactly as far as extended -> restore REST. The grip is NEVER opened: a held item keeps
    its Barcode MeshCollider live as a trigger (RetailItemRuntimeService.cs), so the sweep scans
    without dropping. There is no drop-scan fallback (deleted by decision 2026-07-22).

    VERIFICATION (no checkout websocket query exists yet - a filed RequestCheckoutState is the durable
    fix): with the hand back at REST (out of frame, no occlusion), center_to_screen locks the POS
    monitor and read_text_in_box OCRs the receipt inside its own bbox. A NEW receipt line vs
    `baseline` (the receipt lines seen BEFORE this scan) => scanned. The diff is FUZZY (_fuzzy_new_lines)
    so OCR char jitter across frames does not fake a scan. `baseline` MUST be threaded across a
    multi-item session (the receipt accumulates - the previous item's line is still on screen); pass
    the returned `receipt` as the next call's baseline. baseline=None treats the checkout as empty
    (correct for the first scan of a fresh checkout, the common gate case).

    Returns {'scanned': bool, 'still_holding': bool, 'receipt': [lines], 'new_lines': [lines],
    'screen_detected': bool, 'reason': str} - surfaced to the actor/learner as `last_scan`. `scanned`
    is the MEASURED signal (OCR delta); the gate also records the human `scanned_verified` separately
    and never promotes one to the other.
    """
    # 'auto' resolves to the hand that IS holding (left-first when both hold - name 'right' to scan
    # the other item); an explicit empty hand is refused. Subsumes the old empty-hand guard.
    from manip.manipulation import resolve_release_hand
    hand, refuse_reason = resolve_release_hand(hand)
    if hand is None:
        print(f"[scan_held_item] REFUSED: {refuse_reason}")
        return {"scanned": False, "still_holding": False, "receipt": [], "new_lines": [],
                "screen_detected": False, "reason": refuse_reason + " - nothing to scan"}
    extend_fn = _XTNFWD_LEFT_ if hand == "left" else _XTNFWD_RIGHT_
    pull_fn = _PLLBCK_LEFT_ if hand == "left" else _PLLBCK_RIGHT_
    grip_key = "leftGrippedState" if hand == "left" else "rightGrippedState"
    trans_key = "leftTranslation" if hand == "left" else "rightTranslation"

    def _hand_state():
        return TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    set_hand_pose("grab", hand=hand)
    moved = 0
    try:
        prev = _hand_state()[trans_key]
        for _ in range(max_extend):                # full extension; the stall guard stops earlier
            cur = extend_fn()[trans_key]
            if sum((cur[k] - prev[k]) ** 2 for k in range(3)) ** 0.5 <= 1e-4:
                break                              # hand clamped at its reach limit
            prev = cur
            moved += 1
    finally:
        for _ in range(moved):                     # retract exactly as far as we extended
            pull_fn()
        set_hand_pose("rest", hand=hand)

    still_holding = bool(_hand_state().get(grip_key))

    # Measured confirmation: centre the POS screen, region-OCR the receipt, diff vs the baseline.
    res = center_to_screen(debug_dir=debug_dir)
    box = res.get("box")
    receipt = read_text_in_box(box, source_image=res.get("_source_image")) if box else []
    new = _fuzzy_new_lines(baseline or [], receipt, ratio=fuzzy_ratio)
    scanned = any(any(c.isdigit() for c in l) for l in new)

    if not box:
        reason = ("swept; POS screen not detected so the scan is UNCONFIRMED (measured channel "
                  "unavailable) - re-centre the screen or verify by eye")
    elif scanned:
        reason = f"scanned (new receipt line: {new[0]})"
    else:
        reason = ("swept but no new receipt line - not aligned on the pad, or out of the scan "
                  "zone; re-run align_to_scanner and retry")
    if not still_holding:
        reason = "the grip OPENED during the sweep (should not happen) - " + reason

    print(f"[scan_held_item] hand={hand} scanned={scanned} still_holding={still_holding} "
          f"screen_detected={bool(box)} new_lines={new}")
    return {"scanned": scanned, "still_holding": still_holding, "receipt": receipt,
            "new_lines": new, "screen_detected": bool(box), "reason": reason, "hand": hand}


# The hand can clip THROUGH the counter/tray in the sim (a physics limitation we cannot fix - user,
# 2026-07-23), so a full-extension release would drop the item INSIDE or beyond the counter. Measured
# release rule (user, personally measured): stop and open the grip when the hand's forward translation
# z reaches (centre-LiDAR distance - this standoff), i.e. 0.1 m SHORT of the measured surface, so the
# item falls ONTO the surface, not through it. Applies to the drop only; the scan sweep still extends
# fully (an item passing through the scan zone is fine, and it is never released there).
PLACE_CLIP_STANDOFF_M = 0.1

# How off-centre a tray lock may be and still release (user, 2026-07-23): the drop does NOT need to be
# dead-centre, only ON the tray and not out at the lip. The LiDAR samples FRAME CENTRE (= the drop
# point); we accept when that point lands within the inner `edge_margin` fraction of the tray's OWN
# bounding box - so at 0.6 the ray may sit up to 60% of the way from the box centre toward an edge,
# leaving a 40% margin to the lip. Bigger = more permissive (fewer aborts) but riskier near the lip
# (the aim confound - items catching the near lip roll off). Measured against the TRAY box, so an
# off-centre accept can NOT drift onto the scanner (that was finding-8's center_to_COUNTER failure).
PLACE_TRAY_EDGE_MARGIN = 0.6


def _ray_in_box(box, edge_margin):
    """True if the LiDAR sample point (frame centre) falls within the inner `edge_margin` fraction of
    `box`. Robust to aim_norm: computed from the box + the fixed frame centre, not the residual."""
    if not box:
        return False
    ray_x, ray_y = ORIGINAL_WIDTH / 2.0, ORIGINAL_HEIGHT / 2.0
    half_w = max(1.0, (box["xmax"] - box["xmin"]) / 2.0)
    half_h = max(1.0, (box["ymax"] - box["ymin"]) / 2.0)
    return (abs(ray_x - box["cx"]) <= half_w * edge_margin
            and abs(ray_y - box["cy"]) <= half_h * edge_margin)


def place_held_item(hand="auto", aim_norm=None, max_approach_iters=4, max_extend=25,
                    clip_standoff=PLACE_CLIP_STANDOFF_M, edge_margin=PLACE_TRAY_EDGE_MARGIN,
                    debug_dir=None):
    """Release the held item onto the checkout bagging tray - the ONE deliberate release in the whole
    checkout chain (everywhere else the grip stays closed). Phase 6.2.

    PRECONDITION: at the counter (go_to_counter) and holding the (already scanned) item at REST. This
    tool centres the tray itself, so the caller need not pre-centre.

    Sequence: guard (must hold; REFUSE otherwise) -> center_to_tray; ONLY when centring SUCCEEDS
    (outcome=='success') is the depth gate trusted -> plan_place; if the verdict is 'move'/'back',
    creep the body and re-centre, up to max_approach_iters, re-planning each time (the deterministic
    depth gate owns the standoff - no dead-reckoning); once 'placeable' UNDER a successful centre,
    GRAB-extend the held item OVER the tray, open the grip, retract, restore REST. A non-approachable
    verdict (crouch/bail/recenter/unavailable) OR a centre that never locks aborts WITHOUT releasing -
    better to keep carrying than drop the item off the tray.

    RELEASE-GATE (fixed 2026-07-23 after a live run dropped on the scanner pad; loosened same day):
    a STALLED/INCOMPLETE centre used to fall through to a release, dropping the item wherever the
    camera gave up. The release no longer requires a dead-centre lock - it requires the LiDAR ray (the
    drop point) to fall within the inner `edge_margin` of the TRAY box (_ray_in_box): off-centre is
    fine, near-the-lip is not, so a stall with the tray still well in frame is accepted. If the ray is
    out near the lip / off the box, it re-centres; repeated failures abort HOLDING. Safe because the
    check is against center_to_tray's OWN box, so an off-centre accept can't drift onto the scanner
    (finding-8 was center_to_counter, whose bbox stalled near the scanner).

    CLIP-STANDOFF (user measured, 2026-07-23): the hand can clip THROUGH the counter in the sim, so the
    release does NOT extend to the reach clamp - it stops when the hand's forward z reaches
    (plan['distance'] - clip_standoff), 0.1 m short of the measured surface, so the item drops ONTO the
    tray rather than through it. If the hand stalls before that depth, it releases there (still short,
    so still no clip-through).

    Honest scoring: `placed` (the MEASURED signal) is True iff the grip actually opened AND the release
    happened under a 'placeable' plan_place verdict at a SUCCESSFULLY-centred tray. It is NOT proof the
    item is physically in the tray: the gate records `placed_verified` (a screenshot) separately and
    never promotes this to it. UNVERIFIED: PLACE_ENVELOPE.place_max was measured on the COUNTER SURFACE,
    not the tray - re-measure against this target. Aim confound: pass aim_norm=(0.5, y>0.5) to dial the
    release deeper if the gate shows items catching the tray's near lip.

    Returns {'placed', 'released', 'verdict', 'distance', 'surface_height', 'iters', 'reason'} -
    surfaced as `last_place`."""
    from manip.manipulation import plan_place, PLACE_ENVELOPE, set_hand_pose as _set_pose, resolve_release_hand
    # 'auto' resolves to the hand that IS holding (left-first when both hold - name 'right' to place
    # the other item); an explicit empty hand is refused. Subsumes the old empty-hand guard.
    hand, refuse_reason = resolve_release_hand(hand)
    if hand is None:
        print(f"[place_held_item] REFUSED: {refuse_reason}")
        return {"placed": False, "released": False, "verdict": "no_item", "distance": None,
                "surface_height": None, "iters": 0, "reason": refuse_reason + " - nothing to place"}
    extend_fn = _XTNFWD_LEFT_ if hand == "left" else _XTNFWD_RIGHT_
    pull_fn = _PLLBCK_LEFT_ if hand == "left" else _PLLBCK_RIGHT_
    grip_fn = ToggleLeftGrip if hand == "left" else ToggleRightGrip
    grip_key = "leftGrippedState" if hand == "left" else "rightGrippedState"
    trans_key = "leftTranslation" if hand == "left" else "rightTranslation"
    aim = aim_norm or TRAY_AIM_NORM

    def _hand_state():
        return TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    # ---- Approach: centre the tray, depth-gate the standoff, creep until 'placeable' -------------
    # The release ONLY fires on a 'placeable' verdict taken under a SUCCESSFUL centre (ready=True).
    # A stalled/incomplete centre means the LiDAR ray is not reliably on the tray, so we do NOT plan a
    # move or a release off it - we re-centre; if it never locks, the loop exhausts and we abort holding.
    plan = None
    iters = 0
    ready = False
    for iters in range(1, max_approach_iters + 1):
        res = center_to_tray(aim_norm=aim, debug_dir=debug_dir)
        box = res.get("box")
        if res.get("outcome") == "detection_error":
            return {"placed": False, "released": False, "verdict": "detection_error",
                    "distance": None, "surface_height": None, "iters": iters,
                    "reason": "bagging-tray detector returned malformed output after retries; "
                    "target visibility is unknown, so nothing was released"}
        if res.get("outcome") == "not_detected" or box is None:
            return {"placed": False, "released": False, "verdict": "not_detected", "distance": None,
                    "surface_height": None, "iters": iters,
                    "reason": "bagging tray not detected - re-approach the counter or pan it into view"}
        if not _ray_in_box(box, edge_margin):
            # The drop point (frame centre) is out near the tray lip / off the box - not dead-centre is
            # fine, but this is too far. Re-centre; do NOT release here.
            off = (round(ORIGINAL_WIDTH / 2.0 - box["cx"]), round(ORIGINAL_HEIGHT / 2.0 - box["cy"]))
            print(f"  [place] drop point near the tray edge (ray offset {off} px, outcome "
                  f"{res.get('outcome')}) - re-centring (iter {iters}/{max_approach_iters}), NOT releasing")
            continue
        plan = plan_place(RequestLidarCenter(), PLACE_ENVELOPE)
        v = plan["verdict"]
        if v == "placeable":
            ready = True
            break
        if v == "move":
            move_forward(plan["move_steps"]); continue
        if v == "back":
            move_backward(plan["move_steps"]); continue
        # crouch / bail / recenter / unavailable: don't release off the tray.
        return {"placed": False, "released": False, "verdict": v, "distance": plan["distance"],
                "surface_height": plan["surface_height"], "iters": iters,
                "reason": f"cannot place from here: {plan['reason']}"}
    if not ready:
        last_v = plan["verdict"] if plan else "centre_never_locked"
        return {"placed": False, "released": False, "verdict": last_v,
                "distance": plan["distance"] if plan else None,
                "surface_height": plan["surface_height"] if plan else None, "iters": iters,
                "reason": f"not placeable after {max_approach_iters} approach step(s) (last: {last_v}) "
                          f"- still holding rather than drop off the tray"}

    # ---- Release: extend toward the tray but STOP 0.1 m short of the surface, then open the grip ---
    # target_z is the hand-forward depth to release at: (centre distance - clip_standoff). Extending
    # to the reach clamp instead would let the hand clip THROUGH the counter and drop the item beyond
    # it (user-measured, 2026-07-23). Stop on reaching target_z OR a stall, whichever is first.
    _set_pose("grab", hand=hand)
    target_z = float(plan["distance"]) - clip_standoff
    moved = 0
    released = False
    stop = "target_depth"
    try:
        prev = _hand_state()[trans_key]
        while moved < max_extend:
            if prev[2] >= target_z:                # hand forward z at (centre dist - standoff): stop short of the surface
                break
            cur = extend_fn()[trans_key]
            if sum((cur[k] - prev[k]) ** 2 for k in range(3)) ** 0.5 <= 1e-4:
                stop = "stall"                     # reach clamp before target depth - release here (still short)
                break
            prev = cur
            moved += 1
        release_z = _hand_state()[trans_key][2]
        released = not grip_fn().get("gripped")    # open the hand; released iff the grip is now off
    finally:
        for _ in range(moved):                     # retract exactly as far as we extended
            pull_fn()
        _set_pose("rest", hand=hand)

    placed = bool(released)                         # released UNDER a 'placeable' verdict (we broke on it)
    reason = (f"released 0.1 m short of the tray (hand z={release_z:.2f} vs slant {plan['distance']:.2f} m, "
              f"stop={stop})" if placed else "extended but the grip did NOT open - still holding the item")
    print(f"[place_held_item] hand={hand} placed={placed} released={released} "
          f"verdict={plan['verdict']} slant={plan['distance']:.2f} release_z={release_z:.2f} stop={stop}")
    return {"placed": placed, "released": released, "verdict": plan["verdict"],
            "distance": plan["distance"], "surface_height": plan["surface_height"],
            "release_z": release_z, "iters": iters, "reason": reason, "hand": hand}


def detect_object_via_gemini(target_name):
    # config.env is loaded once at module import; the redundant CWD-relative reload here was a no-op.
    client = CLIENT  # UCL qwen (name kept for call sites; Gemini retired with OpenRouter)
    model_name = MODEL_NAME
    original_width = 1920
    original_height = 1080
    screenshot = RequestScreenshot(save_image=True)
    im = Image.open(BytesIO(screenshot["image"]))
    im.load()
    prompt = (f"Detect the {target_name} in the image. The box_2d should be [xmin, ymin, xmax, ymax] in the image normalized to 0-1000. "
              "The top left corner of the image is the origin. The x and y axis go horizontally and vertically, respectively. "
              "Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 1 object only. "
              "Here is an example output:\n\n"
              "```json\n"
              "{'box_2d': box_2d, 'label': label_name}\n"
              "```\n\n")

    with token_meter.role(token_meter.ROLE_PERCEPTION):
        resp = call_with_api_retries(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": [
                    _encode_image(im),
                    {"type": "text", "text": prompt},
                ]}],
                temperature=0.5,
                max_tokens=400,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}},
            )
        )
    annotated_bbox = resp.choices[0].message.content

    json_pattern = r'```json\n(.*?)\n```'
    match = re.search(json_pattern, annotated_bbox, re.DOTALL)
    if match:
        json_str = match.group(1)
        box_2d = ast.literal_eval(json_str)
        if type(box_2d) == list:
            box_2d = box_2d[0]
        coords = box_2d.get('box_2d') or box_2d.get('bbox_2d')
        xmin = coords[0] / 1000 * original_width   # [xmin, ymin, xmax, ymax] - Qwen order
        ymin = coords[1] / 1000 * original_height
        xmax = coords[2] / 1000 * original_width
        ymax = coords[3] / 1000 * original_height
    else:
        return None
    annotate_target(ymin, xmin, ymax, xmax, source_image=im)
    return {'box': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}}


_md_model = None

def _get_md_model():
    global _md_model
    if _md_model is None:
        import moondream as md
        _md_model = md.vl(api_key=os.getenv("MDREAM_API_KEY"))
    return _md_model


def detect_object_via_moondream(target_name):
    screenshot = RequestScreenshot(save_image=True)
    image = Image.open(BytesIO(screenshot["image"]))
    image.load()

    model = _get_md_model()
    result = model.detect(image, target_name)
    objects = result.get("objects", [])

    if not objects:
        return None

    def dist_to_center(obj):
        cx = (obj["x_min"] + obj["x_max"]) / 2
        cy = (obj["y_min"] + obj["y_max"]) / 2
        return (cx - 0.5) ** 2 + (cy - 0.5) ** 2

    best = min(objects, key=dist_to_center)

    xmin = best["x_min"] * ORIGINAL_WIDTH
    ymin = best["y_min"] * ORIGINAL_HEIGHT
    xmax = best["x_max"] * ORIGINAL_WIDTH
    ymax = best["y_max"] * ORIGINAL_HEIGHT

    annotate_target(ymin, xmin, ymax, xmax, source_image=image)
    return {"box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}}


LEGACY_DEPTH_API_URL = "http://202.92.159.242:8000/estimate-depth"


def request_rgbd_image(depth_api_url=None, timeout=30.0):
    """Request a depth map for the exact frame returned by the simulator.

    The old write-then-read through ClientScreenshot.png could silently upload another worker's
    frame. The returned websocket bytes are now the sole functional source. SARI_DEPTH_API_URL
    makes the dormant endpoint deployable without changing ordinary local behavior.
    """
    screenshot = RequestScreenshot(save_image=False)
    endpoint = depth_api_url or os.getenv("SARI_DEPTH_API_URL") or LEGACY_DEPTH_API_URL
    response = requests.post(
        endpoint,
        files={"file": ("ClientScreenshot.png", screenshot["image"], "image/png")},
        timeout=timeout,
    )

    response.raise_for_status()

    depth_img = Image.open(BytesIO(response.content))
    depth_path = artifact_path("depth_image.png", legacy_base="")
    os.makedirs(os.path.dirname(os.path.abspath(depth_path)), exist_ok=True)
    depth_img.save(depth_path)
    return depth_img


def estimate_steps_from_depth(bbox, depth_source):
    import numpy as np
    if isinstance(depth_source, np.ndarray):
        h, w = depth_source.shape[:2]
        cx = int((bbox["xmin"] + bbox["xmax"]) / 2 * w / 1920)
        cy = int((bbox["ymin"] + bbox["ymax"]) / 2 * h / 1080)
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        distance = float(depth_source[cy, cx])
        est_steps = round(distance / 0.1)
        print(f"[DEPTH] Array ({cx},{cy})={distance:.3f}m, steps={est_steps}")
        return est_steps
    # Fallback: colorized image from request_rgbd_image (RGB heuristic)
    img_width, img_height = depth_source.size
    pixels = depth_source.convert("RGB").load()
    cx = int((bbox["xmin"] + bbox["xmax"]) / 2 * img_width / 1920)
    cy = int((bbox["ymin"] + bbox["ymax"]) / 2 * img_height / 1080)
    cx = max(0, min(cx, img_width - 1))
    cy = max(0, min(cy, img_height - 1))
    r, g, b = pixels[cx, cy]
    closeness = (r - b + 255) / 510.0
    est_distance = 8.0 * (1 - closeness)
    est_steps = round(est_distance / 0.1)
    print(f"[DEPTH] RGB=({r},{g},{b}), closeness={closeness:.2f}, est_dist={est_distance:.2f}, steps={est_steps}")
    return est_steps


def approach_target(target_name, annotate=False):
    item = detect_object_via_moondream(target_name)
    box = item["box"]

    # depth.py (local monocular fallback) was REMOVED in Phase 4.2 - navigation distance is
    # LiDAR's job now, and the manipulation phase owns choosing a replacement for
    # distance-to-item (see phase4.2 plan, REMOVED #3). Only the remote API path remains here.
    depth_img = request_rgbd_image()
    steps = estimate_steps_from_depth(box, depth_img)

    x_center_before = (box["xmin"] + box["xmax"]) / 2
    y_center_before = (box["ymin"] + box["ymax"]) / 2
    if annotate:
        from vision.annotation_tools import annotate_boxes
        annotate_boxes(item)
    print("Y Center of bbox: ", y_center_before)
    print("X Center of bbox: ", x_center_before)
    y_cam_movement = -1 * (y_center_before - 540) / 19.2
    x_cam_movement = -1 * (x_center_before - 960) / 19.2

    print("Movement Y: ", y_cam_movement)
    print("Movement X: ", x_cam_movement)
    TransformAgent((0,0,0), ((y_center_before - 540) / 19.2,(x_center_before - 960) / 19.2, 0))

    state = move_forward(units=steps)
    return state, box


def face_cardinal_direction(angle_deg):
    current_angle = TransformAgent((0, 0, 0), (0, 0, 0))['rotation'][1]
    TransformAgent((0,0,0), (0,angle_deg-current_angle, 0))
    print(f"[ROTATE] Facing {angle_deg} degrees")


def strafe_to_center(bbox, image_width=1920):
    center_x = (bbox["xmin"] + bbox["xmax"]) / 2
    offset_px = center_x - (image_width / 2)
    offset_units = 0.1 * (offset_px / 19.2)

    print(f"[STRAFE] Object offset: {offset_px} px -> {offset_units:.2f} units")

    movement_count = int(offset_units // 1)
    for _ in range(movement_count):
        move_right(units=1)


def define_cardinal_direction(current_yaw_angle):
    cardinals = [0, 90, 180, 270, 360]
    resultant = [abs(c - current_yaw_angle) for c in cardinals]
    min_value = min(resultant)
    min_index = resultant.index(min_value) % 4
    return cardinals[min_index]


def retrieve_item(target_name, cardinal_deg=None, annotate=False):
    item = detect_object_via_gemini(target_name)
    if item is None:
        print("[RETRIEVE] Target not detected in current view. Navigate closer.")
        return None

    # depth.py fallback removed (Phase 4.2) - see approach_target's note.
    depth_img = request_rgbd_image()
    steps = estimate_steps_from_depth(item["box"], depth_img)
    est_distance = steps * 0.1
    if est_distance > GRAB_DISTANCE_THRESHOLD:
        print(f"[RETRIEVE] Too far to grab: {est_distance:.2f} units (threshold={GRAB_DISTANCE_THRESHOLD}). Navigate closer first.")
        return None

    state, box = approach_target(target_name, annotate=annotate)

    if not cardinal_deg:
        current_yaw_angle = TransformAgent((0,0,0), (0,0,0))['rotation'][1]
        cardinal_deg = define_cardinal_direction(current_yaw_angle)
    face_cardinal_direction(cardinal_deg)

    item = detect_object_via_gemini(target_name)
    if item:
        box = item["box"]
        if annotate:
            from vision.annotation_tools import annotate_boxes
            annotate_boxes(item)
        strafe_to_center(box)

    item = detect_object_via_gemini(target_name)
    box = item["box"]
    x_center_before = (box["xmin"] + box["xmax"]) / 2
    y_center_before = (box["ymin"] + box["ymax"]) / 2
    if annotate:
        from vision.annotation_tools import annotate_boxes
        annotate_boxes(item)
    print("Y Center of bbox: ", y_center_before)
    print("X Center of bbox: ", x_center_before)
    y_cam_movement = -1 * (y_center_before - 540) / 19.2
    x_cam_movement = -1 * (x_center_before - 960) / 19.2

    print("Movement Y: ", y_cam_movement)
    print("Movement X: ", x_cam_movement)
    TransformAgent((0,0,0), ((y_center_before - 540) / 19.2,(x_center_before - 960) / 19.2, 0))

    print("Final readout:", grab_and_read_item(text_read_fn=read_text))
    return state
