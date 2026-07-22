import ast
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
# Repo-root api.env, resolved from __file__ so it loads regardless of CWD.
load_dotenv(Path(__file__).resolve().parent.parent / 'api.env')

GRAB_DISTANCE_THRESHOLD = 2.0  # units; beyond this, retrieve_item refuses to grab

# Agent runtime = UCL qwen (user directive 2026-07-19; OpenRouter retired on 402). This is
# the bounding-box/centering client - qwen-VL replaces Gemini here, identically in BOTH A/B
# arms; bbox quality vs Gemini is unmeasured and shared, so it cannot skew the arms.
from agent import _ucl_creds
_UCL_HOST, _UCL_KEY = _ucl_creds()
MODEL_NAME = "Qwen/Qwen3.6-27B"
CLIENT = OpenAI(
    base_url=f"http://{_UCL_HOST}:8000/v1",
    api_key=_UCL_KEY,
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
                     "Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to one object only. Do not put the JSON inside a list/array. "
                     "Example output:\n\n"
                     "```json\n"
                     "{'box_2d': box_2d, 'label': target_object}\n"
                     "```\n\n")
PERCEPTION_PROMPT_MULTI = ("Detect up to 12 instances of the <target_object> from the provided info about it - the ones CLOSEST to the centre of the image. "
                           "Each box_2d is [xmin, ymin, xmax, ymax] normalized to 0-1000, origin at the top-left, x horizontal and y vertical. "
                           "Return a JSON array with one entry per instance and nothing else. Never return masks or extra code fencing. "
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

from env import *
from manipulation import *


_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang='en')
    return _ocr


def _encode_image(image: Image.Image) -> dict:
    buf = BytesIO()
    image.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def read_text(image_path='screenshots/ClientScreenshot.png'):
    result = _get_ocr().ocr(image_path)
    return "\n".join([line[1][0] for line in result[0]]) if result else ""

def extract_text_from_image(image_path):
    result = _get_ocr().ocr(image_path, cls=True)
    if len(result) == 0:
        return "", []
    try:
        final_result = "\n".join([line[1][0] for line in result[0]] if result else "")
    except Exception as e:
        final_result = ""
    return final_result, result

def find_most_similar_bbox_to_target_name(target_name, ocr_result):
    bboxes = '\n'.join([f'* {box}' for box in ocr_result])
    resp = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"{FIND_MOST_SIMILAR_OCR_BBOX_PROMPT}\n\ntarget_name={target_name}\n\n{bboxes}"}],
        temperature=0.5,
        max_tokens=400,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
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


def annotate_target(ymin, xmin, ymax, xmax, file_path='screenshots/ClientScreenshot.png'):
    """Draw the detected box on the screenshot (debug eyeballing only). Inputs are PIXEL coords -
    every caller passes pixels (bbox already scaled by ORIGINAL_W/H). The previous body re-divided
    by 1000 and re-multiplied by ORIGINAL_*, treating pixels as if normalised, so it drew a box
    shrunk ~1.08x/1.92x near the top-left and CRASHED PIL whenever an inverted box made y1<y0.
    Now: sort so min<=max and clamp to the image, so a malformed detection can't kill the run."""
    image = Image.open(file_path)
    draw = ImageDraw.Draw(image)
    W, H = image.size
    # Inputs are in the fixed ORIGINAL_WIDTH x ORIGINAL_HEIGHT virtual frame (detections are
    # 0-1000 normalised * ORIGINAL_*), so scale to the ACTUAL frame before drawing. Without this a
    # higher-res screenshot (e.g. 4K) paints the box/crosshair in the top-left quadrant instead of
    # over the target - the drawing is now resolution-dynamic, off the real image.size.
    sx, sy = W / ORIGINAL_WIDTH, H / ORIGINAL_HEIGHT
    x0, x1 = sorted((int(xmin * sx), int(xmax * sx)))
    y0, y1 = sorted((int(ymin * sy), int(ymax * sy)))
    x0, x1 = max(0, min(x0, W - 1)), max(0, min(x1, W - 1))
    y0, y1 = max(0, min(y0, H - 1)), max(0, min(y1, H - 1))
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    draw.text((x0, max(0, y0 - 12)), "Target", fill="red")
    image.save('screenshots/annotated_target.png')


def _draw_debug_frame(frame_path, boxes, chosen, aim_xy, out_path):
    """Debug-only: save `frame_path` with EVERY VLM candidate box (thin yellow), the chosen /
    tracked instance (thick red), and the aim crosshair (green). This is the "what did the
    detector actually return, and which one are we driving to centre" picture. Best-effort - a
    drawing error must never take down a centring run."""
    try:
        img = Image.open(frame_path).convert("RGB")
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
        img.save(out_path)
    except Exception as e:
        print(f"[CENTER] debug frame save failed: {type(e).__name__}: {e}")


def _detect_bbox_px(image, target_info, temperature=0.0):
    """Detect ONE target box and return it in PIXELS as
    {'xmin','ymin','xmax','ymax','cx','cy','label'}, or None if nothing parseable.

    temperature defaults to 0.0 on purpose: this is a LOCALISATION call, not a creative
    one. At the old 0.5 the same frame returned different coordinates run to run, which a
    closed centring loop would chase as if the object itself were moving. Deterministic box
    in, deterministic correction out. Pulled out of center_object_on_screen so it (and the
    projection math) can be exercised offline on saved PNGs - see center_offline_check.py."""
    resp = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": [
            _encode_image(image),
            {"type": "text", "text": f"{PERCEPTION_PROMPT}\n\ntarget_info={target_info}\n\n"},
        ]}],
        temperature=temperature,
        max_tokens=400,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )
    annotated_bbox = resp.choices[0].message.content
    print(f"[DETECT OBJECT IN FRAME] Response: {annotated_bbox}")

    match = re.search(EXTRACTABLE_JSON_PATTERN, annotated_bbox)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None
    box_2d = parsed[0] if isinstance(parsed, list) else parsed
    if not isinstance(box_2d, dict):
        return None
    coords = box_2d.get('box_2d') or box_2d.get('bbox_2d')  # Qwen sometimes uses bbox_2d
    if not coords or len(coords) != 4:
        return None
    label = box_2d.get('label', 'unknown')
    xmin = coords[0] / 1000 * ORIGINAL_WIDTH   # [xmin, ymin, xmax, ymax] - Qwen order (see prompt note)
    ymin = coords[1] / 1000 * ORIGINAL_HEIGHT
    xmax = coords[2] / 1000 * ORIGINAL_WIDTH
    ymax = coords[3] / 1000 * ORIGINAL_HEIGHT
    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'cx': (xmin + xmax) / 2.0, 'cy': (ymin + ymax) / 2.0, 'label': label}


def _detect_boxes_px(image, target_info, temperature=0.0):
    """ALL matching instances as a list of box dicts (same shape as _detect_bbox_px's return);
    [] if none. Lets the centring loop pick and TRACK one instance instead of taking whatever
    single box the model happens to return - the fix for the loop hopping between identical
    items on a dense shelf (measured 2026-07-21). temperature 0.0 for the same reason as
    _detect_bbox_px: a stable candidate set look to look."""
    resp = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": [
            _encode_image(image),
            {"type": "text", "text": f"{PERCEPTION_PROMPT_MULTI}\n\ntarget_info={target_info}\n\n"},
        ]}],
        temperature=temperature,
        max_tokens=1500,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )
    content = resp.choices[0].message.content
    # Tolerant parse: pull each {...} object out on its own rather than literal_eval-ing the whole
    # array. A long instance list can truncate at max_tokens with no closing ``` fence; the
    # fence-based parse then returns nothing. Per-object extraction still yields every COMPLETE
    # box and simply drops a half-written trailing one.
    body = content.split("```json", 1)[-1].split("```", 1)[0]
    out = []
    for m in re.finditer(r'\{[^{}]*\}', body):
        try:
            box_2d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if not isinstance(box_2d, dict):
            continue
        coords = box_2d.get('box_2d') or box_2d.get('bbox_2d')  # Qwen sometimes uses bbox_2d
        if not coords or len(coords) != 4:
            continue
        xmin = coords[0] / 1000 * ORIGINAL_WIDTH   # [xmin, ymin, xmax, ymax] - Qwen order (see prompt note)
        ymin = coords[1] / 1000 * ORIGINAL_HEIGHT
        xmax = coords[2] / 1000 * ORIGINAL_WIDTH
        ymax = coords[3] / 1000 * ORIGINAL_HEIGHT
        out.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                    'cx': (xmin + xmax) / 2.0, 'cy': (ymin + ymax) / 2.0,
                    'label': box_2d.get('label', 'unknown')})
    return out


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
    (success | not_detected | stalled | incomplete) and 'center_message' (a human-readable line
    the runner surfaces to the agent as `last_center`, so the actor and the episodic learner know
    whether centring worked instead of guessing)."""
    aim_x = aim_norm[0] * ORIGINAL_WIDTH
    aim_y = aim_norm[1] * ORIGINAL_HEIGHT
    state = TransformAgent((0, 0, 0), (0, 0, 0))  # read current pose (no-op move)
    residual = (None, None)
    centered = False
    detected = False        # did the detector ever return the target?
    stalled = False         # did the residual stop shrinking (target likely at the frame edge)?
    locked = None           # predicted (x, y) of the tracked instance; None until look 1 picks one
    prev_mag = None
    no_improve = 0
    i = 0
    for i in range(1, max_iters + 2):   # max_iters rotations + one measurement-only look
        RequestScreenshot(save_image=True)
        filepath = os.path.join("screenshots", "ClientScreenshot.png")
        with open(filepath, "rb") as image_file:
            image = Image.open(BytesIO(image_file.read()))
        boxes = _detect_boxes_px(image, target_info)
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
        annotate_target(box['ymin'], box['xmin'], box['ymax'], box['xmax'])
        if debug_dir:
            _draw_debug_frame(filepath, boxes, box, (aim_x, aim_y),
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
                  'iters': i, 'outcome': outcome, 'center_message': message})
    return state


def detect_object_via_gemini(target_name):
    # api.env is loaded once at module import; the redundant CWD-relative reload here was a no-op.
    client = CLIENT  # UCL qwen (name kept for call sites; Gemini retired with OpenRouter)
    model_name = MODEL_NAME
    original_width = 1920
    original_height = 1080
    RequestScreenshot(save_image=True)
    file_path = os.path.join("screenshots", "ClientScreenshot.png")
    im = Image.open(BytesIO(open(file_path, "rb").read()))
    prompt = (f"Detect the {target_name} in the image. The box_2d should be [xmin, ymin, xmax, ymax] in the image normalized to 0-1000. "
              "The top left corner of the image is the origin. The x and y axis go horizontally and vertically, respectively. "
              "Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 1 object only. "
              "Here is an example output:\n\n"
              "```json\n"
              "{'box_2d': box_2d, 'label': label_name}\n"
              "```\n\n")

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [
            _encode_image(im),
            {"type": "text", "text": prompt},
        ]}],
        temperature=0.5,
        max_tokens=400,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
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
    annotate_target(ymin, xmin, ymax, xmax)
    return {'box': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}}


_md_model = None

def _get_md_model():
    global _md_model
    if _md_model is None:
        import moondream as md
        _md_model = md.vl(api_key=os.getenv("MDREAM_API_KEY"))
    return _md_model


def detect_object_via_moondream(target_name):
    RequestScreenshot(save_image=True)
    file_path = os.path.join("screenshots", "ClientScreenshot.png")
    image = Image.open(BytesIO(open(file_path, "rb").read()))

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

    annotate_target(ymin, xmin, ymax, xmax)
    return {"box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}}


def request_rgbd_image():
    RequestScreenshot(save_image=True)
    DEPTH_API = "http://202.92.159.242:8000/estimate-depth"

    with open("screenshots/ClientScreenshot.png", "rb") as file:
        response = requests.post(DEPTH_API, files={"file": file})

    response.raise_for_status()

    depth_img = Image.open(BytesIO(response.content))
    depth_img.save("depth_image.png")
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
        from annotation_tools import annotate_boxes
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
            from annotation_tools import annotate_boxes
            annotate_boxes(item)
        strafe_to_center(box)

    item = detect_object_via_gemini(target_name)
    box = item["box"]
    x_center_before = (box["xmin"] + box["xmax"]) / 2
    y_center_before = (box["ymin"] + box["ymax"]) / 2
    if annotate:
        from annotation_tools import annotate_boxes
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
