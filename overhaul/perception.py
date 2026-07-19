import ast
import random
import os
import base64
from dotenv import load_dotenv

from openai import OpenAI
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import ast
import re
load_dotenv('../api.env')

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
PERCEPTION_PROMPT = ("Detect the <target_object> from the provided info about it. The box_2d should be [ymin, xmin, ymax, xmax] in the image normalized to 0-1000. "
                     "The top-left corner of the image is the origin. The x- and y-axes go horizontally and vertically, respectively. "
                     "Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to one object only. Do not put the JSON inside a list/array. "
                     "Example output:\n\n"
                     "```json\n"
                     "{'box_2d': box_2d, 'label': target_object}\n"
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
    image = Image.open(file_path)
    draw = ImageDraw.Draw(image)

    ymin_pixel = int(ymin / 1000 * ORIGINAL_HEIGHT)
    xmin_pixel = int(xmin / 1000 * ORIGINAL_WIDTH)
    ymax_pixel = int(ymax / 1000 * ORIGINAL_HEIGHT)
    xmax_pixel = int(xmax / 1000 * ORIGINAL_WIDTH)
    draw.rectangle([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], outline="red", width=3)
    draw.text((xmin_pixel, ymin_pixel), "Target", fill="red")
    annotated_image_path = 'screenshots/annotated_target.png'
    image.save(annotated_image_path)


def center_object_on_screen(target_info):
    RequestScreenshot(save_image=True)
    filepath = os.path.join("screenshots", "ClientScreenshot.png")
    with open(filepath, "rb") as image_file:
        image_bytes = image_file.read()
    image = Image.open(BytesIO(image_bytes))

    resp = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": [
            _encode_image(image),
            {"type": "text", "text": f"{PERCEPTION_PROMPT}\n\ntarget_info={target_info}\n\n"},
        ]}],
        temperature=0.5,
        max_tokens=400,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )
    annotated_bbox = resp.choices[0].message.content
    print(f"[DETECT OBJECT IN FRAME] Response: {annotated_bbox}")

    match = re.search(EXTRACTABLE_JSON_PATTERN, annotated_bbox)
    if match:
        extracted = match.group(1)
        parsed = ast.literal_eval(extracted)
        box_2d = parsed[0] if isinstance(parsed, list) else parsed
        target_object = box_2d.get('label', 'unknown')
        box_2d = box_2d['box_2d']

        print(f"Detected bounding box: {box_2d}, Target Object: {target_object}")
        ymin = box_2d[0] / 1000 * ORIGINAL_HEIGHT
        xmin = box_2d[1] / 1000 * ORIGINAL_WIDTH
        ymax = box_2d[2] / 1000 * ORIGINAL_HEIGHT
        xmax = box_2d[3] / 1000 * ORIGINAL_WIDTH

        annotate_target(ymin, xmin, ymax, xmax)

        x_center_before = (xmin + xmax) / 2
        y_center_before = (ymin + ymax) / 2
        agent_x_rot = (x_center_before - 960) / 19.2
        agent_y_rot = (y_center_before - 540) / 19.2

        state = TransformAgent((0, 0, 0), (agent_y_rot, agent_x_rot, 0))
        return state
    return TransformAgent((0,0,0), (0,0,0))


def detect_object_via_gemini(target_name):
    import dotenv
    dotenv.load_dotenv('api.env')

    client = CLIENT  # UCL qwen (name kept for call sites; Gemini retired with OpenRouter)
    model_name = MODEL_NAME
    original_width = 1920
    original_height = 1080
    RequestScreenshot(save_image=True)
    file_path = os.path.join("screenshots", "ClientScreenshot.png")
    im = Image.open(BytesIO(open(file_path, "rb").read()))
    prompt = (f"Detect the {target_name} in the image. The box_2d should be [ymin, xmin, ymax, xmax] in the image normalized to 0-1000. "
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
        ymin = box_2d['box_2d'][0] / 1000 * original_height
        xmin = box_2d['box_2d'][1] / 1000 * original_width
        ymax = box_2d['box_2d'][2] / 1000 * original_height
        xmax = box_2d['box_2d'][3] / 1000 * original_width
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
