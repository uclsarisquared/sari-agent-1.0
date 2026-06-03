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

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL_NAME = "google/gemini-2.5-pro-preview-05-06"
CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
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
    item = detect_object_via_gemini(target_info)
    if item is None:
        return TransformAgent((0, 0, 0), (0, 0, 0))

    box = item["box"]
    y_center = (box["ymin"] + box["ymax"]) / 2

    strafe_to_center(box)

    agent_y_rot = (y_center - 540) / 19.2
    state = TransformAgent((0, 0, 0), (agent_y_rot, 0, 0))
    return state


def detect_object_via_gemini(target_name):
    import dotenv
    dotenv.load_dotenv('api.env')

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    model_name = "google/gemini-2.5-pro-preview-05-06"
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

    try:
        depth_img = request_rgbd_image()
        steps = estimate_steps_from_depth(box, depth_img)
    except Exception as e:
        from depth import estimate_depth
        from env import _REQUEST_SCREENSHOT_
        from io import BytesIO
        import base64

        imagebytes = _REQUEST_SCREENSHOT_()['image']
        imageb64 = base64.b64encode(imagebytes).decode('utf-8')
        imageb64 = imageb64.encode('utf-8')
        screenshot = base64.b64decode(imageb64)
        screenshot = BytesIO(screenshot)
        _, depth_array = estimate_depth(screenshot)
        steps = estimate_steps_from_depth(box, depth_array)

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


def scan_shelf_left(n: int, target_name: str) -> dict:
    """
    Strafe left one step at a time, checking for target_name after each step.
    Stops early if the target is found.
    Returns {'found': True, 'box': bbox} on detection, {'found': False} otherwise.
    """
    for i in range(n):
        move_left(units=1)
        result = detect_object_via_moondream(target_name)
        if result is not None:
            print(f"[SCAN_LEFT] Target '{target_name}' found at step {i + 1}/{n}")
            return {'found': True, 'box': result['box']}
    print(f"[SCAN_LEFT] Target '{target_name}' not found after {n} steps.")
    return {'found': False}


def scan_shelf_right(n: int, target_name: str) -> dict:
    """
    Strafe right one step at a time, checking for target_name after each step.
    Stops early if the target is found.
    Returns {'found': True, 'box': bbox} on detection, {'found': False} otherwise.
    """
    for i in range(n):
        move_right(units=1)
        result = detect_object_via_moondream(target_name)
        if result is not None:
            print(f"[SCAN_RIGHT] Target '{target_name}' found at step {i + 1}/{n}")
            return {'found': True, 'box': result['box']}
    print(f"[SCAN_RIGHT] Target '{target_name}' not found after {n} steps.")
    return {'found': False}


def retrieve_item(target_name, cardinal_deg=None, annotate=False):
    item = detect_object_via_gemini(target_name)
    if item is None:
        print("[RETRIEVE] Target not detected in current view. Navigate closer.")
        return None

    try:
        depth_img = request_rgbd_image()
        steps = estimate_steps_from_depth(item["box"], depth_img)
    except Exception:
        from depth import estimate_depth
        from env import _REQUEST_SCREENSHOT_
        imagebytes = BytesIO(base64.b64decode(base64.b64encode(_REQUEST_SCREENSHOT_()['image'])))
        _, depth_array = estimate_depth(imagebytes)
        steps = estimate_steps_from_depth(item["box"], depth_array)
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
