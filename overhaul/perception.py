import ast
import random
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import ast
import re
from paddleocr import PaddleOCR

load_dotenv('../api.env')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = "gemini-2.5-pro-preview-05-06"
CLIENT = genai.Client(api_key=GEMINI_API_KEY)
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


ocr = PaddleOCR(use_angle_cls=True, lang='en')

def read_text(image_path='screenshots/ClientScreenshot.png'):
    result = ocr.ocr(image_path)
    return "\n".join([line[1][0] for line in result[0]]) if result else ""

def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    if len(result) == 0:
        return "", []
    try:
        final_result = "\n".join([line[1][0] for line in result[0]] if result else "")
    except Exception as e:
        final_result = ""
    return final_result, result

def find_most_similar_bbox_to_target_name(target_name, ocr_result):
    bboxes = '\n'.join([f'* {box}' for box in ocr_result])
    response = CLIENT.models.generate_content(
        model=MODEL_NAME,
        contents=[FIND_MOST_SIMILAR_OCR_BBOX_PROMPT, f"target_name={target_name}\n\n", bboxes],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.5
        )
    )
    annotated_bbox = response.text
    
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
    """
    Annotate the target in the image with a bounding box.
    """
    image = Image.open(file_path)
    draw = ImageDraw.Draw(image)
    
    # Convert coordinates to pixel values
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

    response = CLIENT.models.generate_content(
        model=MODEL_NAME,
        contents=[PERCEPTION_PROMPT, f"target_info={target_info}\n\n", image],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.5
        )
    )
    print(f"[DETECT OBJECT IN FRAME] Response: {response.text}")
    annotated_bbox = response.text
    match = re.search(EXTRACTABLE_JSON_PATTERN, annotated_bbox)
    if match:
        extracted = match.group(1)
        try:
            box_2d = ast.literal_eval(extracted)[0]
        except Exception as e:
            box_2d = ast.literal_eval(extracted)
        box_2d = box_2d['box_2d']
        
        target_object = ast.literal_eval(extracted)['label']

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
    from google import genai
    from google.genai import types
    from PIL import Image
    from io import BytesIO
    import re
    import ast
    import dotenv

    dotenv.load_dotenv('api.env')

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    model_name = "gemini-2.5-pro-preview-05-06"
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

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, im],
        config=types.GenerateContentConfig(
            temperature=0.5
        )
    )

    annotated_bbox = response.text

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


def request_rgbd_image():
    RequestScreenshot(save_image=True)
    DEPTH_API = "http://202.92.159.242:8000/estimate-depth"

    with open("screenshots/ClientScreenshot.png", "rb") as file:
        response = requests.post(DEPTH_API, files={"file": file})

    # Ensure request succeeded
    response.raise_for_status()

    # Read image from response directly
    depth_img = Image.open(BytesIO(response.content))
    depth_img.save("depth_image.png")  # Save for inspection
    return depth_img


def estimate_steps_from_depth(bbox, depth_img):
    """
    Given a PIL grayscale depth image and bbox, compute how many 0.1-unit steps to move.
    """
    pixels = depth_img.convert("L").load()
    cx = int((bbox["xmin"] + bbox["xmax"]) / 2)
    cy = int((bbox["ymin"] + bbox["ymax"]) / 2)
    intensity = pixels[cx, cy]  # 0 = far, 255 = near

    # Normalize: 1.0 = max visible range assumed, scaled by (1 - intensity/255)
    est_distance = (8.0) * (1 - intensity / 255.0)
    est_steps = round(est_distance / 0.1)  # each step = 0.1 units
    print(f"[DEPTH] center intensity={intensity}, est_dist={est_distance:.2f}, steps={est_steps}")
    return est_steps


def approach_target(target_name, annotate=False):
    # 1) Detect object and set bounding box
    item = detect_object_via_gemini(target_name)
    box = item["box"]

    # 2) grab depth image and estimate steps
    try:
        depth_img = request_rgbd_image()
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
        depth_img = estimate_depth(screenshot)
    steps = estimate_steps_from_depth(box, depth_img)

    # 3) Pan camera on object
    x_center_before = (box["xmin"] + box["xmax"]) / 2
    y_center_before = (box["ymin"] + box["ymin"]) / 2
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

    # 4) estimate steps & move once
    state = move_forward(units=steps)

    # 5) read/grab if desired
    # print("Final readout:", grab_and_read_item(text_read_fn=read_text))
    return state, box


def face_cardinal_direction(angle_deg):
    """
    Set camera to one of the cardinal angles: 0 (north/+z), 90 (east/+x), etc.
    """
    # Only affects rotation (pitch, yaw, roll)
    current_angle = TransformAgent((0, 0, 0), (0, 0, 0))['rotation'][1]
    TransformAgent((0,0,0), (0,angle_deg-current_angle, 0))
    print(f"[ROTATE] Facing {angle_deg} degrees")


def strafe_to_center(bbox, image_width=1920):
    """
    Move sideways until the object is horizontally centered.
    """
    center_x = (bbox["xmin"] + bbox["xmax"]) / 2
    offset_px = center_x - (image_width / 2)
    offset_units = 0.1 * (offset_px / 19.2)  # same scaling as before

    print(f"[STRAFE] Object offset: {offset_px} px -> {offset_units:.2f} units")

    # Move sideways: +Z is right, -Z is left
    # TransformAgent((offset_units, 0, 0), (0, 0, 0))
    # Or do it gradually
    movement_count = int(offset_units // 1)
    for _ in range(movement_count):
        move_right(units=1)


def define_cardinal_direction(current_yaw_angle):
    """ This would work better if we had pose estimation
    on the object. We're only getting the direction of the
    closest cardinal based on the current target angle.
    """
    cardinals = [0, 90, 180, 270, 360]
    resultant = [abs(c - current_yaw_angle) for c in cardinals]
    min_value = min(resultant)
    min_index = resultant.index(min_value) % 4
    return cardinals[min_index]
    

def retrieve_item(target_name, cardinal_deg=None, annotate=False):
    # Step 1: Use rough approach
    state, box = approach_target(target_name, annotate=annotate)

    # Step 2: Lock camera to face cardinal direction
    if not cardinal_deg:
        current_yaw_angle = TransformAgent((0,0,0), (0,0,0))['rotation'][1]
        cardinal_deg = define_cardinal_direction(current_yaw_angle)
    face_cardinal_direction(cardinal_deg)

    # Step 3: Take new image & re-detect object
    item = detect_object_via_gemini(target_name)
    if item:
        box = item["box"]
        if annotate:
            from annotation_tools import annotate_boxes
            annotate_boxes(item)

        # Step 4: Strafe left/right until object is centered
        strafe_to_center(box)

    # Step 5: Take fresh RGBD image & estimate new approach steps
    item = detect_object_via_gemini(target_name)
    box = item["box"]
    x_center_before = (box["xmin"] + box["xmax"]) / 2
    y_center_before = (box["ymin"] + box["ymin"]) / 2
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

    # Step 6: Move forward to final position
    # state = move_forward(units=steps)

    # Step 7: Grab and read
    print("Final readout:", grab_and_read_item(text_read_fn=read_text))
    return state
