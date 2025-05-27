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


def center_bbox_on_screen(bbox):
    """
    Center the bounding box on the screen.
    """
    x_center_before = (bbox["xmin"] + bbox["xmax"]) / 2
    y_center_before = (bbox["ymin"] + bbox["ymax"]) / 2
    return TransformAgent((0, 0, 0), ((y_center_before - 540) / 19.2, (x_center_before - 960) / 19.2, 0))

def detect_object_in_frame(target_info):
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
        seek_steps = random.randint(3, 7)
        state = move_forward(units=seek_steps)
        RequestScreenshot()
        filepath = 'screenshots/ClientScreenshot.png'
        _, paddle_result = extract_text_from_image(filepath)
        bboxes = transform_paddle_result_to_coco_label_format(paddle_result)
        most_similar_bbox_response = find_most_similar_bbox_to_target_name(target_object, bboxes)
        if most_similar_bbox_response:
            bbox = {"xmin": float(most_similar_bbox_response[0]),
                    "ymin": float(most_similar_bbox_response[1]),
                    "xmax": float(most_similar_bbox_response[2]),
                    "ymax": float(most_similar_bbox_response[3])}
        else:
            bbox = {"xmin": float(bboxes[0][0]),
                    "ymin": float(bboxes[0][1]),
                    "xmax": float(bboxes[0][2]),
                    "ymax": float(bboxes[0][3])}
        print(f"Most similar bounding box to target '{target_object}': {bbox}")
        center_bbox_on_screen(bbox)
        close_steps = random.randint(3, 7)
        state = move_forward(units=close_steps)
        grab_and_read_item(text_read_fn=read_text)
        return state