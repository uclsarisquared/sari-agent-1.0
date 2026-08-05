import ast
import random
import os
import requests
from io import BytesIO
from env import RequestScreenshot, TransformAgent
from openai import OpenAI
from paddleocr import PaddleOCR
from env import *
from annotation_tools import annotate_boxes

from dotenv import load_dotenv
from manipulation import grab_and_read_item
from PIL import Image


load_dotenv("config.env")
from overhaul.perception import detect_object_via_moondream

def face_cardinal_direction(angle_deg):
    """
    Set camera to one of the cardinal angles: 0 (north/+z), 90 (east/+x), etc.
    """
    # Only affects rotation (pitch, yaw, roll)
    current_angle = TransformAgent((0, 0, 0), (0, 0, 0))['rotation'][1]
    TransformAgent((0,0,0), (0, angle_deg-current_angle, 0))
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


def approach_cardinal(target_name, cardinal_deg=270, annotate=False):
    # Step 1: Use rough approach
    state, box = approach_target(target_name, annotate=annotate)

    # Step 2: Lock camera to face cardinal direction
    face_cardinal_direction(cardinal_deg)

    # Step 3: Take new image & re-detect object
    item = detect_object_via_moondream(target_name)
    if item:
        box = item["box"]
        if annotate:
            from annotation_tools import annotate_boxes
            annotate_boxes(item)

        # Step 4: Strafe left/right until object is centered
        strafe_to_center(box)

    # Step 5: Take fresh RGBD image & estimate new approach steps
    item = detect_object_via_moondream(target_name)
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



ocr = PaddleOCR(use_angle_cls=True, lang='en')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
    item = detect_object_via_moondream(target_name)
    box = item["box"]

    # 2) grab depth image and estimate steps
    depth_img = request_rgbd_image()
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



def read_text(image_path="screenshots/ClientScreenshot.png"):
    result = ocr.ocr(image_path)
    return "\n".join([line[1][0] for line in result[0]]) if len(result) else "" 


def extract_text_from_image(image_path):
    result = ocr.ocr(image_path, cls=True)
    if len(result) == 0:
        return "", []
    try:
        final_result = "\n".join([line[1][0] for line in result[0]]) if result else ""
    except Exception as e:
        final_result = ""
    return final_result, result  


def detect_object(target_name, threshold=10):
    OWL_VIT_URL = "http://202.92.159.242:8000/locate-owl-vit"
    RequestScreenshot(save_image=True)

    file_path = os.path.join("screenshots", "ClientScreenshot.png")
    print(file_path)
    with open(file_path, "rb") as file:
        image_data = {"file": file}
        prompt = target_name
        response = requests.post(OWL_VIT_URL, data={"prompt": prompt}, files=image_data)

        if response.status_code == 200:
            roi = response.json()
        else:
            print("Error in locating items.", response.text)
        
        if len(roi) == 0:
            return TransformAgent((0,0,0), (0,0,0))
        roi = sorted(roi, key=lambda d: d['score'])
        return roi[:threshold]


def center_item_on_screen(target_name, annotate=False):
    item = detect_object_via_moondream(target_name)
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
    state = TransformAgent((0,0,0), ((y_center_before - 540) / 19.2,(x_center_before - 960) / 19.2, 0))
    return state, box


def detect_object_via_gemini(target_name):
    from google import genai
    from google.genai import types
    from PIL import Image
    from io import BytesIO
    import re
    import ast
    import dotenv

    dotenv.load_dotenv('config.env')

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

def detect_object_in_frame_gemini(target_name):
    from google import genai
    from google.genai import types
    from PIL import Image
    from io import BytesIO
    import re
    import ast
    import dotenv

    dotenv.load_dotenv('config.env')

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
        box_2d = ast.literal_eval(json_str)[0]
        box_2d = box_2d['box_2d']
        print("Box 2D: ", box_2d)
        print("Type of box_2d: ", type(box_2d))
        ymin = box_2d[0] / 1000 * original_height
        xmin = box_2d[1] / 1000 * original_width
        ymax = box_2d[2] / 1000 * original_height
        xmax = box_2d[3] / 1000 * original_width

        annotate_target(ymin, xmin, ymax, xmax)

        x_center_before = (xmin + xmax) / 2
        y_center_before = (ymin + ymax) / 2
        y_cam_movement = -1 * (y_center_before - 540) / 19.2
        x_cam_movement = -1 * (x_center_before - 960) / 19.2
        print("Y Center of bbox: ", y_center_before)
        print("X Center of bbox: ", x_center_before)
        print("Movement Y: ", y_cam_movement)
        print("Movement X: ", x_cam_movement)
        state = TransformAgent((0,0,0), ((y_center_before - 540) / 19.2,(x_center_before - 960) / 19.2, 0))
        # seeking
        seek_steps = random.randint(3,5)
        state = move_forward(units=seek_steps)
        RequestScreenshot()
        image_path = "screenshots/ClientScreenshot.png"
        _, paddle_result = extract_text_from_image(image_path)
        bboxes = transform_paddle_result_to_coco_label_format(paddle_result)
        try:
            most_similar_bbox = ast.literal_eval(find_most_similar_ocr_bbox(bboxes, goal="box of cereal")['response'])
        except Exception:
            most_similar_bbox = bboxes[0]
        bbox = {
            "xmin": float(most_similar_bbox[0]),
            "ymin": float(most_similar_bbox[1]),
            "xmax": float(most_similar_bbox[2]),
            "ymax": float(most_similar_bbox[3])
        }
        center_bbox_on_screen(bbox)
        # closing
        close_steps = random.randint(5,10)
        state = move_forward(units=close_steps)
        print("Things read:")
        print(grab_and_read_item(text_read_fn=read_text))
        return state, bbox

def annotate_target(ymin, xmin, ymax, xmax, file_path="screenshots/ClientScreenshot.png"):
    from PIL import ImageDraw
    from PIL import Image
    image = Image.open(file_path)
    draw = ImageDraw.Draw(image)
    ymin = ymin / 1000 * image.size[1]
    xmin = xmin / 1000 * image.size[0]
    ymax = ymax / 1000 * image.size[1]
    xmax = xmax / 1000 * image.size[0]
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    draw.text((xmin, ymin), "Detected", fill="red")
    image.save("annotations/annotated_image.png")


def center_bbox_on_screen(bbox):
    x_center_before = (bbox["xmin"] + bbox["xmax"]) / 2
    y_center_before = (bbox["ymin"] + bbox["ymin"]) / 2
    print("Y Center of bbox: ", y_center_before)
    print("X Center of bbox: ", x_center_before)
    y_cam_movement = -1 * (y_center_before - 540) / 19.2
    x_cam_movement = -1 * (x_center_before - 960) / 19.2

    print("Movement Y: ", y_cam_movement)
    print("Movement X: ", x_cam_movement)
    return TransformAgent((0,0,0), ((y_center_before - 540) / 19.2,(x_center_before - 960) / 19.2, 0)), bbox


def transform_paddle_result_to_coco_label_format(paddle_result):
    return [(b[0][0][0],b[0][0][1], b[0][2][0], b[0][2][1], b[1][0]) for b in paddle_result[0]]


def find_most_similar_ocr_bbox(paddle_result, goal):
    bboxes = "\n".join([f"* {box}" for box in paddle_result])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are the eyes of an AI agent in a virtual grocery store envionment. You will be given "
                            "a <GOAL> and bounding boxes with text read from an OCR tool. Your goal is "
                            "to find the bounding box with text that is most semantically related with the <GOAL>. "
                            "The reason is that I will be using the selected bbox to center my agent's camera and "
                            "move forward accordingly. \n\n"
                            "For example:\n\n"
                            "<GOAL>: Box of cereal\n"
                            "<BOUNDING BOXES>:\n"
                            "* (123,45,127,50,'Corn Flakes')"
                            "* (56,123,150,131,'Coca')"
                            "* (440,172,200,445,'Cheetos')"
                            "* (700,45,707,47,'Cream delights')\n\n"
                            "Response: (123,45,127,50,'Corn Flakes')"
                        )
                    },
                ]
            }, {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"<GOAL>: {goal}\n"
                            "<BOUNDING BOXES>:\n"
                            + bboxes + "\n"
                            "Response: "
                        )
                    }
                ]
            }
        ],
        max_tokens=256
    )

    return {"response": response.choices[0].message.content}


# if __name__ == "__main__":
#     # Example usage
#     target_name = "Pillows snack"
#     detect_object_in_frame_gemini(target_name)
