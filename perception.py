import os
import requests
from env import RequestScreenshot, TransformAgent


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
    item = detect_object(target_name)[0]
    box = item["box"]
    x_center_before = (box["xmin"] + box["xmax"]) / 2
    y_center_before = (box["ymin"] + box["ymin"]) / 2
    if annotate:
        from tools import annotate_boxes
        annotate_boxes(item)
    print("Y Center of bbox: ", y_center_before)
    print("X Center of bbox: ", x_center_before)
    y_cam_movement = -1 * (y_center_before - 540) / 19.2
    x_cam_movement = -1 * (x_center_before - 960) / 19.2

    print("Movement Y: ", y_cam_movement)
    print("Movement X: ", x_cam_movement)
    return TransformAgent((0,0,0), ((y_center_before - 540) / 19.2,(x_center_before - 960) / 19.2, 0))

def detect_object_in_frame_gemini(target_name):
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
        ymin = box_2d['box_2d'][0] / 1000 * original_height
        xmin = box_2d['box_2d'][1] / 1000 * original_width
        ymax = box_2d['box_2d'][2] / 1000 * original_height
        xmax = box_2d['box_2d'][3] / 1000 * original_width

        annotate_target(ymin, xmin, ymax, xmax)

        x_center_before = (xmin + xmax) / 2
        y_center_before = (ymin + ymax) / 2
        y_cam_movement = -1 * (y_center_before - 540) / 19.2
        x_cam_movement = -1 * (x_center_before - 960) / 19.2
        print("Y Center of bbox: ", y_center_before)
        print("X Center of bbox: ", x_center_before)
        print("Movement Y: ", y_cam_movement)
        print("Movement X: ", x_cam_movement)
        return TransformAgent((0,0,0), ((y_center_before - 540) / 19.2,(x_center_before - 960) / 19.2, 0))

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