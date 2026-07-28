import requests
from io import BytesIO
from env import RequestScreenshot, TransformAgent, move_forward
from PIL import Image
from overhaul.perception import detect_object_via_moondream as detect_object_via_gemini
from dotenv import load_dotenv
from annotation_tools import annotate_boxes
from manipulation import grab_and_read_item
from Tests.perception import read_text

load_dotenv("config.env")


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
    est_distance = (1.0) * (1 - intensity / 255.0)
    est_steps = round(est_distance / 0.1)  # each step = 0.1 units
    print(f"[DEPTH] center intensity={intensity}, est_dist={est_distance:.2f}, steps={est_steps}")
    return est_steps

def approach_target(target_name, annotate=False):
    # 1) detect + center
    item = detect_object_via_gemini(target_name)
    box = item["box"]
    if annotate:
        annotate_boxes(item)

    # compute camera tilt to center (unchanged)
    x_ctr = (box["xmin"] + box["xmax"]) / 2
    y_ctr = (box["ymin"] + box["ymax"]) / 2
    TransformAgent(
        (0, 0, 0),
        (-(y_ctr - 540) / 19.2, -(x_ctr - 960) / 19.2, 0)
    )

    # 2) grab depth image
    depth_img = request_rgbd_image()

    # 3) estimate steps & move once
    steps = estimate_steps_from_depth(box, depth_img)
    state = move_forward(units=steps)

    # 4) read/grab if desired
    print("Final readout:", grab_and_read_item(text_read_fn=read_text))
    return state, box
