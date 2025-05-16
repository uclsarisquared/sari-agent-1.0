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


def center_item_on_screen(item, annotate=False):
    item = detect_object(item)[0]
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
