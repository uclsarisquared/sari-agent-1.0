import requests
import os
from PIL import ImageDraw, Image
from env import TransformAgent, RequestScreenshot

def annotate_located_object(image: Image.Image | str, bbox: dict, color="red", radius=20):
    """
    Draws a circle at the best patch center location based on CLIP locate result.

    Args:
        image (PIL.Image): The original image.
        result (dict): The result dict from locate_object_in_frame().
        color (str): Color of the annotation. Default "red".
        radius (int): Radius of the annotation circle. Default 20.

    Returns:
        PIL.Image: Annotated image.
    """
    if type(image) == str:
        image = Image.open(image)
    width, height = image.size

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Draw a circle centered at (x, y)
    draw.rectangle(xy=(bbox), outline=color, width=4)

    # Optional: Draw a smaller crosshair at frame center too
    # frame_center = result["frame_center"]
    fx, fy = width // 2, height // 2
    draw.line((fx - 10, fy, fx + 10, fy), fill="blue", width=2)
    draw.line((fx, fy - 10, fx, fy + 10), fill="blue", width=2)

    return annotated_image


def annotate_boxes(roi, prefix="", file_path="screenshots/ClientScreenshot.png"):
    if type(roi) != list:
        roi = [roi]
    for i, item in enumerate(roi):
        box = item["box"]
        image = annotate_located_object(file_path, (box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        os.makedirs("annotations", exist_ok=True)
        image.save(f"annotations/{prefix+'-' if prefix else ''}{i}.png")
