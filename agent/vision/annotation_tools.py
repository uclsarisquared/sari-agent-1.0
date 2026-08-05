import os
from PIL import ImageDraw, Image
from sim.env import TransformAgent, RequestScreenshot, artifact_path, screenshot_dir

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
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            opened.load()
            image = opened.copy()
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


def annotate_boxes(roi, prefix="", file_path=None, source_image=None, output_dir=None):
    """Annotate boxes without requiring a shared screenshot/output directory.

    Live callers may provide the exact PIL ``source_image`` used for detection. In an orchestrated
    attempt the compatibility defaults resolve under SARI_RUN_DIR; a standalone invocation retains
    screenshots/ and annotations/ in the current directory.
    """
    file_path = file_path or os.path.join(screenshot_dir(), "ClientScreenshot.png")
    output_dir = output_dir or artifact_path("annotations", legacy_base="")
    os.makedirs(output_dir, exist_ok=True)
    if type(roi) != list:
        roi = [roi]
    for i, item in enumerate(roi):
        box = item["box"]
        image = annotate_located_object(
            source_image if source_image is not None else file_path,
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
        )
        image.save(os.path.join(output_dir, f"{prefix+'-' if prefix else ''}{i}.png"))
