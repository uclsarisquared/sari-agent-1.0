import moondream as md
from PIL import Image
import os
model = md.vl(api_key=os.environ.get("MDREAM_API_KEY"))
X_MIN = 0.4810
X_MAX = 0.5190
Y_MIN = 0.6533
Y_MAX = 0.6933
def point_at_centered_object(image_path: str) -> str:
    image = Image.open(image_path)
    result = model.point(image, "Item closest to the center of the screen for pickup")
    points = result["points"]
    for point in points:
        x, y = point["x"], point["y"]
        if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX:
            return (x, y)
    return (None, None)