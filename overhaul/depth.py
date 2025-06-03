from transformers import pipeline
from PIL import Image
import io

depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda")

def estimate_depth(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    depth_est = depth_pipe(image)['depth']

    # return the depth map as a PIL Image
    buf = io.BytesIO()
    depth_est.save(buf, format='PNG')
    buf.seek(0)

    return Image.open(buf)