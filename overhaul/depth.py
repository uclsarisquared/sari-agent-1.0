DEPTH_PACKAGE_AVAILABLE = True
try:
    from transformers import pipeline
except ModuleNotFoundError:
    DEPTH_PACKAGE_AVAILABLE = False

from PIL import Image
import io

if DEPTH_PACKAGE_AVAILABLE:
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda")

def estimate_depth(image_bytes):
    if DEPTH_PACKAGE_AVAILABLE:
        image = Image.open(image_bytes).convert("RGB")
        depth_est = depth_pipe(image)['depth']

        # return the depth map as a PIL Image
        buf = io.BytesIO()
        depth_est.save(buf, format='PNG')
        buf.seek(0)

        return Image.open(buf)
    else:
        from perception import request_rgbd_image
        return request_rgbd_image()
