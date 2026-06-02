import base64
import io
import json
import time
import replicate
from replicate.exceptions import ReplicateError
import httpx
import numpy as np
from PIL import Image

_REPLICATE_MODEL = "vufinder/depth-anything-v3-metric:d2ef7653aa75f87e3b502738a3f57ca2b09ba92f6f286075c8929a69937a013f"
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 10  # seconds; 429 resets in ~6s so 10s is safe
# Replicate inline base64 data URI limit; above this we upload via files API
_MAX_INLINE_BYTES = 4 * 1024 * 1024  # 4 MB


def _upload_image(buf: io.BytesIO) -> str:
    buf.seek(0)
    file_obj = replicate.files.create(buf, filename="depth_input.png")
    return file_obj.urls["get"]


def _run_replicate(image_url: str):
    for attempt in range(_MAX_RETRIES):
        print(f"[DEPTH] Attempting to run Replicate model (attempt {attempt + 1}/{_MAX_RETRIES})...")
        try:
            return replicate.run(
                _REPLICATE_MODEL,
                input={
                    "images": [image_url],
                    "to_base64": True,
                    "num_patches": 24,
                    "return_depth": True,
                    "output_format": "json",
                    "keys_to_exclude": "",
                    "alpha_blend_onto": "white",
                    "processing_resolution": "match_input",
                },
            )
        except ReplicateError as e:
            if e.status == 429 and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (attempt + 1)
                print(f"[DEPTH] Rate limited (429). Retrying in {delay}s... (attempt {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise
        except httpx.HTTPError as e:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (attempt + 1)
                print(f"[DEPTH] Network error: {e}. Retrying in {delay}s... (attempt {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise
    return None  # unreachable


def estimate_depth(image_bytes: io.BytesIO) -> tuple[Image.Image, np.ndarray]:
    image_bytes.seek(0)
    img = Image.open(image_bytes).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    size = buf.getbuffer().nbytes
    print(f"Estimating depth from image of size: {size} bytes")

    if size > _MAX_INLINE_BYTES:
        print("[DEPTH] Image exceeds inline limit, uploading via Replicate files API...")
        image_url = _upload_image(buf)
    else:
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode("utf-8")
        image_url = f"data:image/png;base64,{image_data}"

    output = _run_replicate(image_url)

    if output is None:
        raise RuntimeError("Replicate depth model returned None — check REPLICATE_API_TOKEN and model availability")

    depth_file = output['depth_images'][0]
    depth_data = depth_file.read()
    with open("depth_map.png", "wb") as f:
        f.write(depth_data)
    depth_image = Image.open(io.BytesIO(depth_data)).convert("RGB")

    raw = json.loads(output['data'][0].read())
    entry = raw[0] if isinstance(raw, list) else raw
    depth = entry['depth']
    if isinstance(depth, str):
        depth_array = np.frombuffer(base64.b64decode(depth), dtype=np.float32)
    elif isinstance(depth, dict):
        depth_array = np.frombuffer(base64.b64decode(depth['data']), dtype=np.dtype(depth.get('dtype', 'float32')))
        if 'shape' in depth:
            depth_array = depth_array.reshape(depth['shape'])
    else:
        depth_array = np.array(depth, dtype=np.float32)

    return depth_image, depth_array