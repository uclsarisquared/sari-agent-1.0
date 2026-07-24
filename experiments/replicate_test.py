import replicate
import json
import base64
import numpy as np

from get_view import get_current_view

image_bytes = get_current_view()
image_data = base64.b64encode(image_bytes).decode("utf-8")
data_uri = f"data:image/png;base64,{image_data}"

output = replicate.run(
    "vufinder/depth-anything-v3-metric:d2ef7653aa75f87e3b502738a3f57ca2b09ba92f6f286075c8929a69937a013f",
    input={
        "images": [data_uri],
        "to_base64": True,
        "num_patches": 36,
        "return_depth": True,
        "output_format": "json",
        "keys_to_exclude": "",
        "alpha_blend_onto": "white",
        "processing_resolution": "match_input"
    }
)

# Save colorized depth image
depth_image = output['depth_images'][0]
with open("depth_map.png", "wb") as f:
    f.write(depth_image.read())
print(f"Depth image saved to depth_map.png (URL: {depth_image.url})")

# Parse raw depth data
raw = json.loads(output['data'][0].read())
print("Data keys:", raw[0].keys() if isinstance(raw, list) else raw.keys())

# Decode depth array
entry = raw[0] if isinstance(raw, list) else raw
depth = entry['depth']
print("depth field type:", type(depth), "| value preview:", str(depth)[:200])

if isinstance(depth, str):
    depth_bytes = base64.b64decode(depth)
    depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
elif isinstance(depth, dict):
    # depth is likely {'data': '<base64>', 'shape': [...], 'dtype': '...'}
    print("depth dict keys:", depth.keys())
    depth_bytes = base64.b64decode(depth['data'])
    dtype = np.dtype(depth.get('dtype', 'float32'))
    depth_array = np.frombuffer(depth_bytes, dtype=dtype)
    if 'shape' in depth:
        depth_array = depth_array.reshape(depth['shape'])
else:
    depth_array = np.array(depth, dtype=np.float32)

print(f"Depth array shape: {depth_array.shape}, min: {depth_array.min():.3f}m, max: {depth_array.max():.3f}m")
#save depth array
np.save("depth_array.npy", depth_array)
print("Depth array saved to depth_array.npy")
