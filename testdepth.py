from transformers import pipeline
from PIL import Image
import io

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda")

def estimate_depth(image_path):
    image = Image.open(image_path).convert("RGB")
    results = pipe(image)['depth']
    buf = io.BytesIO()
    results.save(buf, format='PNG')
    buf.seek(0)

    # Save the depth map to a file if needed
    with open('depth_map.png', 'wb') as f:
        f.write(buf.getvalue())

    # return the depth map as a PIL Image
    

estimate_depth(r'C:\Users\EMMANUEL\Desktop\capstone\pantrypal\overhaul\screenshots\SIM_RUNS\Shelf2\000001-ClientScreenshot-05-28-2025-12-35-54.png')