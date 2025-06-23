import requests
from PIL import Image
from io import BytesIO

DEPTH_API = "http://202.92.159.242:8000/estimate-depth"

with open("screenshots/ClientScreenshot.png", "rb") as file:
    response = requests.post(DEPTH_API, files={"file": file})

# Ensure request succeeded
response.raise_for_status()

# Read image from response directly
depth_img = Image.open(BytesIO(response.content))
depth_img.save("depth_image.png")  # Save for inspection
