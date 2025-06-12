from PIL import Image, ImageDraw

# Load the grayscale depth image
depth_img = Image.open("depth_image.png").convert("L")

# Inspect specific pixels
pixels = depth_img.load()

# Initialize min/max trackers
width, height = depth_img.size
min_val, max_val = 255, 0
min_loc, max_loc = (0, 0), (0, 0)

for y in range(height):
    for x in range(width):
        val = pixels[x, y]
        if val < min_val:
            min_val, min_loc = val, (x, y)
        if val > max_val:
            max_val, max_loc = val, (x, y)

print(f"Closest point at {max_loc} with intensity {max_val}")
print(f"Farthest point at {min_loc} with intensity {min_val}")

# Convert to RGB so we can draw in color
depth_img_rgb = depth_img.convert("RGB")
draw = ImageDraw.Draw(depth_img_rgb)

def draw_crosshair(draw, center, color, size=10):
    x, y = center
    draw.line((x - size, y, x + size, y), fill=color, width=1)
    draw.line((x, y - size, x, y + size), fill=color, width=1)

# Draw crosshairs
draw_crosshair(draw, min_loc, color="blue")   # Farthest point
draw_crosshair(draw, max_loc, color="red")    # Closest point

# Save annotated image
depth_img_rgb.save("depth_image_annotated.png")
print("Saved depth_image_annotated.png with crosshairs.")
