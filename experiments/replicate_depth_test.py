import numpy as np
import sys
def get_distance_at_pixel(x_percent, y_percent, array_path="depth_array.npy"):
    """Return metric depth (meters) at pixel (x, y). Origin is top-left."""
    arr = np.load(array_path)
    height, width = arr.shape
    x = int(x_percent * width)
    y = int(y_percent * height)
    return float(arr[y, x])

if __name__ == "__main__":
    #get x and y at 0.5,0.74 of the image (percentage) based on the size of the image (refer to array)
    if len(sys.argv) != 3:
        print("Usage: python get_distance.py <x_percent> <y_percent>")
        sys.exit(1)
    x_percent = float(sys.argv[1])
    y_percent = float(sys.argv[2])
    distance = get_distance_at_pixel(x_percent, y_percent)
    print(f"Distance at pixel ({x_percent}, {y_percent}): {distance:.3f} meters")
    hand_extend = 0.025
    print("You have to extend your hand " + str(distance/hand_extend) + " times to reach the object")