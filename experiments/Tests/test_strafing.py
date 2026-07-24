from perception import detect_object_via_moondream

def detect():
    detect_object_via_moondream("box of Pepero ")    

# Pseudocode:
tol_px = 20
Δx = 0.1  # meters per strafe step

while True:
    frame = get_frame()
    success, box = tracker.update(frame)
    if not success:
        box = detect()
        tracker.init(frame, box)

    cx = box[0] + box[2] / 2
    error_px = cx - (frame.shape[1] / 2)

    if abs(error_px) <= tol_px:
        break     # centered laterally
    elif error_px < 0:
        send("_MOVE_LEFT_", distance=Δx)
    else:
        send("_MOVE_RIGHT_", distance=Δx)
