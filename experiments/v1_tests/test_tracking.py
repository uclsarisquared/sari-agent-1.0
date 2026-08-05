import math


def compute_target_pose(obj_pos, shelf_orientation_deg, offset_distance=1.0):
    """
    Given object position and shelf orientation (0° = +Z), return a pose (x, z, θ)
    in front of the object that the agent can approach.
    """
    direction_map = {
        0:  (0, -1),    # Shelf facing +Z, approach from -Z
        90: (-1, 0),    # Shelf facing +X, approach from -X
        180: (0, 1),  # Shelf facing -Z, approach from +Z
        270: (1, 0),  # Shelf facing -X, approach from +X
    }

    dx, dz = direction_map[shelf_orientation_deg]
    # Offset "backward" to stand in front of shelf
    tx = obj_pos[0] - dx * offset_distance
    tz = obj_pos[1] - dz * offset_distance
    target_orientation = (shelf_orientation_deg + 180) % 360 # Face the shelf

    return (tx, tz, target_orientation)


def angle_diff(target, current):
    diff = (target - current + 180) % 360 - 180
    return diff

def plan_movement(current_pos, current_ori, target_pos, target_ori):
    steps = []
    # Step 1: Rotate toward target point
    dx = target_pos[0] - current_pos[0]
    dz = target_pos[1] - current_pos[1]
    angle_to_target = math.degrees(math.atan2(dx, dz)) % 360
    print(current_ori, angle_to_target)

    diff = angle_diff(angle_to_target, current_ori)
    while abs(diff) >= 10:
        if diff > 0:
            steps.append("_ROTATE_RIGHT_")
            current_ori = (current_ori + 10) % 360
        else:
            steps.append("_ROTATE_LEFT_")
            current_ori = (current_ori - 10) % 360
        diff = angle_diff(angle_to_target, current_ori)

    # Step 2: Move forward in a straight line
    dist = math.hypot(dx, dz)
    print("Dist", dist)
    print("current orientation", current_ori)
    for _ in range(int(dist / 0.5)):  # assume step size = 0.5 units
        steps.append("_MOVE_FWD_")
        print("sin", -0.5 * math.sin(math.radians(current_ori)))
        current_pos = (
            current_pos[0] + 0.5 * math.cos(math.radians(current_ori)),
            current_pos[1] - 0.5 * math.sin(math.radians(current_ori))
        )

    # Step 3: Final rotation to face shelf
    diff =  (target_ori, current_ori)
    while abs(diff) >= 10:
        if diff > 0:
            steps.append("_ROTATE_RIGHT_")
            current_ori = (current_ori + 10) % 360
        else:
            steps.append("_ROTATE_LEFT_")
            current_ori = (current_ori - 10) % 360
        diff = angle_diff(target_ori, current_ori)

    print("Final pos", current_pos)

    return steps


current_pos = (6.91, 1.79)
current_ori = 180  # Facing +Z
obj_pos = (4, 3.0)
shelf_ori = 90  # Shelf faces +X

target_pose = compute_target_pose(obj_pos, shelf_ori)
print("target pose", target_pose)
steps = plan_movement(current_pos, current_ori, (target_pose[0], target_pose[1]), target_pose[2])

for step in steps:
    print(step)
