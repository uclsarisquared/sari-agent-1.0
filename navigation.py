from env import TransformAgent, move_forward 


def align_with_cardinal_direction(direction="N"):
    state = TransformAgent((0,0,0),(0,0,0))
    yaw = state['rotation'][1]
    if direction == "N":
        dYaw = yaw * -1
    elif direction == "E":
        dYaw = 90 + yaw * -1
    elif direction == "S":
        dYaw = 180 + yaw * -1
    elif direction == "W":
        dYaw = 270 + yaw * -1
    else:
        dYaw = 0

    state = TransformAgent((0,0,0), (0,dYaw,0))
    return state


def exit_aisle(direction="N", units=5):
    state = align_with_cardinal_direction(direction)
    state = move_forward(10)
    return state
