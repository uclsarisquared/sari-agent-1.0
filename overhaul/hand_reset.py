import math
import numpy as np

from env import *

def euler_to_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Build a local-to-world rotation matrix from Tait-Bryan angles
    in radians: pitch around X, yaw around Y, roll around Z.
    """
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    sr, cr = math.sin(roll), math.cos(roll)

    # Rx (pitch); Ry (yaw); Rz (roll)
    Rx = np.array([[1, 0, 0],
                   [0, cp, -sp],
                   [0, sp, cp]])
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    Rz = np.array([[cr, -sr, 0],
                   [sr, cr, 0],
                   [0, 0, 1]])

    R = Rx @ Rz @ Ry

    return R


def world_to_local(delta_ws: tuple, pitch: float, yaw: float, roll: float) -> tuple[float]:
    """
    Convert a world-space vector into the agent's local frame
    using the full pitch/yaw/roll.
    """
    # build a rotation matrix (R): local -> world
    R = euler_to_matrix(pitch, yaw, roll)
    # transpose of rotation matrix (R^T): world -> local
    delta_ws = np.array(delta_ws).reshape(3, 1)
    
    # multiply R^T * [dx, dy, dz]
    local_coords = np.matmul(R.T, delta_ws).reshape(-1).tolist()
    return (local_coords[0], local_coords[1], local_coords[2])


def reset_hands(
    forward_dist: float = 0.4,
    hand_spacing: float = 0.1,
    extra_elevation: float = -0.1
) -> None:
    """
    Resets both hands to sit forward_dist units in front of the agent
    along the camera's pitch & yaw, and extra_elevation above that line.

    Args:
        forward_dist (float): Distance along the camera's forward ray (including pitch).
        hand_spacing (float): Horizontal distance between the hands.
        extra_elevation (float): Additional height above the forward line.
    """

    # not moving, just getting state
    init_hands_state = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    left_hand_state, right_hand_state = init_hands_state['leftTranslation'], init_hands_state['rightTranslation']
    init_agent_state = TransformAgent((0, 0, 0), (0, 0, 0))
    agent_pos, agent_rot = init_agent_state['translation'], init_agent_state['rotation']

    # get pitch, yaw, roll
    pitch = math.radians(agent_rot[0])
    yaw = math.radians(agent_rot[1])
    roll = math.radians(agent_rot[2])

    # build a full 3D forward vector
    fx = math.cos(pitch) * math.sin(yaw)
    fy = math.sin(pitch)
    fz = math.cos(pitch) * math.cos(yaw)

    # building a lateral right vector
    lateral_right = (fz, 0, -fx)

    # normalize lateral right
    lr_len = math.sqrt(lateral_right[0]**2 + lateral_right[2]**2)
    lateral_right = tuple(c / lr_len for c in lateral_right)

    # find midpoint in world-space: agent_pos + forward * dist
    mid_ws = (agent_pos[0] + fx * forward_dist,
              agent_pos[1] + fy * forward_dist + extra_elevation,
              agent_pos[2] + fz * forward_dist)

    half = hand_spacing / 2.0
    left_target = (mid_ws[0] - lateral_right[0] * half,
                   mid_ws[1] - lateral_right[1] * half,
                   mid_ws[2] - lateral_right[2] * half)
    right_target = (mid_ws[0] + lateral_right[0] * half,
                    mid_ws[1] + lateral_right[1] * half,
                    mid_ws[2] + lateral_right[2] * half)

    delta_left_ws = tuple(left_target[i] - left_hand_state[i] for i in range(3))
    delta_right_ws = tuple(right_target[i] - right_hand_state[i] for i in range(3))

    delta_left_loc = world_to_local(delta_left_ws, pitch, yaw, roll)
    delta_right_loc = world_to_local(delta_right_ws, pitch, yaw, roll)

    # position hands in front
    TransformHands(delta_left_loc, (0, 0, 0), delta_right_loc, (0, 0, 0))
