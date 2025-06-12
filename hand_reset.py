import math
from env import *


def reset_hands_in_front(
    forward_dist: float = 0.4,
    hand_spacing: float = 0.1,
    extra_elevation: float = 0.0
):
    """
    Resets both hands to sit forward_dist units in front of the agent
    along the camera’s pitch & yaw, and extra_elevation above that line.
    
    Parameters
    ----------
    forward_dist : float
        Distance along the camera’s forward ray (including pitch).
    hand_spacing : float
        Lateral distance between left & right hands.
    extra_elevation : float
        Additional vertical offset (in world‐units) on top of the pitch.
    """

    # 1) snapshot current state
    hs  = TransformHands((0,0,0),(0,0,0),(0,0,0),(0,0,0))
    left_cur, right_cur = hs['leftTranslation'], hs['rightTranslation']

    as_ = TransformAgent((0,0,0),(0,0,0))
    pos, rot = as_['translation'], as_['rotation']
    
    pitch = math.radians(rot[0])
    yaw   = math.radians(rot[1])
    roll  = math.radians(rot[2])

    # 2) build a full 3D forward vector from pitch & yaw
    fx = math.cos(pitch) * math.sin(yaw)
    fy = math.sin(pitch)
    fz = math.cos(pitch) * math.cos(yaw)
    
    # 3) build a lateral‐right vector via cross(forward, up)
    lateral_right = (
        fz * 1.0 - fy * 0.0,     # = fz
        fx * 0.0 - fz * 0.0,     # = 0
        fy * 0.0 - fx * 1.0      # = -fx
    )
    # normalize lateral_right
    lr_len = math.sqrt(lateral_right[0]**2 + lateral_right[1]**2 + lateral_right[2]**2)
    lateral_right = tuple(c / lr_len for c in lateral_right)

    # 4) find midpoint in world‐space: agent_pos + forward*dist
    mid_ws = (
        pos[0] + fx * forward_dist,
        pos[1] + fy * forward_dist + extra_elevation,
        pos[2] + fz * forward_dist
    )

    half = hand_spacing / 2.0
    left_target  = (
        mid_ws[0] - lateral_right[0]*half,
        mid_ws[1] - lateral_right[1]*half,
        mid_ws[2] - lateral_right[2]*half
    )
    right_target = (
        mid_ws[0] + lateral_right[0]*half,
        mid_ws[1] + lateral_right[1]*half,
        mid_ws[2] + lateral_right[2]*half
    )

    # 5) world→local converter (inverse yaw‑only rotation)
    def euler_to_matrix(pitch: float, yaw: float, roll: float):
        """
        Build a local‑to‑world rotation matrix from Tait‑Bryan angles
        in radians: pitch around X, yaw around Y, roll around Z.
        """
        sp, cp = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(yaw),   math.cos(yaw)
        sr, cr = math.sin(roll),  math.cos(roll)

        # Rx (pitch)
        Rx = [
            [1,   0,    0   ],
            [0,  cp,  -sp   ],
            [0,  sp,   cp   ],
        ]
        # Ry (yaw)
        Ry = [
            [ cy, 0,  sy ],
            [  0, 1,   0 ],
            [-sy, 0,  cy ],
        ]
        # Rz (roll)
        Rz = [
            [ cr, -sr, 0 ],
            [ sr,  cr, 0 ],
            [  0,   0, 1 ],
        ]

        # R = Ry @ Rx @ Rz
        # First RxRz, then Ry * (RxRz)
        RxRz = [
            [
                Rx[i][0]*Rz[0][j] + Rx[i][1]*Rz[1][j] + Rx[i][2]*Rz[2][j]
                for j in range(3)
            ]
            for i in range(3)
        ]
        R = [
            [
                Ry[i][0]*RxRz[0][j] + Ry[i][1]*RxRz[1][j] + Ry[i][2]*RxRz[2][j]
                for j in range(3)
            ]
            for i in range(3)
        ]
        return R

    def world_to_local(delta_ws, pitch, yaw, roll):
        """
        Convert a world‐space vector into the agent’s local frame
        using the full pitch/yaw/roll.
        """
        # Build rotation matrix local→world
        R = euler_to_matrix(pitch, yaw, roll)
        # world→local is R^T
        dx, dy, dz = delta_ws
        # Multiply R^T * [dx,dy,dz]
        local_x = R[0][0]*dx + R[1][0]*dy + R[2][0]*dz
        local_y = R[0][1]*dx + R[1][1]*dy + R[2][1]*dz
        local_z = R[0][2]*dx + R[1][2]*dy + R[2][2]*dz
        return (local_x, local_y, local_z)

    # 6) compute world-space deltas from current hand positions
    delta_left_ws  = tuple(left_target[i]  - left_cur[i]  for i in range(3))
    delta_right_ws = tuple(right_target[i] - right_cur[i] for i in range(3))


    # 7) convert to agent‐local frame
    delta_left_loc  = world_to_local(delta_left_ws,  pitch, yaw, roll)
    delta_right_loc = world_to_local(delta_right_ws, pitch, yaw, roll)

    # 8) one atomic transform call
    print("Left initial:", left_cur)
    print("Left final:", delta_left_loc)
    print("Right initial:", right_cur)
    print("Right final:", delta_right_loc)
    print("Yaw (in degrees): ", rot[1])
    TransformHands(delta_left_loc,  (0,0,0), delta_right_loc, (0,0,0))


def reset_hands_in_front2(
    forward_dist: float = 0.4,
    extra_elevation: float = 0.0,
    hand: str = "both"  # options: "left", "right", "both"
):
    """
    Resets one or both hands to sit forward_dist units in front of the agent
    along the camera’s pitch & yaw, and extra_elevation above that line.

    Parameters
    ----------
    forward_dist : float
        Distance along the camera’s forward ray (including pitch).
    extra_elevation : float
        Additional vertical offset (in world‐units) on top of the pitch.
    hand : str
        Which hand(s) to reset: "left", "right", or "both"
    """
    assert hand in {"left", "right", "both"}, "hand must be 'left', 'right', or 'both'"

    hs = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    left_cur, right_cur = hs['leftTranslation'], hs['rightTranslation']

    as_ = TransformAgent((0, 0, 0), (0, 0, 0))
    pos, rot = as_['translation'], as_['rotation']

    pitch = math.radians(rot[0])
    yaw = math.radians(rot[1])
    roll = math.radians(rot[2])

    # Forward vector
    fx = math.cos(pitch) * math.sin(yaw)
    fy = math.sin(pitch)
    fz = math.cos(pitch) * math.cos(yaw)

    # Midpoint = position in front of the agent
    mid_ws = (
        pos[0] + fx * forward_dist,
        pos[1] + fy * forward_dist + extra_elevation,
        pos[2] + fz * forward_dist
    )

    # Rotation helpers
    def euler_to_matrix(pitch: float, yaw: float, roll: float):
        sp, cp = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(yaw), math.cos(yaw)
        sr, cr = math.sin(roll), math.cos(roll)

        Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
        Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
        Rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]

        RxRz = [
            [Rx[i][0] * Rz[0][j] + Rx[i][1] * Rz[1][j] + Rx[i][2] * Rz[2][j] for j in range(3)]
            for i in range(3)
        ]
        return [
            [Ry[i][0] * RxRz[0][j] + Ry[i][1] * RxRz[1][j] + Ry[i][2] * RxRz[2][j] for j in range(3)]
            for i in range(3)
        ]

    def world_to_local(delta_ws, pitch, yaw, roll):
        R = euler_to_matrix(pitch, yaw, roll)
        dx, dy, dz = delta_ws
        local_x = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz
        local_y = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz
        local_z = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz
        return (local_x, local_y, local_z)

    # Calculate deltas and apply transforms
    delta_left_loc = (0, 0, 0)
    delta_right_loc = (0, 0, 0)

    if hand in {"left", "both"}:
        delta_left_ws = tuple(mid_ws[i] - left_cur[i] for i in range(3))
        delta_left_loc = world_to_local(delta_left_ws, pitch, yaw, roll)

    if hand in {"right", "both"}:
        delta_right_ws = tuple(mid_ws[i] - right_cur[i] for i in range(3))
        delta_right_loc = world_to_local(delta_right_ws, pitch, yaw, roll)

    # Apply the transform
    print("Yaw (deg):", rot[1])
    print("Left move:", delta_left_loc)
    print("Right move:", delta_right_loc)
    TransformHands(delta_left_loc, (0, 0, 0), delta_right_loc, (0, 0, 0))