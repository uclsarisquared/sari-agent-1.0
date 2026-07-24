"""
DA3-BASE depth estimation test using the current Sari sandbox view.

Captures 3 frames (minimum for DA3 multi-frame inference) by moving
forward slightly between shots, then runs DA3-BASE and reports results.

Usage:
    python tool_test.py

Requires:
    - Sari sandbox (Unity) running at ws://localhost:8080/commands
    - DA3_PATH env var set (or in api.env) pointing to Depth-Anything-3/src
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

load_dotenv("api.env")
da3_path = os.getenv("DA3_PATH")
if da3_path:
    sys.path.insert(0, da3_path)

import env
from SariVoxeLLMap.pointcloud.da3_wrapper import DA3Wrapper

# Unity sandbox camera: 1920x1080, vertical FOV = 60° (FOVAxisMode: 0)
# Source: Assets/Resources/Prefabs/XR Origin (XR Rig).prefab + all scene files
_IMG_W, _IMG_H = 1920, 1080
_FOV_V_DEG = 60.0
_FY = (_IMG_H / 2) / np.tan(np.radians(_FOV_V_DEG / 2))  # ≈ 935.3
_FX = _FY  # square pixels
_CX, _CY = _IMG_W / 2, _IMG_H / 2


def make_intrinsics() -> np.ndarray:
    return np.array(
        [[_FX, 0.0, _CX],
         [0.0, _FY, _CY],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def state_to_extrinsics(state: dict) -> np.ndarray:
    """Unity agent state → 4×4 world-to-camera extrinsics matrix."""
    t = state["translation"]   # [x, y, z]
    r = state["rotation"]      # [pitch, yaw, roll] degrees, Unity YXZ order

    pitch = np.radians(r[0])
    yaw   = np.radians(r[1])
    roll  = np.radians(r[2])

    Rx = np.array([
        [1, 0, 0],
        [0,  np.cos(pitch), -np.sin(pitch)],
        [0,  np.sin(pitch),  np.cos(pitch)],
    ])
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [0,            1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0,             0,            1],
    ])

    R = Ry @ Rx @ Rz  # Unity YXZ Euler order

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3]  = t

    return np.linalg.inv(c2w)  # world-to-camera


def capture_frames(n: int = 3, step_units: int = 1) -> list[dict]:
    """Capture n screenshots from the sandbox with small forward steps between them."""
    save_dir = "test_frames"
    os.makedirs(save_dir, exist_ok=True)
    frames = []

    for i in range(n):
        ts = datetime.now().strftime("%H%M%S%f")
        env.RequestScreenshot(
            prefix=f"frame{i}",
            suffix=ts,
            folder_name=save_dir,
            save_image=True,
        )
        img_path = os.path.join(save_dir, f"frame{i}-ClientScreenshot-{ts}.png")

        raw_state = env.RequestJson()
        state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state

        frames.append({"img_path": img_path, "state": state})
        print(f"  Frame {i + 1}/{n} captured → {img_path}")
        print(f"    position: {state.get('translation')}, rotation: {state.get('rotation')}")

        if i < n - 1:
            env.move_forward(step_units)

    return frames


def run_da3_test() -> object:
    print("=== DA3-BASE Test: Sari Sandbox ===\n")

    print("[1] Capturing frames from sandbox...")
    frames = capture_frames(n=3, step_units=1)

    image_paths = [f["img_path"] for f in frames]
    K = make_intrinsics()
    intrinsics = np.stack([K] * len(frames))                           # (N, 3, 3)
    extrinsics = np.stack([state_to_extrinsics(f["state"]) for f in frames])  # (N, 4, 4)

    print(f"\n[2] Running DA3-BASE inference on {len(frames)} frames...")
    da3 = DA3Wrapper(model_id="DA3-BASE", process_res=512)
    prediction = da3(image_paths, extrinsics, intrinsics)

    depth = np.array(prediction.depth)
    conf  = np.array(prediction.conf)

    print("\n[3] Results:")
    print(f"  depth shape : {depth.shape}")
    print(f"  conf  shape : {conf.shape}")
    print(f"  depth range : {depth.min():.4f} – {depth.max():.4f}")
    print(f"  conf  range : {conf.min():.4f} – {conf.max():.4f}")

    _save_depth_vis(depth, image_paths)
    return prediction


def _save_depth_vis(depth: np.ndarray, image_paths: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = depth.shape[0] if depth.ndim == 3 else 1
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        if n == 1:
            axes = [axes]

        for idx, ax in enumerate(axes):
            frame = depth[idx] if depth.ndim == 3 else depth
            im = ax.imshow(frame, cmap="plasma")
            ax.set_title(f"Frame {idx} Depth")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        out_path = os.path.join("test_frames", "da3_depth_result.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"\n  Depth map saved → {out_path}")
    except ImportError:
        print("\n  (matplotlib not available — skipping depth visualization)")


if __name__ == "__main__":
    run_da3_test()
