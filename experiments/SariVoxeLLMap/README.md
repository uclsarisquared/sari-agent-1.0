# SariVoxeLLMap

> **Archived prototype.** This directory is a partial snapshot of the original experiment and is
> not part of the live agent. The 2026-07-24 repository refactor did not retain its `locator/`,
> `LoGeR/`, or `test/` directories, so `server.py` does not run as checked in. The design and usage
> notes below are preserved as historical context; see [`../README.md`](../README.md) for the
> current status and replacement.

Real-time 3D heightmap server for robotic/agent navigation. Receives RGB camera frames from a Unity client, estimates depth using [Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-V3), builds a live point cloud, and voxelizes it into a 2D heightmap that is streamed back to Unity over WebSocket.

---

## Overview

```
Unity Client  ──►  JSON messages over WebSocket
                        │
          server.py (aiohttp + WebSocket)
                        │
              ┌─────────▼──────────┐
              │   DynamicPointcloud │  pointcloud/
              │   (DA3 depth est.)  │
              └─────────┬──────────┘
                        │  3D points + colors
              ┌─────────▼──────────┐
              │    Voxelizator      │  voxelization/
              │  preprocess →       │
              │  remove outliers →  │
              │  remove floor/ceil →│
              │  histogram2d →      │
              │  height estimates → │
              │  density mask →     │
              │  height mask →      │
              │  filtered heightmap │
              └─────────┬──────────┘
                        │  heightmap event
              ◄──────── Unity Client
```

---

## Project Structure

```
SariVoxeLLMap/
├── server.py              # Top-level async server (main entry point)
├── locator/
│   ├── __init__.py        # Exports pixel_to_world and batch_pixel_to_world
│   └── pixel_to_world.py  # Utilities: back-project pixels → 3D world coordinates
│
├── pointcloud/            # Point cloud building
│   ├── core.py            # DynamicPointcloud — incremental DA3-based point cloud
│   ├── da3_wrapper.py     # Thin wrapper around the Depth-Anything-3 API
│   ├── inputs.py          # PointcloudInputs — stores extrinsics, intrinsics, image paths
│   └── outputs.py         # PointcloudOutputs — stores depth maps and confidence maps
│
├── voxelization/          # Heightmap pipeline
│   └── core.py            # All voxelization functions + Voxelizator data class
│
├── save/                  # File-saving utilities
│   ├── save.py            # save_scatter, save_2d_grid, save_bar3d, save_npz, save_scene_glb
│   └── queue.py           # SaveTask — async worker pool for non-blocking saves
│
├── LoGeR/                 # Long-Context Geometric Reconstruction submodule (see its README)
├── runs/                  # Output directory for per-run saves
└── test/                  # Test scripts
```

---

## Dependencies

- **Python ≥ 3.11**
- [Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-V3) — depth estimation backbone
- `aiohttp` — async WebSocket server
- `numpy`, `scipy` — point cloud math
- `matplotlib` — plot saving (scatter, 2D grid, 3D bar)
- `trimesh` — GLB export
- `torch` — DA3 inference

Install dependencies:

```bash
pip install aiohttp numpy scipy matplotlib trimesh torch
```

> Also ensure Depth-Anything-3 is installed and the path in `pointcloud/core.py` and `save/save.py` is updated to match your local clone.

---

## Running the Server

```bash
cd SariVoxeLLMap
python server.py
```

Starts the server at `http://localhost:5000`.

---

## WebSocket Protocol

Connect to `ws://localhost:5000/ws` and send JSON objects with a `type` field.

### Client → Server

| Type | Payload | Description |
|---|---|---|
| `run_id` | `{ "type": "run_id", "run_id": str }` | Initializes a new run session. Must be sent before any frames. |
| `frame_data` | `{ "type": "frame_data", "data": { "image_path": str, "extrinsics": [[...]], "intrinsics": [[...]] } }` | Sends a single camera frame for processing. |
| `ping_from_client` | `{ "type": "ping_from_client" }` | Latency check; server responds with `pong_from_server`. |

### Server → Client

| Type | Payload | Description |
|---|---|---|
| `heightmap` | `{ "type": "heightmap", "data": [[x, height, -y], ...], "min_z": float, "bin_width": float }` | Updated heightmap after each frame. Each entry is a Unity world-space `(x, height, -y)` point at a non-zero bin. |
| `pong_from_server` | `{ "type": "pong_from_server" }` | Response to `ping_from_client`. |

---

## Heightmap Payload

```json
{
  "data": [
    [1.2, 0.85, -0.5],
    [1.4, 0.92, -0.5],
    ...
  ],
  "min_z": 0.312,
  "bin_width": 0.75
}
```

- **`data`** — list of `[x, height, -y]` triples in Unity world space for every occupied bin in the filtered heightmap
- **`min_z`** — minimum z value of the point cloud (floor reference)
- **`bin_width`** — spatial size of each voxel bin in world units

---

## Voxelization Pipeline

Each frame goes through the following steps inside `_process_frame_queue`:

1. **Depth estimation** — DA3 infers depth + confidence for the new frame; point cloud is updated incrementally (with sliding window overlap).
2. **Preprocess** — coordinate correction (swap axes for Unity ↔ world convention), height-sort, non-negative shift.
3. **Remove outliers** — statistical outlier removal.
4. **Remove floor/ceiling** — strips the lowest and highest percentile of points.
5. **Histogram2D** — bins points into a 2D grid with a fixed `bin_width`.
6. **Height estimates** — for each occupied bin, computes a representative height as the z-percentile of points in that bin (vectorized via `np.digitize` + sort + split).
7. **Density mask** — filters bins with too few points.
8. **Height mask** — filters bins whose height is below `height_threshold × max(heights)`.
9. **Filtered heightmap** — applies `density_mask & height_mask` to the height estimates.

Steps 6 and 7 run concurrently via `asyncio.gather`. Step 8 is sequential (depends on step 6's output).

---

## Save Flags

Controlled in `Server.__init__`:

| Flag | Default | Saves |
|---|---|---|
| `save_glb` | `False` | Per-frame `.glb` 3D scene export |
| `save_preprocessed` | `False` | Preprocessed point cloud scatter plot |
| `save_no_outlier` | `False` | Outlier-removed scatter plot |
| `save_no_floor_ceil` | `True` | Floor/ceiling-removed scatter plot |
| `save_histogram` | `True` | 3D bar plot of height estimates per frame |
| `save_heightmap` | `True` | 3D bar plot of filtered heightmap per frame |

All per-frame saves are deferred and flushed as a batch on client disconnect. A final `.npz` with all arrays is always saved at disconnect.

Final `data_final.npz` includes:
- `input_intrinsics` (camera intrinsics received from client)
- `output_intrinsics` (intrinsics used by DA3 outputs)
- `intrinsics` (legacy compatibility key; same as `output_intrinsics`)
- `extrinsics`, `depths`, `confs`, `processed_images`
- `points`, `colors`, `preprocessed_points`, `no_outlier_points`, `no_floor_ceil_points`
- `histogram2d`, `xedges`, `yedges`, `height_estimates`, `heightmap`

Saved files follow the naming pattern `{frame_name}_{order}_{plot_type}.{ext}` inside `runs/{run_id}/`.

---

## Utility: pixel_to_world

`locator/pixel_to_world.py` provides a helper to back-project a pixel coordinate from a recorded frame into 3D world space.

Single pixel (`None` if invalid depth/confidence):

```python
from locator import pixel_to_world
import numpy as np

data = np.load("runs/my_run/data_final.npz")
world_pt = pixel_to_world(
    depth=data['depths'],
    intrinsics=data['output_intrinsics'] if 'output_intrinsics' in data.files else data['intrinsics'],
    extrinsics=data['extrinsics'],
    frame_idx=0,
    u=320, v=240,          # pixel column, row
    conf=data['confs'],
    conf_thr=0.5,
)
# world_pt is a (3,) float32 array [X, Y, Z] in world space, or None if invalid
```

### Locator Test Script

Use `locator/test.py` to validate pixel-to-world conversion directly from a saved run (`data_final.npz`).

What it does:
- Loads the latest run automatically (or a provided run directory)
- Converts one or more `(u, v)` pixels to world coordinates using `pixel_to_world`
- Prints per-point one-line summaries: pixel, depth, confidence, world coordinate
- Optionally displays `processed_images[frame_idx]` with annotated points
- Prints timing breakdown: load, conversion, display, total

Run from project root:

```bash
python locator/test.py --frame-idx 0 --u 320 --v 240
```

Multiple points:

```bash
python locator/test.py --frame-idx 0 --points "320,240;200,180;100,120"
```

Use an explicit run directory and confidence threshold:

```bash
python locator/test.py --run-dir runs/my_run --frame-idx 0 --points "320,240;200,180" --conf-thr 0.5
```

Notes:
- If `--points` is omitted, the script uses `--u` and `--v`.
- If `processed_images` is missing in the NPZ, conversion still runs and display is skipped.
- OpenCV (`cv2`) is required for annotated image display in the current script.

---

## Submodules

- **[Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-V3)** — Monocular depth estimation model (external clone; path must be set locally).
