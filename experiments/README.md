# experiments/ — dormant explorations (May 2026)

Untracked experiment cluster moved here from the repo root 2026-07-24. Nothing in the live
`overhaul/` stack imports any of this.

## SariVoxeLLMap / VMap (monocular-depth heightmap)

- `VMap_Plan.md` — the integration plan: build a live 2D heightmap from RGB frames +
  Depth-Anything-3 and feed it to the agent for path planning.
- `SariVoxeLLMap/` — the heightmap package (voxelization, pointcloud, server).
- `Tests/` — v1-era test scripts and module copies from the same exploration.
- `tool_test.py` — DA3-BASE multi-frame depth test against the live sim.

Status: never wired into the agent. The problem it targeted (persistent metric spatial memory,
obstacle map, path planning) was solved in `overhaul/slamtest/` with **real LiDAR** instead of
monocular depth — the frozen occupancy grid + checkpoint graph the current agent navigates on.
If resumed, note VMap_Plan.md's file references predate the 2026-07-24 reorg.

## Monocular depth probes

- `get_view.py` — capture the current sim view over WebSocket (standalone).
- `replicate_test.py`, `replicate_depth_test.py`, `real_depth.py` — Replicate-hosted depth
  estimation tests; `depth_array.npy` / `depth_map.png` / `real_depth_array.npy` /
  `real_depth_map.png` are their outputs (the big .npy is ~44 MB — that's why they were
  never tracked).
- `center_object.py` — moondream pointing snippet.
- `temp.py` — env-var echo scratch.
