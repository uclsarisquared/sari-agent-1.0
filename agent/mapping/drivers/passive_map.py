#!/usr/bin/env python3
"""
Passive mapping: build the occupancy grid while you drive the agent with
SariSandboxMY's own built-in WASD/arrow-key controls (AgentControllerBase.
HandleMovement, Input.GetKey-driven) instead of sending movement commands
from Python.

Matches explore.py's whole output contract, just with you driving instead of the
frontier planner: it clears --output-dir first, folds scans into a 3D VoxelGrid
that collapses to the 2D grid with occupied-priority (so shelf faces read solid
instead of gappy/speckled - see plans/phase1.1_voxelized_mapping.md), stows the
hand prefabs for the run (SetHandsActive, so the agent's own hands don't map as a
phantom close-range obstacle), and on exit saves grid_final + point cloud AND
extracts the base topology to topology_final.json. So the output is a drop-in for
build_shelf_graph.py (Phase 2 shelf checkpoints) exactly like an explore run -
hand-driving every corner yourself is a way to get a clean, fully-covered map
without any of the frontier explorer's navigation limits.

This script issues NO TransformAgent translation/rotation deltas of its own
(only zero-delta polls to read pose, which are no-ops in Unity's
TranslateAgent). Drive by clicking into the Play-mode Game window and using
WASD + arrow keys directly; this script just polls pose + LiDAR over the
WebSocket command server and folds scans into the grid.

Scans are only taken once the agent has been (approximately) stationary for
--settle-time seconds, not continuously while moving: the pose used to
project a scan comes from a separate WebSocket round-trip than the scan
itself, so if the agent is still moving between the two, the projected pose
is already stale by the time the scan is actually captured. Drive, then
pause briefly to let a scan land, then continue.

Run from anywhere; requires SariSandboxV2 in Play mode with the WebSocket
command server running (default ws://localhost:8080/commands), agent
interaction style set so WASD isn't being eaten by manual hand control
(don't hold Shift, which switches to hand-control mode in the Unity script).

    python mapping/drivers/passive_map.py

Ctrl+C to stop and save the final grid, point cloud, and topology_final.json.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/drivers
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)
from sim.env import TransformAgent, SetHandsActive  # noqa: E402

from voxel_grid import VoxelGrid
from pointcloud_map import PointCloudMap
from lidar_client import RequestLidarScan
from mapping import scan_to_world_points_3d, SENSOR_HEIGHT_OFFSET_M, SELF_EXCLUSION_RANGE_M
from topology import (
    extract_topology, save_topology,
    DEFAULT_MIN_BRANCH_LENGTH_M, DEFAULT_DOORWAY_MAX_WIDTH_M, DEFAULT_DOORWAY_WIDTH_RATIO,
)


def save_snapshot(grid, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"grid_{tag}.npy"), grid.log_odds)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    display = np.full(grid.log_odds.shape, 0.5)
    display[grid.log_odds < grid.FREE_THRESHOLD] = 1.0
    display[grid.log_odds > grid.OCCUPIED_THRESHOLD] = 0.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(display.T, origin="lower", cmap="gray", vmin=0, vmax=1)

    # Crop to the explored region (plus a 1m margin) instead of the full grid extent - most
    # of a 60m grid is still "unknown" gray on any run that hasn't mapped the whole store, so
    # a full-extent plot renders the actual store as a small blob in the middle. Matches the
    # crop build_shelf_graph.py uses, so the occupancy PNG and the graph PNG frame the same area.
    known_cells = np.argwhere(display.T != 0.5)
    if len(known_cells) > 0:
        margin = int(round(1.0 / grid.res))  # 1m padding
        y0, x0 = known_cells.min(axis=0) - margin
        y1, x1 = known_cells.max(axis=0) + margin
        ax.set_xlim(max(0, x0), min(grid.log_odds.shape[0], x1))
        ax.set_ylim(max(0, y0), min(grid.log_odds.shape[1], y1))

    ax.set_title(f"Occupancy grid ({tag})")
    fig.savefig(os.path.join(output_dir, f"grid_{tag}.png"), dpi=150)
    plt.close(fig)


def _clear_output_dir(output_dir):
    """Delete every file already in output_dir before a new run starts (same as explore.py's
    own _clear_output_dir), so the folder only ever holds this run's grid_*/points_*/topology_*
    outputs instead of a previous run's leftovers (e.g. a higher scan count, or a stale
    grid_final.*). Only removes files directly inside output_dir (not subdirectories); a no-op
    if output_dir doesn't exist yet."""
    if not os.path.isdir(output_dir):
        return
    removed = 0
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
    if removed:
        print(f"[passive_map] cleared {removed} file(s) from a previous run in {output_dir}")


def run(args):
    if args.clear_output:
        _clear_output_dir(args.output_dir)
    # Same pipeline as explore.py: scans update a 3D VoxelGrid, collapsed each scan to
    # voxel.grid (the 2D OccupancyGrid consumers read). The occupied-priority collapse is what
    # keeps shelf faces solid instead of the gappy read a flat 2D integrate produces.
    voxel = VoxelGrid(
        size_m=args.size, resolution=args.resolution,
        min_obstacle_height=args.min_obstacle_height,
        max_obstacle_height=args.max_obstacle_height,
        sensor_height_offset=args.sensor_height_offset,
    )
    grid = voxel.grid
    cloud = PointCloudMap()

    if args.stow_hands:
        # Deactivate the hand prefabs so the agent's own hands don't map as a phantom
        # close-range obstacle (the LiDAR otherwise catches a persistent ~0.6-0.7m self-hit -
        # same reason explore.py stows them). Re-enabled on exit in the finally below.
        SetHandsActive(False, uri=args.uri)

    last_scan_pos = None
    last_poll_pos = None
    still_polls = 0
    scans_taken = 0
    # TransformAgent's reply and the LiDAR scan are two separate WebSocket round-trips: if the
    # agent is still moving between them (live WASD control), the pose used to project the scan
    # is already stale by the time the scan is actually captured, smearing the grid. Instead of
    # scanning on every poll where the agent has moved, wait until the agent has been
    # (approximately) stationary for settle_time seconds — pose and scan can't drift apart if
    # nothing is moving — then take the scan.
    settle_polls_needed = max(1, int(round(args.settle_time / args.poll_interval)))

    print("[passive_map] polling pose + LiDAR — drive the agent in the Unity Game window "
          "with WASD/arrow keys, then stop and hold still for a moment to trigger a scan. "
          "Ctrl+C to stop and save.")

    try:
        while True:
            state = TransformAgent((0, 0, 0), (0, 0, 0), uri=args.uri)
            pos, rot = state["translation"], state["rotation"]

            poll_delta = (
                0.0 if last_poll_pos is None
                else math.hypot(pos[0] - last_poll_pos[0], pos[2] - last_poll_pos[2])
            )
            still_polls = still_polls + 1 if poll_delta < args.settle_threshold else 0
            last_poll_pos = pos

            moved_since_scan = (
                last_scan_pos is None
                or math.hypot(pos[0] - last_scan_pos[0], pos[2] - last_scan_pos[2]) >= args.move_threshold
            )

            if moved_since_scan and still_polls >= settle_polls_needed:
                scan = RequestLidarScan(args.uri)
                # 3D-integrate into the voxel grid, then collapse to the 2D `grid`. collapse()
                # re-projects the WHOLE accumulated voxel grid each scan, so the map builds up
                # correctly across the run just like explore.py's per-step integrate/collapse.
                n_hits = voxel.integrate(scan, pos, rot[1], self_exclusion_range=args.self_exclusion_range)
                voxel.collapse()
                cloud.add(scan_to_world_points_3d(scan, pos, rot[1], sensor_height_offset=args.sensor_height_offset))
                last_scan_pos = pos
                scans_taken += 1
                print(
                    f"[passive_map] scan={scans_taken} pos=({pos[0]:.2f},{pos[2]:.2f}) "
                    f"yaw={rot[1]:.1f} hits={n_hits}"
                )
                if scans_taken % args.save_every == 0:
                    save_snapshot(grid, args.output_dir, scans_taken)
                    cloud.save(args.output_dir, scans_taken)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        # Save first (the artifacts matter most), then restore hands - so a failure re-enabling
        # them can't cost you the map. Runs on Ctrl+C and on any unexpected error alike.
        save_snapshot(grid, args.output_dir, "final")
        cloud.save(args.output_dir, "final")
        print(f"\n[passive_map] saved final grid + point cloud to {args.output_dir}")
        if args.extract_topology:
            # Same base topology explore.py extracts (aisle ends / junctions / doorway pinch
            # points), so a passive map is a drop-in input for build_shelf_graph.py.
            # min_checkpoint_clearance_m=body_radius drops checkpoints the agent's body can't
            # fit at, matching explore.
            topology = extract_topology(
                grid, connectivity=args.connectivity,
                min_branch_length_m=args.topology_min_branch_length_m,
                doorway_max_width_m=args.topology_doorway_max_width_m,
                doorway_width_ratio=args.topology_doorway_width_ratio,
                min_checkpoint_clearance_m=args.body_radius,
            )
            topology_path = save_topology(topology, args.output_dir, "final")
            kind_counts = {}
            for c in topology.checkpoints:
                kind_counts[c.kind] = kind_counts.get(c.kind, 0) + 1
            print(f"[passive_map] saved topology ({len(topology.checkpoints)} checkpoints "
                  f"{kind_counts}, {len(topology.edges)} edges) to {topology_path}")
        if args.stow_hands:
            SetHandsActive(True, uri=args.uri)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Passive occupancy-grid mapping while driving via Unity's built-in controls."
    )
    parser.add_argument("--uri", default="ws://localhost:8080/commands")
    parser.add_argument("--size", type=float, default=60.0, help="Grid extent in meters (square)")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid cell size in meters")
    parser.add_argument(
        "--min-obstacle-height", type=float, default=0.05,
        help="Meters above the agent's root; hits below this are floor noise, ignored",
    )
    parser.add_argument(
        "--max-obstacle-height", type=float, default=2.0,
        help="Meters above the agent's root; hits above this (e.g. ceiling signage) are ignored",
    )
    parser.add_argument(
        "--sensor-height-offset", type=float, default=SENSOR_HEIGHT_OFFSET_M,
        help="Meters the LiDAR sensor sits above the agent's reported root position",
    )
    parser.add_argument(
        "--self-exclusion-range", type=float, default=SELF_EXCLUSION_RANGE_M,
        help="Meters within which LiDAR hits are ignored as sensor-housing/self noise",
    )
    parser.add_argument(
        "--stow-hands", action=argparse.BooleanOptionalAction, default=True,
        help="Deactivate the hand prefabs for the run so they don't map as a phantom close-range "
             "obstacle (same as explore.py). Re-enabled on exit. Pass --no-stow-hands to leave "
             "the Unity hand state untouched.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.15,
        help="Seconds between pose polls (each poll is a zero-delta TransformAgent no-op)",
    )
    parser.add_argument(
        "--move-threshold", type=float, default=0.1,
        help="Minimum meters moved since the last scan before requesting a new one",
    )
    parser.add_argument(
        "--settle-time", type=float, default=0.3,
        help="Seconds the agent must stay within --settle-threshold before a scan is taken, "
             "so the pose used to project the scan can't have drifted from the pose Unity "
             "actually captured it at",
    )
    parser.add_argument(
        "--settle-threshold", type=float, default=0.03,
        help="Max meters of movement between consecutive polls to still count as stationary",
    )
    parser.add_argument("--save-every", type=int, default=20, help="Save a snapshot every N scans")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_MAPPING_DIR, "output"),
    )
    parser.add_argument(
        "--clear-output", action=argparse.BooleanOptionalAction, default=True,
        help="Delete every file already in --output-dir before this run starts (same as "
             "explore.py), so the folder only holds this run's outputs instead of a previous "
             "run's leftovers. Pass --no-clear-output to keep them.",
    )
    parser.add_argument(
        "--extract-topology", action=argparse.BooleanOptionalAction, default=True,
        help="After the run, thin the resolved grid to a checkpoint/adjacency graph (aisle ends, "
             "junctions, doorway pinch points) and save topology_final.json (same as explore.py) - "
             "the input to build_shelf_graph.py. Pass --no-extract-topology for grid + cloud only.",
    )
    parser.add_argument(
        "--body-radius", type=float, default=0.3,
        help="Agent footprint radius (m); topology checkpoints in gaps narrower than this are "
             "dropped as spots the body can't fit at, matching explore.py.",
    )
    parser.add_argument(
        "--connectivity", type=int, default=8, choices=[4, 8],
        help="Neighbor connectivity for topology extraction",
    )
    parser.add_argument(
        "--topology-min-branch-length-m", type=float, default=DEFAULT_MIN_BRANCH_LENGTH_M,
        help="Drop dead-end skeleton spurs shorter than this (thinning artifacts, not real stubs)",
    )
    parser.add_argument(
        "--topology-doorway-max-width-m", type=float, default=DEFAULT_DOORWAY_MAX_WIDTH_M,
        help="A corridor's narrowest point must be below this width to qualify as a doorway",
    )
    parser.add_argument(
        "--topology-doorway-width-ratio", type=float, default=DEFAULT_DOORWAY_WIDTH_RATIO,
        help="Narrowest point must also be below this fraction of the corridor's own median width",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
