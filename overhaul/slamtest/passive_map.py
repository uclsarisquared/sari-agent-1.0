#!/usr/bin/env python3
"""
Passive mapping test: build the occupancy grid while you drive the agent
with SariSandboxMY's own built-in WASD/arrow-key controls (AgentControllerBase.
HandleMovement, Input.GetKey-driven) instead of sending movement commands
from Python.

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

    python slamtest/passive_map.py

Ctrl+C to stop and save the final grid.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TransformAgent  # noqa: E402

from occupancy_grid import OccupancyGrid
from pointcloud_map import PointCloudMap
from lidar_client import RequestLidarScan
from mapping import scan_to_world_points, scan_to_world_points_3d


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

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display.T, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Occupancy grid ({tag})")
    fig.savefig(os.path.join(output_dir, f"grid_{tag}.png"), dpi=150)
    plt.close(fig)


def run(args):
    grid = OccupancyGrid(size_m=args.size, resolution=args.resolution)
    cloud = PointCloudMap()

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
                hits = scan_to_world_points(scan, pos, rot[1])
                grid.integrate((pos[0], pos[2]), hits)
                cloud.add(scan_to_world_points_3d(scan, pos, rot[1]))
                last_scan_pos = pos
                scans_taken += 1
                print(
                    f"[passive_map] scan={scans_taken} pos=({pos[0]:.2f},{pos[2]:.2f}) "
                    f"yaw={rot[1]:.1f} hits={len(hits)}"
                )
                if scans_taken % args.save_every == 0:
                    save_snapshot(grid, args.output_dir, scans_taken)
                    cloud.save(args.output_dir, scans_taken)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        save_snapshot(grid, args.output_dir, "final")
        cloud.save(args.output_dir, "final")
        print(f"\n[passive_map] saved final grid + point cloud to {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Passive occupancy-grid mapping while driving via Unity's built-in controls."
    )
    parser.add_argument("--uri", default="ws://localhost:8080/commands")
    parser.add_argument("--size", type=float, default=60.0, help="Grid extent in meters (square)")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid cell size in meters")
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
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
