"""
Frontier-based occupancy-grid exploration using the LiDAR scanner.

Run from anywhere (this script fixes up sys.path itself):

    python slamtest/explore.py --max-steps 20          # smoke test
    python slamtest/explore.py --max-steps 300          # full mapping run

Requires the Unity SariSandboxV2 scene to be in Play mode with the
WebSocket command server running (default ws://localhost:8080/commands)
and the V1 compatibility layer enabled, since this reuses env.TransformAgent
for movement (text-response parsing).

Movement assumption to validate in the smoke test: TranslateAgent applies
translation in the agent's local (facing-relative) frame, matching how
env.py's _MOVE_FWD_/_PAN_LEFT_ helpers behave. Each loop iteration rotates
to face the target frontier, then steps forward.

Obstacle avoidance is proactive, not reactive: TranslateAgent moves the
agent by adding deltaTranslation directly to world position (not rotated
into the agent's local/facing frame — the step vector below is computed
in world space to match). The agent's Rigidbody is dynamic but uses
Discrete collision detection with no collider on its own GameObject (body
shape comes from child skeleton bone colliders built for IK/grab, not a
sealed locomotion capsule), so the "collided" flag it returns is not a
reliable safety net — a step can tunnel through geometry with zero
collision event ever firing. Every step is instead capped by
swept_clearance_ahead() reading the LiDAR scan already in hand, so the agent
should stop short of anything before ever touching it. "collided" is only
logged as a bonus signal; its absence proves nothing.

Hands are stowed for the whole run by default (--stow-hands, on by default)
via the Unity-side SetHandsActive command: hands carry their own colliders/
physics-activation trigger that the body-only clearance check never models,
so a swinging hand could still clip a shelf item even with the body kept
clear.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import TransformAgent, SetHandsActive  # noqa: E402

from occupancy_grid import OccupancyGrid
from pointcloud_map import PointCloudMap
from lidar_client import RequestLidarScan
from mapping import (
    scan_to_world_points,
    scan_to_world_points_3d,
    normalize_deg,
    angle_to_deg,
    swept_clearance_ahead,
)


def step_agent(delta_translation, delta_rotation, uri):
    state = TransformAgent(delta_translation, delta_rotation, uri=uri)
    return state["translation"], state["rotation"], state["isColliding"]


def pick_frontier(frontier_cells, cur_cell, recent_positions, stuck_dist=0.3):
    dists = np.hypot(
        frontier_cells[:, 0] - cur_cell[0],
        frontier_cells[:, 1] - cur_cell[1],
    )
    order = np.argsort(dists)

    stuck = False
    if len(recent_positions) >= 2:
        span = max(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a in recent_positions
            for b in recent_positions
        )
        stuck = span < stuck_dist

    if stuck and len(order) > 1:
        idx = order[min(len(order) - 1, np.random.randint(1, min(5, len(order))))]
    else:
        idx = order[0]
    return tuple(frontier_cells[idx])


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

    pos, rot, _ = step_agent((0, 0, 0), (0, 0, 0), args.uri)
    recent_positions = []

    if args.stow_hands:
        # Hands carry their own colliders/physics-activation trigger; stowing them for
        # the whole exploration run removes a collision surface the clearance check
        # below never accounts for (it only models the body via --body-radius).
        SetHandsActive(False, uri=args.uri)

    try:
        _explore_loop(args, grid, cloud, pos, rot, recent_positions)
    finally:
        if args.stow_hands:
            SetHandsActive(True, uri=args.uri)

    save_snapshot(grid, args.output_dir, "final")
    cloud.save(args.output_dir, "final")
    print(f"[explore] saved final grid + point cloud to {args.output_dir}")


def _explore_loop(args, grid, cloud, pos, rot, recent_positions):
    for step in range(args.max_steps):
        scan = RequestLidarScan(args.uri)
        hits = scan_to_world_points(scan, pos, rot[1])
        grid.integrate((pos[0], pos[2]), hits)
        cloud.add(scan_to_world_points_3d(scan, pos, rot[1]))

        frontier_cells = grid.frontiers()
        if len(frontier_cells) == 0:
            print(f"[explore] map complete after {step} scans, {len(hits)} hits in final scan")
            break

        cur_cell = grid.cell(pos[0], pos[2])
        target_cell = pick_frontier(frontier_cells, cur_cell, recent_positions)
        target_x, target_z = grid.to_world(*target_cell)

        dx, dz = target_x - pos[0], target_z - pos[2]
        dist = math.hypot(dx, dz)
        delta_yaw = normalize_deg(angle_to_deg(dx, dz) - rot[1])

        # Clearance comes from the scan taken this iteration, before rotating,
        # so delta_yaw doubles as the sensor-local heading to check. Swept (not
        # angular-cone) clearance so the check scales with the agent's actual
        # body width instead of a fixed degree window that under-covers close range.
        # vertical_band_deg no longer exists on swept_clearance_ahead() - mapping.py switched to
        # height-band filtering (min_obstacle_height/max_obstacle_height) shared with
        # slamtest_frontier/. This old script picks up those new defaults automatically; only
        # this call site needed touching to stop it crashing, no decision logic changed.
        clearance = swept_clearance_ahead(scan, delta_yaw, body_radius=args.body_radius)
        safe_step = max(0.0, clearance - args.safety_margin)
        step_len = min(args.step_size, dist, safe_step)

        # Rotate to face the target (keeps the LiDAR's forward aligned with
        # travel direction for the next scan), but note this does NOT steer
        # the translation below — see world_dx/world_dz.
        pos, rot, _ = step_agent((0, 0, 0), (0, delta_yaw, 0), args.uri)

        if step_len < args.min_step:
            blocked_x = pos[0] + math.sin(math.radians(rot[1])) * max(clearance, 0.05)
            blocked_z = pos[2] + math.cos(math.radians(rot[1])) * max(clearance, 0.05)
            grid.mark_occupied((blocked_x, blocked_z))
            collided = False
            print(f"[explore] path blocked (clearance={clearance:.2f}m); holding position and rescanning")
        else:
            # TranslateAgent applies deltaTranslation directly to world
            # position (AgentControllerBase.cs: rigidbody.MovePosition(rigidbody.position
            # + delta)) — it is NOT rotated into the agent's local/facing frame.
            # So the step vector must already point at the target in world space;
            # sending (0, 0, step_len) here would move along raw world +Z instead.
            world_dx = dx / dist * step_len
            world_dz = dz / dist * step_len
            pos, rot, collided = step_agent((world_dx, 0, world_dz), (0, 0, 0), args.uri)
            if collided:
                # Bonus signal only — the agent's Rigidbody uses Discrete collision
                # detection with no root collider (body shape comes from child bone
                # colliders), so this can miss real contact entirely. Its absence does
                # NOT mean the step was safe; the clearance check above is what matters.
                ahead_x = pos[0] + math.sin(math.radians(rot[1])) * 0.2
                ahead_z = pos[2] + math.cos(math.radians(rot[1])) * 0.2
                grid.mark_occupied((ahead_x, ahead_z))
                print(f"[explore] collision reported near ({pos[0]:.2f}, {pos[2]:.2f})")

        recent_positions.append((pos[0], pos[2]))
        if len(recent_positions) > args.stuck_window:
            recent_positions.pop(0)

        print(
            f"[explore] step={step} pos=({pos[0]:.2f},{pos[2]:.2f}) "
            f"yaw={rot[1]:.1f} frontiers={len(frontier_cells)} collided={collided}"
        )

        if step % args.save_every == 0:
            save_snapshot(grid, args.output_dir, step)
            cloud.save(args.output_dir, step)


def build_parser():
    parser = argparse.ArgumentParser(description="Frontier-exploration mapping via the LiDAR scanner.")
    parser.add_argument("--uri", default="ws://localhost:8080/commands")
    parser.add_argument("--size", type=float, default=60.0, help="Grid extent in meters (square)")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid cell size in meters")
    parser.add_argument("--step-size", type=float, default=0.5, help="Max meters moved per iteration")
    parser.add_argument(
        "--safety-margin", type=float, default=0.65,
        help="Meters of clearance to keep from any LiDAR-detected obstacle. Physical collision "
             "(Rigidbody, Discrete detection, no root collider) is not a reliable fallback here, "
             "so this margin is the only real safety net — kept above the ~0.4m hand-proximity "
             "radius that activates item physics, plus extra cushion for scan/pose error.",
    )
    parser.add_argument("--min-step", type=float, default=0.1, help="Below this, skip the move and rescan instead")
    parser.add_argument(
        "--body-radius", type=float, default=0.3,
        help="Agent footprint radius (m) used to sweep the clearance check, so an obstacle "
             "off-axis but within the agent's actual width still blocks the step. Tune to the "
             "sim's real shoulder/hand width — too small under-protects, too large starves "
             "frontier picking near narrow aisles.",
    )
    parser.add_argument(
        "--stow-hands", action=argparse.BooleanOptionalAction, default=True,
        help="Deactivate the hand prefabs (via the SetHandsActive command) for the whole run, "
             "since the clearance check only models the body and never accounts for hands "
             "swinging past shelf items while walking. Re-enabled on exit, including on error.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stuck-window", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
