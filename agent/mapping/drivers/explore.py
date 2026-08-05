"""
Frontier-based occupancy-grid exploration using the LiDAR scanner, driven by
FrontierPlanner (clustering + A* + goal commitment).

Run from anywhere (this script fixes up sys.path itself):

    python mapping/drivers/explore.py --max-steps 20    # smoke test
    python mapping/drivers/explore.py --max-steps 300   # full mapping run

Requires the Unity SariSandboxV2 scene to be in Play mode with the
WebSocket command server running (default ws://localhost:8080/commands)
and the V1 compatibility layer enabled, since this reuses env.TransformAgent
for movement (text-response parsing).

Movement convention (confirmed against AgentControllerBase.cs / SariAgentCommandBehavior.cs):
TranslateAgent applies translation EGOCENTRICALLY - the sim rotates the (x=right, y=up, z=forward)
vector into world space itself via EgocentricToWorldTranslation using the agent's current facing -
and applies rotation as a delta added to the current transform. So this loop always rotates the
agent to face its travel direction first, then steps straight forward with the body-relative vector
(0, 0, step_len); it must NOT pre-rotate the step into world space (the sim would rotate it a second
time). DE-SYNC 2026-07-22: the sim flipped this contract from world-space to egocentric
(SariSandboxV2 3940ce7, merged 2026-07-21), and this loop's old world-space step vectors then came
out rotated by the agent's yaw - the "forward isn't forward" bug. Re-verify with
probe_translation.py if the sim's dispatcher contract changes again.

Obstacle avoidance relies entirely on LiDAR, never on the Rigidbody's
isColliding/collided flag: the agent's Rigidbody uses Discrete collision
detection with no collider on its own GameObject (body shape comes from
child skeleton bone colliders built for IK/grab, not a sealed locomotion
capsule), and this signal has been confirmed unreliable in practice - it can
miss real contact, and a false positive would inject a phantom obstacle into
the map. Every step is instead capped by swept_clearance_ahead() reading the
LiDAR scan already in hand, and all occupancy-grid updates come from
grid.integrate() (LiDAR raycast hits) plus the LiDAR-clearance-driven
"blocked" mark below. step_agent()'s collided return value is intentionally
discarded.

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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/drivers
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from sim.env import TransformAgent, SetHandsActive  # noqa: E402

from occupancy_grid import OccupancyGrid  # noqa: E402
from voxel_grid import VoxelGrid  # noqa: E402
from pointcloud_map import PointCloudMap  # noqa: E402
from lidar_client import RequestLidarScan  # noqa: E402
from mapping import (  # noqa: E402
    scan_to_world_points_3d,
    normalize_deg,
    angle_to_deg,
    swept_clearance_ahead,
    SENSOR_HEIGHT_OFFSET_M,
    SELF_EXCLUSION_RANGE_M,
)
from frontier_planner import FrontierPlanner  # noqa: E402
from topology import (  # noqa: E402
    extract_topology,
    save_topology,
    DEFAULT_MIN_BRANCH_LENGTH_M,
    DEFAULT_DOORWAY_MAX_WIDTH_M,
    DEFAULT_DOORWAY_WIDTH_RATIO,
)


def step_agent(delta_translation, delta_rotation, uri):
    state = TransformAgent(delta_translation, delta_rotation, uri=uri)
    return state["translation"], state["rotation"], state["isColliding"]


def _format_clearance_debug(clearance_debug):
    if not clearance_debug:
        return " [no offending hit found - clearance capped by dist/step-size instead]"
    return (
        f" [closest hit: ch={clearance_debug['channel']} v_deg={clearance_debug['v_deg']:.1f} "
        f"range={clearance_debug['range']:.2f} height_above_root={clearance_debug['height_above_root']:.2f} "
        f"az_rel={clearance_debug['az_rel_deg']:.1f} lateral={clearance_debug['lateral']:.2f}]"
    )


def blocked_position(pos, rot_deg, clearance, clearance_debug):
    """World (x, z) of the obstruction actually responsible for a blocked step.

    Uses the real offending ray's bearing/range (clearance_debug) when available,
    rather than assuming the obstruction is straight ahead along rot_deg - the
    closest hit is frequently well off-axis (e.g. a shelf to one side, not dead
    ahead). Confirmed live: marking straight-ahead when the real obstruction was
    ~48 degrees off to the side poisoned empty space instead of the real
    obstacle, so the grid never learned where it actually was and A* kept
    re-confirming routes through it, even toward different distant goals.
    """
    if clearance_debug:
        bearing = rot_deg + clearance_debug["az_rel_deg"]
        rng = clearance_debug["range"]
    else:
        # No offending hit found (blocked by dist/step-size instead) - straight-ahead
        # at the reported clearance is the best guess available in this edge case.
        bearing = rot_deg
        rng = max(clearance, 0.05)
    return (
        pos[0] + math.sin(math.radians(bearing)) * rng,
        pos[2] + math.cos(math.radians(bearing)) * rng,
    )


def find_clear_heading(scan, desired_heading_deg, min_needed_step, safety_margin, *,
                        body_radius, min_obstacle_height, max_obstacle_height,
                        sensor_height_offset, self_exclusion_range,
                        max_nudge_deg=30.0, nudge_step_deg=5.0):
    """Search for a heading near desired_heading_deg with enough swept_clearance_ahead
    to actually move min_needed_step, trying the smallest deviations first (alternating
    left/right) before giving up.

    This is the local reactive steering the loop was otherwise missing entirely: without
    it, the only two behaviors are "walk exactly straight at the waypoint" or "stop dead
    and wait for the global A* planner to find a whole different route" - even when the
    real-time scan already shows the obstruction is off to one side (see
    clearance_debug's az_rel_deg/lateral) and a small heading adjustment would clear it.

    Returns (heading_deg, clearance, clearance_debug, offset_deg) for the first candidate
    that clears, or None if nothing within max_nudge_deg works (caller falls back to
    treating the step as blocked, unchanged from before this existed).
    """
    offsets = [0.0]
    steps = max(1, int(round(max_nudge_deg / nudge_step_deg)))
    for i in range(1, steps + 1):
        offsets.append(i * nudge_step_deg)
        offsets.append(-i * nudge_step_deg)

    for offset in offsets:
        heading = normalize_deg(desired_heading_deg + offset)
        clearance, debug = swept_clearance_ahead(
            scan, heading, body_radius=body_radius,
            min_obstacle_height=min_obstacle_height, max_obstacle_height=max_obstacle_height,
            sensor_height_offset=sensor_height_offset, self_exclusion_range=self_exclusion_range,
            debug=True,
        )
        if max(0.0, clearance - safety_margin) >= min_needed_step:
            return heading, clearance, debug, offset
    return None


def find_escape_heading(scan, safety_margin, min_step, *, body_radius, min_obstacle_height,
                        max_obstacle_height, sensor_height_offset, self_exclusion_range,
                        sweep_step_deg=10.0):
    """Full-circle search for the single most-open heading, used to break out of a wedge:
    a spot where the straight-at-waypoint heading AND every small find_clear_heading nudge
    are all blocked because the obstruction spans the whole forward arc and the only open
    space is off to the side or behind (a shelf-corner pocket, an aisle dead-end).

    Where find_clear_heading returns the SMALLEST deviation that merely clears one step
    (staying as close to the intended waypoint as possible), this returns the heading of
    MAXIMUM clearance regardless of direction - a decisive step toward the most open space
    to physically leave the pocket, after which the planner (already forced back to
    NEED_GOAL by notify_blocked) re-picks a goal from the new, un-wedged cell. The raw scan
    is a full 360deg sweep (confirmed live: 1024 samples over 360deg), so every heading has
    a real range measurement in the enclosed store - no unsensed direction can masquerade as
    "most open".

    Headings are scan-local (relative to the pose the scan was taken at), the same
    convention as delta_yaw. Returns (heading_deg, clearance, clearance_debug) for the
    most-open heading, or None if even that heading can't fit a min_step move (the agent is
    genuinely boxed in on every side - caller falls back to holding position, which then
    trips the planner's no-movement circuit breaker and ends the run).
    """
    best = None
    n = max(1, int(round(360.0 / sweep_step_deg)))
    for i in range(n):
        heading = normalize_deg(i * sweep_step_deg)
        clearance, debug = swept_clearance_ahead(
            scan, heading, body_radius=body_radius,
            min_obstacle_height=min_obstacle_height, max_obstacle_height=max_obstacle_height,
            sensor_height_offset=sensor_height_offset, self_exclusion_range=self_exclusion_range,
            debug=True,
        )
        if best is None or clearance > best[1]:
            best = (heading, clearance, debug)
    if best is None or max(0.0, best[1] - safety_margin) < min_step:
        return None
    return best


def save_snapshot(grid, output_dir, tag, interactive_backend=False, show_now=False):
    """Save a grid snapshot (.npy + .png). By default forces the Agg backend
    (no display needed, safe on any machine including headless ones). Pass
    interactive_backend=True to skip that - required for show_now=True to
    actually pop up a window, since matplotlib locks in its backend on the
    first `import matplotlib.pyplot` in the process; if any earlier call in
    this run forced Agg, a later show_now=True call here would silently do
    nothing. Callers must pass the SAME interactive_backend value on every
    save_snapshot() call within a run for this to work correctly.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"grid_{tag}.npy"), grid.log_odds)
    try:
        import matplotlib
        if not interactive_backend:
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
    if show_now:
        print("[explore] displaying final map - close the plot window to exit")
        plt.show()
        plt.close("all")  # belt-and-braces: release the GUI backend's figure/window state too
    else:
        plt.close(fig)


def _clear_output_dir(output_dir):
    """Delete every file already in output_dir before a new run starts, so the folder only ever
    shows this run's grid_*/points_* snapshots - otherwise a previous run's leftovers (e.g. a
    higher step count, or stale grid_final.*) linger alongside the new ones and make the
    directory listing confusing to read. Only removes files directly inside output_dir (not the
    directory itself, and not subdirectories); a no-op if output_dir doesn't exist yet."""
    if not os.path.isdir(output_dir):
        return
    removed = 0
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
    if removed:
        print(f"[explore] cleared {removed} file(s) from a previous run in {output_dir}")


def run(args):
    if args.clear_output:
        _clear_output_dir(args.output_dir)

    # Phase 1.1: scans update a 3D VoxelGrid, collapsed each step to voxel.grid - the same
    # OccupancyGrid every consumer below reads. The band ([min,max]_obstacle_height, above-
    # root frame) must match scan_to_world_points' own filter so the voxel column spans
    # exactly the heights the 2D map cares about. See plans/phase1.1_voxelized_mapping.md.
    voxel = VoxelGrid(
        size_m=args.size, resolution=args.resolution,
        min_obstacle_height=args.min_obstacle_height,
        max_obstacle_height=args.max_obstacle_height,
        sensor_height_offset=args.sensor_height_offset,
    )
    grid = voxel.grid
    cloud = PointCloudMap()
    planner = FrontierPlanner(
        grid,
        min_cluster_size=args.min_cluster_size,
        connectivity=args.connectivity,
        goal_arrival_radius=args.goal_arrival_radius,
        waypoint_arrival_radius=args.waypoint_arrival_radius,
        replan_stuck_steps=args.replan_stuck_steps,
        replan_no_progress_steps=args.replan_no_progress_steps,
        min_progress_m=args.min_progress_m,
        goal_blocklist_steps=args.goal_blocklist_steps,
        window_slack=args.window_slack,
        min_window_cells=args.min_window_cells,
        astar_max_window_cells=args.astar_max_window_cells,
        max_replans_without_moving=args.max_replans_without_moving,
        body_radius=args.body_radius,
        debug=args.debug_planner,
    )

    pos, rot, _ = step_agent((0, 0, 0), (0, 0, 0), args.uri)

    if args.stow_hands:
        # Hands carry their own colliders/physics-activation trigger; stowing them for
        # the whole exploration run removes a collision surface the clearance check
        # below never accounts for (it only models the body via --body-radius).
        SetHandsActive(False, uri=args.uri)

    try:
        _explore_loop(args, voxel, grid, cloud, pos, rot, planner)
    finally:
        if args.stow_hands:
            SetHandsActive(True, uri=args.uri)

    save_snapshot(grid, args.output_dir, "final", interactive_backend=args.show_map, show_now=args.show_map)
    cloud.save(args.output_dir, "final")
    print(f"[explore] saved final grid + point cloud to {args.output_dir}")

    if args.extract_topology:
        topology = extract_topology(
            grid, connectivity=args.connectivity,
            min_branch_length_m=args.topology_min_branch_length_m,
            doorway_max_width_m=args.topology_doorway_max_width_m,
            doorway_width_ratio=args.topology_doorway_width_ratio,
            # Drop checkpoints the agent's body can't fit at (spurious junctions/doorways/
            # ends the skeleton threads into a ragged shelf face) - same radius A* plans with.
            min_checkpoint_clearance_m=args.body_radius,
        )
        topology_path = save_topology(topology, args.output_dir, "final")
        kind_counts = {}
        for c in topology.checkpoints:
            kind_counts[c.kind] = kind_counts.get(c.kind, 0) + 1
        print(f"[explore] saved topology ({len(topology.checkpoints)} checkpoints {kind_counts}, "
              f"{len(topology.edges)} edges) to {topology_path}")


def _explore_loop(args, voxel, grid, cloud, pos, rot, planner):
    total_escapes = 0
    for step in range(args.max_steps):
        scan = RequestLidarScan(args.uri)
        # 3D-integrate into the voxel grid, then collapse to the 2D `grid` (same object) the
        # planner/topology read. collapse() runs before planner.update() below so the planner
        # always sees a grid consistent with the latest scan (and with any blocked-mark from
        # the previous step, which lands in the voxel grid and surfaces on this collapse).
        n_hits = voxel.integrate(scan, pos, rot[1], self_exclusion_range=args.self_exclusion_range)
        voxel.collapse()
        cloud.add(scan_to_world_points_3d(scan, pos, rot[1], sensor_height_offset=args.sensor_height_offset))

        nav = planner.update((pos[0], pos[2]), grid.cell(pos[0], pos[2]))
        if nav.kind == "done":
            print(f"[explore] map complete after {step} scans, {n_hits} hits in final scan")
            break

        target_x, target_z = nav.target_world_xz
        dx, dz = target_x - pos[0], target_z - pos[2]
        dist = math.hypot(dx, dz)
        delta_yaw = normalize_deg(angle_to_deg(dx, dz) - rot[1])

        # Clearance comes from the scan taken this iteration, before rotating,
        # so delta_yaw doubles as the sensor-local heading to check. Swept (not
        # angular-cone) clearance so the check scales with the agent's actual
        # body width instead of a fixed degree window that under-covers close range.
        clearance, clearance_debug = swept_clearance_ahead(
            scan, delta_yaw, body_radius=args.body_radius,
            min_obstacle_height=args.min_obstacle_height,
            max_obstacle_height=args.max_obstacle_height,
            sensor_height_offset=args.sensor_height_offset,
            self_exclusion_range=args.self_exclusion_range,
            debug=True,
        )
        safe_step = max(0.0, clearance - args.safety_margin)
        step_len = min(args.step_size, dist, safe_step)

        # The straight-at-waypoint heading is blocked - before giving up on this waypoint
        # entirely (holding position for a full replan), try small heading nudges around it.
        # The real-time scan already tells us which side the obstruction is on
        # (clearance_debug's az_rel_deg/lateral); a few-degree sidestep is often enough to
        # clear it while still making progress toward the same waypoint, rather than
        # discarding the whole path over an obstruction the global planner can't react to
        # this granularly.
        nudge_deg = 0.0
        if step_len < args.min_step and args.max_nudge_deg > 0:
            nudged = find_clear_heading(
                scan, delta_yaw, args.min_step, args.safety_margin,
                body_radius=args.body_radius, min_obstacle_height=args.min_obstacle_height,
                max_obstacle_height=args.max_obstacle_height, sensor_height_offset=args.sensor_height_offset,
                self_exclusion_range=args.self_exclusion_range,
                max_nudge_deg=args.max_nudge_deg, nudge_step_deg=args.nudge_step_deg,
            )
            if nudged is not None:
                delta_yaw, clearance, clearance_debug, nudge_deg = nudged
                safe_step = max(0.0, clearance - args.safety_margin)
                step_len = min(args.step_size, dist, safe_step)

        # Rotate to face the target (keeps the LiDAR's forward aligned with travel
        # direction for the next scan) - or the nudged heading above, if one was used.
        pos, rot, _ = step_agent((0, 0, 0), (0, delta_yaw, 0), args.uri)

        if step_len < args.min_step:
            blocked_x, blocked_z = blocked_position(pos, rot[1], clearance, clearance_debug)
            # Height of the offending hit, so the mark lands in the right voxel bin (see
            # VoxelGrid.mark_blocked_region). Fall back to sensor height when there's no
            # offending hit (blocked by dist/step-size, not an obstacle) - a mid-band bin.
            blocked_h = clearance_debug["height_above_root"] if clearance_debug else args.sensor_height_offset
            # A region, not a single point: one real obstacle spans many cells, and marking
            # only the exact hit point took many repeated near-identical blocks to build up
            # enough occupied area for inflation-aware A* to actually avoid the danger zone
            # (or exclude a candidate goal too close to it). body_radius as the mark radius
            # keeps this consistent with the same clearance the agent itself needs. Marked in
            # the voxel grid (not the 2D grid, which the next collapse() would overwrite).
            voxel.mark_blocked_region((blocked_x, blocked_z), blocked_h, radius_m=args.body_radius)
            # notify_blocked forces the planner back to NEED_GOAL (replan next step) and
            # blocklists the current goal, so whether we escape or hold below, the doomed
            # waypoint is abandoned rather than re-committed to.
            planner.notify_blocked((blocked_x, blocked_z))

            # Wedge escape. The straight-at-waypoint heading is blocked AND every small
            # find_clear_heading nudge above already failed - so the obstruction spans the
            # whole forward arc and the only open space is off to the side/behind (a shelf-
            # corner pocket, an aisle dead-end). Holding position just rescans the identical
            # pose until the planner's no-movement circuit breaker ends the run early with
            # the map half-built (the recurring "blocked path premature end"). Instead, step
            # toward the most-open direction to physically leave the pocket; the forced replan
            # above then plans from the new, un-wedged cell, and the blocked-region mark
            # deters A* from routing right back in. Bounded by --max-escapes so a pathological
            # approach<->escape oscillation (e.g. a frontier only reachable through the
            # A*-vs-clearance dead band) still terminates instead of thrashing to --max-steps.
            #
            # Gate on safe_step (clearance - safety_margin) < min_step, NOT just step_len <
            # min_step: step_len is ALSO driven to ~0 when the agent simply arrives at its
            # waypoint (dist-to-waypoint ~0) with the path's last waypoint un-advanceable -
            # e.g. parked at an observation vantage. That is not a wedge (clearance is wide
            # open there); escaping it would waste a 0.5m detour and burn escape budget. Only
            # a genuine close obstruction (clearance < safety_margin + min_step) is a wedge.
            escaped = False
            if args.escape_when_wedged and total_escapes < args.max_escapes and safe_step < args.min_step:
                escape = find_escape_heading(
                    scan, args.safety_margin, args.min_step,
                    body_radius=args.body_radius, min_obstacle_height=args.min_obstacle_height,
                    max_obstacle_height=args.max_obstacle_height,
                    sensor_height_offset=args.sensor_height_offset,
                    self_exclusion_range=args.self_exclusion_range,
                    sweep_step_deg=args.escape_sweep_step_deg,
                )
                if escape is not None:
                    esc_heading, esc_clear, _esc_debug = escape
                    # esc_heading is scan-local; the agent currently faces (scan pose +
                    # delta_yaw) after the rotation above, so rotate by the difference to face
                    # the escape heading, then step out along it. step magnitude uses the same
                    # clearance - safety_margin formula as a normal move (never toward the
                    # waypoint here - we're deliberately leaving it).
                    pos, rot, _ = step_agent(
                        (0, 0, 0), (0, normalize_deg(esc_heading - delta_yaw), 0), args.uri)
                    esc_step = min(args.step_size, max(0.0, esc_clear - args.safety_margin))
                    # The agent now faces the escape heading, so a body-relative forward step
                    # travels along it - the sim rotates (0, 0, esc_step) into world space itself.
                    pos, rot, _ = step_agent((0, 0, esc_step), (0, 0, 0), args.uri)
                    total_escapes += 1
                    escaped = True
                    print(f"[explore] path blocked (clearance={clearance:.2f}m) toward waypoint "
                          f"{nav.target_world_xz}; wedged -> escaped {esc_step:.2f}m toward the open "
                          f"side (clearance {esc_clear:.2f}m, escape {total_escapes}/{args.max_escapes})"
                          f"{_format_clearance_debug(clearance_debug)}")
            if not escaped:
                print(f"[explore] path blocked (clearance={clearance:.2f}m) toward waypoint "
                      f"{nav.target_world_xz}; holding position and rescanning"
                      f"{_format_clearance_debug(clearance_debug)}")
        else:
            # The agent was already rotated above to face its travel direction - either
            # straight at the waypoint (normal) or the nudged heading that steered around a
            # close obstruction (nudge_deg != 0; see find_clear_heading). Either way "forward"
            # now points where we want to go, so a body-relative step is the same (0, 0,
            # step_len) in both cases: the sim rotates it into world space itself via
            # EgocentricToWorldTranslation (it must NOT be pre-rotated here, or the sim would
            # rotate it a second time). On the nudge branch the next iteration recomputes
            # delta_yaw fresh from the new position, re-correcting toward the actual waypoint
            # once past the obstruction.
            #
            # step_agent's `collided` return is intentionally discarded (`_`) - it comes from
            # the Rigidbody's Discrete collision detection with no root collider, which is known
            # unreliable (misses real contact; a false positive would inject a phantom obstacle
            # into the map). LiDAR (integrate() + swept_clearance_ahead) is the only source of
            # truth for both mapping and step-safety.
            pos, rot, _ = step_agent((0, 0, step_len), (0, 0, 0), args.uri)

        nudge_str = f" nudge={nudge_deg:+.1f}deg" if nudge_deg else ""
        print(
            f"[explore] step={step} pos=({pos[0]:.2f},{pos[2]:.2f}) yaw={rot[1]:.1f} "
            f"goal={nav.goal_world_xz} waypoint={nav.target_world_xz} replanned={nav.replanned} "
            f"clearance={clearance:.2f}m{nudge_str}{_format_clearance_debug(clearance_debug)}"
        )

        if step % args.save_every == 0:
            # interactive_backend must match the final save_snapshot() call's setting for
            # the whole run (see save_snapshot's docstring) - show_now stays False here,
            # only the final call at end-of-run actually pops a window.
            save_snapshot(grid, args.output_dir, step, interactive_backend=args.show_map)
            # include_ply=False: save_ply() re-serializes the WHOLE accumulated cloud with a
            # per-point Python write() loop every call (see PointCloudMap.save_ply), which
            # gets slower as the run goes on and was the actual source of the periodic
            # multi-second pauses at each --save-every boundary. The .npy (fast, vectorized)
            # still gets written each time; .ply is reserved for the one final export below.
            cloud.save(args.output_dir, step, include_ply=False)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Frontier-exploration mapping via the LiDAR scanner, using FrontierPlanner "
                     "(clustering + A* + goal commitment) instead of greedy nearest-cell picking."
    )
    parser.add_argument("--uri", default="ws://localhost:8080/commands")
    parser.add_argument("--size", type=float, default=60.0, help="Grid extent in meters (square)")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid cell size in meters")
    parser.add_argument("--step-size", type=float, default=0.5, help="Max meters moved per iteration")
    parser.add_argument(
        "--safety-margin", type=float, default=0.65,
        help="Meters of clearance to keep from any LiDAR-detected obstacle. This margin (driven "
             "entirely by swept_clearance_ahead's LiDAR reading, never by physical collision "
             "reporting) is the only real safety net - kept above the ~0.4m hand-proximity "
             "radius that activates item physics, plus extra cushion for scan/pose error.",
    )
    parser.add_argument("--min-step", type=float, default=0.1, help="Below this, skip the move and rescan instead")
    parser.add_argument(
        "--max-nudge-deg", type=float, default=30.0,
        help="When the straight-at-waypoint heading is blocked, try heading nudges up to this "
             "many degrees off it (smallest deviation first) before giving up on the waypoint "
             "entirely. Lets the agent sidestep a close obstruction detected off to one side "
             "instead of always stopping dead and waiting for a full replan. Set to 0 to disable "
             "and restore the old stop-or-go-straight-only behavior.",
    )
    parser.add_argument(
        "--nudge-step-deg", type=float, default=5.0,
        help="Degree increment between tried headings within --max-nudge-deg.",
    )
    parser.add_argument(
        "--escape-when-wedged", action=argparse.BooleanOptionalAction, default=True,
        help="When a step is blocked AND no --max-nudge-deg nudge around it clears either "
             "(the obstruction spans the whole forward arc - a shelf-corner pocket or aisle "
             "dead-end), turn to the most-open direction from a full 360deg clearance sweep "
             "and physically step out, instead of holding position and rescanning the identical "
             "pose until the planner's no-movement circuit breaker ends the run early (the "
             "recurring 'blocked path premature end'). The planner replans from the escaped "
             "cell (notify_blocked already forced it to NEED_GOAL). Disable to restore the old "
             "hold-and-wait-only behavior.",
    )
    parser.add_argument(
        "--max-escapes", type=int, default=15,
        help="Total wedge-escapes allowed per run before falling back to hold-and-wait (which "
             "then trips the no-movement circuit breaker and ends the run). Bounds a pathological "
             "approach<->escape oscillation - e.g. a frontier only reachable through the "
             "A*-vs-clearance dead band - so it terminates instead of thrashing to --max-steps. "
             "Healthy runs use only a few; raise for a large store with many genuine dead-end "
             "pockets.",
    )
    parser.add_argument(
        "--escape-sweep-step-deg", type=float, default=10.0,
        help="Degree increment for the full-circle clearance sweep that picks the escape heading "
             "(see --escape-when-wedged). Smaller = finer heading resolution at more scan cost.",
    )
    parser.add_argument(
        "--body-radius", type=float, default=0.3,
        help="Agent footprint radius (m). Used both to sweep the real-time clearance check (so "
             "an obstacle off-axis but within the agent's actual width still blocks the step) "
             "AND to inflate occupied cells before A* path planning (so the planner never "
             "confirms a route through a gap narrower than the agent can actually fit through - "
             "the two must agree, or the planner keeps re-confirming paths the real-time check "
             "correctly refuses). Tune to the sim's real shoulder/hand width - too small "
             "under-protects, too large starves frontier picking near narrow aisles.",
    )
    parser.add_argument(
        "--min-obstacle-height", type=float, default=0.05,
        help="Meters above the agent's root position; hits below this are treated as ordinary "
             "floor noise and ignored, not obstacles.",
    )
    parser.add_argument(
        "--max-obstacle-height", type=float, default=2.0,
        help="Meters above the agent's root position (matches the NavMesh-baked agentHeight); "
             "hits above this - e.g. overhanging ceiling signage - are physically out of the "
             "agent's reach and ignored, both for mapping and clearance.",
    )
    parser.add_argument(
        "--sensor-height-offset", type=float, default=SENSOR_HEIGHT_OFFSET_M,
        help="Meters the LiDAR sensor sits above the agent's reported root position (it tracks "
             "the live camera/head, not the root - see mapping.SENSOR_HEIGHT_OFFSET_M). Override "
             "if live validation shows the rig-derived default is off.",
    )
    parser.add_argument(
        "--self-exclusion-range", type=float, default=SELF_EXCLUSION_RANGE_M,
        help="Meters within which LiDAR hits are ignored outright, regardless of height. Kept "
             "small (default) deliberately - widening this blanket-excludes real close obstacles "
             "too, confirmed live to let the agent scrape an actual shelf undetected when this was "
             "briefly raised to 0.8m to mask a suspected self-hit (stowed hand/arm/shoulder). Only "
             "raise this once the self-hit is confirmed and distinguished from real obstacles more "
             "precisely (e.g. via --no-stow-hands comparison), not as a general-purpose margin.",
    )
    parser.add_argument(
        "--stow-hands", action=argparse.BooleanOptionalAction, default=True,
        help="Deactivate the hand prefabs (via the SetHandsActive command) for the whole run, "
             "since the clearance check only models the body and never accounts for hands "
             "swinging past shelf items while walking. Re-enabled on exit, including on error.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_MAPPING_DIR, "output"),
    )
    parser.add_argument(
        "--clear-output", action=argparse.BooleanOptionalAction, default=True,
        help="Delete every file already in --output-dir before this run starts, so the folder "
             "only ever holds this run's grid_*/points_* snapshots instead of accumulating "
             "leftovers (e.g. a higher step count) from a previous run. Disable to keep "
             "appending across runs instead.",
    )
    parser.add_argument(
        "--show-map", action=argparse.BooleanOptionalAction, default=False,
        help="Pop up a matplotlib window with the final occupancy grid when the run ends (in "
             "addition to always saving grid_final.png/.npy). Requires a working interactive "
             "matplotlib backend (e.g. TkAgg) - if none is available, this silently has no "
             "effect (falls back to Agg, same as --no-show-map). Disable for headless runs.",
    )

    planner_group = parser.add_argument_group("frontier planner")
    planner_group.add_argument(
        "--debug-planner", action=argparse.BooleanOptionalAction, default=False,
        help="Print timing/progress through FrontierPlanner._pick_and_plan()/_plan_to_cluster() "
             "on every replan: frontier/cluster counts, which cluster is being tried, and how "
             "long each astar() window-growth attempt takes. The pure-Python A* retry loop over "
             "many candidate clusters can run for a while with no other console output in "
             "between, which looks identical to a hang unless you can see it's still working - "
             "use this to tell the two apart and see exactly where time is going.",
    )
    planner_group.add_argument(
        "--min-cluster-size", type=int, default=4,
        help="Drop raw-frontier clusters smaller than this many cells before they're eligible as goals",
    )
    planner_group.add_argument(
        "--connectivity", type=int, default=8, choices=[4, 8],
        help="Neighbor connectivity for both frontier clustering and A* path search",
    )
    planner_group.add_argument(
        "--goal-arrival-radius", type=float, default=0.4,
        help="Meters to the committed goal counted as arrived, triggering a new goal pick",
    )
    planner_group.add_argument(
        "--waypoint-arrival-radius", type=float, default=0.25,
        help="Meters to the current path waypoint counted as reached, advancing to the next one",
    )
    planner_group.add_argument(
        "--replan-stuck-steps", type=int, default=1,
        help="Consecutive blocked steps before forcing a replan to a different goal cluster. "
             "Defaults to 1 (replan immediately) - while blocked the agent doesn't move or "
             "rotate, so retries rescan identical geometry from an identical pose and get an "
             "identical reading; waiting for a streak just burns scans with no new information "
             "in this static, deterministic sim. Raise only if scans are expected to be noisy "
             "or the environment can change between retries.",
    )
    planner_group.add_argument(
        "--replan-no-progress-steps", type=int, default=8,
        help="Consecutive steps without meaningful distance-to-goal improvement before replanning",
    )
    planner_group.add_argument(
        "--min-progress-m", type=float, default=0.15,
        help="Minimum meters of distance-to-goal improvement per step to not count as stagnant",
    )
    planner_group.add_argument(
        "--goal-blocklist-steps", type=int, default=40,
        help="Steps an abandoned goal cluster stays ineligible for re-selection",
    )
    planner_group.add_argument(
        "--window-slack", type=float, default=1.5,
        help="Multiplier on straight-line start-goal cell distance used to size the initial A* search window",
    )
    planner_group.add_argument(
        "--min-window-cells", type=int, default=100,
        help="Floor on the A* search window radius, in cells",
    )
    planner_group.add_argument(
        "--astar-max-window-cells", type=int, default=300,
        help="Ceiling on window growth before giving up and blocklisting a goal as unreachable "
             "(default 300 = the full 60m grid at 0.1m resolution)",
    )
    planner_group.add_argument(
        "--max-replans-without-moving", type=int, default=10,
        help="Circuit breaker: end the run if the planner replans this many times in a row "
             "from the exact same cell with no actual movement in between. Without this, a "
             "cluster that A* can reach on paper (through open/unknown space) but that keeps "
             "getting blocked in real time - e.g. by a self-hit not reflected as a static "
             "occupied cell in the grid - can get re-picked and re-blocked identically forever.",
    )

    topology_group = parser.add_argument_group("topology extraction")
    topology_group.add_argument(
        "--extract-topology", action=argparse.BooleanOptionalAction, default=True,
        help="After the run finishes, thin the resolved grid's free space to a skeleton and "
             "reduce it to a small checkpoint/adjacency graph (aisle ends, junctions, doorway "
             "pinch points) saved as topology_final.json - see topology.py. Purely derived "
             "from what was actually observed (free vs occupied), not the true Unity layout.",
    )
    topology_group.add_argument(
        "--topology-min-branch-length-m", type=float, default=DEFAULT_MIN_BRANCH_LENGTH_M,
        help="Drop dead-end skeleton spurs shorter than this - typically a thinning artifact "
             "from a jagged shelf corner rather than a real, walkable aisle stub.",
    )
    topology_group.add_argument(
        "--topology-doorway-max-width-m", type=float, default=DEFAULT_DOORWAY_MAX_WIDTH_M,
        help="A corridor's narrowest point must be below this width to ever qualify as a "
             "doorway checkpoint.",
    )
    topology_group.add_argument(
        "--topology-doorway-width-ratio", type=float, default=DEFAULT_DOORWAY_WIDTH_RATIO,
        help="A corridor's narrowest point must also be below this fraction of its own median "
             "width to qualify as a doorway - so a uniformly-narrow corridor doesn't get a "
             "spurious doorway checkpoint just for being narrow throughout.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
    # Hard exit: TransformAgent/RequestLidarScan each call asyncio.get_event_loop() per
    # call but never close it, and (with --show-map) the matplotlib GUI backend's own
    # event loop can leave a non-daemon thread alive - either can silently block normal
    # interpreter shutdown even though all real work (files saved, plot closed) is done.
    # All output is already written to disk above, so skipping normal cleanup here is safe.
    os._exit(0)
