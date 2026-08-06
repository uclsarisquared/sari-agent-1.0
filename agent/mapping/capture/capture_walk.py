"""Drive to checkpoints on a frozen map, face each one's perpendicular, capture a screenshot.

This does two jobs at once:

1. **The Phase-3 pre-flight content check.** Before any VLM code is written, this answers the
   questions that decide whether Phase 3 is even viable: are the shelves actually stocked with
   distinguishable products? are the ceiling category signs legible at this distance? does a
   shelf FIT in frame from reading distance (the camera sits ~1.485m up and this only varies
   yaw, so a tall shelf may not fit floor-to-top)? is product behind fridge glass readable
   through glare? Cheap to run, and a "no" here saves building an annotator against unusable
   content.

2. **The reusable nav + face + capture spine of the Phase 3 annotator.** Walking to a
   checkpoint, turning to face its perpendicular, and grabbing the frame is exactly what the
   annotator does - it is just this plus a VLM call on the captured image. So this module is
   the Stage-1 scaffolding, built and validated without needing the model server.

Facing uses the bearing from a checkpoint's `cell` toward its `shelf_cell` - the perpendicular
Phase 2 already computed (see phase3_vlm_annotation_pass.md, "Perpendicular-first capture").
Only `kind="shelf"` checkpoints have a perpendicular; junction/end/doorway checkpoints are
structural skeleton points sitting in open space with no surface to face, so they're skipped
here by default (Phase 3 gives them a multi-angle non_shelf annotation instead).

Navigation reuses explore.py's proven executor pieces (A* over the body-radius-inflated mask,
swept-clearance-gated steps, the +-nudge, and the wedge escape) rather than reinventing them -
so the same safety properties hold, and shelf corners/dead ends behave the way explore does.

Requires the sim in Play mode. Requires NO model server. Reads a frozen map; never re-maps.

Captures are STANDING + CROUCHED, not yaw-swept: at each checkpoint the agent faces the shelf,
shoots level from standing height, crouches (halving the view height) and shoots level again, then
stands back up. Both views look at the SAME shelf face - they are not a "surroundings" set. This
exists because the camera sits at ~1.485m: at a 1.0m reading distance a tall shelf no longer fits
in one frame, and the bottom rows - real product - fall outside it. Crouching beats tilting down,
which sees those rows only from above (cans show their lids, labels foreshorten away).

    python mapping/capture/capture_walk.py mapping/output --limit 5
    python mapping/capture/capture_walk.py mapping/output --ids 42 43 --angles 2
"""
import argparse
import json
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/capture
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from sim.env import TransformAgent, SetHandsActive, SetCrouch, RequestScreenshot  # noqa: E402

from occupancy_grid import OccupancyGrid  # noqa: E402
from lidar_client import RequestLidarScan  # noqa: E402
from mapping import (  # noqa: E402
    normalize_deg, angle_to_deg, swept_clearance_ahead,
    SENSOR_HEIGHT_OFFSET_M, SELF_EXCLUSION_RANGE_M,
)
from frontier_planner import astar, simplify_path, _inflate_occupied  # noqa: E402
# Reuse explore.py's local steering rather than duplicating it: same nudge + wedge-escape
# semantics the mapping loop is already validated against.
from explore import find_clear_heading, find_escape_heading, step_agent  # noqa: E402


def load_grid(output_dir, tag, resolution):
    log_odds = np.load(os.path.join(output_dir, f"grid_{tag}.npy"))
    grid = OccupancyGrid(size_m=log_odds.shape[0] * resolution, resolution=resolution)
    grid.log_odds = log_odds
    return grid


def perpendicular_yaw(checkpoint):
    """World yaw that faces a shelf checkpoint's perpendicular: the bearing from its `cell`
    toward its `shelf_cell`. Cell deltas map to world deltas by a positive scale (see
    OccupancyGrid.to_world), so the bearing can be taken straight off the cell delta."""
    cx, cz = checkpoint["cell"]
    sx, sz = checkpoint["shelf_cell"]
    return angle_to_deg(sx - cx, sz - cz)


def goto(args, grid, inflated, target_world_xz, pos, rot):
    """Drive to target_world_xz with explore.py's executor semantics: replan A* each step
    (cheap, and robust to the agent ending up off-path), aim at the next real waypoint, and
    gate every step on swept LiDAR clearance with the same nudge -> escape fallbacks.

    Replans from scratch each step rather than committing to a path: this is a short hop to a
    known reachable point, not exploration, so there's no goal-commitment state worth keeping,
    and a fresh plan can't drift stale. Returns (pos, rot, reached)."""
    tx, tz = target_world_xz
    for _ in range(args.max_steps_per_leg):
        if math.hypot(pos[0] - tx, pos[2] - tz) <= args.arrival_radius:
            return pos, rot, True

        scan = RequestLidarScan(args.uri)
        path = astar(grid, grid.cell(pos[0], pos[2]), grid.cell(tx, tz),
                     connectivity=args.connectivity, occupied_mask=inflated)
        if path is None:
            return pos, rot, False
        waypoints = [grid.to_world(*c) for c in simplify_path(path, grid, occupied_mask=inflated)]
        # First waypoint we haven't effectively reached; fall back to the target itself.
        target = next(
            (w for w in waypoints[1:]
             if math.hypot(pos[0] - w[0], pos[2] - w[1]) > args.waypoint_arrival_radius),
            (tx, tz),
        )

        dx, dz = target[0] - pos[0], target[1] - pos[2]
        dist = math.hypot(dx, dz)
        delta_yaw = normalize_deg(angle_to_deg(dx, dz) - rot[1])
        clearance, _dbg = swept_clearance_ahead(
            scan, delta_yaw, body_radius=args.body_radius,
            min_obstacle_height=args.min_obstacle_height,
            max_obstacle_height=args.max_obstacle_height,
            sensor_height_offset=args.sensor_height_offset,
            self_exclusion_range=args.self_exclusion_range, debug=True,
        )
        step_len = min(args.step_size, dist, max(0.0, clearance - args.safety_margin))

        nudged_heading = None
        if step_len < args.min_step:
            nudged = find_clear_heading(
                scan, delta_yaw, args.min_step, args.safety_margin,
                body_radius=args.body_radius, min_obstacle_height=args.min_obstacle_height,
                max_obstacle_height=args.max_obstacle_height,
                sensor_height_offset=args.sensor_height_offset,
                self_exclusion_range=args.self_exclusion_range,
                max_nudge_deg=args.max_nudge_deg, nudge_step_deg=args.nudge_step_deg,
            )
            if nudged is not None:
                delta_yaw, clearance, _dbg, _off = nudged
                step_len = min(args.step_size, dist, max(0.0, clearance - args.safety_margin))
                nudged_heading = delta_yaw

        pos, rot, _ = step_agent((0, 0, 0), (0, delta_yaw, 0), args.uri)

        if step_len >= args.min_step:
            # The agent now faces its travel direction (straight at the waypoint, or the nudged
            # heading), so a body-relative forward step (0, 0, step_len) travels along it. The sim
            # applies translation EGOCENTRICALLY (EgocentricToWorldTranslation) and rotates it into
            # world space itself - it must NOT be pre-rotated here, or the sim rotates it twice
            # (the "forward isn't forward" de-sync; see explore.py's module docstring).
            pos, rot, _ = step_agent((0, 0, step_len), (0, 0, 0), args.uri)
            continue

        # Wedged: straight-ahead blocked and no nudge cleared. Same escape explore.py uses -
        # step toward the most-open heading so the next replan starts from an un-wedged cell.
        escape = find_escape_heading(
            scan, args.safety_margin, args.min_step,
            body_radius=args.body_radius, min_obstacle_height=args.min_obstacle_height,
            max_obstacle_height=args.max_obstacle_height,
            sensor_height_offset=args.sensor_height_offset,
            self_exclusion_range=args.self_exclusion_range, sweep_step_deg=args.escape_sweep_step_deg,
        )
        if escape is None:
            return pos, rot, False
        esc_heading, esc_clear, _ = escape
        pos, rot, _ = step_agent((0, 0, 0), (0, normalize_deg(esc_heading - delta_yaw), 0), args.uri)
        esc_step = min(args.step_size, max(0.0, esc_clear - args.safety_margin))
        # Facing the escape heading now, so step straight forward body-relative; the sim rotates
        # (0, 0, esc_step) into world space itself (do NOT pre-rotate - see the note above).
        pos, rot, _ = step_agent((0, 0, esc_step), (0, 0, 0), args.uri)

    return pos, rot, False


def face(args, pos, rot, world_yaw):
    """Rotate in place to an absolute world yaw (step_agent applies rotation as a delta)."""
    pos, rot, _ = step_agent((0, 0, 0), (0, normalize_deg(world_yaw - rot[1]), 0), args.uri)
    return pos, rot


PITCH_LEVEL_DEG = 0.0
DEFAULT_PITCH_STEP_DEG = 20.0


def pitch_to(args, rot, target_pitch_deg):
    """Tilt the camera to an ABSOLUTE pitch: 0 = level, POSITIVE = down, NEGATIVE = up.

    The sign convention is env.py's, not a choice made here - see `_TILT_DOWN_`/`_TILT_UP_`, which
    are TransformAgent((0,0,0), (+2.5,0,0)) and (-2.5,0,0). So rotation's X axis is pitch, applied
    as a delta exactly like yaw; no new websocket command is needed for any of this.

    Absolute rather than relative on purpose: pitching by a delta from wherever we happen to be
    lets rounding drift accumulate across a long walk. Reading the live pitch back and correcting
    to the target keeps every capture at the same angle, and makes returning to level exact."""
    delta = normalize_deg(target_pitch_deg - rot[0])
    pos, rot, _ = step_agent((0, 0, 0), (delta, 0, 0), args.uri)
    return pos, rot


def capture_plan(n_angles):
    """Which (label, crouched) captures to take, given --angles. 1 -> standing only.
    2 -> standing + crouched. The camera is LEVEL for both.

    Crouching rather than tilting down, measured: a down-pitched camera looks at the lower rows
    from above, so cans present their LIDS and the labels foreshorten away - it proves a row exists
    without letting you read it. Crouching halves the view height (Unity's CurrentLocalViewHeight)
    and keeps the camera level, so the same rows are seen face-on and their labels stay readable.

    There is no up shot: measured at cp050, a +20 deg up frame was ~65% ceiling tiles and its only
    product row was already covered by the standing view."""
    plan = [("primary", False)]
    if n_angles >= 2:
        plan.append(("crouch", True))
    return plan


def capture(args, out_path):
    """Grab the agent's current view. save_image=False so we own the filename/location rather
    than letting env.py drop it in its own screenshots folder.

    Capture-walk is the ONE exception to the pipeline's 1080p storage cap: it writes the sim's
    NATIVE full-res bytes verbatim, no downscale_for_storage. These PNGs are the annotator's VLM
    input, and legibility is bottlenecked on effective resolution after the encoder's own
    downscale - so every pixel the sim renders is worth keeping on disk here, even though it makes
    these the bulk of the pipeline's image storage. Everything else in the pipeline still caps at
    1080p."""
    result = RequestScreenshot(save_image=False, uri=args.uri)
    with open(out_path, "wb") as f:
        f.write(result["image"])
    return out_path


def select_checkpoints(topology, args):
    kinds = {k.strip() for k in args.kind.split(",")}
    cps = [c for c in topology["checkpoints"] if c.get("kind") in kinds]
    # A perpendicular is what makes a checkpoint capturable. Shelf AND landmark nodes both carry
    # shelf_cell (a landmark faces its structure the same way a shelf faces its face), so the same
    # facing code serves both; structural nodes have none and are dropped.
    cps = [c for c in cps if c.get("shelf_cell")]
    if args.ids:
        wanted = set(args.ids)
        cps = [c for c in cps if c["id"] in wanted]
    cps.sort(key=lambda c: c["id"])
    if args.limit:
        # Spread the sample across the graph instead of taking the first N (which would all
        # sit on one shelf and tell you nothing about the rest of the store).
        step = max(1, len(cps) // args.limit)
        cps = cps[::step][:args.limit]
    return cps


def run(args, on_captured=None, skip_ids=()):
    """Walk the selected checkpoints and capture each one.

    `on_captured(cp, shots)` is the fused capture+annotate hook (capture_annotate_walk.py): called
    on the walk thread right after a checkpoint's shots are all on disk AND the agent is back
    standing/level - so the sim is safe to keep moving no matter what the callback does. `shots`
    is the saved paths in capture_plan order (primary first). Not called for unreachable or
    shot-less checkpoints. `skip_ids` drops checkpoints from the walk entirely (the fused
    driver's --resume: a checkpoint already annotated needs no re-capture either)."""
    grid = load_grid(args.output_dir, args.grid_tag, args.resolution)
    with open(os.path.join(args.output_dir, f"topology_{args.topology_tag}.json")) as f:
        topology = json.load(f)
    inflated = _inflate_occupied(grid, args.body_radius)

    targets = select_checkpoints(topology, args)
    if skip_ids:
        n_all = len(targets)
        targets = [c for c in targets if c["id"] not in skip_ids]
        if len(targets) != n_all:
            print(f"[capture_walk] skipping {n_all - len(targets)} already-annotated checkpoint(s)")
    if not targets:
        print(f"[capture_walk] no '{args.kind}' checkpoints matched - nothing to do")
        return
    os.makedirs(args.capture_dir, exist_ok=True)

    pos, rot, _ = step_agent((0, 0, 0), (0, 0, 0), args.uri)
    if args.stow_hands:
        SetHandsActive(False, uri=args.uri)

    reached = failed = 0
    try:
        print(f"[capture_walk] walking {len(targets)} '{args.kind}' checkpoints -> {args.capture_dir}")
        for cp in targets:
            world = grid.to_world(*cp["cell"])
            pos, rot, ok = goto(args, grid, inflated, world, pos, rot)
            if not ok:
                failed += 1
                print(f"[capture_walk] id={cp['id']:3d} UNREACHABLE (gave up at "
                      f"({pos[0]:.2f},{pos[2]:.2f})) - skipping")
                continue
            reached += 1

            shots = []
            if cp.get("shelf_cell"):
                yaw = perpendicular_yaw(cp)
                pos, rot = face(args, pos, rot, yaw)
                try:
                    # Level the camera before shooting. Not paranoia: the agent was found sitting
                    # at 16.1 deg pitch from an earlier manual-control session, which would have
                    # silently tilted every "straight" capture of the run.
                    pos, rot = pitch_to(args, rot, PITCH_LEVEL_DEG)
                    # Both views look at THIS shelf from the same spot and yaw - one standing, one
                    # crouched. Neither is "surroundings"; both are item-bearing.
                    for label, crouched in capture_plan(args.angles):
                        SetCrouch(crouched, uri=args.uri)
                        shots.append(capture(args, os.path.join(
                            args.capture_dir, f"cp{cp['id']:03d}_{label}.png")))
                finally:
                    # Stand back up and re-level before ANY further movement, even if a capture
                    # threw. This is a safety property, not tidiness: goto() gates every step on
                    # swept_clearance_ahead, and a LiDAR scan taken while crouched or pitched
                    # reports the floor as an obstacle ahead - the executor would either wedge or,
                    # worse, clear a step that isn't actually clear.
                    SetCrouch(False, uri=args.uri)
                    pos, rot = pitch_to(args, rot, PITCH_LEVEL_DEG)

            print(f"[capture_walk] id={cp['id']:3d} at ({pos[0]:.2f},{pos[2]:.2f}) "
                  f"yaw={rot[1]:.0f} pitch={rot[0]:.1f} -> {len(shots)} shot(s)")
            if on_captured and shots:
                on_captured(cp, shots)
    finally:
        if args.stow_hands:
            SetHandsActive(True, uri=args.uri)
        print(f"[capture_walk] done: {reached} reached, {failed} unreachable -> {args.capture_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Walk a frozen map's checkpoints, face each perpendicular, capture screenshots."
    )
    parser.add_argument("output_dir", help="Directory holding grid_<tag>.npy and topology_<tag>.json")
    parser.add_argument("--uri", default="ws://localhost:8080/commands")
    parser.add_argument("--grid-tag", default="final")
    parser.add_argument("--topology-tag", default="final_shelf")
    parser.add_argument("--capture-dir", default=None,
                        help="Where to write PNGs (default: <output_dir>/captures)")
    parser.add_argument("--kind", default="shelf,landmark",
                        help="Comma-separated checkpoint kinds to walk (default: shelf,landmark - "
                             "the kinds that carry a perpendicular to face)")
    parser.add_argument("--ids", type=int, nargs="*", default=None, help="Only these checkpoint ids")
    parser.add_argument("--limit", type=int, default=5,
                        help="Sample this many checkpoints, spread across the graph (0 = all)")
    parser.add_argument("--angles", type=int, default=1, choices=[1, 2],
                        help="Captures per checkpoint, both level and facing the same shelf: "
                             "1 = standing only; 2 = standing + crouched. The crouched view halves "
                             "the camera height to see the lower rows face-on. "
                             "Files are cp<id>_primary.png / cp<id>_crouch.png")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--arrival-radius", type=float, default=0.3,
                        help="Meters from a checkpoint counted as arrived")
    parser.add_argument("--waypoint-arrival-radius", type=float, default=0.25)
    parser.add_argument("--max-steps-per-leg", type=int, default=80,
                        help="Give up on a checkpoint after this many steps")
    # Mirrors explore.py's executor knobs so navigation behaves identically.
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--safety-margin", type=float, default=0.65)
    parser.add_argument("--min-step", type=float, default=0.1)
    parser.add_argument("--body-radius", type=float, default=0.3)
    parser.add_argument("--max-nudge-deg", type=float, default=30.0)
    parser.add_argument("--nudge-step-deg", type=float, default=5.0)
    parser.add_argument("--escape-sweep-step-deg", type=float, default=10.0)
    parser.add_argument("--min-obstacle-height", type=float, default=0.05)
    parser.add_argument("--max-obstacle-height", type=float, default=2.0)
    parser.add_argument("--sensor-height-offset", type=float, default=SENSOR_HEIGHT_OFFSET_M)
    parser.add_argument("--self-exclusion-range", type=float, default=SELF_EXCLUSION_RANGE_M)
    parser.add_argument("--stow-hands", action=argparse.BooleanOptionalAction, default=True,
                        help="Stow the hand prefabs for the walk so they don't clip shelf items "
                             "or clutter the captured frame. Re-enabled on exit.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.capture_dir is None:
        args.capture_dir = os.path.join(args.output_dir, "captures")
    run(args)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
