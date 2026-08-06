"""Frontier exploration driven by a VLM instead of A* - the measured control for explore.py.

    python mapping/drivers/explore_vlm.py --max-steps 20                     # VLM smoke test
    python mapping/drivers/explore_vlm.py --max-steps 300                    # full VLM run
    python mapping/drivers/explore_vlm.py --planner astar --max-steps 300    # the A* control arm

Requires the Unity scene in Play mode (same as explore.py) and the Qwen server reachable at
$OPENAI_API_URL/v1 (the configured endpoint root includes its port).

WHAT THIS IS FOR
----------------
CLAUDE.md's principle - "the graph owns spatial truth; the VLM only judges what is directly in
front of it" - is currently justified by an assertion (NavReasonPlan.md:5) rather than a
measurement. This turns it into a measurement. See vlm_planner.py's module docstring for the
experimental design and, in particular, for why the VLM here is given MORE information than A*
rather than less (a blind-VLM-vs-mapped-A* comparison would be confounded and prove nothing).

HOW THE COMPARISON IS KEPT HONEST - by construction, not by discipline
---------------------------------------------------------------------
This module does not re-implement explore.py's run loop; it IMPORTS it:

  * `build_parser()` is explore.py's own parser, so every physics/safety/mapping default
    (--step-size, --safety-margin, --body-radius, --min-obstacle-height, the lot) is not merely
    "the same value" but literally the same argparse definition. They cannot drift apart.
  * `_explore_loop()` is explore.py's own loop - the same scan integration, the same
    swept_clearance_ahead cap, the same nudge/wedge-escape logic, the same collided-flag
    discipline. It already takes the planner as a parameter, so no fork was needed.
  * The only difference between the arms is which object is passed as `planner`. That is the
    entire independent variable, and it is visible in one line of run() below.

`--planner astar` exists so the control arm is measured through this same instrumentation rather
than by parsing explore.py's stdout. It should reproduce explore.py's behaviour exactly; if it
ever does not, that is a bug in this harness, and it is worth running once as a sanity check
before trusting any VLM comparison.

WHERE RUNS ARE WRITTEN
----------------------
--output-dir defaults to mapping/output_vlm/<run-tag>/ - a PER-TAG subdirectory (tag defaults to
the --planner value), never mapping/output. Two reasons, both load-bearing:

  * Arms must not overwrite each other. Several outputs are UNTAGGED - grid_final.npy/.png,
    points_final.*, and every periodic grid_<step>.npy written by _explore_loop - so two arms
    sharing one directory silently clobber each other's maps, and --clear-output (default true)
    wipes the earlier arm's reports outright. A flat default would let this harness destroy its
    own control arm with no warning, which is the one failure that would quietly invalidate every
    number it produces.
  * The frozen baseline is protected. CLAUDE.md is explicit that output/ is the working baseline
    and that --clear-output renumbers every checkpoint, invalidating all downstream captures and
    annotations. run() hard refuses to clear it even if pointed there explicitly - a mistyped flag
    must not cost a re-map plus a re-capture plus a re-annotate.

An explicit --output-dir is honoured exactly as given (the per-tag default applies only when the
flag is omitted). Clearing only removes files directly inside the target, never subdirectories, so
sibling arm directories are safe from one another.
"""
import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/drivers
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from sim.env import SetHandsActive  # noqa: E402

# Imported, not reimplemented - this is what makes the two arms comparable. See the docstring.
from explore import (  # noqa: E402
    _clear_output_dir,
    _explore_loop,
    build_parser,
    save_snapshot,
    step_agent,
)
from frontier_planner import FrontierPlanner  # noqa: E402
from nav_metrics import (  # noqa: E402
    InstrumentedPlanner,
    coverage_stats,
    coverage_vs_reference,
    load_reference,
    write_run_report,
)
from pointcloud_map import PointCloudMap  # noqa: E402
from topology import extract_topology, save_topology  # noqa: E402
from vlm_planner import DEFAULT_MODEL, VLMAdvisedPlanner, VLMFrontierPlanner, VLMGoalPlanner  # noqa: E402
from voxel_grid import VoxelGrid  # noqa: E402

DEFAULT_OUTPUT_ROOT = os.path.join(_MAPPING_DIR, "output_vlm")
FROZEN_OUTPUT_DIR = os.path.join(_MAPPING_DIR, "output")


def build_vlm_parser():
    """explore.py's parser, plus the VLM knobs. Inheriting rather than redefining is deliberate:
    a hand-copied duplicate would let a default drift and silently turn a navigation comparison
    into a comparison of step sizes."""
    parser = build_parser()
    parser.description = ("Frontier exploration navigated by a VLM instead of A*. Shares "
                          "explore.py's parser and run loop; only the planner differs.")
    # None is a sentinel meaning "not chosen explicitly" - run() then resolves it to a PER-TAG
    # subdirectory. It cannot default to a fixed directory: the arms write several UNTAGGED files
    # (grid_final.npy/.png, points_final.*, and every periodic grid_<step>.npy from
    # _explore_loop), so two arms sharing one directory silently overwrite each other's maps, and
    # --clear-output (default true) additionally wipes the earlier arm's reports outright. The
    # A/B this whole harness exists for would destroy its own control, with no warning.
    parser.set_defaults(output_dir=None)
    for action in parser._actions:
        # argparse exposes no public way to amend an inherited action's help, and this flag's
        # behaviour genuinely differs from explore.py's now, so documenting it is worth the
        # private access - an undocumented silent default is what caused the problem above.
        if action.dest == "output_dir":
            action.help = (f"Where this run writes. Default: "
                           f"{DEFAULT_OUTPUT_ROOT}{os.sep}<run-tag> - a per-tag subdirectory, so "
                           f"arms never overwrite each other. An explicit path is used exactly "
                           f"as given.")
            break

    g = parser.add_argument_group("vlm navigation")
    g.add_argument(
        "--planner", choices=["vlm", "vlm-advised", "vlm-goal", "astar"], default="vlm",
        help="vlm: the model picks the frontier AND emits every waypoint (no A* at all) - the "
             "arm that actually tests the thesis claim. vlm-advised: same as vlm, but each ask "
             "carries A*'s own suggestion as an explicitly ADVISORY line - the most generous "
             "baseline constructible (map + coordinates + the answer), with agreement recorded "
             "so obedience and navigation stay distinguishable. vlm-goal: the model picks WHICH "
             "frontier, A* still computes HOW to get there - isolates goal choice from path "
             "planning, which is what tells you *where* the VLM fails rather than just *that* "
             "it does. astar: the control, identical to explore.py, run through this same "
             "instrumentation.",
    )
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--base-url", default=None, help="Default: $OPENAI_API_URL, +/v1")
    g.add_argument("--api-key", default=None,
                   help="Bearer for the qwen server (default: $OPENAI_API_KEY, then sari_env_old's "
                        "conda state). The server 401s without it - measured 2026-07-19.")
    g.add_argument(
        "--vlm-mode", choices=["both", "map", "ego"], default="both",
        help="What the model sees. both (default): top-down map image + first-person camera - "
             "strictly more than A* gets, which is the point. map: map image only. ego: camera "
             "only - the naive 'VLM navigates from first person' setup NavReasonPlan.md describes; "
             "useful as the lower rung of the ablation, NOT as the headline result, since on its "
             "own it re-introduces the 'no global view' confound.",
    )
    g.add_argument(
        "--ascii-map-res", type=float, default=0.5,
        help="Cell size (m) of the coarse TEXT occupancy map included in every prompt. 0 disables. "
             "On by default because it removes the strongest rebuttal to a negative result - that "
             "the model simply could not SEE the map after the vision encoder's downscale (the "
             "measured cause of the fridge annotation failure; see CLAUDE.md). Text is immune to "
             "that. Coarsening marks a cell '#' if ANY fine cell in it is occupied, which slightly "
             "over-reports walls - see render_ascii_map's docstring before tuning.",
    )
    g.add_argument(
        "--map-crop-m", type=float, default=None,
        help="Render the map image cropped to +/- this many meters around the robot. Off by "
             "default: cropping raises legibility but removes the global view, re-introducing the "
             "very confound this experiment exists to eliminate. An ablation knob, not a default.",
    )
    g.add_argument(
        "--vlm-replan-every", type=int, default=1,
        help="Steps between VLM calls; 1 (default) re-asks every step. 1 is both the most "
             "responsive setting for the model and the most expensive - capability is deliberately "
             "not traded away for cost here. Raise it to measure the cost/quality curve.",
    )
    g.add_argument("--vlm-think", action=argparse.BooleanOptionalAction, default=False,
                   help="Qwen3.x reasoning. OFF by default on measured grounds: a probe run spent "
                        "its whole budget thinking, looped, and never answered.")
    g.add_argument("--vlm-timeout", type=float, default=180.0)
    g.add_argument("--vlm-retries", type=int, default=2)
    g.add_argument("--vlm-max-tokens", type=int, default=1024)
    g.add_argument("--vlm-temperature", type=float, default=0.0)
    g.add_argument("--save-vlm-inputs", action=argparse.BooleanOptionalAction, default=False,
                   help="Save every image sent to the model into <output-dir>/vlm_inputs. Large, "
                        "but this is the qualitative evidence for the writeup: the exact frames "
                        "beside the model's own stated reasoning for what it did with them.")
    g.add_argument("--debug-vlm", action=argparse.BooleanOptionalAction, default=False,
                   help="Print every decision. Flagged (bad) waypoints always print regardless.")
    g.add_argument(
        "--reference-grid", default=os.path.join(FROZEN_OUTPUT_DIR, "grid_final.npy"),
        help="A trusted grid_final.npy to score coverage against. Defaults to the frozen map, "
             "which CLAUDE.md already treats as the known-good baseline. Absolute cell counts are "
             "a weak metric (the 60m grid is mostly padding); recall against this is the real one. "
             "Pass '' to skip.",
    )
    g.add_argument("--run-tag", default=None,
                   help="Names the report files. Defaults to the --planner value.")
    return parser


def build_planner(args, grid):
    """The entire independent variable of the experiment lives in this function."""
    if args.planner == "astar":
        return FrontierPlanner(
            grid,
            min_cluster_size=args.min_cluster_size, connectivity=args.connectivity,
            goal_arrival_radius=args.goal_arrival_radius,
            waypoint_arrival_radius=args.waypoint_arrival_radius,
            replan_stuck_steps=args.replan_stuck_steps,
            replan_no_progress_steps=args.replan_no_progress_steps,
            min_progress_m=args.min_progress_m, goal_blocklist_steps=args.goal_blocklist_steps,
            window_slack=args.window_slack, min_window_cells=args.min_window_cells,
            astar_max_window_cells=args.astar_max_window_cells,
            max_replans_without_moving=args.max_replans_without_moving,
            body_radius=args.body_radius, debug=args.debug_planner,
        )

    capture_dir = os.path.join(args.output_dir, "vlm_inputs") if args.save_vlm_inputs else None
    common = dict(
        uri=args.uri, base_url=args.base_url, model=args.model, api_key=args.api_key,
        timeout=args.vlm_timeout, retries=args.vlm_retries, mode=args.vlm_mode,
        think=args.vlm_think, max_tokens=args.vlm_max_tokens, temperature=args.vlm_temperature,
        ascii_map_res=args.ascii_map_res, map_crop_m=args.map_crop_m,
        capture_dir=capture_dir, debug=args.debug_vlm,
    )
    if args.planner == "vlm-goal":
        return VLMGoalPlanner(
            grid, **common,
            min_cluster_size=args.min_cluster_size, connectivity=args.connectivity,
            goal_arrival_radius=args.goal_arrival_radius,
            waypoint_arrival_radius=args.waypoint_arrival_radius,
            replan_stuck_steps=args.replan_stuck_steps,
            replan_no_progress_steps=args.replan_no_progress_steps,
            min_progress_m=args.min_progress_m, goal_blocklist_steps=args.goal_blocklist_steps,
            window_slack=args.window_slack, min_window_cells=args.min_window_cells,
            astar_max_window_cells=args.astar_max_window_cells,
            max_replans_without_moving=args.max_replans_without_moving,
            body_radius=args.body_radius, debug=args.debug_planner,
        )
    cls = VLMAdvisedPlanner if args.planner == "vlm-advised" else VLMFrontierPlanner
    return cls(
        grid, **common,
        min_cluster_size=args.min_cluster_size, connectivity=args.connectivity,
        body_radius=args.body_radius, goal_arrival_radius=args.goal_arrival_radius,
        replan_every=args.vlm_replan_every,
        max_replans_without_moving=args.max_replans_without_moving,
    )


def run(args):
    tag = args.run_tag or args.planner
    # Resolve the sentinel before anything reads args.output_dir (the frozen-dir guard below, the
    # clear, and build_planner's capture_dir all depend on it).
    if args.output_dir is None:
        args.output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, tag)

    # A mistyped --output-dir should not be able to cost a re-map + re-capture + re-annotate.
    # CLAUDE.md: output/ is the frozen baseline and --clear-output renumbers every checkpoint.
    if (args.clear_output and os.path.abspath(args.output_dir) == os.path.abspath(FROZEN_OUTPUT_DIR)):
        sys.exit(f"refusing to --clear-output the frozen baseline map at {FROZEN_OUTPUT_DIR}.\n"
                 f"That directory is the working baseline (see CLAUDE.md): wiping it renumbers "
                 f"every checkpoint and invalidates all captures/annotations keyed to them.\n"
                 f"Use a different --output-dir (default: {DEFAULT_OUTPUT_ROOT}{os.sep}<run-tag>), "
                 f"or pass --no-clear-output if you really mean to write alongside it.")
    if args.clear_output:
        _clear_output_dir(args.output_dir)

    voxel = VoxelGrid(
        size_m=args.size, resolution=args.resolution,
        min_obstacle_height=args.min_obstacle_height,
        max_obstacle_height=args.max_obstacle_height,
        sensor_height_offset=args.sensor_height_offset,
    )
    grid = voxel.grid
    cloud = PointCloudMap()

    planner = InstrumentedPlanner(build_planner(args, grid), label=tag)
    print(f"[explore-vlm] planner={args.planner} mode={args.vlm_mode} model={args.model} "
          f"ascii_map_res={args.ascii_map_res} -> {args.output_dir}")

    pos, rot, _ = step_agent((0, 0, 0), (0, 0, 0), args.uri)

    if args.stow_hands:
        SetHandsActive(False, uri=args.uri)
    try:
        # explore.py's loop, unmodified, with only `planner` differing between arms.
        _explore_loop(args, voxel, grid, cloud, pos, rot, planner)
    finally:
        if args.stow_hands:
            SetHandsActive(True, uri=args.uri)

    save_snapshot(grid, args.output_dir, "final", interactive_backend=args.show_map,
                  show_now=args.show_map)
    cloud.save(args.output_dir, "final")

    summary = planner.summary()
    summary["config"] = {
        "planner": args.planner, "model": args.model, "vlm_mode": args.vlm_mode,
        "ascii_map_res": args.ascii_map_res, "map_crop_m": args.map_crop_m,
        "vlm_replan_every": args.vlm_replan_every, "vlm_think": args.vlm_think,
        "max_steps": args.max_steps, "step_size": args.step_size,
        "safety_margin": args.safety_margin, "body_radius": args.body_radius,
    }
    summary["coverage"] = coverage_stats(grid)

    if args.reference_grid and os.path.isfile(args.reference_grid):
        try:
            summary["coverage_vs_reference"] = coverage_vs_reference(
                grid, load_reference(args.reference_grid))
            summary["coverage_vs_reference"]["reference"] = args.reference_grid
        except ValueError as e:
            # Shape mismatch means the reference was built at a different --size/--resolution, so
            # the comparison would be meaningless. Say so instead of emitting a wrong number.
            summary["coverage_vs_reference"] = {"error": str(e)}
    elif args.reference_grid:
        summary["coverage_vs_reference"] = {"error": f"not found: {args.reference_grid}"}

    report = write_run_report(args.output_dir, tag, summary, planner)
    if hasattr(planner.planner, "write_report"):
        decisions = planner.planner.write_report(args.output_dir, tag)
        print(f"[explore-vlm] wrote {len(planner.planner.decisions)} VLM decisions to {decisions}")

    if args.extract_topology:
        topology = extract_topology(
            grid, connectivity=args.connectivity,
            min_branch_length_m=args.topology_min_branch_length_m,
            doorway_max_width_m=args.topology_doorway_max_width_m,
            doorway_width_ratio=args.topology_doorway_width_ratio,
            min_checkpoint_clearance_m=args.body_radius,
        )
        path = save_topology(topology, args.output_dir, tag)
        print(f"[explore-vlm] saved topology ({len(topology.checkpoints)} checkpoints, "
              f"{len(topology.edges)} edges) to {path}")

    print(f"[explore-vlm] saved final grid + point cloud to {args.output_dir}")
    print(f"[explore-vlm] report -> {report}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run(build_vlm_parser().parse_args())
    # Same hard exit as explore.py, for the same measured reason: env.py's TransformAgent /
    # RequestLidarScan call asyncio.get_event_loop() per call and never close it, which can block
    # normal interpreter shutdown long after all output is safely on disk.
    os._exit(0)
