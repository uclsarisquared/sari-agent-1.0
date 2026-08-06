"""
Phase 2 CLI: run splice_shelf_checkpoints() against a grid already saved by explore.py,
producing an enriched topology with shelf-hugging checkpoints and ladder-rung adjacency,
plus a PNG so the result can actually be eyeballed. See
mapping/plans/phase2_shelf_hugging_coverage.md for the design this implements.

Recomputes the topology graph fresh from grid_<tag>.npy via extract_topology() rather
than loading the saved topology_<tag>.json back - that JSON intentionally omits each
edge's raw skeleton path (in-memory-only data, see topology.TopologyEdge), which
splice_shelf_checkpoints() needs to do anything. Re-running extract_topology() is cheap
(a one-time post-process, not a hot loop) and deterministic, so it reproduces the exact
same checkpoint/edge structure already in the saved JSON, just with `path` populated
too - checked explicitly below rather than assumed, since a mismatch would mean this run
used different --topology-* flags than explore.py's own defaults.

No live Unity connection needed - this is pure offline post-processing over files
explore.py already wrote.

Run from anywhere:
    python mapping/graph/build_shelf_graph.py mapping/output --tag final
"""
import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/graph
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from occupancy_grid import OccupancyGrid  # noqa: E402
from topology import (  # noqa: E402
    extract_topology, save_topology,
    DEFAULT_MIN_BRANCH_LENGTH_M, DEFAULT_DOORWAY_MAX_WIDTH_M, DEFAULT_DOORWAY_WIDTH_RATIO,
)
from shelf_coverage import (  # noqa: E402
    splice_shelf_checkpoints, splice_landmark_checkpoints,
    DEFAULT_CHECKPOINT_INTERVAL_M, DEFAULT_SHELF_SEARCH_RADIUS_M,
    MIN_READING_DISTANCE_M, MAX_READING_DISTANCE_M,
)

_KIND_COLORS = {"junction": "red", "end": "dodgerblue", "doorway": "limegreen",
                "landmark": "gold"}
_SHELF_SIDE_COLORS = {"left": "orange", "right": "magenta"}


def _load_grid(output_dir, tag, resolution):
    log_odds = np.load(os.path.join(output_dir, f"grid_{tag}.npy"))
    grid = OccupancyGrid(size_m=log_odds.shape[0] * resolution, resolution=resolution)
    grid.log_odds = log_odds
    return grid


def _warn_if_topology_mismatch(output_dir, tag, graph):
    """Best-effort sanity check: if explore.py already saved topology_<tag>.json for this
    same grid, the freshly re-extracted graph should match it exactly. A mismatch doesn't
    stop the run (the fresh extraction is still used - it's the one with `path`
    populated), it just means this script's topology-extraction flags didn't match
    whatever explore.py was actually invoked with, which is worth knowing."""
    path = os.path.join(output_dir, f"topology_{tag}.json")
    if not os.path.exists(path):
        return
    import json
    with open(path) as f:
        saved = json.load(f)
    saved_kinds, fresh_kinds = {}, {}
    for c in saved["checkpoints"]:
        saved_kinds[c["kind"]] = saved_kinds.get(c["kind"], 0) + 1
    for c in graph.checkpoints:
        fresh_kinds[c.kind] = fresh_kinds.get(c.kind, 0) + 1
    if len(saved["checkpoints"]) != len(graph.checkpoints) or len(saved["edges"]) != len(graph.edges) \
            or saved_kinds != fresh_kinds:
        print(f"[build_shelf_graph] WARNING: freshly re-extracted topology does not match "
              f"{path} (saved: {len(saved['checkpoints'])} checkpoints {saved_kinds}, "
              f"{len(saved['edges'])} edges vs. fresh: {len(graph.checkpoints)} checkpoints "
              f"{fresh_kinds}, {len(graph.edges)} edges) - this script's --topology-* flags "
              f"probably don't match what explore.py was run with; pass matching flags for "
              f"an apples-to-apples result.")
    else:
        print(f"[build_shelf_graph] fresh topology matches {path} - good, same extraction params.")


def _plot(grid, graph, output_dir, out_tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[build_shelf_graph] matplotlib not available, skipping PNG")
        return None

    display = np.full(grid.log_odds.shape, 0.5)
    display[grid.log_odds < grid.FREE_THRESHOLD] = 1.0
    display[grid.log_odds > grid.OCCUPIED_THRESHOLD] = 0.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(display.T, origin="lower", cmap="gray", vmin=0, vmax=1)

    # Crop to the explored region (plus a margin) instead of the full grid extent - most
    # of a 60m grid is still "unknown" gray on any run that hasn't mapped the whole store,
    # and a plot that's 90% empty makes the actual checkpoint graph unreadable.
    known_cells = np.argwhere(display.T != 0.5)
    if len(known_cells) > 0:
        margin = int(round(1.0 / grid.res))  # 1m padding
        y0, x0 = known_cells.min(axis=0) - margin
        y1, x1 = known_cells.max(axis=0) + margin
        ax.set_xlim(max(0, x0), min(grid.log_odds.shape[0], x1))
        ax.set_ylim(max(0, y0), min(grid.log_odds.shape[1], y1))

    by_id = {c.id: c for c in graph.checkpoints}
    for e in graph.edges:
        a, b = by_id[e.a], by_id[e.b]
        is_rung = a.kind == "shelf" and b.kind == "shelf" and a.shelf_side != b.shelf_side
        style = dict(color="cyan", linewidth=0.8, linestyle="--", alpha=0.7) if is_rung \
            else dict(color="yellow", linewidth=0.8, alpha=0.6)
        ax.plot([a.cell[0], b.cell[0]], [a.cell[1], b.cell[1]], **style, zorder=1)

    by_kind = {}
    for c in graph.checkpoints:
        key = c.kind if c.kind != "shelf" else f"shelf_{c.shelf_side}"
        by_kind.setdefault(key, []).append(c)

    for kind, kind_checkpoints in by_kind.items():
        xs = [c.cell[0] for c in kind_checkpoints]
        ys = [c.cell[1] for c in kind_checkpoints]
        if kind.startswith("shelf_"):
            color = _SHELF_SIDE_COLORS[kind.split("_", 1)[1]]
            ax.scatter(xs, ys, c=color, s=6, label=kind, zorder=2)
        else:
            ax.scatter(xs, ys, c=_KIND_COLORS[kind], s=30, edgecolors="black", linewidths=0.5,
                       label=kind, zorder=3)

    ax.set_title(f"Shelf-hugging checkpoint graph ({out_tag})")
    ax.legend(loc="upper right", fontsize=7, markerscale=1.5)
    out_path = os.path.join(output_dir, f"shelf_graph_{out_tag}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phase 2: extend a saved Phase 1 topology with shelf-hugging checkpoints."
    )
    parser.add_argument("output_dir", help="Directory containing grid_<tag>.npy (from explore.py)")
    parser.add_argument("--tag", default="final", help="Input tag, e.g. grid_<tag>.npy (default: final)")
    parser.add_argument("--out-tag", default=None, help="Output tag (default: <tag>_shelf)")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--min-branch-length-m", type=float, default=DEFAULT_MIN_BRANCH_LENGTH_M)
    parser.add_argument("--doorway-max-width-m", type=float, default=DEFAULT_DOORWAY_MAX_WIDTH_M)
    parser.add_argument("--doorway-width-ratio", type=float, default=DEFAULT_DOORWAY_WIDTH_RATIO)
    parser.add_argument("--checkpoint-interval-m", type=float, default=DEFAULT_CHECKPOINT_INTERVAL_M)
    parser.add_argument("--search-radius-m", type=float, default=DEFAULT_SHELF_SEARCH_RADIUS_M)
    parser.add_argument("--min-reading-distance-m", type=float, default=MIN_READING_DISTANCE_M)
    parser.add_argument("--max-reading-distance-m", type=float, default=MAX_READING_DISTANCE_M)
    parser.add_argument("--body-radius", type=float, default=0.3)
    parser.add_argument("--no-ladder-rungs", action="store_true", help="Disable cross-aisle rung adjacency")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Skip the landmark pass (unobserved free-standing structures such as the checkout counter)")
    parser.add_argument("--no-plot", action="store_true", help="Skip the PNG visualization")
    return parser


def main():
    args = build_parser().parse_args()
    out_tag = args.out_tag or f"{args.tag}_shelf"

    grid = _load_grid(args.output_dir, args.tag, args.resolution)
    graph = extract_topology(
        grid, connectivity=args.connectivity, min_branch_length_m=args.min_branch_length_m,
        doorway_max_width_m=args.doorway_max_width_m, doorway_width_ratio=args.doorway_width_ratio,
        # Match explore.py: drop checkpoints inside sub-body-width shelf gaps so this
        # re-extraction reproduces the same graph explore.py saved (see the mismatch warning).
        min_checkpoint_clearance_m=args.body_radius,
    )
    _warn_if_topology_mismatch(args.output_dir, args.tag, graph)
    base_checkpoints, base_edges = len(graph.checkpoints), len(graph.edges)

    splice_shelf_checkpoints(
        grid, graph,
        interval_m=args.checkpoint_interval_m, search_radius_m=args.search_radius_m,
        min_reading_distance_m=args.min_reading_distance_m, max_reading_distance_m=args.max_reading_distance_m,
        body_radius=args.body_radius, add_ladder_rungs=not args.no_ladder_rungs,
    )

    landmarks = []
    if not args.no_landmarks:
        landmarks = splice_landmark_checkpoints(grid, graph, body_radius=args.body_radius)
        for cp in landmarks:
            print(f"[build_shelf_graph] landmark cp{cp.id} at {cp.cell} facing {cp.shelf_cell} "
                  f"({cp.reading_distance_m:.2f}m) -> anchored to cp{cp.neighbors[0]}")

    out_path = save_topology(graph, args.output_dir, out_tag)

    shelf_checkpoints = [c for c in graph.checkpoints if c.kind == "shelf"]
    by_side = {"left": 0, "right": 0}
    for c in shelf_checkpoints:
        by_side[c.shelf_side] += 1
    by_id = {c.id: c for c in graph.checkpoints}
    rung_edges = sum(
        1 for e in graph.edges
        if by_id[e.a].kind == "shelf" and by_id[e.b].kind == "shelf"
        and by_id[e.a].shelf_side != by_id[e.b].shelf_side
    )
    print(f"[build_shelf_graph] base topology: {base_checkpoints} checkpoints, {base_edges} edges")
    print(f"[build_shelf_graph] +{len(shelf_checkpoints)} shelf checkpoints "
          f"(left={by_side['left']}, right={by_side['right']}), {rung_edges} ladder rungs")
    print(f"[build_shelf_graph] final: {len(graph.checkpoints)} checkpoints, {len(graph.edges)} edges "
          f"-> {out_path}")

    if not args.no_plot:
        png_path = _plot(grid, graph, args.output_dir, out_tag)
        if png_path:
            print(f"[build_shelf_graph] saved visualization -> {png_path}")


if __name__ == "__main__":
    main()
