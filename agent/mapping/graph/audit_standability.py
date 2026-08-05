"""Offline audit: which checkpoints on a frozen map can the agent actually navigate to?

topology.py's narrowness filter drops checkpoints whose clearance is below body_radius
(0.3m) - spots the agent's body physically can't occupy. But explore.py's executor is
stricter in practice: it refuses to move within `safety_margin + min_step` (0.65 + 0.1 =
0.75m) of an obstacle ahead. So a checkpoint sitting in the 0.3-0.75m band survives the
filter yet can be unreachable in the real loop - the same dead band behind the wedge/escape
investigation. Phase 3's annotator has to physically drive to every checkpoint, so it's worth
knowing which ones are at risk BEFORE it tries and stalls on them.

This is a pure offline post-process (no sim, no VLM): it re-derives the per-cell clearance
field from the saved grid and reports each checkpoint's clearance against the standoff.

    python mapping/graph/audit_standability.py mapping/output
    python mapping/graph/audit_standability.py mapping/output --topology-tag final_shelf

A flagged checkpoint is NOT automatically broken - clearance here is distance to the nearest
occupied cell in ANY direction, while the executor's real check is directional (forward,
along travel). So this is a conservative "worth a look" list, not a verdict: the useful
signals are the COUNT (a handful is normal near shelf faces; dozens means the graph is
threading places the agent can't go) and any checkpoint whose clearance is far below the
standoff.
"""
import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/graph
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from occupancy_grid import OccupancyGrid  # noqa: E402
from topology import _distance_to_occupied  # noqa: E402

DEFAULT_STANDOFF_M = 0.75
"""The clearance explore.py's executor effectively demands ahead of it before it will move at
all: --safety-margin (0.65) + --min-step (0.1). Checkpoints below this are the ones the
annotator's navigator may not be able to reach."""


def load_grid(output_dir, tag, resolution):
    log_odds = np.load(os.path.join(output_dir, f"grid_{tag}.npy"))
    grid = OccupancyGrid(size_m=log_odds.shape[0] * resolution, resolution=resolution)
    grid.log_odds = log_odds
    return grid


def audit(grid, checkpoints, standoff_m, connectivity=8):
    """Per-checkpoint clearance (distance to nearest occupied cell, in meters) against
    standoff_m. Returns a list of dicts sorted worst-clearance-first."""
    occupied = grid.log_odds > grid.OCCUPIED_THRESHOLD
    clearance = _distance_to_occupied(occupied, grid.res, connectivity)
    rows = []
    for c in checkpoints:
        cx, cz = int(c["cell"][0]), int(c["cell"][1])
        clr = float(clearance[cx, cz])
        rows.append({
            "id": c["id"], "kind": c.get("kind", "?"), "cell": (cx, cz),
            "clearance_m": clr, "at_risk": clr < standoff_m,
        })
    rows.sort(key=lambda r: r["clearance_m"])
    return rows


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit whether a frozen map's checkpoints are reachable by explore.py's executor."
    )
    parser.add_argument("output_dir", help="Directory holding grid_<tag>.npy and topology_<tag>.json")
    parser.add_argument("--grid-tag", default="final", help="Loads grid_<grid-tag>.npy (default: final)")
    parser.add_argument("--topology-tag", default="final_shelf",
                        help="Loads topology_<topology-tag>.json (default: final_shelf - the "
                             "shelf-enriched graph Phase 3 actually walks)")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--standoff-m", type=float, default=DEFAULT_STANDOFF_M,
                        help="Clearance the executor demands before moving (safety_margin + min_step)")
    parser.add_argument("--body-radius", type=float, default=0.3,
                        help="Reported as a second reference line - the narrowness filter's own threshold")
    parser.add_argument("--list-all", action="store_true", help="List every checkpoint, not just at-risk ones")
    return parser


def main():
    args = build_parser().parse_args()
    grid = load_grid(args.output_dir, args.grid_tag, args.resolution)
    topo_path = os.path.join(args.output_dir, f"topology_{args.topology_tag}.json")
    with open(topo_path) as f:
        topology = json.load(f)
    checkpoints = topology["checkpoints"]

    rows = audit(grid, checkpoints, args.standoff_m, args.connectivity)
    at_risk = [r for r in rows if r["at_risk"]]
    below_body = [r for r in rows if r["clearance_m"] < args.body_radius]

    by_kind = {}
    for r in rows:
        k = r["kind"]
        by_kind.setdefault(k, {"n": 0, "at_risk": 0})
        by_kind[k]["n"] += 1
        by_kind[k]["at_risk"] += 1 if r["at_risk"] else 0

    print(f"[audit] {topo_path}")
    print(f"[audit] grid_{args.grid_tag}.npy, standoff={args.standoff_m:.2f}m "
          f"(executor), body_radius={args.body_radius:.2f}m (narrowness filter)")
    print(f"[audit] {len(rows)} checkpoints; {len(at_risk)} below standoff; "
          f"{len(below_body)} below body_radius")
    print("[audit] by kind:")
    for k, v in sorted(by_kind.items()):
        print(f"    {k:9s} {v['n']:3d} total, {v['at_risk']:3d} below standoff")

    shown = rows if args.list_all else at_risk
    if shown:
        print(f"[audit] {'all checkpoints' if args.list_all else 'at-risk checkpoints'} "
              f"(worst clearance first):")
        for r in shown:
            flag = "  <-- BELOW body_radius" if r["clearance_m"] < args.body_radius else ""
            print(f"    id={r['id']:3d} {r['kind']:9s} cell={r['cell']} "
                  f"clearance={r['clearance_m']:.2f}m{flag}")
    else:
        print("[audit] no checkpoints below the standoff - every checkpoint looks reachable.")


if __name__ == "__main__":
    main()
