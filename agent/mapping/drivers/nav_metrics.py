"""Identical instrumentation for both navigation arms, plus the coverage metrics that decide the
comparison.

The point of putting this in its own module, rather than inside the VLM planner, is that a metric
collected differently for each arm is not a metric - it is a second independent variable. So
InstrumentedPlanner wraps ANY object with the FrontierPlanner surface (update/notify_blocked) and
records the same fields the same way, whether the thing inside it is A* or a 27B VLM.

Nothing here modifies explore.py or frontier_planner.py. Both remain the frozen, working baseline
(CLAUDE.md: "Do not run explore.py casually"), and the comparison is assembled around them rather
than by editing them.
"""
import json
import math
import os
import time

import numpy as np


class InstrumentedPlanner:
    """Transparent wrapper recording what a planner did, at a per-step granularity.

    Delegates unknown attributes to the wrapped planner, so a VLM planner's own stats/decisions
    and a FrontierPlanner's internals both stay reachable through the wrapper. explore.py's loop
    only ever calls update()/notify_blocked(), so wrapping is invisible to it.
    """

    def __init__(self, planner, label):
        self.planner = planner
        self.label = label
        self.steps = []
        self.blocked = []
        self.n_updates = 0
        self.n_replans = 0
        self.n_blocked = 0
        self.planner_time_s = 0.0
        self.distance_travelled_m = 0.0
        self._last_pos = None
        self._t_start = time.time()

    def __getattr__(self, name):
        # Only called when normal lookup fails, so the wrapper's own attributes always win.
        return getattr(self.planner, name)

    def update(self, cur_world_xz, cur_cell):
        if self._last_pos is not None:
            self.distance_travelled_m += math.hypot(cur_world_xz[0] - self._last_pos[0],
                                                    cur_world_xz[1] - self._last_pos[1])
        self._last_pos = cur_world_xz

        t0 = time.perf_counter()
        nav = self.planner.update(cur_world_xz, cur_cell)
        dt = time.perf_counter() - t0

        self.n_updates += 1
        self.planner_time_s += dt
        if getattr(nav, "replanned", False):
            self.n_replans += 1
        self.steps.append({
            "i": self.n_updates,
            "pos": [round(cur_world_xz[0], 3), round(cur_world_xz[1], 3)],
            "kind": nav.kind,
            "target": list(nav.target_world_xz) if nav.target_world_xz else None,
            "goal": list(nav.goal_world_xz) if nav.goal_world_xz else None,
            "replanned": bool(getattr(nav, "replanned", False)),
            "planner_s": round(dt, 3),
            "dist_m": round(self.distance_travelled_m, 3),
        })
        return nav

    def notify_blocked(self, blocked_cell):
        self.n_blocked += 1
        self.blocked.append({"i": self.n_updates,
                             "at": [round(float(blocked_cell[0]), 3),
                                    round(float(blocked_cell[1]), 3)]})
        return self.planner.notify_blocked(blocked_cell)

    def summary(self):
        wall = time.time() - self._t_start
        out = {
            "label": self.label,
            "updates": self.n_updates,
            "replans": self.n_replans,
            "blocked_steps": self.n_blocked,
            "blocked_rate": round(self.n_blocked / max(1, self.n_updates), 4),
            "distance_travelled_m": round(self.distance_travelled_m, 2),
            "planner_time_s": round(self.planner_time_s, 1),
            "wall_time_s": round(wall, 1),
            "planner_time_frac": round(self.planner_time_s / max(1e-9, wall), 3),
        }
        vlm_stats = getattr(self.planner, "stats", None)
        if vlm_stats:
            s = dict(vlm_stats)
            calls = max(1, s.get("vlm_calls", 0))
            s["vlm_latency_s_mean"] = round(s.get("vlm_latency_s_total", 0.0) / calls, 2)
            # Rates are over waypoints the model actually PROPOSED (parsed, in-bounds decisions),
            # not over steps: coasted/held steps involve no proposal and would dilute the
            # denominator into meaninglessness.
            proposed = s.get("waypoint_proposals", 0)
            if proposed:
                # The headline. A* structurally cannot emit a waypoint whose straight line crosses
                # a recorded obstacle (simplify_path/_line_of_sight forbid it), so its value here
                # is 0.0 by construction and the VLM's is the whole finding.
                s["crosses_obstacle_rate"] = round(
                    s.get("waypoint_crosses_obstacle", 0) / proposed, 4)
                # The like-for-like number: A* is held to the inflated standard too.
                s["unexecutable_rate"] = round(
                    (s.get("waypoint_crosses_obstacle", 0) + s.get("waypoint_crosses_inflation", 0)
                     + s.get("waypoint_into_obstacle", 0) + s.get("waypoint_into_inflation", 0))
                    / proposed, 4)
                s["clean_rate"] = round(s.get("waypoint_ok", 0) / proposed, 4)
            out["vlm"] = s
        return out


def coverage_stats(grid):
    """Absolute map coverage. `resolved` = free|occupied, matching how every other consumer of the
    grid (save_snapshot's display, OccupancyGrid.frontiers' `unknown`) partitions it - a cell
    grazed by one ray sits at small nonzero log-odds and is NOT resolved, which is deliberate and
    documented in OccupancyGrid.frontiers.
    """
    free = grid.log_odds < grid.FREE_THRESHOLD
    occupied = grid.log_odds > grid.OCCUPIED_THRESHOLD
    resolved = free | occupied
    cell_m2 = grid.res ** 2
    return {
        "free_cells": int(free.sum()),
        "occupied_cells": int(occupied.sum()),
        "unknown_cells": int((~resolved).sum()),
        "resolved_cells": int(resolved.sum()),
        "free_m2": round(float(free.sum()) * cell_m2, 2),
        "resolved_m2": round(float(resolved.sum()) * cell_m2, 2),
    }


def coverage_vs_reference(grid, reference_log_odds):
    """Fraction of a trusted reference map that this run also resolved.

    Absolute cell counts are a weak comparison: the 60m grid is mostly empty padding around a much
    smaller store, so "resolved_cells" is dominated by how much void each run happened to sweep.
    This instead scores against the frozen map in mapping/output - the baseline CLAUDE.md already
    treats as good ("no grey holes in the store interior") - and asks the question that matters:
    of the store A* is known to have mapped, how much did this run map?

    recall_resolved is the headline; recall_free is the walkable area specifically, which is what
    downstream topology/shelf extraction actually consumes. precision_extra flags cells this run
    resolved that the reference did not - normally near-zero, and a large value means the two runs
    are not describing the same store (e.g. a different scene or a moved origin), i.e. the
    comparison is invalid rather than interesting.
    """
    if reference_log_odds.shape != grid.log_odds.shape:
        raise ValueError(f"reference grid {reference_log_odds.shape} != run grid "
                         f"{grid.log_odds.shape}; both must come from the same --size/--resolution")
    ref_free = reference_log_odds < grid.FREE_THRESHOLD
    ref_occ = reference_log_odds > grid.OCCUPIED_THRESHOLD
    ref_resolved = ref_free | ref_occ

    run_free = grid.log_odds < grid.FREE_THRESHOLD
    run_occ = grid.log_odds > grid.OCCUPIED_THRESHOLD
    run_resolved = run_free | run_occ

    n_ref = int(ref_resolved.sum())
    n_ref_free = int(ref_free.sum())
    return {
        "reference_resolved_cells": n_ref,
        "recall_resolved": round(float((ref_resolved & run_resolved).sum()) / max(1, n_ref), 4),
        "recall_free": round(float((ref_free & run_free).sum()) / max(1, n_ref_free), 4),
        "extra_resolved_cells": int((run_resolved & ~ref_resolved).sum()),
    }


def write_run_report(output_dir, tag, summary, instrumented):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"nav_report_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    trail = os.path.join(output_dir, f"nav_steps_{tag}.jsonl")
    with open(trail, "w", encoding="utf-8") as f:
        for s in instrumented.steps:
            f.write(json.dumps(s) + "\n")
    return path


def load_reference(path):
    return np.load(path)
