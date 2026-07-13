"""
Frontier clustering, A* path planning, and a goal-commitment state machine for
slamtest_frontier/explore.py.

Replaces slamtest/explore.py's old pick_frontier(): greedy nearest-raw-cell
picking with no clustering or commitment, which (combined with the separately
fixed OccupancyGrid.frontiers() np.roll wraparound bug) caused the agent to
circle near the origin instead of exploring outward. This module never reads
the Rigidbody isColliding/collided signal (known unreliable) - all obstacle
awareness here comes from the LiDAR-built OccupancyGrid passed in by the
caller.
"""
import os
import sys
import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # overhaul/slamtest/slamtest_frontier
_SLAMTEST_DIR = os.path.dirname(_THIS_DIR)                     # overhaul/slamtest
_OVERHAUL_DIR = os.path.dirname(_SLAMTEST_DIR)                 # overhaul
for _p in (_OVERHAUL_DIR, _SLAMTEST_DIR, _THIS_DIR):
    # Idempotent: slamtest/ and slamtest_frontier/ both happen to contain a file named
    # explore.py. If this insert re-ran unconditionally every time this module loads, a
    # redundant re-insert (e.g. triggered by explore.py importing this module) would shove
    # slamtest/ back ahead of slamtest_frontier/ in sys.path - meaning a bare `import explore`
    # anywhere would silently resolve to the OLD explore.py instead of the new one. Skipping
    # already-present entries preserves whichever script bootstrapped first.
    if _p not in sys.path:
        sys.path.insert(0, _p)

from occupancy_grid import _bresenham_line  # noqa: E402


CONNECTIVITY_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CONNECTIVITY_8 = CONNECTIVITY_4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _neighbor_offsets(connectivity):
    if connectivity == 4:
        return CONNECTIVITY_4
    if connectivity == 8:
        return CONNECTIVITY_8
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


@dataclass
class FrontierCluster:
    cells: np.ndarray      # Kx2 int array of member cells
    centroid_cell: tuple   # (cx, cz), snapped to a real member cell
    size: int


def cluster_frontiers(frontier_cells, connectivity=8, min_cluster_size=4):
    """Group raw frontier cells (as returned by OccupancyGrid.frontiers())
    into connected clusters via BFS flood-fill. Clusters smaller than
    min_cluster_size are dropped as sensor noise. Returns clusters sorted by
    size descending.
    """
    offsets = _neighbor_offsets(connectivity)
    remaining = {(int(c[0]), int(c[1])) for c in frontier_cells}
    clusters = []

    while remaining:
        start = next(iter(remaining))
        remaining.discard(start)
        members = [start]
        queue = deque([start])
        while queue:
            cx, cz = queue.popleft()
            for dx, dz in offsets:
                nb = (cx + dx, cz + dz)
                if nb in remaining:
                    remaining.discard(nb)
                    members.append(nb)
                    queue.append(nb)

        if len(members) < min_cluster_size:
            continue

        arr = np.array(members, dtype=int)
        mean = arr.mean(axis=0)
        # A concave/arc-shaped cluster's raw mean can land outside the cluster
        # (e.g. off in occupied/free-interior space) - snap to the actual
        # member cell nearest the mean instead.
        dists = np.hypot(arr[:, 0] - mean[0], arr[:, 1] - mean[1])
        centroid_cell = tuple(int(v) for v in arr[np.argmin(dists)])
        clusters.append(FrontierCluster(cells=arr, centroid_cell=centroid_cell, size=len(members)))

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def score_cluster(cluster, cur_cell, blocklist=None, size_weight=2.0, size_cap=200, revisit_penalty=1000.0):
    """Lower is better: distance minus a size bonus (bigger unexplored region
    = more valuable), plus a heavy penalty if this goal is still blocklisted
    (just abandoned - avoid immediately re-picking it)."""
    dist = math.hypot(cluster.centroid_cell[0] - cur_cell[0], cluster.centroid_cell[1] - cur_cell[1])
    penalty = revisit_penalty if blocklist and cluster.centroid_cell in blocklist else 0.0
    return dist - size_weight * min(cluster.size, size_cap) + penalty


def _octile_heuristic(a, b):
    dx, dz = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dz) + (math.sqrt(2) - 1) * min(dx, dz)


def _inflate_occupied(grid, body_radius):
    """Boolean n x n array: True wherever a cell is occupied, or within
    body_radius meters of an occupied cell - i.e. where the agent's actual
    body (not just its center point) could not fit, not just the single
    occupied cell itself. Padded (not wrapped) at the grid boundary, matching
    the frontiers() wraparound fix.

    Without this, plain cell-occupancy checks let A* (and simplify_path's
    line-of-sight shortcuts) confirm a path through a gap that's technically
    free cell-by-cell but narrower than the agent's real footprint - confirmed
    live: the planner kept re-confirming a path through a shelf-corner gap
    that real-time swept_clearance_ahead correctly refused to let the agent
    actually pass, producing a plan-block-replan loop stuck at the same spot.
    """
    occupied = grid.log_odds > grid.OCCUPIED_THRESHOLD
    if not body_radius:
        return occupied
    r_cells = int(math.ceil(body_radius / grid.res))
    if r_cells <= 0:
        return occupied
    padded = np.pad(occupied, r_cells, mode="constant", constant_values=False)
    inflated = np.zeros_like(occupied)
    for dx in range(-r_cells, r_cells + 1):
        for dz in range(-r_cells, r_cells + 1):
            if dx * dx + dz * dz > r_cells * r_cells:
                continue
            inflated |= padded[r_cells + dx: r_cells + dx + grid.n, r_cells + dz: r_cells + dz + grid.n]
    return inflated


def _passable(occupied_mask, grid, cell):
    cx, cz = cell
    if not grid.in_bounds(cx, cz):
        return False
    # Free and unknown cells are both passable - unknown is passable because
    # routing toward/through unexplored space near a frontier goal is the
    # point. swept_clearance_ahead() (LiDAR, checked every physical step by
    # the caller) is the real-time safety net; occupied_mask (optionally
    # inflated by body_radius - see _inflate_occupied) is what keeps the
    # PLANNER from confirming paths the body can't actually fit through.
    return not occupied_mask[cx, cz]


def astar(grid, start_cell, goal_cell, connectivity=8, window_radius=None, occupied_mask=None):
    """8-connected (or 4-connected) A* over the occupancy grid. Cost 1.0 for
    cardinal moves, sqrt(2) for diagonal. Returns a list of cells from start
    to goal inclusive, or None if unreachable (or goal_cell falls outside
    window_radius cells of the start/goal bounding box, if given).

    occupied_mask defaults to plain per-cell occupancy if not given - pass an
    _inflate_occupied() result to make routing body-radius-aware.
    """
    if occupied_mask is None:
        occupied_mask = grid.log_odds > grid.OCCUPIED_THRESHOLD
    if window_radius is not None:
        cx_min = min(start_cell[0], goal_cell[0]) - window_radius
        cx_max = max(start_cell[0], goal_cell[0]) + window_radius
        cz_min = min(start_cell[1], goal_cell[1]) - window_radius
        cz_max = max(start_cell[1], goal_cell[1]) + window_radius

        def in_window(cell):
            return cx_min <= cell[0] <= cx_max and cz_min <= cell[1] <= cz_max
    else:
        def in_window(cell):
            return True

    if not in_window(goal_cell):
        return None

    offsets = _neighbor_offsets(connectivity)
    open_heap = []
    counter = 0
    g_score = {start_cell: 0.0}
    came_from = {}
    closed = set()

    heapq.heappush(open_heap, (_octile_heuristic(start_cell, goal_cell), counter, start_cell))

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)

        for dx, dz in offsets:
            neighbor = (current[0] + dx, current[1] + dz)
            if neighbor in closed or not in_window(neighbor):
                continue
            if not _passable(occupied_mask, grid, neighbor):
                continue
            step_cost = math.sqrt(2) if dx != 0 and dz != 0 else 1.0
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                counter += 1
                f = tentative_g + _octile_heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (f, counter, neighbor))

    return None


def _line_of_sight(grid, a, b, occupied_mask=None):
    if occupied_mask is None:
        occupied_mask = grid.log_odds > grid.OCCUPIED_THRESHOLD
    for cx, cz in _bresenham_line(a[0], a[1], b[0], b[1]):
        if not _passable(occupied_mask, grid, (cx, cz)):
            return False
    return True


def simplify_path(path_cells, grid, occupied_mask=None):
    """Line-of-sight waypoint pruning ("string pulling"): collapse a raw
    per-cell A* path into a shorter list of true direction-change waypoints,
    so the local controller isn't re-issuing a rotate+step for every single
    grid cell.

    occupied_mask defaults to plain per-cell occupancy if not given - pass the
    SAME (optionally body-radius-inflated) mask used for the astar() call that
    produced path_cells, so a shortcut never cuts through a gap the agent's
    actual body couldn't fit through even though the raw A* path avoided it.
    """
    if occupied_mask is None:
        occupied_mask = grid.log_odds > grid.OCCUPIED_THRESHOLD
    path_cells = list(path_cells)
    if len(path_cells) <= 2:
        return path_cells

    simplified = [path_cells[0]]
    anchor_idx = 0
    for i in range(1, len(path_cells)):
        if not _line_of_sight(grid, path_cells[anchor_idx], path_cells[i], occupied_mask):
            simplified.append(path_cells[i - 1])
            anchor_idx = i - 1
    if simplified[-1] != path_cells[-1]:
        simplified.append(path_cells[-1])
    return simplified


class PlannerState(Enum):
    NEED_GOAL = auto()
    FOLLOWING = auto()
    DONE = auto()


@dataclass
class NavCommand:
    kind: str                         # "goto_waypoint" | "done"
    target_world_xz: tuple = None
    goal_world_xz: tuple = None
    replanned: bool = False


class FrontierPlanner:
    """Stateful orchestrator: clusters frontiers, commits to a goal, plans an
    A* path to it, and hands explore.py one waypoint at a time. Call update()
    once per step after integrating the latest scan; call notify_blocked()
    when a step was capped near-zero by swept_clearance_ahead().
    """

    def __init__(self, grid, *, min_cluster_size=4, connectivity=8,
                 goal_arrival_radius=0.4, waypoint_arrival_radius=0.25,
                 replan_stuck_steps=1, replan_no_progress_steps=8,
                 min_progress_m=0.15, goal_blocklist_steps=40,
                 window_slack=1.5, min_window_cells=100, astar_max_window_cells=300,
                 max_replans_without_moving=10, body_radius=0.3, debug=False):
        self.grid = grid
        # Prints timing/progress through _pick_and_plan()/_plan_to_cluster() - the A*
        # window-growth retry loop over many candidate clusters is pure-Python and can run
        # for a long time with zero console output in between, which is indistinguishable
        # from a true hang unless you can see it's still working. Off by default (noisy);
        # explore.py's --debug-planner flag turns this on.
        self.debug = debug
        self.min_cluster_size = min_cluster_size
        self.connectivity = connectivity
        self.goal_arrival_radius = goal_arrival_radius
        self.waypoint_arrival_radius = waypoint_arrival_radius
        self.replan_stuck_steps = replan_stuck_steps
        self.replan_no_progress_steps = replan_no_progress_steps
        self.min_progress_m = min_progress_m
        self.goal_blocklist_steps = goal_blocklist_steps
        self.window_slack = window_slack
        self.min_window_cells = min_window_cells
        self.astar_max_window_cells = astar_max_window_cells
        self.max_replans_without_moving = max_replans_without_moving
        # Inflates occupied cells by this much before planning (see _inflate_occupied) so
        # A*/simplify_path never confirm a path through a gap narrower than the agent's
        # actual footprint - must match the real body_radius passed to
        # swept_clearance_ahead() for the planner and the real-time safety check to agree
        # on what's actually passable.
        self.body_radius = body_radius

        self.state = PlannerState.NEED_GOAL
        self.goal_cluster = None
        self.path_world = []       # simplified waypoints, world (x, z) coords
        self.waypoint_idx = 0
        self._blocklist = {}       # centroid_cell -> steps remaining
        self._stuck_counter = 0
        self._best_dist_to_goal = math.inf
        self._steps_since_progress = 0
        self._last_cur_cell = None
        self._no_progress_pick_cell = None   # last cell _pick_and_plan was called from
        self._no_progress_pick_count = 0     # consecutive _pick_and_plan calls from that cell

    def _dbg(self, msg):
        if self.debug:
            print(f"[planner] {msg}", flush=True)

    def _decay_blocklist(self):
        expired = [cell for cell, remaining in self._blocklist.items() if remaining <= 1]
        for cell in expired:
            del self._blocklist[cell]
        for cell in self._blocklist:
            self._blocklist[cell] -= 1

    def _blocklist_current_goal(self):
        if self.goal_cluster is not None:
            self._blocklist[self.goal_cluster.centroid_cell] = self.goal_blocklist_steps

    def _plan_to_cluster(self, cur_cell, cluster):
        straight_dist = math.hypot(cluster.centroid_cell[0] - cur_cell[0],
                                    cluster.centroid_cell[1] - cur_cell[1])
        window = max(self.min_window_cells, straight_dist * self.window_slack)
        self._dbg(f"  _plan_to_cluster: size={cluster.size} centroid={cluster.centroid_cell} "
                  f"straight_dist={straight_dist:.1f}cells initial_window={window:.0f}")

        # Inflated by body_radius so A*/simplify_path never confirm a path through a gap
        # narrower than the agent can actually fit through (see _inflate_occupied) -
        # computed once per call, reused across every candidate goal cell and window-growth
        # retry below rather than recomputed per astar() call.
        t0 = time.perf_counter()
        occupied_mask = _inflate_occupied(self.grid, self.body_radius)
        self._dbg(f"  _inflate_occupied took {time.perf_counter() - t0:.3f}s")

        # Try the centroid first, then a couple of member cells nearest the
        # centroid, in case the centroid itself is awkward for A* to reach.
        candidates = [cluster.centroid_cell]
        if len(cluster.cells) > 1:
            dists = np.hypot(cluster.cells[:, 0] - cluster.centroid_cell[0],
                              cluster.cells[:, 1] - cluster.centroid_cell[1])
            for idx in np.argsort(dists)[:3]:
                cell = tuple(int(v) for v in cluster.cells[idx])
                if cell not in candidates:
                    candidates.append(cell)

        for goal_cell in candidates:
            w = window
            t0 = time.perf_counter()
            path = astar(self.grid, cur_cell, goal_cell, connectivity=self.connectivity,
                         window_radius=w, occupied_mask=occupied_mask)
            self._dbg(f"  astar candidate={goal_cell} window={w:.0f} -> "
                      f"{'found len=' + str(len(path)) if path is not None else 'None'} "
                      f"({time.perf_counter() - t0:.3f}s)")
            while path is None and w < self.astar_max_window_cells:
                w = min(w * 2, self.astar_max_window_cells)
                t0 = time.perf_counter()
                path = astar(self.grid, cur_cell, goal_cell, connectivity=self.connectivity,
                             window_radius=w, occupied_mask=occupied_mask)
                self._dbg(f"  astar candidate={goal_cell} window={w:.0f} (grown) -> "
                          f"{'found len=' + str(len(path)) if path is not None else 'None'} "
                          f"({time.perf_counter() - t0:.3f}s)")
            if path is not None:
                t0 = time.perf_counter()
                simplified = simplify_path(path, self.grid, occupied_mask=occupied_mask)
                self._dbg(f"  simplify_path {len(path)} -> {len(simplified)} waypoints "
                          f"({time.perf_counter() - t0:.3f}s)")
                return [self.grid.to_world(*c) for c in simplified], goal_cell
        self._dbg(f"  _plan_to_cluster: no candidate reachable, giving up on this cluster")
        return None, None

    def _pick_and_plan(self, cur_cell):
        pick_t0 = time.perf_counter()
        self._dbg(f"_pick_and_plan start cur_cell={cur_cell} "
                  f"no_progress_count={self._no_progress_pick_count}")

        # Circuit breaker: the blocklist-fallback pass below (needed so a merely-cooling-down
        # cluster doesn't get mistaken for "nothing left to explore") can otherwise loop
        # forever. body_radius inflation (see _plan_to_cluster/_inflate_occupied) fixes the
        # main known cause - A* confirming a path through a gap too narrow for the agent's
        # actual footprint, which real-time swept_clearance_ahead then correctly refuses -
        # but this stays as a general safety net for any other way "A* says reachable, real
        # time says blocked" could happen. If we're replanning repeatedly from the exact same
        # cell with no progress, stop retrying and end the run instead of looping forever.
        if cur_cell == self._no_progress_pick_cell:
            self._no_progress_pick_count += 1
        else:
            self._no_progress_pick_cell = cur_cell
            self._no_progress_pick_count = 0
        if self._no_progress_pick_count >= self.max_replans_without_moving:
            self._dbg("_pick_and_plan: circuit breaker tripped -> done")
            self.state = PlannerState.DONE
            return NavCommand(kind="done")

        t0 = time.perf_counter()
        frontier_cells = self.grid.frontiers()
        self._dbg(f"grid.frontiers(): {len(frontier_cells)} cells ({time.perf_counter() - t0:.3f}s)")
        if len(frontier_cells) == 0:
            self._dbg("_pick_and_plan: no frontier cells -> done")
            self.state = PlannerState.DONE
            return NavCommand(kind="done")

        t0 = time.perf_counter()
        clusters = cluster_frontiers(frontier_cells, connectivity=self.connectivity,
                                      min_cluster_size=self.min_cluster_size)
        self._dbg(f"cluster_frontiers(): {len(clusters)} clusters ({time.perf_counter() - t0:.3f}s)")
        if not clusters:
            self._dbg("_pick_and_plan: no clusters survived min_cluster_size -> done")
            self.state = PlannerState.DONE
            return NavCommand(kind="done")

        clusters.sort(key=lambda c: score_cluster(c, cur_cell, self._blocklist))

        # Two passes: prefer non-blocklisted clusters first, but fall back to blocklisted
        # ones rather than declaring DONE if that's all that's left. A blocklist entry means
        # "deprioritize this for goal_blocklist_steps," not "gone forever" - confirmed live,
        # treating "every known cluster happens to be blocklisted right now" as "exploration
        # complete" was a real bug: with replan_stuck_steps=1, a run near a persistent close
        # obstruction blocklisted every cluster it knew about in quick succession and quit with
        # large unexplored areas still visible in the saved map, long before the blocklist
        # entries would have naturally expired.
        for consider_blocklisted in (False, True):
            self._dbg(f"pass consider_blocklisted={consider_blocklisted}: {len(clusters)} clusters to try")
            for i, cluster in enumerate(clusters):
                if not consider_blocklisted and cluster.centroid_cell in self._blocklist:
                    continue
                self._dbg(f" cluster {i + 1}/{len(clusters)} size={cluster.size} "
                          f"centroid={cluster.centroid_cell} blocklisted={cluster.centroid_cell in self._blocklist}")
                cluster_t0 = time.perf_counter()
                path_world, reached_cell = self._plan_to_cluster(cur_cell, cluster)
                self._dbg(f" cluster {i + 1}/{len(clusters)} done in {time.perf_counter() - cluster_t0:.3f}s "
                          f"-> {'reachable' if path_world is not None else 'unreachable'}")
                if path_world is not None:
                    self.goal_cluster = cluster
                    self.path_world = path_world
                    # index 0 is the start cell itself - aim at the next waypoint.
                    self.waypoint_idx = 1 if len(path_world) > 1 else 0
                    self.state = PlannerState.FOLLOWING
                    self._best_dist_to_goal = math.inf
                    self._steps_since_progress = 0
                    self._stuck_counter = 0
                    goal_world = self.grid.to_world(*cluster.centroid_cell)
                    self._dbg(f"_pick_and_plan committed to cluster {i + 1} in "
                              f"{time.perf_counter() - pick_t0:.3f}s total")
                    return NavCommand(kind="goto_waypoint", target_world_xz=self.path_world[self.waypoint_idx],
                                       goal_world_xz=goal_world, replanned=True)
                if not consider_blocklisted:
                    # Unreachable even at the max window - blocklist so we don't retry every
                    # call. Only set on the first pass, so the fallback pass doesn't reset the
                    # timer on a cluster we're now retrying anyway.
                    self._blocklist[cluster.centroid_cell] = self.goal_blocklist_steps

        self._dbg(f"_pick_and_plan: every cluster unreachable in both passes -> done "
                  f"({time.perf_counter() - pick_t0:.3f}s total)")
        self.state = PlannerState.DONE
        return NavCommand(kind="done")

    def _goal_region_dissolved(self):
        """Cheap proxy for "is the committed goal cluster's region still
        unexplored?" - checks raw frontier-cell membership near the goal
        centroid instead of re-running full clustering every FOLLOWING step
        (cluster_frontiers()/astar() are the expensive parts, reserved for
        actual replans; grid.frontiers() itself is a cheap vectorized numpy op)."""
        frontier_set = {(int(c[0]), int(c[1])) for c in self.grid.frontiers()}
        gx, gz = self.goal_cluster.centroid_cell
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                if (gx + dx, gz + dz) in frontier_set:
                    return False
        return True

    def update(self, cur_world_xz, cur_cell):
        self._decay_blocklist()
        if self._last_cur_cell is not None and cur_cell != self._last_cur_cell:
            self._stuck_counter = 0
        self._last_cur_cell = cur_cell

        if self.state == PlannerState.DONE:
            return NavCommand(kind="done")

        if self.state == PlannerState.NEED_GOAL:
            return self._pick_and_plan(cur_cell)

        # FOLLOWING
        goal_world = self.grid.to_world(*self.goal_cluster.centroid_cell)
        dist_to_goal = math.hypot(cur_world_xz[0] - goal_world[0], cur_world_xz[1] - goal_world[1])

        if dist_to_goal <= self.goal_arrival_radius:
            self.goal_cluster = None
            self.state = PlannerState.NEED_GOAL
            return self._pick_and_plan(cur_cell)

        if self._goal_region_dissolved():
            self.goal_cluster = None
            self.state = PlannerState.NEED_GOAL
            return self._pick_and_plan(cur_cell)

        target = self.path_world[self.waypoint_idx]
        wp_dist = math.hypot(cur_world_xz[0] - target[0], cur_world_xz[1] - target[1])
        if wp_dist <= self.waypoint_arrival_radius and self.waypoint_idx < len(self.path_world) - 1:
            self.waypoint_idx += 1
            target = self.path_world[self.waypoint_idx]

        if dist_to_goal < self._best_dist_to_goal - self.min_progress_m:
            self._best_dist_to_goal = dist_to_goal
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
            if self._steps_since_progress >= self.replan_no_progress_steps:
                self._blocklist_current_goal()
                self.goal_cluster = None
                self.state = PlannerState.NEED_GOAL
                return self._pick_and_plan(cur_cell)

        return NavCommand(kind="goto_waypoint", target_world_xz=target, goal_world_xz=goal_world, replanned=False)

    def notify_blocked(self, blocked_cell):
        """Call when swept_clearance_ahead capped the step near-zero. Default
        replan_stuck_steps=1 (replan on the very first block, not after a
        streak): while blocked, the agent doesn't move or rotate, so a retry
        rescans the identical geometry from the identical pose and gets the
        identical reading - waiting for repeated confirmation just burns scans
        without learning anything new. Only raise replan_stuck_steps above 1
        if the environment is expected to change between retries (e.g. a
        moving obstacle) or scans are noisy enough that one reading could be
        a fluke - neither is true for this static, deterministic sim."""
        self._stuck_counter += 1
        if self._stuck_counter >= self.replan_stuck_steps:
            self._blocklist_current_goal()
            self.goal_cluster = None
            self.state = PlannerState.NEED_GOAL
            self._stuck_counter = 0
