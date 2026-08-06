"""
Standalone unit tests for frontier_planner.py and the frontiers() wraparound
fix in occupancy_grid.py. No live Unity/WebSocket connection needed - builds
synthetic OccupancyGrid instances directly.

Run with:
    uv run pytest validation/tests/mapping/test_frontier_planner.py
"""
import os
import sys
import unittest

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # validation/tests/mapping
_MAPPING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))), "agent", "mapping")
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from occupancy_grid import OccupancyGrid, _bresenham_line  # noqa: E402
from frontier_planner import (  # noqa: E402
    cluster_frontiers,
    score_cluster,
    astar,
    simplify_path,
    _line_of_sight,
    _inflate_occupied,
    FrontierCluster,
    FrontierPlanner,
)


class TestFrontiersWraparoundFix(unittest.TestCase):
    def test_no_wraparound_at_grid_edge(self):
        grid = OccupancyGrid(size_m=2.0, resolution=0.1)  # n = 20
        # Rows 0-2 free; rows 3..19 stay unknown (0.0, the default).
        grid.log_odds[0:3, :] = -1.0
        frontier_cells = grid.frontiers()
        frontier_rows = {int(c[0]) for c in frontier_cells}

        self.assertNotIn(
            0, frontier_rows,
            "row 0 has no real neighbor at row -1; pre-fix np.roll wrapped that "
            "lookup to row n-1 (unknown), manufacturing a false frontier here",
        )
        self.assertIn(2, frontier_rows, "row 2 genuinely borders unknown row 3")


class TestMarkOccupiedRegion(unittest.TestCase):
    def test_marks_a_disk_not_just_the_nearest_cell(self):
        # Regression: marking only the single exact hit point took many repeated
        # near-identical blocks before enough of a real obstacle's true footprint was
        # registered for inflation-aware A* to actually treat the area as blocked.
        grid = OccupancyGrid(size_m=2.0, resolution=0.1)  # n = 20
        center = grid.to_world(10, 10)

        grid.mark_occupied_region(center, radius_m=0.2)  # r_cells = 2

        self.assertGreater(grid.log_odds[10, 10], grid.OCCUPIED_THRESHOLD)
        self.assertGreater(grid.log_odds[10, 12], grid.OCCUPIED_THRESHOLD, "2 cells away is within radius_m")
        self.assertLessEqual(grid.log_odds[10, 15], grid.OCCUPIED_THRESHOLD,
                             "5 cells away is well outside radius_m and must stay untouched")

    def test_out_of_bounds_neighbors_are_skipped_not_erroring(self):
        grid = OccupancyGrid(size_m=2.0, resolution=0.1)  # n = 20
        corner = grid.to_world(0, 0)

        grid.mark_occupied_region(corner, radius_m=0.3)  # would extend past the grid edge

        self.assertGreater(grid.log_odds[0, 0], grid.OCCUPIED_THRESHOLD)


class TestClusterFrontiers(unittest.TestCase):
    def test_two_known_size_blobs_with_noise(self):
        blob_a = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]  # 6 cells
        blob_b = [(20 + i, 20 + j) for i in range(3) for j in range(5)]  # 15 cells
        noise = [(50, 50), (60, 60), (70, 70)]  # isolated singles, below min_cluster_size

        frontier_cells = np.array(blob_a + blob_b + noise, dtype=int)
        clusters = cluster_frontiers(frontier_cells, connectivity=8, min_cluster_size=4)

        self.assertEqual(len(clusters), 2)
        sizes = sorted(c.size for c in clusters)
        self.assertEqual(sizes, [6, 15])

        for cluster in clusters:
            members = {tuple(c) for c in cluster.cells}
            self.assertIn(cluster.centroid_cell, members)

    def test_connectivity_wiring(self):
        # (5,5) and (6,6) touch only diagonally.
        frontier_cells = np.array([(5, 5), (6, 6)], dtype=int)

        clusters_4 = cluster_frontiers(frontier_cells, connectivity=4, min_cluster_size=1)
        clusters_8 = cluster_frontiers(frontier_cells, connectivity=8, min_cluster_size=1)

        self.assertEqual(len(clusters_4), 2)
        self.assertEqual(len(clusters_8), 1)


class TestAStar(unittest.TestCase):
    def _wall_with_gap_grid(self):
        grid = OccupancyGrid(size_m=6.0, resolution=0.1)  # n = 60
        grid.log_odds[:, :] = -1.0  # all free
        grid.log_odds[25, :] = 5.0  # occupied wall along cx=25
        grid.log_odds[25, 5] = -1.0  # single-cell gap ("doorway") at cz=5
        return grid

    def test_routes_around_obstacle_through_gap(self):
        grid = self._wall_with_gap_grid()
        start, goal = (10, 10), (40, 40)
        path = astar(grid, start, goal, connectivity=8, window_radius=None)

        self.assertIsNotNone(path)
        for cx, cz in path:
            self.assertLessEqual(grid.log_odds[cx, cz], grid.OCCUPIED_THRESHOLD)

        gap_crossings = [c for c in path if c[0] == 25]
        self.assertTrue(gap_crossings, "path never crosses cx=25 at all")
        self.assertTrue(all(abs(c[1] - 5) <= 1 for c in gap_crossings),
                         "path crossed the wall row somewhere other than the gap")

        # Sanity bound so a badly broken heuristic (e.g. wandering the whole
        # grid) would fail this - octile distance start->goal is ~42.4 cells;
        # detouring to the gap at cz=5 is a longer but still bounded route.
        self.assertLess(len(path), 150)

    def test_window_bounding(self):
        grid = self._wall_with_gap_grid()
        start, goal = (10, 10), (40, 40)

        too_small = astar(grid, start, goal, connectivity=8, window_radius=3)
        self.assertIsNone(too_small, "window_radius=3 should exclude the gap at cz=5")

        large_enough = astar(grid, start, goal, connectivity=8, window_radius=50)
        self.assertIsNotNone(large_enough)

    def test_inflated_occupied_blocks_gap_narrower_than_body_radius(self):
        # Regression: a live run had the planner repeatedly "successfully" routing through
        # a shelf-corner gap that real-time swept_clearance_ahead correctly refused to let
        # the agent actually pass (too narrow for its real footprint), producing an
        # infinite plan-block-replan loop. Plain per-cell occupancy has no concept of body
        # width; inflating occupied cells by body_radius before planning fixes this.
        grid = self._wall_with_gap_grid()
        start, goal = (10, 10), (40, 40)

        # Without inflation, the 1-cell gap at cz=5 is passable (same as test_routes_around_obstacle_through_gap).
        plain_path = astar(grid, start, goal, connectivity=8, window_radius=None)
        self.assertIsNotNone(plain_path)

        # A body_radius covering >=1 cell inflates both walls of the gap enough to close it -
        # a real body couldn't fit through a gap exactly one 0.1m cell wide.
        inflated = _inflate_occupied(grid, body_radius=0.15)  # r_cells = ceil(0.15/0.1) = 2
        inflated_path = astar(grid, start, goal, connectivity=8, window_radius=None, occupied_mask=inflated)
        self.assertIsNone(inflated_path,
                         "a gap only 1 cell wide should become impassable once inflated by body_radius")


class TestInflateOccupied(unittest.TestCase):
    def test_inflates_within_radius_only(self):
        grid = OccupancyGrid(size_m=2.0, resolution=0.1)  # n = 20
        grid.log_odds[10, 10] = 5.0  # single occupied cell

        inflated = _inflate_occupied(grid, body_radius=0.2)  # r_cells = 2

        self.assertTrue(inflated[10, 10], "the occupied cell itself must stay occupied")
        self.assertTrue(inflated[10, 12], "2 cells away is within the inflation radius")
        self.assertFalse(inflated[10, 15], "5 cells away is well outside the inflation radius")

    def test_zero_body_radius_is_a_no_op(self):
        grid = OccupancyGrid(size_m=2.0, resolution=0.1)
        grid.log_odds[10, 10] = 5.0

        plain = _inflate_occupied(grid, body_radius=0)

        self.assertTrue(plain[10, 10])
        self.assertFalse(plain[10, 11], "no inflation should occur when body_radius is 0/falsy")


class TestSimplifyPath(unittest.TestCase):
    def test_straight_corridor_collapses_to_endpoints(self):
        grid = OccupancyGrid(size_m=6.0, resolution=0.1)
        grid.log_odds[:, :] = -1.0  # all free, no obstacles anywhere

        path = [(i, i) for i in range(11)]  # diagonal straight line, 11 cells
        simplified = simplify_path(path, grid)

        self.assertEqual(simplified, [path[0], path[-1]])

    def test_l_shaped_corridor_preserves_turn(self):
        grid = OccupancyGrid(size_m=6.0, resolution=0.1)
        grid.log_odds[:, :] = -1.0  # all free
        # Block the direct diagonal shortcut from (0,0) to (5,5) so the L-shape
        # is a real routing constraint, not just an arbitrary raw-path shape.
        grid.log_odds[2, 2] = 5.0
        grid.log_odds[3, 3] = 5.0

        path = (
            [(i, 0) for i in range(6)]       # (0,0) .. (5,0)
            + [(5, j) for j in range(1, 6)]  # (5,1) .. (5,5)
        )
        simplified = simplify_path(path, grid)

        self.assertEqual(simplified[0], path[0])
        self.assertEqual(simplified[-1], path[-1])
        self.assertGreater(len(simplified), 2,
                            "direct shortcut is blocked; must retain an intermediate waypoint")
        for a, b in zip(simplified, simplified[1:]):
            self.assertTrue(_line_of_sight(grid, a, b),
                             f"simplified waypoints {a}->{b} are not actually clear")

    def test_inflated_start_cell_does_not_produce_degenerate_first_waypoint(self):
        # Regression: with the agent parked against a shelf, its own cell sits inside the
        # body_radius inflation. _line_of_sight tested that start endpoint too, so EVERY
        # shortcut from the start failed and simplify_path emitted the start cell as a
        # DUPLICATE first waypoint. The executor then read to_world(start) == its own
        # position as the target -> dist~0 -> false 'path blocked' (forward arc wide open,
        # clearance 4.17m in the live log) -> replan->same-degenerate-path until the
        # no-movement circuit breaker ended the run with the map half-built.
        grid = OccupancyGrid(size_m=6.0, resolution=0.1)
        grid.log_odds[:, :] = -1.0
        grid.log_odds[30, 20:40] = 5.0                       # shelf column
        occupied = _inflate_occupied(grid, body_radius=0.3)  # r_cells = 3

        start, goal = (33, 30), (55, 30)  # start on the inflation edge; (34,30) is free
        self.assertTrue(occupied[start], "precondition: the start cell must be inflated")

        path = astar(grid, start, goal, connectivity=8, occupied_mask=occupied)
        self.assertIsNotNone(path, "A* can still step out toward the open goal")
        simplified = simplify_path(path, grid, occupied_mask=occupied)

        self.assertNotEqual(simplified[0], simplified[1],
                            "the start cell must not be duplicated as the first waypoint")
        self.assertNotEqual(grid.to_world(*simplified[1]), grid.to_world(*start),
                            "the first waypoint must not equal the agent's own position")
        # Efficient string-pull: with the start-endpoint exemption, the straight shot to the
        # far goal is now visible, so it's one stride, not a 1-cell (0.1m) crawl.
        self.assertEqual(simplified[1], goal,
                         "should string-pull straight to the far goal past the inflated start")

    def test_start_exemption_does_not_enable_corner_cutting(self):
        # Negative control for the fix above: exempting the START endpoint from the
        # passability test must NOT let a shortcut cut a corner through a real obstacle -
        # only `a` is exempt, every cell strictly between a and b is still tested.
        grid = OccupancyGrid(size_m=8.0, resolution=0.1)
        grid.log_odds[:, :] = -1.0
        grid.log_odds[40, 10:70] = 5.0                       # wall blocking the straight line
        occupied = _inflate_occupied(grid, body_radius=0.3)

        start, goal = (43, 40), (20, 40)  # start inside the wall's inflation; goal past it
        self.assertTrue(occupied[start], "precondition: the start cell must be inflated")

        path = astar(grid, start, goal, connectivity=8, occupied_mask=occupied)
        self.assertIsNotNone(path, "goal is reachable by routing around the wall")
        simplified = simplify_path(path, grid, occupied_mask=occupied)

        # The straight start->goal line crosses the wall and must be rejected even with the
        # start exemption on.
        self.assertFalse(
            _line_of_sight(grid, start, goal, occupied_mask=occupied, ignore_start=True),
            "a straight line through the wall must not pass line-of-sight",
        )
        # Every retained segment's interior must stay clear (skip each segment's own start
        # cell, which is the exempted anchor).
        for a, b in zip(simplified, simplified[1:]):
            for cx, cz in _bresenham_line(a[0], a[1], b[0], b[1])[1:]:
                self.assertFalse(occupied[cx, cz],
                                 f"segment {a}->{b} cuts through an occupied/inflated cell at ({cx},{cz})")


class TestFrontierPlannerStateMachine(unittest.TestCase):
    def _grid_with_two_far_clusters(self):
        grid = OccupancyGrid(size_m=10.0, resolution=0.1)  # n = 100
        # Single free cell at agent start - touches unknown on all sides, but
        # its 1-cell cluster is below min_cluster_size and gets filtered.
        grid.log_odds[10, 10] = -1.0
        for cx, cz in ((11, 11), (9, 11), (11, 9)):
            grid.log_odds[cx, cz] = -1.0  # more filtered noise singles nearby

        # Two real candidate clusters (4x4 free squares -> size-12 rings).
        grid.log_odds[50:54, 50:54] = -1.0   # cluster B
        grid.log_odds[50:54, 10:14] = -1.0   # cluster C
        return grid

    def test_goal_commitment_no_oscillation(self):
        grid = self._grid_with_two_far_clusters()
        planner = FrontierPlanner(grid, min_cluster_size=4, goal_arrival_radius=0.4,
                                   waypoint_arrival_radius=0.25)

        pos = grid.to_world(10, 10)
        nav = planner.update(pos, grid.cell(*pos))
        self.assertEqual(nav.kind, "goto_waypoint")
        self.assertTrue(nav.replanned)
        committed_goal = planner.goal_cluster.centroid_cell

        for _ in range(50):
            if nav.kind != "goto_waypoint":
                break
            pos = nav.target_world_xz  # fake agent teleports exactly onto the waypoint
            nav = planner.update(pos, grid.cell(*pos))
            if nav.kind == "goto_waypoint" and not nav.replanned:
                self.assertEqual(
                    planner.goal_cluster.centroid_cell, committed_goal,
                    "goal flipped mid-pursuit without a legitimate replan trigger",
                )
            elif nav.replanned:
                # Legitimate replan (goal reached -> next goal, etc.) - update
                # the tracked goal and keep going.
                committed_goal = planner.goal_cluster.centroid_cell if planner.goal_cluster else None

    def test_recovery_switches_goal_after_repeated_blocks(self):
        grid = self._grid_with_two_far_clusters()
        planner = FrontierPlanner(grid, min_cluster_size=4, replan_stuck_steps=5,
                                   goal_blocklist_steps=40)

        pos = grid.to_world(10, 10)
        cur_cell = grid.cell(*pos)
        nav = planner.update(pos, cur_cell)
        self.assertEqual(nav.kind, "goto_waypoint")
        first_goal = planner.goal_cluster.centroid_cell

        for _ in range(5):
            planner.notify_blocked(cur_cell)

        self.assertIn(first_goal, planner._blocklist)

        nav = planner.update(pos, cur_cell)
        self.assertEqual(nav.kind, "goto_waypoint")
        self.assertIsNotNone(planner.goal_cluster)
        self.assertNotEqual(planner.goal_cluster.centroid_cell, first_goal,
                             "planner should have switched to the other candidate cluster")

    def test_default_replan_stuck_steps_recovers_on_first_block(self):
        # Default replan_stuck_steps=1: a single notify_blocked() should be enough to
        # blocklist and switch goals - retrying an identical blocked reading (same pose,
        # same target) provides no new information, so there's no reason to wait for a streak.
        grid = self._grid_with_two_far_clusters()
        planner = FrontierPlanner(grid, min_cluster_size=4, goal_blocklist_steps=40)

        pos = grid.to_world(10, 10)
        cur_cell = grid.cell(*pos)
        nav = planner.update(pos, cur_cell)
        self.assertEqual(nav.kind, "goto_waypoint")
        first_goal = planner.goal_cluster.centroid_cell

        planner.notify_blocked(cur_cell)
        self.assertIn(first_goal, planner._blocklist)

        nav = planner.update(pos, cur_cell)
        self.assertEqual(nav.kind, "goto_waypoint")
        self.assertIsNotNone(planner.goal_cluster)
        self.assertNotEqual(planner.goal_cluster.centroid_cell, first_goal,
                             "a single block should be enough to switch goals by default")

    def test_blocklisted_cluster_is_not_treated_as_map_complete(self):
        # Regression: a live run declared "map complete" with large unexplored areas still
        # visible in the saved map, right after rapid-fire blocks (replan_stuck_steps=1)
        # blocklisted every cluster the planner currently knew about. A blocklist entry means
        # "deprioritized for goal_blocklist_steps," not "gone forever" - if it's the ONLY
        # known cluster and it's genuinely still reachable, the planner must still go there
        # instead of declaring DONE just because it happens to be on cooldown.
        grid = OccupancyGrid(size_m=10.0, resolution=0.1)
        grid.log_odds[10, 10] = -1.0
        grid.log_odds[50:54, 50:54] = -1.0  # exactly one real candidate cluster

        planner = FrontierPlanner(grid, min_cluster_size=4, goal_blocklist_steps=40)
        cur_cell = grid.cell(*grid.to_world(10, 10))  # (10, 10) here is a CELL index, not world coords

        clusters = cluster_frontiers(grid.frontiers(), connectivity=8, min_cluster_size=4)
        self.assertEqual(len(clusters), 1)
        planner._blocklist[clusters[0].centroid_cell] = 40  # simulate a stale blocklist entry

        nav = planner._pick_and_plan(cur_cell)

        self.assertEqual(nav.kind, "goto_waypoint",
                         "the only known cluster is reachable and must not be skipped just "
                         "because it's blocklisted - falling through to DONE here is the bug")

    def test_circuit_breaker_stops_infinite_replan_loop(self):
        # Regression: a live run got stuck replanning to the same goal from the same position
        # forever - the goal was A*-reachable "on paper" (through open/unknown space) but kept
        # failing in real time (most likely a self-hit never reflected as a static occupied
        # cell in the grid), so the blocklist-fallback pass just kept re-picking it every call.
        # max_replans_without_moving must cap this instead of looping indefinitely.
        grid = OccupancyGrid(size_m=10.0, resolution=0.1)
        grid.log_odds[10, 10] = -1.0
        grid.log_odds[50:54, 50:54] = -1.0

        planner = FrontierPlanner(grid, min_cluster_size=4, max_replans_without_moving=5)
        cur_cell = grid.cell(*grid.to_world(10, 10))

        # cur_cell never changes across these calls, simulating the agent never actually
        # moving because every attempt gets blocked immediately in real time.
        results = [planner._pick_and_plan(cur_cell).kind for _ in range(20)]

        self.assertIn("goto_waypoint", results[:5], "should still try normally at first")
        self.assertEqual(results[-1], "done",
                         "must eventually stop retrying from an unchanged cell rather than loop forever")
        self.assertEqual(planner._pick_and_plan(cur_cell).kind, "done",
                         "once tripped, the breaker should not resume trying")


class TestObservationVantages(unittest.TestCase):
    def _wall_grid(self):
        grid = OccupancyGrid(size_m=12.0, resolution=0.1)  # n = 120
        grid.log_odds[:, :] = -1.0            # everything known-free by default
        grid.log_odds[48, 30:70] = 5.0        # a vertical occupied wall at cx=48
        return grid

    def test_vantages_are_free_body_safe_and_have_line_of_sight(self):
        grid = self._wall_grid()
        planner = FrontierPlanner(grid, body_radius=0.3)
        occ_mask = _inflate_occupied(grid, 0.3)
        cluster = FrontierCluster(cells=np.array([[52, 50]]), centroid_cell=(52, 50), size=1)

        vantages = planner._observation_vantages(cluster, occ_mask, cur_cell=(60, 50))

        self.assertTrue(vantages, "there should be standable, in-view vantage cells east of the wall")
        for vx, vz in vantages:
            self.assertLess(grid.log_odds[vx, vz], grid.FREE_THRESHOLD, "vantage must be known-free")
            self.assertFalse(occ_mask[vx, vz], "vantage must be body-safe (outside the inflation)")
            self.assertGreater(vx, 48, "a vantage west of the wall has no line-of-sight to the target")

    def test_plan_to_cluster_falls_back_to_a_vantage_when_the_frontier_is_unstandable(self):
        # The exact situation this feature targets: a frontier cell that is free (hence a
        # frontier) but sits inside the body-radius inflation next to a shelf, so A* can't
        # stand on it. The planner must reach a vantage instead of abandoning the cluster.
        grid = self._wall_grid()
        planner = FrontierPlanner(grid, body_radius=0.3)
        occ_mask = _inflate_occupied(grid, 0.3)
        centroid = (49, 50)   # free, but one cell off the wall -> inside inflation -> unstandable
        self.assertTrue(occ_mask[centroid], "precondition: the frontier cell is unstandable")
        cluster = FrontierCluster(cells=np.array([list(centroid)]), centroid_cell=centroid, size=1)

        path_world, goal = planner._plan_to_cluster((60, 50), cluster)

        self.assertIsNotNone(path_world, "must reach a vantage rather than declare the cluster unreachable")
        self.assertNotEqual(goal, centroid, "goal must be a standable vantage, not the unstandable frontier cell")

    def test_vantage_excludes_the_cell_the_agent_already_occupies(self):
        # Regression: when a line-of-sight vantage coincides with the cell the agent already
        # stands on (or within the waypoint-arrival radius of it), returning it produced a
        # degenerate "go to your own cell" path -> the executor can't move -> the agent holds
        # every step until the no-movement circuit breaker ends the run (a real stall seen
        # live at a shelf-corner pocket re-picking top-wall frontier slivers already in view).
        # Such self-vantages must be excluded so only real-movement vantages are offered.
        grid = self._wall_grid()
        planner = FrontierPlanner(grid, body_radius=0.3)
        occ_mask = _inflate_occupied(grid, 0.3)
        cluster = FrontierCluster(cells=np.array([[52, 50]]), centroid_cell=(52, 50), size=1)
        cur_cell = (55, 50)  # east of the wall, itself a valid line-of-sight vantage

        vantages = planner._observation_vantages(cluster, occ_mask, cur_cell=cur_cell)

        self.assertTrue(vantages, "real-movement vantages still exist farther from the agent")
        self.assertNotIn(cur_cell, vantages, "a vantage the agent already occupies must be excluded")
        min_move_sq = (planner.waypoint_arrival_radius / grid.res) ** 2
        for vx, vz in vantages:
            d2 = (vx - cur_cell[0]) ** 2 + (vz - cur_cell[1]) ** 2
            self.assertGreater(d2, min_move_sq, "every vantage must be real movement, not an in-place hold")

    def test_no_vantage_when_the_target_is_fully_walled_off(self):
        # A frontier with no line-of-sight from any standable cell (boxed in by occupied on
        # all sides) has no vantage - the planner should still, correctly, give up on it
        # rather than invent an unreachable goal.
        grid = OccupancyGrid(size_m=12.0, resolution=0.1)
        grid.log_odds[:, :] = -1.0
        grid.log_odds[58:63, 58:63] = 5.0     # a solid occupied box
        grid.log_odds[60, 60] = -1.0           # a lone free cell sealed inside it
        planner = FrontierPlanner(grid, body_radius=0.3)
        occ_mask = _inflate_occupied(grid, 0.3)
        cluster = FrontierCluster(cells=np.array([[60, 60]]), centroid_cell=(60, 60), size=1)

        vantages = planner._observation_vantages(cluster, occ_mask, cur_cell=(30, 30))

        self.assertEqual(vantages, [], "a fully-occluded frontier has no line-of-sight vantage")


if __name__ == "__main__":
    unittest.main()
