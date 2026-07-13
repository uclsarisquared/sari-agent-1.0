"""
Standalone unit tests for topology.py's skeleton-to-checkpoint-graph extraction.
No live Unity/WebSocket connection needed - builds synthetic OccupancyGrid
instances directly, the same pattern test_frontier_planner.py uses.

Run with:
    python slamtest/tests/test_topology.py
"""
import os
import sys
import unittest

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # overhaul/slamtest/tests
_SLAMTEST_DIR = os.path.dirname(_THIS_DIR)                     # overhaul/slamtest
_OVERHAUL_DIR = os.path.dirname(_SLAMTEST_DIR)                 # overhaul
for _p in (_OVERHAUL_DIR, _SLAMTEST_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from occupancy_grid import OccupancyGrid  # noqa: E402
import topology as topo  # noqa: E402


def _corridor_grid(size_m=6.0, resolution=0.1):
    grid = OccupancyGrid(size_m=size_m, resolution=resolution)
    grid.log_odds[:, :] = 5.0  # everything occupied by default
    return grid


class TestSkeletonize(unittest.TestCase):
    def test_straight_bar_thins_to_a_single_line(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 8:12] = True  # a 10x4 solid bar

        skeleton = topo.skeletonize(mask)

        # A thinned straight bar should collapse to (approximately) one cell wide - far
        # fewer skeleton cells than the original solid bar's 40.
        self.assertLess(skeleton.sum(), 15)
        self.assertGreater(skeleton.sum(), 5)

    def test_no_wraparound_at_grid_edge(self):
        # Same class of bug as OccupancyGrid.frontiers()' fixed np.roll issue: a shape
        # touching the array boundary must not spuriously connect to the opposite edge.
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:10, 8:12] = True  # touches row 0

        skeleton = topo.skeletonize(mask)  # must not raise, and must stay near the shape

        self.assertFalse(skeleton[:, 0].any())
        self.assertFalse(skeleton[:, 19].any())


class TestExtractTopologyStraightCorridor(unittest.TestCase):
    def test_two_ends_one_edge(self):
        grid = _corridor_grid()
        grid.log_odds[10:50, 28:32] = -1.0  # 4m-long, 0.4m-wide horizontal corridor

        result = topo.extract_topology(grid)

        self.assertEqual(len(result.checkpoints), 2)
        self.assertTrue(all(c.kind == "end" for c in result.checkpoints))
        self.assertEqual(len(result.edges), 1)
        # True carved span is ~3.9m; thinning insets slightly from each open boundary.
        self.assertGreater(result.edges[0].length_m, 3.0)
        self.assertLess(result.edges[0].length_m, 4.0)

    def test_edge_endpoints_reference_real_checkpoint_ids(self):
        grid = _corridor_grid()
        grid.log_odds[10:50, 28:32] = -1.0

        result = topo.extract_topology(grid)

        ids = {c.id for c in result.checkpoints}
        self.assertEqual({result.edges[0].a, result.edges[0].b}, ids)


class TestExtractTopologyCrossJunction(unittest.TestCase):
    def test_one_junction_four_ends_four_edges(self):
        grid = _corridor_grid()
        grid.log_odds[10:50, 28:32] = -1.0   # horizontal corridor
        grid.log_odds[28:32, 10:50] = -1.0   # vertical corridor, crossing at the center

        result = topo.extract_topology(grid)

        kinds = sorted(c.kind for c in result.checkpoints)
        self.assertEqual(kinds, ["end", "end", "end", "end", "junction"])
        self.assertEqual(len(result.edges), 4)

        junction_id = next(c.id for c in result.checkpoints if c.kind == "junction")
        for edge in result.edges:
            self.assertIn(junction_id, (edge.a, edge.b), "every edge must radiate from the one junction")


class TestPruneShortSpurs(unittest.TestCase):
    def _grid_with_side_spur(self):
        grid = _corridor_grid()
        grid.log_odds[10:50, 28:32] = -1.0
        grid.log_odds[18:24, 32:38] = -1.0  # ~0.5-0.6m nub sticking off the side
        return grid

    def test_short_spur_is_pruned_at_a_strict_threshold(self):
        grid = self._grid_with_side_spur()

        result = topo.extract_topology(grid, min_branch_length_m=0.7)

        kinds = sorted(c.kind for c in result.checkpoints)
        self.assertEqual(kinds, ["end", "end", "junction"],
                          "the spur's own dead end must be gone, not just its edge")
        self.assertEqual(len(result.edges), 2)

    def test_same_spur_survives_a_lenient_threshold(self):
        grid = self._grid_with_side_spur()

        result = topo.extract_topology(grid, min_branch_length_m=0.0)

        kinds = sorted(c.kind for c in result.checkpoints)
        self.assertEqual(kinds, ["end", "end", "end", "junction"])
        self.assertEqual(len(result.edges), 3)

    def test_pruning_never_leaves_an_orphaned_checkpoint(self):
        # Regression: pruning used to drop the edge but leave the spur's now-disconnected
        # endpoint checkpoint behind, dangling with no edge referencing it.
        grid = self._grid_with_side_spur()

        result = topo.extract_topology(grid, min_branch_length_m=0.7)

        referenced = {e.a for e in result.edges} | {e.b for e in result.edges}
        self.assertEqual(referenced, {c.id for c in result.checkpoints})


class TestDoorwayDetection(unittest.TestCase):
    def test_narrow_pinch_point_splits_the_edge(self):
        grid = _corridor_grid()
        grid.log_odds[10:50, 10:30] = -1.0    # open area, split into two halves by a wall
        grid.log_odds[10:50, 22:24] = 5.0     # wall
        grid.log_odds[28:30, 22:24] = -1.0    # a narrow ~0.2m gap through the wall

        result = topo.extract_topology(grid, doorway_max_width_m=1.5, doorway_width_ratio=0.7)

        self.assertIn("doorway", [c.kind for c in result.checkpoints])

    def test_uniform_width_corridor_gets_no_spurious_doorway(self):
        grid = _corridor_grid()
        grid.log_odds[10:50, 28:32] = -1.0  # constant width along its whole length

        result = topo.extract_topology(grid)

        self.assertNotIn("doorway", [c.kind for c in result.checkpoints])


class TestSaveTopology(unittest.TestCase):
    def test_round_trips_through_json(self):
        import json
        import tempfile

        grid = _corridor_grid()
        grid.log_odds[10:50, 28:32] = -1.0
        result = topo.extract_topology(grid)

        with tempfile.TemporaryDirectory() as tmp:
            path = topo.save_topology(result, tmp, "final")
            with open(path) as f:
                payload = json.load(f)

        self.assertEqual(len(payload["checkpoints"]), len(result.checkpoints))
        self.assertEqual(len(payload["edges"]), len(result.edges))


if __name__ == "__main__":
    unittest.main()
