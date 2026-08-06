"""
Standalone unit tests for mapping/graph/shelf_coverage.py - the Phase 2 shelf-hugging offset
primitive (find_shelf_checkpoint), the segment-sweeping composition built on top of it
(sweep_edge_side), and the top-level graph-splicing step (splice_shelf_checkpoints). No
live Unity/WebSocket connection needed - builds synthetic OccupancyGrid/TopologyGraph
instances directly, the same pattern test_frontier_planner.py/test_topology.py use.

Run with:
    uv run pytest validation/tests/mapping/test_shelf_coverage.py
"""
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # validation/tests/mapping
_MAPPING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))), "agent", "mapping")
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from occupancy_grid import OccupancyGrid  # noqa: E402
from frontier_planner import _inflate_occupied  # noqa: E402
from topology import Checkpoint, TopologyEdge, TopologyGraph  # noqa: E402
from shelf_coverage import (  # noqa: E402
    find_shelf_checkpoint, sweep_edge_side, splice_shelf_checkpoints, _douglas_peucker,
)


def _open_grid(size_m=6.0, resolution=0.1):
    grid = OccupancyGrid(size_m=size_m, resolution=resolution)
    grid.log_odds[:, :] = -1.0  # everything free by default
    return grid


def _two_checkpoint_graph(grid, path, a_id=0, b_id=1, a_kind="end", b_kind="end"):
    """A minimal one-edge TopologyGraph spanning `path`'s two ends - splice_shelf_checkpoints'
    tests build these directly rather than running the full extract_topology() pipeline,
    the same "construct the graph shape under test by hand" pattern find_shelf_checkpoint's
    and sweep_edge_side's own tests use for grids."""
    a = Checkpoint(id=a_id, cell=path[0], world_xz=grid.to_world(*path[0]), kind=a_kind)
    b = Checkpoint(id=b_id, cell=path[-1], world_xz=grid.to_world(*path[-1]), kind=b_kind)
    a.neighbors = [b_id]
    b.neighbors = [a_id]
    edge = TopologyEdge(a=a_id, b=b_id, length_m=len(path) * grid.res, path=path)
    return TopologyGraph(checkpoints=[a, b], edges=[edge]), a, b, edge


def _reachable_from(start_id, checkpoints):
    by_id = {c.id: c for c in checkpoints}
    seen = {start_id}
    frontier = [start_id]
    while frontier:
        current = frontier.pop()
        for neighbor_id in by_id[current].neighbors:
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                frontier.append(neighbor_id)
    return seen


class TestFindShelfCheckpoint(unittest.TestCase):
    def test_uses_max_reading_distance_when_there_is_room(self):
        grid = _open_grid()
        grid.log_odds[30, 10] = 5.0  # shelf 2.0m away

        result = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right")

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["reading_distance_m"], 1.5, places=6)
        self.assertEqual(result["shelf_cell"], (30, 10))

    def test_clamps_reading_distance_to_the_available_width(self):
        # Narrow aisle: shelf only 1.2m away - must use 1.2m, not the 1.5m max.
        grid = _open_grid()
        grid.log_odds[30, 18] = 5.0  # 1.2m away

        result = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right")

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["reading_distance_m"], 1.2, places=6)

    def test_shelf_at_exactly_max_distance_places_checkpoint_at_the_path_point(self):
        # If the shelf is already at (or within) the max reading distance, the path
        # point itself is already a fine reading distance - no offset needed.
        grid = _open_grid()
        grid.log_odds[30, 18] = 5.0  # 1.2m away

        result = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right")

        self.assertEqual(result["cell"], (30, 30))

    def test_shelf_closer_than_the_floor_yields_no_checkpoint(self):
        grid = _open_grid()
        grid.log_odds[30, 27] = 5.0  # 0.3m away - below the 1.0m floor

        result = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right")

        self.assertIsNone(result)

    def test_no_shelf_anywhere_yields_no_checkpoint(self):
        grid = _open_grid()  # nothing occupied at all

        result = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right")

        self.assertIsNone(result)

    def test_search_radius_caps_how_far_the_shelf_search_looks(self):
        # Regression: without a cap, an open floor with no nearby shelf would reach out
        # and grab an unrelated shelf far across the store.
        grid = _open_grid()
        grid.log_odds[30, 0] = 5.0  # 3.0m away

        capped = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right", search_radius_m=2.0)
        uncapped = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right", search_radius_m=5.0)

        self.assertIsNone(capped, "a shelf beyond the search radius must not be found")
        self.assertIsNotNone(uncapped, "sanity check: the same shelf IS found once the radius covers it")

    def test_left_and_right_search_opposite_directions(self):
        grid = _open_grid()
        grid.log_odds[30, 10] = 5.0  # one side
        grid.log_odds[30, 50] = 5.0  # the other side

        left = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="left")
        right = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right")

        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertNotEqual(left["shelf_cell"], right["shelf_cell"])

    def test_invalid_side_raises(self):
        grid = _open_grid()

        with self.assertRaises(ValueError):
            find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="up")

    def test_obstruction_blocking_the_straight_path_yields_no_checkpoint(self):
        # The real shelf is far enough (2.0m) that nothing sits directly ON the search
        # ray closer than it, so the raw shelf-search still correctly identifies it. But
        # a separate obstruction just off the ray (31, 25) has a body_radius=0.3 inflated
        # footprint that crosses the straight-line path from path_point to the would-be
        # candidate at (30, 20) - this must be caught, not silently ignored.
        grid = _open_grid()
        grid.log_odds[30, 10] = 5.0   # real shelf, 2.0m away
        grid.log_odds[31, 25] = 5.0   # off-ray obstruction whose inflation crosses the path

        result = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right")

        self.assertIsNone(result)

    def test_same_shelf_without_the_obstruction_is_valid(self):
        # Confirms the previous test's rejection was actually caused by the obstruction,
        # not the shelf itself.
        grid = _open_grid()
        grid.log_odds[30, 10] = 5.0

        result = find_shelf_checkpoint(grid, (30, 30), direction=(1, 0), side="right")

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["reading_distance_m"], 1.5, places=6)

    def test_out_of_bounds_direction_yields_no_checkpoint_not_an_error(self):
        grid = _open_grid()
        # Path point right at the grid's low corner; "right" of a +x tangent here
        # searches toward -z, i.e. immediately off the grid.
        result = find_shelf_checkpoint(grid, (0, 0), direction=(1, 0), side="right")

        self.assertIsNone(result)


class TestDouglasPeucker(unittest.TestCase):
    def test_preserves_endpoints_and_a_real_corner(self):
        path = [(x, 0) for x in range(0, 21)] + [(20, z) for z in range(1, 21)]  # an L
        simp = _douglas_peucker(path, epsilon_cells=1.0)
        self.assertEqual(simp[0], (0, 0))
        self.assertEqual(simp[-1], (20, 20))
        self.assertIn((20, 0), simp, "the corner must survive as a vertex")
        self.assertLessEqual(len(simp), 4, "a clean L should reduce to ~3 vertices")

    def test_collapses_sub_tolerance_jitter_to_the_endpoints(self):
        path = [(x, x % 2) for x in range(0, 21)]  # a near-straight line wobbling z 0/1
        self.assertEqual(_douglas_peucker(path, epsilon_cells=2.0), [(0, 0), (20, 0)])

    def test_short_paths_pass_through_unchanged(self):
        self.assertEqual(_douglas_peucker([(3, 3)], 1.0), [(3, 3)])
        self.assertEqual(_douglas_peucker([(1, 1), (5, 5)], 1.0), [(1, 1), (5, 5)])


class TestSweepEdgeSide(unittest.TestCase):
    def test_shelf_hugging_L_is_not_shortcut_diagonally_across_the_aisle(self):
        # Regression for the real bug: an edge that hugs a shelf face then turns. A
        # line-of-sight collapse (the old simplify_path) shortcuts it into a diagonal
        # straight across the open aisle, so the perpendicular sweep stops pointing at the
        # shelf and the whole face gets ~1 checkpoint. Douglas-Peucker keeps the run
        # parallel to the shelf, so the face is covered across its width.
        grid = _open_grid(size_m=12.0)
        grid.log_odds[20:81, 60] = 5.0                 # shelf wall at cz=60, cx 20-80
        seg1 = [(cx, 45) for cx in range(20, 81)]      # runs 1.5m south of the shelf
        seg2 = [(80, cz) for cz in range(44, 20, -1)]  # then turns south at the east end
        path = seg1 + seg2

        result = sweep_edge_side(grid, path, side="left")  # "left" of +x faces +z = the shelf

        on_shelf = sorted(c["cell"][0] for c in result if c["shelf_cell"][1] == 60)
        self.assertGreater(len(on_shelf), 2, "the shelf face must get several checkpoints, not ~1")
        self.assertGreater(max(on_shelf) - min(on_shelf), 30,
                            "checkpoints must span the shelf's width, not cluster at one end")


    def test_straight_shelf_produces_evenly_spaced_checkpoints(self):
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0  # long shelf wall, cz=20, spanning cx 15-85
        path = [(cx, 50) for cx in range(20, 81)]  # straight path, 3.0m from the shelf

        result = sweep_edge_side(grid, path, side="right")

        self.assertEqual(len(result), 7, "6.0m path at 1.0m intervals -> 7 checkpoints")
        for cp in result:
            self.assertAlmostEqual(cp["reading_distance_m"], 1.5, places=6, msg="plenty of room -> uses the max")
        spacings = [
            abs(result[i + 1]["cell"][0] - result[i]["cell"][0]) * grid.res
            for i in range(len(result) - 1)
        ]
        for spacing in spacings:
            self.assertAlmostEqual(spacing, 1.0, places=6)

    def test_checkpoints_are_ordered_along_the_path(self):
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0
        path = [(cx, 50) for cx in range(20, 81)]

        result = sweep_edge_side(grid, path, side="right")

        cxs = [cp["cell"][0] for cp in result]
        self.assertEqual(cxs, sorted(cxs), "checkpoints must come back in path-traversal order")

    def test_no_shelf_anywhere_yields_an_empty_list(self):
        grid = _open_grid(size_m=6.0)
        path = [(cx, 30) for cx in range(20, 41)]

        result = sweep_edge_side(grid, path, side="right")

        self.assertEqual(result, [])

    def test_degenerate_paths_yield_an_empty_list_not_an_error(self):
        grid = _open_grid(size_m=6.0)

        self.assertEqual(sweep_edge_side(grid, [], side="right"), [])
        self.assertEqual(sweep_edge_side(grid, [(30, 30)], side="right"), [])

    def test_precomputed_masks_give_identical_results_to_computing_internally(self):
        # sweep_edge_side accepts precomputed occupied_mask/inflated_mask specifically so
        # a caller sweeping many edges/sides over the same grid doesn't recompute
        # _inflate_occupied's full-grid pass every single call - must not change behavior.
        grid = _open_grid(size_m=6.0)
        grid.log_odds[20:41, 20] = 5.0
        path = [(cx, 35) for cx in range(20, 41)]

        computed_internally = sweep_edge_side(grid, path, side="right")
        occupied_mask = grid.log_odds > grid.OCCUPIED_THRESHOLD
        inflated_mask = _inflate_occupied(grid, 0.3)
        computed_with_precomputed_masks = sweep_edge_side(
            grid, path, side="right", occupied_mask=occupied_mask, inflated_mask=inflated_mask,
        )

        self.assertEqual(computed_internally, computed_with_precomputed_masks)

    def test_a_genuine_direction_change_produces_checkpoints_on_both_legs(self):
        # A path hugging the outside of a large rectangular shelf block's corner at a
        # full 1.5m clearance on both faces - verified (see the topology.py-style
        # empirical check this test is modeled on) to simplify into exactly two straight
        # segments, not a diagonal shortcut that cuts the corner away entirely.
        grid = _open_grid(size_m=14.0)
        grid.log_odds[40:101, 20:81] = 5.0  # big block, cx 40-100, cz 20-80
        seg1 = [(cx, 95) for cx in range(20, 116)]     # north face (cz=80) -> 1.5m away
        seg2 = [(115, cz) for cz in range(94, 4, -1)]  # east face (cx=100) -> 1.5m away
        path = seg1 + seg2

        result = sweep_edge_side(grid, path, side="right")

        self.assertGreater(len(result), 0)
        near_seg1 = [cp for cp in result if cp["cell"][1] > 85]   # high cz -> segment 1's region
        near_seg2 = [cp for cp in result if cp["cell"][0] == 115]  # cx=115 -> segment 2's region
        self.assertTrue(near_seg1, "the first leg (before the turn) must contribute checkpoints too")
        self.assertTrue(near_seg2, "the second leg (after the turn) must contribute checkpoints")
        self.assertLess(
            result.index(near_seg1[-1]), result.index(near_seg2[0]),
            "checkpoints must stay in path-traversal order across the turn",
        )
        for cp in near_seg2:
            self.assertAlmostEqual(cp["reading_distance_m"], 1.5, places=6,
                                    msg="segment 2 stays at the full offset the whole way")


class TestSpliceShelfCheckpoints(unittest.TestCase):
    def test_one_sided_shelf_replaces_the_direct_edge_with_a_chain(self):
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0  # shelf wall, cz=20
        path = [(cx, 50) for cx in range(20, 81)]  # 3.0m from the shelf
        graph, a, b, _ = _two_checkpoint_graph(grid, path)

        result = splice_shelf_checkpoints(grid, graph)

        self.assertIs(result, graph, "should mutate and return the same graph")
        shelf_cps = [c for c in graph.checkpoints if c.kind == "shelf"]
        self.assertEqual(len(shelf_cps), 7)
        self.assertTrue(all(c.shelf_side == "right" for c in shelf_cps))
        self.assertNotIn(b.id, a.neighbors, "the direct a-b link must be replaced, not kept alongside the chain")
        self.assertNotIn(a.id, b.neighbors)
        self.assertEqual(_reachable_from(a.id, graph.checkpoints), {c.id for c in graph.checkpoints},
                          "a must still reach b (and every chain checkpoint) through the spliced-in chain")

    def test_both_sides_produce_two_independent_chains_both_anchored_at_the_endpoints(self):
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0
        grid.log_odds[15:86, 80] = 5.0
        path = [(cx, 50) for cx in range(20, 81)]
        graph, a, b, _ = _two_checkpoint_graph(grid, path, a_kind="junction", b_kind="junction")

        splice_shelf_checkpoints(grid, graph)

        shelf_cps = [c for c in graph.checkpoints if c.kind == "shelf"]
        self.assertEqual({c.shelf_side for c in shelf_cps}, {"left", "right"})
        self.assertEqual(len(a.neighbors), 2, "one chain start per side")
        self.assertEqual(len(b.neighbors), 2, "one chain end per side")
        self.assertEqual(_reachable_from(a.id, graph.checkpoints), {c.id for c in graph.checkpoints})

    def test_no_shelf_anywhere_leaves_the_edge_untouched(self):
        grid = _open_grid(size_m=10.0)
        path = [(cx, 50) for cx in range(20, 81)]
        graph, a, b, edge = _two_checkpoint_graph(grid, path)

        splice_shelf_checkpoints(grid, graph)

        self.assertEqual(len(graph.checkpoints), 2)
        self.assertEqual(graph.edges, [edge])
        self.assertEqual(a.neighbors, [b.id])
        self.assertEqual(b.neighbors, [a.id])

    def test_new_checkpoint_ids_continue_from_the_existing_maximum(self):
        # Endpoint ids deliberately non-contiguous/non-zero-based, so this can't pass by
        # accident from a naive len(checkpoints)-based id scheme.
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0
        path = [(cx, 50) for cx in range(20, 81)]
        graph, a, b, _ = _two_checkpoint_graph(grid, path, a_id=5, b_id=9)

        splice_shelf_checkpoints(grid, graph)

        ids = sorted(c.id for c in graph.checkpoints)
        self.assertEqual(ids[:2], [5, 9])
        self.assertEqual(ids[2:], list(range(10, 10 + len(ids) - 2)))

    def test_a_single_checkpoint_chain_links_directly_to_both_anchors(self):
        # A short wall only near cz=50 means the path's other sample point (cz=55, 0.5m
        # further along) has no shelf within range on this side - exactly one valid
        # checkpoint for the whole edge, so first and last are the same object.
        grid = _open_grid(size_m=10.0)
        grid.log_odds[40, 48:53] = 5.0
        path = [(30, 50), (30, 55)]
        graph, a, b, _ = _two_checkpoint_graph(grid, path)

        splice_shelf_checkpoints(grid, graph)

        shelf_cps = [c for c in graph.checkpoints if c.kind == "shelf"]
        self.assertEqual(len(shelf_cps), 1)
        self.assertEqual(set(shelf_cps[0].neighbors), {a.id, b.id})

    def test_a_parallel_edge_between_the_same_pair_is_handled_independently(self):
        # Two distinct edges between the same two checkpoints (e.g. a loop aisle): only
        # the one that actually finds a shelf (path_near) should have its direct link
        # replaced; the other (path_far, shelf beyond its search radius) must be left
        # alone rather than erroring on an already-removed neighbor entry.
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0
        path_near = [(cx, 50) for cx in range(20, 81)]   # 3.0m from the shelf - in range
        path_far = [(cx, 90) for cx in range(20, 81)]    # 7.0m from the shelf - out of range
        a = Checkpoint(id=0, cell=path_near[0], world_xz=grid.to_world(*path_near[0]), kind="junction")
        b = Checkpoint(id=1, cell=path_near[-1], world_xz=grid.to_world(*path_near[-1]), kind="junction")
        a.neighbors = [1, 1]
        b.neighbors = [0, 0]
        edge_near = TopologyEdge(a=0, b=1, length_m=6.0, path=path_near)
        edge_far = TopologyEdge(a=0, b=1, length_m=6.0, path=path_far)
        graph = TopologyGraph(checkpoints=[a, b], edges=[edge_near, edge_far])

        splice_shelf_checkpoints(grid, graph)

        self.assertIn(1, a.neighbors, "edge_far's direct link must survive")
        self.assertIn(0, b.neighbors)
        shelf_cps = [c for c in graph.checkpoints if c.kind == "shelf"]
        self.assertTrue(shelf_cps)
        self.assertTrue(any(cid in a.neighbors for cid in (c.id for c in shelf_cps)),
                         "edge_near's direct link must be replaced by its chain")

    def test_ladder_rungs_link_every_checkpoint_when_both_sides_fully_overlap(self):
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0  # right, full length
        grid.log_odds[15:86, 80] = 5.0  # left, full length
        path = [(cx, 50) for cx in range(20, 81)]
        graph, a, b, _ = _two_checkpoint_graph(grid, path)

        splice_shelf_checkpoints(grid, graph)

        left_cps = [c for c in graph.checkpoints if c.shelf_side == "left"]
        right_cps = [c for c in graph.checkpoints if c.shelf_side == "right"]
        right_ids = {c.id for c in right_cps}
        for lc in left_cps:
            rung_targets = [n for n in lc.neighbors if n in right_ids]
            self.assertEqual(len(rung_targets), 1, f"checkpoint {lc.id} should have exactly one rung")
            rc = next(c for c in right_cps if c.id == rung_targets[0])
            self.assertEqual(lc.cell[0], rc.cell[0], "a rung should connect checkpoints at the same along-path position")
            self.assertIn(lc.id, rc.neighbors, "rungs must be mutual")

    def test_ladder_rungs_only_connect_positions_present_on_both_sides(self):
        # Left wall only covers cx 15-50 (checkpoints at path positions 20/30/40/50);
        # right wall covers the full 15-86 (checkpoints at 20/30/40/50/60/70/80). Only the
        # 4 overlapping positions should get a rung - the 3 right-only checkpoints must not.
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0   # right, full length
        grid.log_odds[15:51, 80] = 5.0   # left, partial length
        path = [(cx, 50) for cx in range(20, 81)]
        graph, a, b, _ = _two_checkpoint_graph(grid, path)

        splice_shelf_checkpoints(grid, graph)

        left_cps = [c for c in graph.checkpoints if c.shelf_side == "left"]
        right_cps = [c for c in graph.checkpoints if c.shelf_side == "right"]
        self.assertEqual(len(left_cps), 4)
        self.assertEqual(len(right_cps), 7)
        right_ids = {c.id for c in right_cps}
        rung_count = sum(1 for lc in left_cps for n in lc.neighbors if n in right_ids)
        self.assertEqual(rung_count, 4, "only the 4 overlapping positions should get a rung")
        unrunged_right = [c for c in right_cps if c.cell[0] > 50]
        self.assertEqual(len(unrunged_right), 3)
        for rc in unrunged_right:
            self.assertFalse(any(n in {c.id for c in left_cps} for n in rc.neighbors))

    def test_ladder_rungs_can_be_disabled(self):
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0
        grid.log_odds[15:86, 80] = 5.0
        path = [(cx, 50) for cx in range(20, 81)]
        graph, a, b, _ = _two_checkpoint_graph(grid, path)

        splice_shelf_checkpoints(grid, graph, add_ladder_rungs=False)

        left_ids = {c.id for c in graph.checkpoints if c.shelf_side == "left"}
        right_ids = {c.id for c in graph.checkpoints if c.shelf_side == "right"}
        for c in graph.checkpoints:
            if c.id in left_ids:
                self.assertFalse(any(n in right_ids for n in c.neighbors))
        rung_edges = [e for e in graph.edges
                      if (e.a in left_ids and e.b in right_ids) or (e.a in right_ids and e.b in left_ids)]
        self.assertEqual(rung_edges, [])

    def test_precomputed_masks_are_not_required_and_defaults_are_used_internally(self):
        # Regression guard: splice_shelf_checkpoints must build its own occupied/inflated
        # masks once and reuse them across every edge/side, not silently recompute (or
        # omit) them per call - covered indirectly by every other test succeeding, this
        # just pins that calling it twice on equivalent graphs is deterministic.
        grid = _open_grid(size_m=10.0)
        grid.log_odds[15:86, 20] = 5.0
        path = [(cx, 50) for cx in range(20, 81)]

        graph_a, *_ = _two_checkpoint_graph(grid, path)
        graph_b, *_ = _two_checkpoint_graph(grid, path)
        splice_shelf_checkpoints(grid, graph_a)
        splice_shelf_checkpoints(grid, graph_b)

        cells_a = [c.cell for c in graph_a.checkpoints]
        cells_b = [c.cell for c in graph_b.checkpoints]
        self.assertEqual(cells_a, cells_b)


if __name__ == "__main__":
    unittest.main()
