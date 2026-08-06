"""
Standalone unit tests for mapping/core/voxel_grid.py - Phase 1.1 voxelized mapping. No live
Unity/WebSocket connection needed, same synthetic pattern as the other mapping tests.

Starts with _bresenham_line_3d, the foundational 3D ray-march primitive everything else in
voxel_grid composes on. Its properties are asserted against random lines (not a handful of
hand-picked cases) since a subtle tie-breaking or axis-dominance bug only shows up on a
fraction of directions.

Run with:
    uv run pytest validation/tests/mapping/test_voxel_grid.py
"""
import os
import random
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # validation/tests/mapping
_MAPPING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))), "agent", "mapping")
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

import numpy as np  # noqa: E402

from occupancy_grid import OccupancyGrid, _bresenham_line  # noqa: E402
from mapping import scan_to_world_points  # noqa: E402
from voxel_grid import VoxelGrid, _bresenham_line_3d  # noqa: E402

_FT, _OT = OccupancyGrid.FREE_THRESHOLD, OccupancyGrid.OCCUPIED_THRESHOLD


def _cls(log_odds, cell):
    """Trinary class of a collapsed 2D cell, by the same thresholds every consumer uses."""
    v = log_odds[cell]
    return "OCC" if v > _OT else ("FREE" if v < _FT else "UNK")


def _two_ray_shelf_gap_scan():
    """Two rays at the same azimuth (0 -> +Z), different vertical angles: a short level-hit
    ray (v=0, r=1.0) landing in the near column, and a longer gap ray (v=10, r=2.0) that
    passes over/through the near column at a higher height and hits a wall further out. This
    is the minimal reproduction of the shelf-gap conflation the whole phase exists to fix."""
    return {
        "channels": 2, "azimuth_samples": 1,
        "ranges": [1.0, 2.0],
        "max_range": 20.0, "min_range": 0.05,
        "azimuth_start_deg": 0.0, "azimuth_step_deg": 1.0,
        "vertical_angles_deg": [0.0, 10.0],
    }


def _chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


class TestBresenhamLine3D(unittest.TestCase):
    def test_endpoints_are_inclusive(self):
        pts = _bresenham_line_3d(1, 2, 3, 9, -4, 7)
        self.assertEqual(pts[0], (1, 2, 3))
        self.assertEqual(pts[-1], (9, -4, 7))

    def test_single_point_line(self):
        self.assertEqual(_bresenham_line_3d(5, 5, 5, 5, 5, 5), [(5, 5, 5)])

    def test_pure_axis_lines_have_expected_length(self):
        # A pure-axis line visits every integer cell inclusive: |delta| + 1 points.
        self.assertEqual(len(_bresenham_line_3d(0, 0, 0, 10, 0, 0)), 11)
        self.assertEqual(len(_bresenham_line_3d(0, 0, 0, 0, 7, 0)), 8)
        self.assertEqual(len(_bresenham_line_3d(0, 0, 0, 0, 0, 4)), 5)

    def test_no_gaps_over_random_lines(self):
        # Every consecutive pair must be a unit step in at least one axis and never more
        # than one in any axis - otherwise a ray could tunnel through an obstacle voxel.
        # Covers all three axis-dominance branches via random directions.
        rng = random.Random(0)
        for _ in range(5000):
            a = (rng.randint(-40, 40), rng.randint(-40, 40), rng.randint(-40, 40))
            b = (rng.randint(-40, 40), rng.randint(-40, 40), rng.randint(-40, 40))
            pts = _bresenham_line_3d(*a, *b)
            self.assertEqual(pts[0], a)
            self.assertEqual(pts[-1], b)
            for p, q in zip(pts, pts[1:]):
                self.assertEqual(_chebyshev(p, q), 1)

    def test_planar_reduction_matches_2d_up_to_tie_breaking(self):
        # With y0 == y1 the (x,z) projection is the SAME line as the 2D _bresenham_line:
        # same length, same endpoints, and every cell within one of the 2D line. It is NOT
        # asserted to be the identical sequence - the two formulations break diagonal ties
        # in opposite directions, so a single cell can differ by one on the driven axis.
        # That's immaterial to free-marching (every cell is decremented regardless) and
        # this test pins exactly that weaker-but-true guarantee so a future change can't
        # silently widen the gap past one cell.
        rng = random.Random(1)
        for _ in range(5000):
            x0, z0 = rng.randint(-50, 50), rng.randint(-50, 50)
            x1, z1 = rng.randint(-50, 50), rng.randint(-50, 50)
            proj = [(p[0], p[2]) for p in _bresenham_line_3d(x0, 0, z0, x1, 0, z1)]
            line2d = _bresenham_line(x0, z0, x1, z1)
            self.assertEqual(len(proj), len(line2d))
            self.assertEqual(proj[0], line2d[0])
            self.assertEqual(proj[-1], line2d[-1])
            for a, b in zip(proj, line2d):
                self.assertLessEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_constant_driven_axis_stays_constant_in_planar_line(self):
        # A planar (constant-y) line must keep y fixed the whole way - a stray y wobble
        # would mean the ray leaks into an adjacent height bin it never physically crossed.
        rng = random.Random(2)
        for _ in range(2000):
            y = rng.randint(-9, 9)
            pts = _bresenham_line_3d(rng.randint(-40, 40), y, rng.randint(-40, 40),
                                     rng.randint(-40, 40), y, rng.randint(-40, 40))
            self.assertTrue(all(p[1] == y for p in pts))

    def test_reversing_endpoints_gives_the_same_line_up_to_tie_breaking(self):
        # Driving-axis Bresenham is NOT reversal-symmetric - it breaks diagonal ties by
        # direction, so a->b and b->a can differ by a cell. Free-marching never needs
        # symmetry (rays always go sensor->hit), so rather than force it, this pins the
        # real, benign behavior: same length, and any differing cell within one of the
        # other line - so a future change can't silently turn a tie-break difference into
        # a genuine divergence.
        rng = random.Random(3)
        for _ in range(3000):
            a = (rng.randint(-40, 40), rng.randint(-40, 40), rng.randint(-40, 40))
            b = (rng.randint(-40, 40), rng.randint(-40, 40), rng.randint(-40, 40))
            fwd = _bresenham_line_3d(*a, *b)
            rev = _bresenham_line_3d(*b, *a)
            self.assertEqual(len(fwd), len(rev))
            for c in fwd:
                self.assertTrue(any(_chebyshev(c, d) <= 1 for d in rev))


class TestVoxelGridHeightBins(unittest.TestCase):
    def test_bin_count_and_endpoints(self):
        vg = VoxelGrid(size_m=6.0, resolution=0.1, min_obstacle_height=0.05, max_obstacle_height=2.0)
        self.assertEqual(vg.n_y, 20)                       # (2.0-0.05)/0.1 -> 19, +1 inclusive
        self.assertEqual(vg._height_bin(0.05), 0)          # band floor -> first bin
        self.assertEqual(vg._height_bin(2.0), vg.n_y - 1)  # band ceiling -> last bin

    def test_bin_is_monotonic_and_clamped(self):
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        self.assertLess(vg._height_bin(0.5), vg._height_bin(1.5))
        self.assertEqual(vg._height_bin(-10.0), 0)          # below floor clamps up
        self.assertEqual(vg._height_bin(99.0), vg.n_y - 1)  # above ceiling clamps down


class TestCollapse(unittest.TestCase):
    def test_occupied_level_beats_free_in_the_same_column(self):
        # A thin shelf level (one confidently-occupied bin) must win over free/gap bins at
        # other heights in the same column - even when the free evidence outweighs it in sum.
        # This is the shelf-face fix, and it's why OCCUPIED is tested per-voxel, not by sum.
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        vg.voxels[10, 10, 5] = OccupancyGrid.OCCUPIED_INCREMENT   # one level hit
        vg.voxels[10, 10, 8] = -5.0                               # lots of gap free evidence
        vg.collapse()
        self.assertEqual(_cls(vg.grid.log_odds, (10, 10)), "OCC")

    def test_free_evidence_split_thin_across_heights_still_reads_free(self):
        # Regression (the fan-shaped 'unknown cone' bug): every column is crossed by ~all
        # channels at that azimuth, each at a DIFFERENT height, so one ray-pass drops a single
        # -FREE_DECREMENT into each of many separate bins, none individually crossing
        # FREE_THRESHOLD. The column SUM must decide free, matching how the flat 2D grid piled
        # every pass into one cell. An earlier any-bin<threshold test read this as unknown.
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        for b in (3, 6, 9, 12, 15):
            vg.voxels[20, 20, b] = -OccupancyGrid.FREE_DECREMENT  # one pass each, 5 heights
        vg.collapse()
        self.assertEqual(_cls(vg.grid.log_odds, (20, 20)), "FREE")

    def test_a_single_ray_pass_is_still_not_enough_to_read_free(self):
        # ...but one pass total (one bin, one -0.4) is genuinely too little evidence -> unknown,
        # exactly as the 2D grid also needs >1 pass to cross FREE_THRESHOLD. Guards the
        # column-sum fix against over-correcting into marking barely-grazed columns free.
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        vg.voxels[25, 25, 7] = -OccupancyGrid.FREE_DECREMENT
        vg.collapse()
        self.assertEqual(_cls(vg.grid.log_odds, (25, 25)), "UNK")

    def test_fully_free_and_untouched_columns(self):
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        vg.voxels[30, 30, :] = -2.0    # an all-freed column
        vg.collapse()
        self.assertEqual(_cls(vg.grid.log_odds, (30, 30)), "FREE")
        self.assertEqual(_cls(vg.grid.log_odds, (40, 40)), "UNK")  # untouched

    def test_mutates_the_same_grid_array_in_place_and_returns_it(self):
        # explore.py/the planner hold a reference to the inner OccupancyGrid and read its
        # .log_odds; collapse must update THAT array, not swap in a new one.
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        same_array = vg.grid.log_odds
        vg.voxels[5, 5, 2] = 3.0
        returned = vg.collapse()
        self.assertIs(returned, vg.grid)
        self.assertIs(vg.grid.log_odds, same_array)
        self.assertEqual(_cls(vg.grid.log_odds, (5, 5)), "OCC")


class TestIntegrate(unittest.TestCase):
    def test_single_ray_frees_the_path_and_occupies_the_endpoint(self):
        # One level ray straight along +Z: sensor (0,0)->cell(30,30), hit (0,1.0)->cell(30,40),
        # both at sensor height (bin 14). The path is freed, the endpoint occupied, and the
        # endpoint is not also freed (both-ends-inclusive line, free-marched with [:-1]).
        scan = {
            "channels": 1, "azimuth_samples": 1, "ranges": [1.0],
            "max_range": 20.0, "min_range": 0.05,
            "azimuth_start_deg": 0.0, "azimuth_step_deg": 1.0, "vertical_angles_deg": [0.0],
        }
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        vg.integrate(scan, (0.0, 0.0, 0.0), 0.0)

        self.assertAlmostEqual(vg.voxels[30, 40, 14], OccupancyGrid.OCCUPIED_INCREMENT, places=5)
        self.assertAlmostEqual(vg.voxels[30, 35, 14], -OccupancyGrid.FREE_DECREMENT, places=5)
        # the occupied endpoint column has no free bins
        self.assertFalse((vg.voxels[30, 40] < 0).any())

    def test_out_of_bounds_hit_is_skipped_without_error(self):
        # A hit projecting outside the (small) grid must be dropped, not raise.
        scan = {
            "channels": 1, "azimuth_samples": 1, "ranges": [50.0],  # way past a 6m grid
            "max_range": 100.0, "min_range": 0.05,
            "azimuth_start_deg": 0.0, "azimuth_step_deg": 1.0, "vertical_angles_deg": [0.0],
        }
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        vg.integrate(scan, (0.0, 0.0, 0.0), 0.0)  # must not raise
        self.assertFalse((vg.voxels > 0).any(), "no in-bounds occupied hit expected")

    def test_shelf_gap_reads_occupied_where_the_2d_grid_muddles_it(self):
        scan = _two_ray_shelf_gap_scan()
        pos, yaw = (0.0, 0.0, 0.0), 0.0

        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        vg.integrate(scan, pos, yaw)
        vg.collapse()
        near = vg.grid.cell(0.0, 1.0)

        # The voxel grid recovers the shelf face...
        self.assertEqual(_cls(vg.grid.log_odds, near), "OCC")
        # ...while the current 2D pipeline, on the identical scan, cannot (the +0.85 level hit
        # and the -0.4 gap-pass sum to +0.45 in one flat cell -> not occupied).
        g2 = OccupancyGrid(size_m=6.0, resolution=0.1)
        g2.integrate((pos[0], pos[2]), scan_to_world_points(scan, pos, yaw))
        self.assertNotEqual(_cls(g2.log_odds, near), "OCC")

    def test_shelf_gap_keeps_level_and_gap_at_distinct_heights(self):
        # The mechanism, not just the outcome: the occupied level and the freed gap-pass must
        # land in DIFFERENT height bins of the same column - that separation is the whole
        # point (a flat 2D cell cannot hold both).
        vg = VoxelGrid(size_m=6.0, resolution=0.1)
        vg.integrate(_two_ray_shelf_gap_scan(), (0.0, 0.0, 0.0), 0.0)
        col = vg.voxels[vg.grid.cell(0.0, 1.0)[0], vg.grid.cell(0.0, 1.0)[1]]
        occ_bins = set(np.where(col > 0)[0])
        free_bins = set(np.where(col < 0)[0])
        self.assertTrue(occ_bins and free_bins)
        self.assertFalse(occ_bins & free_bins, "level and gap must occupy different bins")


if __name__ == "__main__":
    unittest.main()
