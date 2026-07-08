"""
Standalone unit tests for the pure-logic helpers in slamtest_frontier/explore.py.
No live Unity/WebSocket connection needed.

Run with:
    python slamtest/slamtest_frontier/test_explore.py
"""
import math
import os
import sys
import tempfile
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # overhaul/slamtest/slamtest_frontier
_SLAMTEST_DIR = os.path.dirname(_THIS_DIR)                     # overhaul/slamtest
_OVERHAUL_DIR = os.path.dirname(_SLAMTEST_DIR)                 # overhaul
for _p in (_OVERHAUL_DIR, _SLAMTEST_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("explore_frontier", os.path.join(_THIS_DIR, "explore.py"))
explore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(explore)


class TestBlockedPosition(unittest.TestCase):
    def test_straight_ahead_when_no_debug_info(self):
        pos = (0.0, 0.0, 0.0)
        bx, bz = explore.blocked_position(pos, rot_deg=0.0, clearance=1.5, clearance_debug=None)

        self.assertAlmostEqual(bx, 0.0, places=6)
        self.assertAlmostEqual(bz, 1.5, places=6)

    def test_uses_off_axis_hit_not_straight_ahead(self):
        # Regression: az_rel=-48.2 was observed live while mark_occupied() assumed the
        # obstruction was straight ahead (rot_deg alone) - poisoning empty space instead of
        # the real obstacle. The marked position must follow the actual ray's bearing/range.
        pos = (0.0, 0.0, 0.0)
        rot_deg = 24.0
        debug = {"az_rel_deg": -48.2, "range": 0.40}

        bx, bz = explore.blocked_position(pos, rot_deg, clearance=0.27, clearance_debug=debug)

        expected_bearing = math.radians(rot_deg + debug["az_rel_deg"])
        expected_x = math.sin(expected_bearing) * debug["range"]
        expected_z = math.cos(expected_bearing) * debug["range"]
        self.assertAlmostEqual(bx, expected_x, places=6)
        self.assertAlmostEqual(bz, expected_z, places=6)

        # And explicitly NOT the straight-ahead (old, buggy) position.
        straight_ahead_x = math.sin(math.radians(rot_deg)) * debug["range"]
        self.assertNotAlmostEqual(bx, straight_ahead_x, places=2)

    def test_offset_from_nonzero_position(self):
        pos = (2.0, 0.0, -3.0)  # (x, y, z) - blocked_position reads pos[0]/pos[2]
        debug = {"az_rel_deg": 10.0, "range": 2.0}
        bx, bz = explore.blocked_position(pos, rot_deg=90.0, clearance=1.0, clearance_debug=debug)

        bearing = math.radians(90.0 + 10.0)
        self.assertAlmostEqual(bx, 2.0 + math.sin(bearing) * 2.0, places=6)
        self.assertAlmostEqual(bz, -3.0 + math.cos(bearing) * 2.0, places=6)


def _make_ring_scan(obstacle_azimuths_and_ranges, max_range=20.0, azimuth_step_deg=1.0):
    """Single channel (v_deg=0, i.e. height == sensor_height_offset, safely in-band),
    one azimuth sample per degree around the full circle, all at max_range (no hit)
    except the given (azimuth_deg, range) overrides - lets a test place an obstacle
    at an exact bearing without needing a dense real sensor layout."""
    azimuth_samples = int(round(360.0 / azimuth_step_deg))
    ranges = [max_range] * azimuth_samples
    for az_deg, r in obstacle_azimuths_and_ranges:
        idx = int(round(az_deg / azimuth_step_deg)) % azimuth_samples
        ranges[idx] = r
    return {
        "channels": 1,
        "azimuth_samples": azimuth_samples,
        "min_range": 0.05,
        "max_range": max_range,
        "azimuth_start_deg": 0.0,
        "azimuth_step_deg": azimuth_step_deg,
        "sequence": 0,
        "timestamp_seconds": 0.0,
        "vertical_angles_deg": [0.0],
        "ranges": ranges,
    }


class TestFindClearHeading(unittest.TestCase):
    def test_returns_zero_offset_when_already_clear(self):
        scan = _make_ring_scan([])  # nothing anywhere

        result = explore.find_clear_heading(
            scan, desired_heading_deg=0.0, min_needed_step=0.3, safety_margin=0.1,
            body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
            sensor_height_offset=1.485, self_exclusion_range=0.1,
        )

        self.assertIsNotNone(result)
        heading, clearance, debug, offset = result
        self.assertEqual(offset, 0.0, "should not nudge at all when the straight heading is already clear")

    def test_finds_smallest_nudge_that_clears_an_off_axis_obstacle(self):
        # Obstacle dead ahead (azimuth 0) at 1.0m: with body_radius=0.3, clearing it needs
        # sin(offset) > 0.3/1.0 = 0.3 -> offset > ~17.46 degrees. With nudge_step_deg=5, the
        # first offset magnitude that actually clears it is 20 degrees, not smaller ones.
        # safety_margin=0.8 (unrealistically large, deliberately) so that even the raw 1.0m
        # straight-ahead clearance (0.2m of margin) fails min_needed_step=0.3 and the search
        # is actually forced to happen, rather than trivially succeeding at offset=0.
        scan = _make_ring_scan([(0.0, 1.0)])

        result = explore.find_clear_heading(
            scan, desired_heading_deg=0.0, min_needed_step=0.3, safety_margin=0.8,
            body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
            sensor_height_offset=1.485, self_exclusion_range=0.1,
            max_nudge_deg=30.0, nudge_step_deg=5.0,
        )

        self.assertIsNotNone(result)
        heading, clearance, debug, offset = result
        self.assertEqual(abs(offset), 20.0, "15 degrees is not enough to clear this obstacle; 20 is")
        self.assertGreaterEqual(clearance - 0.1, 0.3, "the returned heading must actually satisfy the safety margin")

    def test_returns_none_when_nothing_within_range_clears(self):
        # Obstacle closer than body_radius itself: no heading adjustment can move it
        # outside the swept-cylinder radius (even a 90-degree turn barely helps, and that's
        # far outside any reasonable nudge budget) - this must be reported as unresolvable
        # by nudging, not silently "succeed" with a heading that still isn't actually safe.
        scan = _make_ring_scan([(0.0, 0.2)])

        result = explore.find_clear_heading(
            scan, desired_heading_deg=0.0, min_needed_step=0.3, safety_margin=0.1,
            body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
            sensor_height_offset=1.485, self_exclusion_range=0.1,
            max_nudge_deg=30.0, nudge_step_deg=5.0,
        )

        self.assertIsNone(result)


class TestClearOutputDir(unittest.TestCase):
    def test_removes_leftover_files_from_a_previous_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("grid_0.npy", "grid_0.png", "grid_final.npy", "points_final.ply"):
                open(os.path.join(tmp, name), "w").close()

            explore._clear_output_dir(tmp)

            self.assertEqual(os.listdir(tmp), [])

    def test_leaves_subdirectories_alone(self):
        # Defensive: only ever remove files directly inside output_dir, never recurse into
        # (let alone delete) a subdirectory that happens to live there.
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "keep_me")
            os.makedirs(sub)
            open(os.path.join(sub, "untouched.txt"), "w").close()
            open(os.path.join(tmp, "grid_0.npy"), "w").close()

            explore._clear_output_dir(tmp)

            self.assertEqual(os.listdir(tmp), ["keep_me"])
            self.assertEqual(os.listdir(sub), ["untouched.txt"])

    def test_missing_output_dir_is_a_silent_no_op(self):
        explore._clear_output_dir(os.path.join(tempfile.gettempdir(), "definitely-does-not-exist-12345"))

    def test_empty_output_dir_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            explore._clear_output_dir(tmp)  # must not raise or print a bogus "cleared 0 files"

            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main()
