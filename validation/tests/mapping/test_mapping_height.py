"""
Standalone unit tests for the height-band LiDAR obstacle filtering in
mapping.py. No live Unity/WebSocket connection needed - builds synthetic
scan dicts by hand.

Run with:
    uv run pytest validation/tests/mapping/test_mapping_height.py
"""
import math
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # validation/tests/mapping
_MAPPING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))), "agent", "mapping")
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from mapping import (  # noqa: E402
    _hit_height_above_root,
    scan_to_world_points,
    scan_to_world_points_3d,
    swept_clearance_ahead,
)


def _make_scan(hits, max_range=20.0, azimuth_step_deg=10.0):
    """Build a minimal synthetic scan dict. `hits` is a list of
    (v_deg, az_deg, r) tuples; each becomes its own channel with a single
    azimuth sample so hits can be placed at arbitrary, independent angles
    without needing a dense real sensor grid."""
    channels = len(hits)
    azimuth_samples = 1
    vertical_angles_deg = [v for v, _az, _r in hits]
    ranges = [r for _v, _az, r in hits]
    # azimuth_start_deg varies per-channel in a real scan, but here each
    # channel only has one azimuth sample, so encode each hit's azimuth via
    # a per-call override isn't supported by the real format - instead give
    # every channel the SAME azimuth_start_deg/step and rely on callers only
    # needing one hit's azimuth at a time (tests below use single-hit scans
    # for azimuth-sensitive checks, multi-hit scans only for angle-independent
    # height checks at azimuth 0).
    az_degs = {az for _v, az, _r in hits}
    assert len(az_degs) == 1, "helper only supports uniform azimuth across hits in one scan"
    azimuth_start_deg = next(iter(az_degs))
    return {
        "channels": channels,
        "azimuth_samples": azimuth_samples,
        "min_range": 0.05,
        "max_range": max_range,
        "azimuth_start_deg": azimuth_start_deg,
        "azimuth_step_deg": azimuth_step_deg,
        "sequence": 0,
        "timestamp_seconds": 0.0,
        "vertical_angles_deg": vertical_angles_deg,
        "ranges": ranges,
    }


class TestHitHeightAboveRoot(unittest.TestCase):
    def test_horizontal_ray_equals_offset(self):
        self.assertAlmostEqual(_hit_height_above_root(0.0, 5.0, sensor_height_offset=1.485), 1.485)

    def test_known_angle_and_range(self):
        v_deg, r, offset = 30.0, 4.0, 1.5
        expected = r * math.sin(math.radians(v_deg)) + offset
        self.assertAlmostEqual(_hit_height_above_root(v_deg, r, sensor_height_offset=offset), expected)


class TestScanToWorldPoints3DBugfix(unittest.TestCase):
    def test_horizontal_hit_includes_sensor_offset(self):
        scan = _make_scan([(0.0, 0.0, 5.0)])
        world_pos = (0.0, 0.9, 0.0)
        sensor_height_offset = 1.485

        points = scan_to_world_points_3d(scan, world_pos, yaw_deg=0.0,
                                          sensor_height_offset=sensor_height_offset)

        self.assertEqual(len(points), 1)
        _wx, wy, _wz = points[0]
        self.assertAlmostEqual(wy, world_pos[1] + sensor_height_offset,
                                msg="pre-fix this would equal world_pos[1] alone, missing the sensor offset")


class TestSweptClearanceHeightFiltering(unittest.TestCase):
    def _straight_ahead_scan(self, v_deg, r, max_range=20.0):
        # heading_deg=0 in swept_clearance_ahead's convention means azimuth 0
        # is straight ahead; place the single hit there so it's both within
        # max_lateral_deg and (if in-band) within body_radius of the travel line.
        return _make_scan([(v_deg, 0.0, r)], max_range=max_range)

    def test_ceiling_hit_ignored(self):
        # Solve for a vertical angle whose height (at r=3m) clears the default
        # max_obstacle_height=2.0m by a comfortable margin.
        r = 3.0
        target_height = 3.0  # well above the 2.0m default ceiling cutoff
        v_deg = math.degrees(math.asin((target_height - 1.485) / r))
        scan = self._straight_ahead_scan(v_deg, r)

        clearance = swept_clearance_ahead(scan, heading_deg=0.0, body_radius=0.3)

        self.assertEqual(clearance, scan["max_range"], "ceiling-height hit should be ignored entirely")

    def test_floor_hit_ignored(self):
        r = 3.0
        # height ~= 0 relative to root: solve v_deg so ly + offset ~= 0.02 (below 0.05 default min)
        target_height = 0.02
        v_deg = math.degrees(math.asin((target_height - 1.485) / r))
        scan = self._straight_ahead_scan(v_deg, r)

        clearance = swept_clearance_ahead(scan, heading_deg=0.0, body_radius=0.3)

        self.assertEqual(clearance, scan["max_range"], "floor-height hit should be ignored entirely")

    def test_in_band_hit_still_blocks(self):
        r = 3.0
        target_height = 1.0  # a normal obstacle height, comfortably within [0.05, 2.0]
        v_deg = math.degrees(math.asin((target_height - 1.485) / r))
        scan = self._straight_ahead_scan(v_deg, r)

        clearance = swept_clearance_ahead(scan, heading_deg=0.0, body_radius=0.3)

        self.assertAlmostEqual(clearance, r, places=3,
                                msg="an in-band obstacle at range r should reduce clearance to ~r")

    def test_hit_within_min_range_ignored(self):
        # v_deg=0 -> height == sensor_height_offset (1.485), comfortably in-band regardless
        # of range; r=0.03 is below the scan's min_range (0.05 by default in _make_scan) -
        # simulates a self/mount hit too close to be reliable.
        r = 0.03
        scan = self._straight_ahead_scan(0.0, r)

        clearance = swept_clearance_ahead(scan, heading_deg=0.0, body_radius=0.3)

        self.assertEqual(clearance, scan["max_range"], "hit at/below min_range should be ignored")

    def test_default_self_exclusion_range_does_not_blind_a_close_real_obstacle(self):
        # r=0.7 is beyond both min_range (0.05) and the current conservative default
        # self_exclusion_range (0.1) - it should count as a real obstacle. A wider default
        # (0.8, tried briefly) blanket-excluded this range and was confirmed live to let the
        # agent scrape an actual shelf undetected - rolled back for exactly this reason.
        r = 0.7
        scan = self._straight_ahead_scan(0.0, r)

        clearance = swept_clearance_ahead(scan, heading_deg=0.0, body_radius=0.3)

        self.assertAlmostEqual(clearance, r, places=3,
                                msg="a real-obstacle-range hit must not be silently excluded by default")

    def test_self_exclusion_range_can_still_be_widened_explicitly(self):
        # The mechanism itself still works if a caller explicitly widens it (e.g. once a
        # specific self-hit signature/range is confirmed for a given rig) - just not by default.
        r = 0.7
        scan = self._straight_ahead_scan(0.0, r)

        clearance = swept_clearance_ahead(scan, heading_deg=0.0, body_radius=0.3, self_exclusion_range=0.8)

        self.assertEqual(clearance, scan["max_range"],
                         "explicitly widened self_exclusion_range should still exclude the hit")

    def test_debug_info_identifies_closest_hit(self):
        r = 3.0
        target_height = 1.0
        v_deg = math.degrees(math.asin((target_height - 1.485) / r))
        scan = self._straight_ahead_scan(v_deg, r)

        clearance, debug_info = swept_clearance_ahead(scan, heading_deg=0.0, body_radius=0.3, debug=True)

        self.assertAlmostEqual(clearance, r, places=3)
        self.assertIsNotNone(debug_info)
        self.assertAlmostEqual(debug_info["range"], r, places=3)
        self.assertAlmostEqual(debug_info["v_deg"], v_deg, places=3)


class TestScanToWorldPointsHeightFiltering(unittest.TestCase):
    def _make_uniform_azimuth_scan(self, v_degs, r=5.0):
        return _make_scan([(v, 0.0, r) for v in v_degs], max_range=20.0)

    def test_excludes_ceiling_and_floor_includes_in_band(self):
        r = 5.0
        ceiling_v = math.degrees(math.asin((3.0 - 1.485) / r))   # ~3m height -> excluded
        floor_v = math.degrees(math.asin((0.02 - 1.485) / r))    # ~0.02m height -> excluded
        in_band_v = math.degrees(math.asin((1.0 - 1.485) / r))   # ~1m height -> included

        scan = self._make_uniform_azimuth_scan([ceiling_v, floor_v, in_band_v], r=r)
        points = scan_to_world_points(scan, world_pos=(0.0, 0.9, 0.0), yaw_deg=0.0)

        self.assertEqual(len(points), 1, "only the in-band hit should survive height filtering")

    def test_excludes_hit_within_min_range(self):
        # v_deg=0 -> height == sensor_height_offset (in-band regardless of range);
        # r=0.03 is below the scan's default min_range (0.05).
        r = 0.03

        scan = self._make_uniform_azimuth_scan([0.0], r=r)
        points = scan_to_world_points(scan, world_pos=(0.0, 0.9, 0.0), yaw_deg=0.0)

        self.assertEqual(len(points), 0, "a hit at/below min_range should not be mapped")

    def test_horizontal_placement_uses_ground_projected_distance_not_slant_range(self):
        # A downward-angled in-band hit must land at horizontal distance r*cos(v) from the
        # sensor, NOT the slant range r. The pre-fix code used r, over-placing steep channels
        # and smearing a shelf face across depth (read as an over-thick, ragged block). With
        # az=0 and yaw=0 the hit lies straight ahead on +z, so its z-coordinate IS its
        # horizontal distance from the sensor.
        v_deg, r = -20.0, 3.0   # height = 3*sin(-20)+1.485 = 0.46m, comfortably in-band
        scan = self._make_uniform_azimuth_scan([v_deg], r=r)

        points = scan_to_world_points(scan, world_pos=(0.0, 0.9, 0.0), yaw_deg=0.0)

        self.assertEqual(len(points), 1)
        _wx, wz = points[0]
        self.assertAlmostEqual(wz, r * math.cos(math.radians(v_deg)), places=6)
        self.assertLess(wz, r, "ground-projected distance must be strictly less than the slant range")


if __name__ == "__main__":
    unittest.main()
