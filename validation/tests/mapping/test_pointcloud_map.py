"""
Standalone unit tests for pointcloud_map.py's save()/include_ply behavior.

Run with:
    uv run pytest validation/tests/mapping/test_pointcloud_map.py
"""
import os
import sys
import tempfile
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # validation/tests/mapping
_MAPPING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))), "agent", "mapping")
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from pointcloud_map import PointCloudMap  # noqa: E402


class TestPointCloudMapSave(unittest.TestCase):
    def test_include_ply_false_skips_the_ply_file(self):
        # This is the periodic-save path (every --save-every steps): save_ply() re-serializes
        # the WHOLE accumulated cloud with a per-point Python write() loop every call, which
        # was the actual source of the multi-second pauses at each save boundary. Only the
        # fast, vectorized .npy should be written here.
        cloud = PointCloudMap()
        cloud.add([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])

        with tempfile.TemporaryDirectory() as tmp:
            cloud.save(tmp, "5", include_ply=False)

            self.assertTrue(os.path.exists(os.path.join(tmp, "points_5.npy")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "points_5.ply")))

    def test_default_still_writes_both_for_the_final_save(self):
        cloud = PointCloudMap()
        cloud.add([(1.0, 2.0, 3.0)])

        with tempfile.TemporaryDirectory() as tmp:
            cloud.save(tmp, "final")

            self.assertTrue(os.path.exists(os.path.join(tmp, "points_final.npy")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "points_final.ply")))

    def test_ply_vertex_count_matches_accumulated_points(self):
        cloud = PointCloudMap()
        cloud.add([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "points_final.ply")
            cloud.save_ply(path)

            with open(path) as f:
                contents = f.read()
            self.assertIn("element vertex 3", contents)


if __name__ == "__main__":
    unittest.main()
