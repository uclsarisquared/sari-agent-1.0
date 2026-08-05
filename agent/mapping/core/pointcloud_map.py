import os

import numpy as np


class PointCloudMap:
    """Accumulates 3D world-frame LiDAR hits for visualization/export.

    Kept separate from OccupancyGrid: navigation/frontier logic stays on the
    existing 2D grid, this just piles up raw XYZ points alongside it.
    """

    def __init__(self):
        self._points = []

    def add(self, points_xyz):
        self._points.extend(points_xyz)

    def __len__(self):
        return len(self._points)

    def to_array(self):
        if not self._points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(self._points, dtype=np.float32)

    def save_npy(self, path):
        np.save(path, self.to_array())

    def save_ply(self, path):
        """Re-serializes the ENTIRE accumulated cloud every call (self._points is never
        pruned), one Python-level f.write() per point - unlike save_npy's vectorized
        np.save, this doesn't scale, and gets slower each call as the run goes on since it
        rewrites points already written out unchanged by an earlier call. Fine for a
        one-off final export; expensive as a periodic per-step save - see save()'s
        include_ply."""
        pts = self.to_array()
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for x, y, z in pts:
                f.write(f"{x} {y} {z}\n")

    def save(self, output_dir, tag, include_ply=True):
        """include_ply=False skips the .ply write - save_ply() re-serializes the WHOLE
        accumulated cloud with a per-point Python write() loop (see its docstring), which
        gets slower every call as the cloud grows across a run. Callers doing frequent
        periodic saves (e.g. every --save-every steps) should pass False and reserve the
        .ply export for the final save."""
        os.makedirs(output_dir, exist_ok=True)
        self.save_npy(os.path.join(output_dir, f"points_{tag}.npy"))
        if include_ply:
            self.save_ply(os.path.join(output_dir, f"points_{tag}.ply"))
