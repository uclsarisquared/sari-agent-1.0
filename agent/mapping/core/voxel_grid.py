"""
Phase 1.1 - voxelized mapping. See mapping/plans/phase1.1_voxelized_mapping.md for the
full design this implements.

The problem this fixes: OccupancyGrid.integrate() flattens every LiDAR scan straight into
a 2D (X,Z) log-odds grid, so rays passing *through* the air gaps between a shelf's stacked
levels free-mark the same 2D cell that rays hitting a level occupy - two different physical
truths at two heights fighting in one cell, and the gaps usually win, so a solid shelf face
reads as free/unknown speckle.

VoxelGrid keeps a 3D log-odds grid as the thing scans actually update, then collapses it to
the same 2D OccupancyGrid every downstream consumer already reads - with an occupied-priority
rule (occupied at any height wins) that resolves the height conflict correctly instead of
letting per-cell vote totals fight. Nothing downstream of OccupancyGrid changes.

This module's foundational piece: _bresenham_line_3d, the 3D extension of
occupancy_grid._bresenham_line used to march free space along a ray through voxels. 2D
Bresenham does not extend to 3D for free; everything else here composes on this.
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/core
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from occupancy_grid import OccupancyGrid  # noqa: E402
from mapping import iter_banded_hits, SENSOR_HEIGHT_OFFSET_M  # noqa: E402


def _bresenham_line_3d(x0, y0, z0, x1, y1, z1):
    """3D integer Bresenham from (x0,y0,z0) to (x1,y1,z1), both endpoints inclusive -
    the 3D sibling of occupancy_grid._bresenham_line, same both-ends-inclusive convention
    (so a caller can free-march path[:-1] and mark the final hit voxel occupied, exactly
    as OccupancyGrid.integrate() does in 2D).

    Drives on whichever axis has the largest absolute delta and accumulates integer error
    for the other two, so consecutive points differ by at most 1 in every axis (no gaps a
    ray could slip an obstacle through).

    In the planar case (y0 == y1) the (x,z) projection is the same line as the 2D
    _bresenham_line - same length and endpoints, every cell within one of the 2D line -
    but NOT necessarily the identical sequence: the two use different formulations
    (driving-axis here vs symmetric there) and break the tie at a diagonal .5 boundary in
    opposite directions, so an occasional single cell differs by one on the driven axis
    (verified: max deviation 1, length always equal). That's immaterial to free-marching -
    every cell on the path is decremented regardless of order or which side of a tie it
    fell on - and a 1-cell diagonal difference is far inside the body-radius inflation
    every consumer already applies.
    """
    x, y, z = x0, y0, z0
    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1
    points = [(x, y, z)]

    if dx >= dy and dx >= dz:          # X is the driving axis
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x != x1:
            x += sx
            if p1 >= 0:
                y += sy
                p1 -= 2 * dx
            if p2 >= 0:
                z += sz
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append((x, y, z))
    elif dy >= dx and dy >= dz:        # Y is the driving axis
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y != y1:
            y += sy
            if p1 >= 0:
                x += sx
                p1 -= 2 * dy
            if p2 >= 0:
                z += sz
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append((x, y, z))
    else:                              # Z is the driving axis
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z != z1:
            z += sz
            if p1 >= 0:
                y += sy
                p1 -= 2 * dz
            if p2 >= 0:
                x += sx
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append((x, y, z))
    return points


class VoxelGrid:
    """3D log-odds occupancy over world XZ plus a height axis, collapsed each scan to a
    plain 2D OccupancyGrid (self.grid) that every downstream consumer reads unchanged.

    The height axis spans exactly the obstacle band scan_to_world_points already keeps
    ([min_obstacle_height, max_obstacle_height], above-root frame - see the phase1.1 plan
    doc for why above-root), so "collapse over the whole column" == "collapse over the
    agent's body-relevant heights," and ceiling signage is excluded for free by the band's
    upper bound.
    """

    def __init__(self, size_m=60.0, resolution=0.1,
                 min_obstacle_height=0.05, max_obstacle_height=2.0,
                 sensor_height_offset=SENSOR_HEIGHT_OFFSET_M):
        self.grid = OccupancyGrid(size_m=size_m, resolution=resolution)  # the collapsed 2D output
        self.res = resolution
        self.n = self.grid.n
        self.min_h = min_obstacle_height
        self.max_h = max_obstacle_height
        self.sensor_height_offset = sensor_height_offset
        # Bins span [min_h, max_h] inclusive at the same resolution as X/Z.
        self.n_y = int((max_obstacle_height - min_obstacle_height) / resolution) + 1
        self.voxels = np.zeros((self.n, self.n, self.n_y), dtype=np.float32)

    def _height_bin(self, height_above_root):
        """Map an above-root height to a voxel bin, clamped into range. Callers pass only
        heights the band filter already accepted, so the clamp is a safety net (a height
        landing exactly on max_h), not the normal path."""
        return min(self.n_y - 1, max(0, int((height_above_root - self.min_h) / self.res)))

    def integrate(self, scan, world_pos, yaw_deg,
                  self_exclusion_range=None):
        """Fold one LiDAR scan into the voxel grid: 3D-march free space from the sensor
        voxel to each in-band hit voxel and mark the hit voxel occupied - the 3D sibling of
        OccupancyGrid.integrate, same FREE_DECREMENT/OCCUPIED_INCREMENT constants and same
        both-ends-inclusive line (free-march path[:-1], occupy the endpoint). Does NOT
        collapse - call collapse() when the 2D projection is next needed (a caller may fold
        several scans first). Reuses mapping.iter_banded_hits so the projection/height-band
        logic can never drift from the 2D path.
        """
        kw = {} if self_exclusion_range is None else {"self_exclusion_range": self_exclusion_range}
        sx, sz = self.grid.cell(world_pos[0], world_pos[2])
        sy = self._height_bin(self.sensor_height_offset)
        free_dec = OccupancyGrid.FREE_DECREMENT
        occ_inc = OccupancyGrid.OCCUPIED_INCREMENT

        n_hits = 0
        for wx, wz, height in iter_banded_hits(
            scan, world_pos, yaw_deg,
            min_obstacle_height=self.min_h, max_obstacle_height=self.max_h,
            sensor_height_offset=self.sensor_height_offset, **kw,
        ):
            hx, hz = self.grid.cell(wx, wz)
            if not self.grid.in_bounds(hx, hz):
                continue
            hy = self._height_bin(height)
            # (X, Z, Y) triples throughout so they index self.voxels directly; the 3D line
            # drives on whichever axis spans most, which for a near-horizontal LiDAR ray is
            # X or Z, with Y accumulating error - exactly the desired behavior.
            for cx, cz, cy in _bresenham_line_3d(sx, sz, sy, hx, hz, hy)[:-1]:
                if 0 <= cx < self.n and 0 <= cz < self.n and 0 <= cy < self.n_y:
                    self.voxels[cx, cz, cy] -= free_dec
            self.voxels[hx, hz, hy] += occ_inc
            n_hits += 1
        return n_hits

    def mark_blocked_region(self, world_xz, height_above_root, radius_m):
        """Record a swept-clearance-detected obstacle (explore.py's stuck branch) as
        occupied evidence in the voxel grid, at the offending hit's observed height, over a
        body-radius disc - the 3D replacement for OccupancyGrid.mark_occupied_region.

        This mark comes from swept_clearance_ahead reading real LiDAR (often a thin edge the
        height-filtered integrate missed), so it's a legitimate observation, not a nav hack -
        writing it into the voxel grid rather than the collapsed 2D grid (which the next
        collapse() would wipe) keeps one source of truth and lets it erode/reinforce like any
        other evidence. A single bin suffices: the occupied-priority collapse ORs across the
        whole column, so one occupied bin makes the 2D cell read occupied. Takes effect on the
        next collapse() - which, since collapse() runs at the top of each explore step before
        the planner reads the grid, is the same timing the old immediate 2D mark had."""
        cx, cz = self.grid.cell(*world_xz)
        cy = self._height_bin(height_above_root)
        r_cells = int(round(radius_m / self.res))
        occ_inc = OccupancyGrid.OCCUPIED_INCREMENT
        for dx in range(-r_cells, r_cells + 1):
            for dz in range(-r_cells, r_cells + 1):
                if dx * dx + dz * dz > r_cells * r_cells:
                    continue
                nx, nz = cx + dx, cz + dz
                if 0 <= nx < self.n and 0 <= nz < self.n:
                    self.voxels[nx, nz, cy] += occ_inc

    def collapse(self):
        """Project the voxel grid down to the 2D self.grid.log_odds in place, by
        occupied-priority (occupied > free > unknown). The two tests are aggregated
        DIFFERENTLY on purpose - this is the crux of the whole phase, see the phase1.1 doc:

        - OCCUPIED is per-voxel: occupied if ANY single bin crosses OCCUPIED_THRESHOLD. A
          shelf level is a thin slab only a few channels catch, so its evidence must not be
          diluted by the gap bins around it - one confidently-occupied bin wins. This is the
          shelf-face fix (a level beats the gaps at other heights in the same column).
        - FREE is by COLUMN SUM: free if the summed free evidence across the whole column
          crosses FREE_THRESHOLD. Free evidence is the OPPOSITE case - every column is
          crossed by ~all channels at that azimuth, each at a DIFFERENT height, so a single
          ray-pass lands one -FREE_DECREMENT in each of ~32 separate bins. Testing any one
          bin (as an earlier version did) never reaches threshold and read open floor as a
          fan-shaped 'unknown' cone; summing the column restores exactly the accumulation the
          flat 2D grid got for free by piling every pass into one cell.

        max() fails both ways at once and is the wrong tool: it dilutes occupied (loses thin
        levels) AND under-frees (a column of sparse single passes maxes to one -0.4, not
        free). Occupied-any + free-by-sum is the rule that gets both right.

        Writes representative trinary values (occupied +1, free -1, unknown 0); every
        downstream consumer only threshold-compares against FREE_/OCCUPIED_THRESHOLD, so the
        graded log-odds that matters for self-correction stays in the voxel grid. Mutates
        self.grid.log_odds in place (same array object every consumer holds), returns self.grid.
        """
        occupied = (self.voxels > OccupancyGrid.OCCUPIED_THRESHOLD).any(axis=2)
        column_sum = self.voxels.sum(axis=2)
        free = ~occupied & (column_sum < OccupancyGrid.FREE_THRESHOLD)
        lo = self.grid.log_odds
        lo.fill(0.0)
        lo[free] = -1.0
        lo[occupied] = 1.0   # occupied-priority: written last so it overrides free
        return self.grid
