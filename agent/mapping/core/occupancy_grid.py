import numpy as np


class OccupancyGrid:
    """2D log-odds occupancy grid over the world XZ (top-down) plane."""

    FREE_THRESHOLD = -0.5
    OCCUPIED_THRESHOLD = 0.5
    FREE_DECREMENT = 0.4
    OCCUPIED_INCREMENT = 0.85

    def __init__(self, size_m=60.0, resolution=0.1):
        self.res = resolution
        self.n = int(size_m / resolution)
        self.origin = -size_m / 2.0
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)

    def cell(self, wx, wz):
        return int((wx - self.origin) / self.res), int((wz - self.origin) / self.res)

    def to_world(self, cx, cz):
        return self.origin + (cx + 0.5) * self.res, self.origin + (cz + 0.5) * self.res

    def in_bounds(self, cx, cz):
        return 0 <= cx < self.n and 0 <= cz < self.n

    def integrate(self, sensor_world_xz, hit_points_xz):
        sx, sz = self.cell(*sensor_world_xz)
        for wx, wz in hit_points_xz:
            hx, hz = self.cell(wx, wz)
            if not self.in_bounds(hx, hz):
                continue
            for cx, cz in _bresenham_line(sx, sz, hx, hz)[:-1]:
                if self.in_bounds(cx, cz):
                    self.log_odds[cx, cz] -= self.FREE_DECREMENT
            self.log_odds[hx, hz] += self.OCCUPIED_INCREMENT

    def mark_occupied(self, world_xz):
        cx, cz = self.cell(*world_xz)
        if self.in_bounds(cx, cz):
            self.log_odds[cx, cz] += self.OCCUPIED_INCREMENT

    def mark_occupied_region(self, world_xz, radius_m):
        """Mark every cell within radius_m of world_xz as occupied, not just the
        single nearest cell. A real obstacle (e.g. a shelf edge) spans far more
        than one 0.1m cell - marking only the exact hit point can take many
        repeated near-identical blocks before enough of the true footprint is
        registered for body-radius-inflated A* planning to actually route
        around it (or exclude a candidate goal too close to it). One call here
        registers the whole local danger zone immediately instead."""
        cx, cz = self.cell(*world_xz)
        r_cells = int(round(radius_m / self.res))
        for dx in range(-r_cells, r_cells + 1):
            for dz in range(-r_cells, r_cells + 1):
                if dx * dx + dz * dz > r_cells * r_cells:
                    continue
                nx, nz = cx + dx, cz + dz
                if self.in_bounds(nx, nz):
                    self.log_odds[nx, nz] += self.OCCUPIED_INCREMENT

    def frontiers(self):
        free = self.log_odds < self.FREE_THRESHOLD
        occupied = self.log_odds > self.OCCUPIED_THRESHOLD
        # "Unknown" must be the complement of free|occupied (matches how save_snapshot's
        # display array - and every other consumer of this grid - treats a cell), not a
        # narrow abs(log_odds) < 1e-6 check for literally-untouched cells. A cell grazed by
        # only one or two rays (e.g. a long-range Bresenham pass-through, or a cell at the
        # edge of the scan fan) sits at a small nonzero log-odds value that never crosses
        # either threshold - under the narrow check that cell is neither "free" nor
        # "unknown", so it can never become a frontier target and a free cell right next to
        # it never sees it as an unexplored neighbor either. Confirmed live: this silently
        # stalled exploration with large map regions still gray/unmapped, because
        # cluster_frontiers() had nothing to find once every remaining unresolved cell fell
        # into that invisible middle zone.
        unknown = ~free & ~occupied
        # Pad with False instead of using np.roll's circular wraparound, so a cell on the
        # grid boundary never sees the opposite edge as a "neighbor" (np.roll would make
        # row 0 see row n-1 and vice versa, manufacturing false frontier cells along the
        # entire perimeter).
        padded = np.pad(unknown, 1, mode="constant", constant_values=False)
        touches_unknown = np.zeros_like(unknown)
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            touches_unknown |= padded[1 + dx : 1 + dx + self.n, 1 + dz : 1 + dz + self.n]
        return np.argwhere(free & touches_unknown)


def _bresenham_line(x0, z0, x1, z1):
    points = []
    dx, dz = abs(x1 - x0), abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    err = dx - dz
    x, z = x0, z0
    while True:
        points.append((x, z))
        if x == x1 and z == z1:
            break
        e2 = 2 * err
        if e2 > -dz:
            err -= dz
            x += sx
        if e2 < dx:
            err += dx
            z += sz
    return points
