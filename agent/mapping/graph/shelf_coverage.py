"""
Phase 2 - shelf-hugging checkpoint generation. See mapping/plans/
phase2_shelf_hugging_coverage.md for the full design this implements.

This module's first, foundational piece: given a point on a (simplified) path and its
local tangent direction, find the nearest shelf face on one side and return a validated
reading-distance checkpoint there - or None if there isn't a usable one. Everything else
in Phase 2 (walking a simplified segment, dropping checkpoints at intervals, splicing
chains into Phase 1's graph) composes from repeated calls to this.
"""
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/graph
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from occupancy_grid import _bresenham_line  # noqa: E402
from frontier_planner import _inflate_occupied, _line_of_sight  # noqa: E402
from topology import Checkpoint, TopologyEdge  # noqa: E402


MIN_READING_DISTANCE_M = 1
MAX_READING_DISTANCE_M = 1.0
"""How far back from a shelf face a checkpoint is placed - i.e. the VLM's reading distance.

Measured history (stepping the agent in at cp067 and re-asking the annotator at each distance):
1.5m sat just outside the legible range - variants came back mostly null and brand logos were
unreadable (the model answered with the product's own name instead). At 1.2m variants came back
correct ("Mild", "Cheesier", "Original", "Sour Cream & Onion") and a product it had missed
entirely ("Platos") appeared. On THAT shelf, 1.0m read no better than 1.2m - dry goods with large
flat labels are already legible at 1.2m. 1.0m is set anyway for the harder cases (small curved
labels, product behind fridge glass) where the extra pixels may matter, and because Phase 3
measured near-zero recall on the refrigerated categories (Juice 0/16, Dairies 2/20).

The floor is NOT this constant - it is explore.py's executor, which refuses to move within
`safety_margin + min_step` (0.65 + 0.1 = 0.75m) of an obstacle ahead. 1.0m leaves only 0.25m of
headroom over that, and it is not free: measured on the frozen grid, 1.2m put ZERO shelf
checkpoints below the executor's standoff, while 1.0m puts a couple of them at ~0.64m clearance -
i.e. nodes the navigator may refuse to reach. 0.8m would leave 0.05m, where LiDAR noise reading
0.74 trips the nudge-then-escape path at every shelf and re-creates the Phase-2 wedge dynamics.

Closer also costs VERTICAL framing: the camera sits at ~1.485m and capture varies yaw only, so the
nearer you stand the more of a tall shelf falls outside the frame (the bottom rows first). The
intended mitigation is a pitched capture (straight + angled down) rather than backing off - that
capability does not exist yet.

Careful when tuning: placement is `min(MAX_READING_DISTANCE_M, dist_to_shelf)` and is then
REJECTED if it lands below MIN_READING_DISTANCE_M. With MAX == MIN == 1.0 there is zero tolerance -
any shelf whose path distance is even slightly under 1.0m loses its checkpoint outright. Lower this
below MIN and every shelf checkpoint silently vanishes."""

DEFAULT_SHELF_SEARCH_RADIUS_M = 2.0
"""Caps how far the perpendicular shelf-search is allowed to look, so a stretch of open
floor with no nearby shelf can't reach out and grab an unrelated shelf far across the
store as "the nearest occupied cell." Lowered from an initial 3.0m after measuring
against a real store layout (Store 2 v2.json): real shelves there often sit flush against
outer walls, so radius alone can't cleanly separate "real shelf" from "bare wall" at any
setting - shrinking far enough to exclude walls also excludes most real shelf coverage
(tested down to 1.5m; that cut shelf checkpoints from 89 to 9 on a real run, clearly too
aggressive). 2.0m was chosen purely to cut overall density (89 -> 45 on that same run)
while Stage 1 of Phase 3's two-stage annotation (see phase3_vlm_annotation_pass.md)
handles the remaining shelf-vs-wall ambiguity cheaply, per-checkpoint, instead."""

DEFAULT_CHECKPOINT_INTERVAL_M = 2.0
"""Spacing between consecutive shelf-hugging checkpoints along one side of an aisle.

History matters here, because this value has been 2.0 -> 1.0 -> 2.0 and the round trip is NOT a
flip-flop - the sampling bug underneath it got fixed in between:

  * It was 2.0m with `round` sampling in sweep_edge_side. That combination lost whole shelves: a
    2.7m edge gave round(2.7/2.0) = 1 step, i.e. only its two ENDPOINTS, skipping the shelf's
    middle. A west-wall shelf measured 0 checkpoints while central shelves next to 7-8m edges
    were densely covered - coverage scaled with EDGE length, not shelf extent.
  * The fix for that was two changes at once: round -> ceil (so the spacing can never EXCEED the
    interval - the same 2.7m edge now gives ceil(2.7/2.0) = 2 steps = 3 sample points), AND
    dropping to 1.0m. Only the first was load-bearing; the second was belt-and-braces.
  * 1.0m then measurably OVER-sampled. Phase 3's real annotation run showed adjacent checkpoints
    re-reading the same products: "Pepero" appeared at 9 checkpoints, "Pringles" at 9, on a store
    with ~9 shelf units total. That duplication is wasted VLM calls and a noisy product index.

So: back to 2.0m, which is safe now that `ceil` guarantees the endpoints-only failure can't
recur. Measured on the frozen grid: 65 checkpoints at 1.0m -> 39 at 2.0m.

Do NOT expect this knob to thin the graph much further - node count is floored by the NUMBER of
shelf-adjacent skeleton edges (each contributes at least its endpoints), not by spacing: 1.5m
gives 41, 2.0m gives 39, 2.5m gives 33. Getting to the ~16 that the store's actual shelf count
implies needs per-shelf-face consolidation (merging the nodes that sit on one physical face),
which is an algorithm change, not a value change.

Whenever this changes, eyeball shelf_graph_<tag>.png before trusting it - the failure mode this
constant guards against (a whole shelf face with no checkpoint) is obvious in the picture and
invisible in the count."""

DEFAULT_SIMPLIFY_EPSILON_M = 0.3
"""Max deviation for the Douglas-Peucker simplification of an edge's skeleton path before
sweeping it. Deliberately NOT frontier_planner.simplify_path() (line-of-sight collapsing):
that shortcuts a run-along-a-shelf-then-turn path into a diagonal straight across the open
aisle - great for a shortest travel path, wrong here, because the perpendicular sweep off a
diagonal no longer points at the shelf the edge was hugging (confirmed live: a shelf's whole
south face got 1 checkpoint instead of a spread across its width). Douglas-Peucker instead
keeps the simplified path within this tolerance of the raw one, so segments follow the
aisle's real shape (and thus the shelf) while still removing per-cell 8-connected tangent
jitter and preserving genuine corners as vertices."""


def _douglas_peucker(points, epsilon_cells):
    """Ramer-Douglas-Peucker polyline simplification: drop points that lie within
    epsilon_cells (perpendicular distance) of the straight line between the kept anchors,
    recursively. Keeps endpoints and any genuine bend, stays within epsilon of the input -
    shape-preserving, unlike a line-of-sight collapse. Iterative stack (not recursion) so a
    long skeleton path can't blow the Python recursion limit."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        ax, az = points[lo]
        bx, bz = points[hi]
        seg_len = math.hypot(bx - ax, bz - az)
        dmax, idx = -1.0, -1
        for i in range(lo + 1, hi):
            px, pz = points[i]
            if seg_len < 1e-9:
                dist = math.hypot(px - ax, pz - az)
            else:
                dist = abs((bx - ax) * (az - pz) - (ax - px) * (bz - az)) / seg_len
            if dist > dmax:
                dmax, idx = dist, i
        if dmax > epsilon_cells:
            keep[idx] = True
            stack.append((lo, idx))
            stack.append((idx, hi))
    return [p for p, k in zip(points, keep) if k]


def _perpendicular(direction, side):
    """Rotate a (dx, dz) path tangent 90 degrees to get the outward search direction for
    the given side. "left"/"right" are just two consistent, opposite choices - grid cell
    space has no canonical compass mapping, so nothing depends on which is which as long
    as they stay opposite (verified by test)."""
    dx, dz = direction
    if side == "left":
        return (-dz, dx)
    if side == "right":
        return (dz, -dx)
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def _find_nearest_occupied_along_ray(occupied_mask, grid, origin_cell, unit_dir, max_radius_m):
    """Nearest occupied cell stepping from origin_cell along unit_dir, capped at
    max_radius_m. Uses the RAW occupancy mask, not body-radius-inflated - this is meant
    to find the true shelf surface (reading distance is measured from the real shelf
    face), not an already safety-padded boundary; inflation only matters later, for
    validating whether a candidate checkpoint is safe to stand at."""
    max_radius_cells = int(round(max_radius_m / grid.res))
    if max_radius_cells <= 0:
        return None
    far_cell = (
        int(round(origin_cell[0] + unit_dir[0] * max_radius_cells)),
        int(round(origin_cell[1] + unit_dir[1] * max_radius_cells)),
    )
    for cx, cz in _bresenham_line(origin_cell[0], origin_cell[1], far_cell[0], far_cell[1]):
        if (cx, cz) == origin_cell:
            continue
        if not grid.in_bounds(cx, cz):
            return None
        if occupied_mask[cx, cz]:
            return (cx, cz)
    return None


def find_shelf_checkpoint(grid, path_point, direction, side, *,
                           search_radius_m=DEFAULT_SHELF_SEARCH_RADIUS_M,
                           min_reading_distance_m=MIN_READING_DISTANCE_M,
                           max_reading_distance_m=MAX_READING_DISTANCE_M,
                           body_radius=0.3,
                           occupied_mask=None, inflated_mask=None):
    """The Phase 2 primitive: given a point on a (simplified) path and its local tangent
    direction, look outward on `side` ("left"/"right") for the nearest shelf face and
    return a validated reading-distance checkpoint there.

    Returns a dict {"cell": (cx, cz), "world_xz": (x, z), "reading_distance_m": float,
    "shelf_cell": (cx, cz)}, or None if there's no shelf within search_radius_m, the
    shelf is closer than min_reading_distance_m, or the resulting point isn't safely
    reachable in a straight line from path_point.

    Reading distance is a RANGE, not one fixed number: use the largest value in
    [min_reading_distance_m, max_reading_distance_m] that still fits before the shelf
    (min(max_reading_distance_m, distance-to-shelf)), clamped down toward the floor only
    by that geometric fit - not by retrying at other distances after a validation
    failure. Retrying at a smaller reading distance on the SAME ray can never rescue a
    validation failure: every point on that ray lies on the straight line from
    path_point through the shelf cell, so if the line-of-sight check fails up to the
    nearer (larger-reading-distance) point, it necessarily also fails for every farther
    point on the same ray (the farther line-of-sight is a strict superset of the nearer
    one). A genuine obstruction between path_point and the shelf means there just isn't a
    usable checkpoint here - the caller should move on to the next point along the path,
    not retry distances on this one.

    Validation reuses frontier_planner's own passability model rather than a bespoke one,
    so this can never disagree with what A*/the real-time safety system consider
    walkable: `_line_of_sight()` over the body-radius-inflated mask checks every cell
    from path_point to the candidate (including the candidate itself) in one pass - this
    is a short, direct sideways step off a path already being walked, not a "does some
    route exist" question, so a straight-line check is the right shape for it.
    """
    if occupied_mask is None:
        occupied_mask = grid.log_odds > grid.OCCUPIED_THRESHOLD
    if inflated_mask is None:
        inflated_mask = _inflate_occupied(grid, body_radius)

    unit_dir = _perpendicular(direction, side)
    norm = math.hypot(*unit_dir)
    if norm < 1e-9:
        return None
    unit_dir = (unit_dir[0] / norm, unit_dir[1] / norm)

    shelf_cell = _find_nearest_occupied_along_ray(occupied_mask, grid, path_point, unit_dir, search_radius_m)
    if shelf_cell is None:
        return None

    dist_to_shelf_m = math.hypot(shelf_cell[0] - path_point[0], shelf_cell[1] - path_point[1]) * grid.res
    reading_distance_m = min(max_reading_distance_m, dist_to_shelf_m)
    if reading_distance_m < min_reading_distance_m:
        return None  # shelf is closer than even the minimum comfortable reading distance

    offset_cells = (dist_to_shelf_m - reading_distance_m) / grid.res
    candidate_cell = (
        int(round(path_point[0] + unit_dir[0] * offset_cells)),
        int(round(path_point[1] + unit_dir[1] * offset_cells)),
    )

    if not grid.in_bounds(*candidate_cell):
        return None
    if not _line_of_sight(grid, path_point, candidate_cell, occupied_mask=inflated_mask):
        return None

    return {
        "cell": candidate_cell,
        "world_xz": grid.to_world(*candidate_cell),
        "reading_distance_m": reading_distance_m,
        "shelf_cell": shelf_cell,
    }


def sweep_edge_side(grid, path_cells, side, *,
                     interval_m=DEFAULT_CHECKPOINT_INTERVAL_M,
                     search_radius_m=DEFAULT_SHELF_SEARCH_RADIUS_M,
                     min_reading_distance_m=MIN_READING_DISTANCE_M,
                     max_reading_distance_m=MAX_READING_DISTANCE_M,
                     body_radius=0.3, simplify_epsilon_m=DEFAULT_SIMPLIFY_EPSILON_M,
                     occupied_mask=None, inflated_mask=None):
    """Walk one shelf face along a Phase 1 TopologyEdge's raw skeleton path (its `path`
    field), returning an ordered chain of validated checkpoints - the Phase 2 unit one
    step up from the single-point find_shelf_checkpoint() primitive.

    First simplifies path_cells into a small number of straight segments via
    shape-preserving Douglas-Peucker (_douglas_peucker, tolerance simplify_epsilon_m)
    instead of working off the raw per-cell skeleton, so each segment has one constant
    direction to offset from rather than a jittery per-point tangent - see "Direction &
    corner handling" in the phase2 plan doc, and _douglas_peucker's docstring for why a
    line-of-sight collapse (frontier_planner.simplify_path) is specifically wrong here (it
    shortcuts a shelf-hugging path diagonally across the aisle, so the perpendicular sweep
    stops pointing at the shelf). Every segment is sampled at interval_m spacing, always
    including its own endpoint - which means a corner (where two segments meet) naturally
    gets its own checkpoint attempt, using the INCOMING segment's direction, with no
    special-case corner logic needed: it's just the last sample of one segment and the
    first of the next (the shared vertex is only sampled once - the outgoing segment skips
    resampling its own start).

    A sampled point with no valid checkpoint on this side (see find_shelf_checkpoint's
    docstring for why retrying it is pointless) is simply skipped, not retried - the next
    sampled point along the path is an independent fresh attempt.

    occupied_mask/inflated_mask can be precomputed once by the caller and passed through
    (recommended when sweeping many edges/sides over the same grid) to avoid recomputing
    _inflate_occupied's full-grid pass on every single checkpoint in the sweep.
    """
    if occupied_mask is None:
        occupied_mask = grid.log_odds > grid.OCCUPIED_THRESHOLD
    if inflated_mask is None:
        inflated_mask = _inflate_occupied(grid, body_radius)

    path_cells = [(int(c[0]), int(c[1])) for c in path_cells]
    if len(path_cells) < 2:
        return []

    simplified = _douglas_peucker(path_cells, simplify_epsilon_m / grid.res)
    if len(simplified) < 2:
        return []

    checkpoints = []
    for seg_idx in range(len(simplified) - 1):
        seg_start = simplified[seg_idx]
        seg_end = simplified[seg_idx + 1]
        direction = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
        seg_length_m = math.hypot(*direction) * grid.res
        if seg_length_m < 1e-9:
            continue

        # ceil, not round: round(2.7m / 2.0m) = 1 samples only a 2.7m segment's two endpoints,
        # skipping its middle - which is exactly how a shelf sitting mid-segment gets 0
        # checkpoints. ceil makes the actual sample spacing never EXCEED interval_m, so no
        # shelf longer than interval_m can fall entirely between two consecutive samples.
        n_steps = max(1, math.ceil(seg_length_m / interval_m))
        first_step = 0 if seg_idx == 0 else 1  # skip resampling the vertex shared with the previous segment
        for step in range(first_step, n_steps + 1):
            t = step / n_steps
            point_cell = (
                int(round(seg_start[0] + direction[0] * t)),
                int(round(seg_start[1] + direction[1] * t)),
            )
            checkpoint = find_shelf_checkpoint(
                grid, point_cell, direction, side,
                search_radius_m=search_radius_m,
                min_reading_distance_m=min_reading_distance_m,
                max_reading_distance_m=max_reading_distance_m,
                body_radius=body_radius,
                occupied_mask=occupied_mask, inflated_mask=inflated_mask,
            )
            if checkpoint is not None:
                # point_cell (the on-path sample this checkpoint was found from, before its
                # perpendicular offset) doesn't depend on `side` - a left and a right sweep
                # over the same path_cells/interval_m sample the exact same point_cell
                # sequence. Stamping it here gives splice_shelf_checkpoints' ladder-rung
                # pairing an exact shared key to match corresponding left/right checkpoints
                # by, instead of a fuzzy nearest-position search.
                checkpoint["path_point"] = point_cell
                checkpoints.append(checkpoint)

    return checkpoints


DEFAULT_LANDMARK_MIN_CELLS = 40
DEFAULT_LANDMARK_OBSERVED_RADIUS_M = 1.2
DEFAULT_LANDMARK_MAX_APPROACH_M = 4.0
DEFAULT_LANDMARK_INTERACTION_GAP_M = 0.20
"""Gap between the agent's BODY SURFACE and a landmark when it is close enough to put items down.

Stated as a surface gap, not a centre distance, because centre distance is the misleading one: the
body is body_radius (0.3m) wide, so an agent whose CENTRE is 0.3m from the counter is already
touching it, and anything nearer is inside the counter. A 0.1-0.3m reach gap therefore means a
centre distance of 0.4-0.6m, which is what this produces (0.3 + 0.2 = 0.5m by default).

That distance is deliberately INSIDE explore.py's 0.75m executor standoff, so a normal goto() will
not drive here - it stops ~0.75m out. Closing the last 0.25m is a docking move: the same goto()
with --safety-margin lowered (the standoff is a parameter, not a hard limit). Phase 4 owns that;
the pose is computed and stored now so it does not have to be re-derived later."""

DEFAULT_LANDMARK_MIN_CLEARANCE_M = 0.75
"""Clearance a landmark's standing cell needs from EVERY obstacle, not just the one it reads.

This is explore.py's executor standoff (safety_margin 0.65 + min_step 0.1) and it is here because
the first version omitted it: the counter's node was placed 1.00m from the counter but 0.57m from
an unrelated structure behind it, and audit_standability immediately flagged it as a node the
navigator would refuse to walk to. Being at reading distance from the target is necessary and not
sufficient - the spot also has to be somewhere the agent can actually stand."""
DEFAULT_LANDMARK_MIN_FREE_PERIMETER = 0.65
"""Fraction of a structure's perimeter that must border FREE space for it to count as a landmark
rather than a wall. On the frozen grid the separation is wide: the checkout counter's cluster
scores 0.71, wall segments 0.44-0.57.

Raised 0.60 -> 0.65 (2026-07-31) after a measured false positive: on run_0724_164652 the shelf
sweep read the counter itself (so the counter dropped out of this pass as "observed"), and the
leftover unobserved SW WALL-CORNER cluster scored 0.62 - over the old bar - so the run's one
"landmark" faced a bare wall, not the counter. Re-scored across five grids (frozen + four fresh
explores), every wall/corner rump lands in 0.41-0.62 while the true island cluster, when it exists
at all, scores 0.71. 0.65 splits those cleanly; do not lower it back without re-measuring all five.

KNOWN LIMIT, measured, do not re-attempt geometrically: when a shelf checkpoint reads the counter
(which the current 2.0m sweep does on every fresh explore since 07-24), the counter is eroded as
"observed" and this pass CANNOT recover it - the graph then gets no landmark from geometry alone.
Five discriminators were tried against the four fresh grids and all failed: raw free-perimeter
(0.45-0.62, overlaps walls), perimeter of the reconstructed full structure (merges into the wall
component - connectivity to the wall is run-dependent: 4-and-8-connected, 8-only, or separate,
across the five grids), a 0.4-0.8m free-space shell (counter sits too close to the west wall),
interior-vs-exterior unknown (the counter's unknown fringe drains around the wall to the outside),
and 1-cell morphological erosion cores (the counter scans too thin on some runs - 46-56 occupied
cells over its 10x16 footprint - and dissolves). The robust recovery is SEMANTIC, not geometric:
the annotator reliably reads the sim's "Welcome to Self Checkout!" signage at that node, and
annotate_pass.promote_checkout_landmarks() promotes it to kind="landmark" in the topology. This
geometric pass remains the path for maps where the sweep does NOT claim the counter (the frozen
baseline), and the two are deduped there by promote's same-structure guard."""


def _unobserved_clusters(occupied, free, observed_cells, res, *, min_cells, observed_radius_m,
                         min_free_perimeter):
    """Occupied structures that no checkpoint reads AND that look like islands, not walls.

    Order matters here and it is the whole trick. Clustering FIRST and then asking "is this
    observed?" fails on this store: the checkout counter is 8-connected to the west wall, so a
    plain flood fill swallows it into one giant wall-plus-shelves component that obviously IS
    observed, and the counter vanishes (measured: that version found zero landmarks). Marking each
    occupied cell observed/unobserved first, and only then clustering the leftovers, keeps the
    counter as its own cluster because the wall cells beside it are read by shelf checkpoints and
    drop out before clustering ever happens.
    """
    from collections import deque

    h, w = occupied.shape
    r2 = (observed_radius_m / res) ** 2
    unobserved = occupied.copy()
    for x, z in zip(*(a.tolist() for a in occupied.nonzero())):
        for sx, sz in observed_cells:
            if (sx - x) ** 2 + (sz - z) ** 2 <= r2:
                unobserved[x, z] = False
                break

    seen = unobserved.copy() & False
    out = []
    for x0, z0 in zip(*(a.tolist() for a in unobserved.nonzero())):
        if seen[x0, z0]:
            continue
        seen[x0, z0] = True
        queue, cells = deque([(int(x0), int(z0))]), []
        while queue:
            x, z = queue.popleft()
            cells.append((x, z))
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nx, nz = x + dx, z + dz
                    if 0 <= nx < h and 0 <= nz < w and unobserved[nx, nz] and not seen[nx, nz]:
                        seen[nx, nz] = True
                        queue.append((nx, nz))
        if len(cells) < min_cells:
            continue

        perimeter = bordering_free = 0
        for x, z in cells:
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, nz = x + dx, z + dz
                if 0 <= nx < h and 0 <= nz < w and not occupied[nx, nz]:
                    perimeter += 1
                    if free[nx, nz]:
                        bordering_free += 1
        if perimeter and bordering_free / perimeter >= min_free_perimeter:
            out.append(cells)
    return out


def dock_interaction_xz(grid, occupied, structure_cells, stand_cell, target_cell, *,
                        body_radius=0.3,
                        interaction_gap_m=DEFAULT_LANDMARK_INTERACTION_GAP_M,
                        max_approach_m=DEFAULT_LANDMARK_MAX_APPROACH_M):
    """The dock pose: step from `stand_cell` toward `target_cell` until the body surface is
    interaction_gap_m from the structure (`structure_cells`, its occupied cells). Walked one cell
    at a time rather than solved, since "distance to the structure" means distance to its nearest
    cell, which changes as you move. Returns world (x, z), or None if the walk hits an obstacle or
    runs out of range first.

    Shared by find_landmark_checkpoints() (dock toward a discovered cluster's centroid) and
    annotate_pass.promote_checkout_landmarks() (dock an existing checkpoint toward its shelf_cell
    when the annotator identifies the checkout) so the two landmark paths derive the same pose."""
    want_centre_dist = body_radius + interaction_gap_m
    vx, vz = target_cell[0] - stand_cell[0], target_cell[1] - stand_cell[1]
    norm = math.hypot(vx, vz) or 1.0
    vx, vz = vx / norm, vz / norm
    for step in range(1, int(math.ceil(max_approach_m / grid.res)) + 1):
        px = int(round(stand_cell[0] + vx * step))
        pz = int(round(stand_cell[1] + vz * step))
        if not grid.in_bounds(px, pz) or occupied[px, pz]:
            break
        gap_m = min(math.hypot(a - px, b - pz) for a, b in structure_cells) * grid.res
        if gap_m <= want_centre_dist:
            return grid.to_world(px, pz)
    return None


def find_landmark_checkpoints(grid, graph, *,
                              min_cells=DEFAULT_LANDMARK_MIN_CELLS,
                              observed_radius_m=DEFAULT_LANDMARK_OBSERVED_RADIUS_M,
                              max_approach_m=DEFAULT_LANDMARK_MAX_APPROACH_M,
                              min_free_perimeter=DEFAULT_LANDMARK_MIN_FREE_PERIMETER,
                              min_clearance_m=DEFAULT_LANDMARK_MIN_CLEARANCE_M,
                              interaction_gap_m=DEFAULT_LANDMARK_INTERACTION_GAP_M,
                              min_reading_distance_m=MIN_READING_DISTANCE_M,
                              max_reading_distance_m=MAX_READING_DISTANCE_M,
                              body_radius=0.3):
    """Find occupied structures that NO existing checkpoint observes, and propose a checkpoint
    facing each. Returns a list of dicts, same shape find_shelf_checkpoint returns plus "n_cells".

    Why this exists, measured: the store's checkout counter (a 0.7 x 1.6m island, 92 occupied
    cells, mapped perfectly well by the LiDAR) had ZERO checkpoints looking at it, and the nearest
    one faced the opposite wall. It was 1.65m from the nearest path point - well inside the 2.0m
    search radius - so range was not the reason.

    The reason is the shape of the shelf sweep: it steps along a path and looks PERPENDICULAR to
    the direction of travel, which finds surfaces that FLANK a corridor. A free-standing island in
    open floor is never perpendicular to anything the agent walks past, so it is never a candidate
    however close it sits. That is correct behaviour for finding shelf faces and a blind spot for
    finding landmarks, and no amount of annotation fixes it - you cannot annotate a place you never
    stand at.

    So this is a SECOND pass with a different question. The shelf sweep asks "walking here, what is
    beside me?"; this asks "what did nobody end up looking at?" - then approaches it head-on from
    the nearest reachable free cell instead of sideways from a path.
    """
    occupied = grid.log_odds > grid.OCCUPIED_THRESHOLD
    inflated = _inflate_occupied(grid, body_radius)
    # A second, wider mask: cells this close to ANY obstacle are ones the executor will not walk
    # into, whatever their distance to the structure being read.
    too_tight = _inflate_occupied(grid, min_clearance_m)
    free = grid.log_odds < grid.FREE_THRESHOLD

    observed = [c.shelf_cell for c in graph.checkpoints if c.shelf_cell]

    proposals = []
    for cells in _unobserved_clusters(occupied, free, observed, grid.res, min_cells=min_cells,
                                      observed_radius_m=observed_radius_m,
                                      min_free_perimeter=min_free_perimeter):
        # Approach head-on: for each free cell within reach, aim at the nearest cell of the
        # structure and keep the standable candidate that ends up closest to the ideal reading
        # distance. Line-of-sight is checked against the same inflated mask A* uses, so a landmark
        # node is reachable by construction - the counter is not much use if the executor won't go.
        cx = sum(c[0] for c in cells) / len(cells)
        cz = sum(c[1] for c in cells) / len(cells)
        reach = int(math.ceil(max_approach_m / grid.res))
        best = None
        for x in range(int(cx) - reach, int(cx) + reach + 1):
            for z in range(int(cz) - reach, int(cz) + reach + 1):
                if not grid.in_bounds(x, z) or not free[x, z] or inflated[x, z]:
                    continue
                if too_tight[x, z]:
                    continue
                target = min(cells, key=lambda t: math.hypot(t[0] - x, t[1] - z))
                d_m = math.hypot(target[0] - x, target[1] - z) * grid.res
                # Accept a RANGE and score toward the ideal, rather than demanding an exact
                # distance. find_shelf_checkpoint can insist on max_reading_distance_m because it
                # CONSTRUCTS a point at that distance along a ray; this pass searches existing
                # cells, and with MIN == MAX (both 1.0m today) an equality test is a zero-width
                # band that no discrete cell ever satisfies - measured: it found nothing at all.
                if not (min_reading_distance_m <= d_m <= max_approach_m):
                    continue
                # Visibility is checked to the free cell one step SHORT of the structure, never to
                # the structure cell itself: that cell is occupied by definition, so a line-of-
                # sight test against it can never pass (measured: it rejected all 1290 otherwise
                # valid standing positions). find_shelf_checkpoint has the same shape - it checks
                # LOS to the candidate standing point, not to the shelf face it reads.
                approach = _bresenham_line(x, z, target[0], target[1])
                if len(approach) >= 2 and not _line_of_sight(
                        grid, (x, z), approach[-2], occupied_mask=occupied):
                    continue
                score = abs(d_m - max_reading_distance_m)
                if best is None or score < best[0]:
                    best = (score, (x, z), target, d_m)
        if best is None:
            continue
        _score, cell, target, d_m = best

        # FACE THE CENTROID, not the nearest cell. Standing off the nearest SURFACE is right (that
        # is what keeps the reading distance honest), but aiming at that same nearest cell points
        # the camera at whichever corner happens to be closest, so a compact structure lands in the
        # corner of frame. The centroid is inside the structure and therefore occupied, which is
        # fine - shelf_cell is only ever used as a bearing target.
        centre = (int(round(sum(c[0] for c in cells) / len(cells))),
                  int(round(sum(c[1] for c in cells) / len(cells))))

        interaction_xz = dock_interaction_xz(
            grid, occupied, cells, cell, centre, body_radius=body_radius,
            interaction_gap_m=interaction_gap_m, max_approach_m=max_approach_m)

        proposals.append({
            "cell": cell,
            "world_xz": grid.to_world(*cell),
            "reading_distance_m": d_m,
            "shelf_cell": centre,
            "interaction_xz": interaction_xz,
            "n_cells": len(cells),
        })
    return proposals


def splice_landmark_checkpoints(grid, graph, **kwargs):
    """Add a kind="landmark" Checkpoint for each structure find_landmark_checkpoints() proposes,
    wired to the nearest existing checkpoint it has line of sight to.

    A landmark hangs off the graph as a leaf rather than splicing into an edge the way shelf chains
    do: it is a destination ("go to the counter"), not a corridor to pass through, so it should not
    become part of anyone's route between two other places. It carries shelf_cell like a shelf
    checkpoint does, so capture_walk's perpendicular facing works on it unchanged."""
    inflated = _inflate_occupied(grid, kwargs.get("body_radius", 0.3))
    next_id = max((c.id for c in graph.checkpoints), default=-1) + 1
    added = []

    for prop in find_landmark_checkpoints(grid, graph, **kwargs):
        cp = Checkpoint(
            id=next_id, cell=prop["cell"], world_xz=prop["world_xz"], kind="landmark",
            shelf_side=None, reading_distance_m=prop["reading_distance_m"],
            shelf_cell=prop["shelf_cell"], interaction_xz=prop["interaction_xz"],
        )
        anchor = min(
            (c for c in graph.checkpoints
             if _line_of_sight(grid, cp.cell, c.cell, occupied_mask=inflated)),
            key=lambda c: math.hypot(c.world_xz[0] - cp.world_xz[0], c.world_xz[1] - cp.world_xz[1]),
            default=None,
        )
        if anchor is None:
            continue  # nothing can see it; a node nobody can walk to is worse than no node
        cp.neighbors.append(anchor.id)
        anchor.neighbors.append(cp.id)
        length_m = math.hypot(anchor.world_xz[0] - cp.world_xz[0], anchor.world_xz[1] - cp.world_xz[1])
        graph.edges.append(TopologyEdge(a=anchor.id, b=cp.id, length_m=length_m))
        graph.checkpoints.append(cp)
        added.append(cp)
        next_id += 1
    return added


def splice_shelf_checkpoints(grid, graph, *,
                              interval_m=DEFAULT_CHECKPOINT_INTERVAL_M,
                              search_radius_m=DEFAULT_SHELF_SEARCH_RADIUS_M,
                              min_reading_distance_m=MIN_READING_DISTANCE_M,
                              max_reading_distance_m=MAX_READING_DISTANCE_M,
                              body_radius=0.3, add_ladder_rungs=True):
    """Phase 2's top-level entry point: extend Phase 1's `graph` (a topology.TopologyGraph)
    in place with shelf-hugging checkpoints, per the "Subdivision, not a parallel
    structure" section of the phase2 plan doc.

    For every existing TopologyEdge, sweeps both sides (sweep_edge_side()) over that
    edge's raw skeleton path. Each side that produces at least one checkpoint becomes a
    chain of new kind="shelf" Checkpoints, wired to each other in path order and spliced
    onto the edge's own two endpoint checkpoints at its ends - so `a` and `b` end up
    connected THROUGH the chain(s) instead of by the direct edge, which is removed. An
    edge where neither side has a nearby shelf (open floor, a central walkway) is left
    untouched: no chain to splice in means the direct edge is still the only connection
    between `a` and `b`, and removing it would disconnect the graph for nothing.

    When both sides produce a chain and add_ladder_rungs is True (the default - this was
    the phase2 doc's "still open" question, now decided), corresponding left/right
    checkpoints also get a direct "rung" link across the aisle - a person standing at one
    point on the left shelf can just turn and look at the right shelf. "Corresponding" is
    an EXACT match, not a nearest-position guess: sweep_edge_side() samples the identical
    point_cell sequence on both sides (sampling doesn't depend on `side`, only validation
    does), so two checkpoints share a rung iff they were found from the same path_point.
    A side missing a checkpoint at a given path_point (its validation failed there, or
    fewer/more of its samples survived) simply gets no rung at that position - independent
    per-side clamping means the two chains aren't guaranteed to have equal counts.

    Both `graph.checkpoints` and `graph.edges` are updated (extended/replaced in place)
    and `graph` itself is returned for convenience. `edges` stays the metric source of
    truth exactly as topology.py's own `_populate_neighbors` establishes - every new
    adjacency (chain links and ladder rungs alike) gets a backing TopologyEdge, not just a
    `neighbors` entry, so a caller that only reads `graph.edges` still sees the whole graph.
    """
    occupied_mask = grid.log_odds > grid.OCCUPIED_THRESHOLD
    inflated_mask = _inflate_occupied(grid, body_radius)
    by_id = {c.id: c for c in graph.checkpoints}
    next_id = max((c.id for c in graph.checkpoints), default=-1) + 1

    kept_edges = []
    new_checkpoints = []
    new_edges = []

    for edge in graph.edges:
        a, b = by_id[edge.a], by_id[edge.b]
        sweeps = {}
        for side in ("left", "right"):
            result = sweep_edge_side(
                grid, edge.path, side,
                interval_m=interval_m, search_radius_m=search_radius_m,
                min_reading_distance_m=min_reading_distance_m,
                max_reading_distance_m=max_reading_distance_m,
                body_radius=body_radius,
                occupied_mask=occupied_mask, inflated_mask=inflated_mask,
            )
            if result:
                sweeps[side] = result

        if not sweeps:
            kept_edges.append(edge)
            continue

        # The chain(s) built below re-establish a<->b connectivity, so the direct edge is
        # replaced, not kept alongside it. Guarded with a membership check (not a bare
        # .remove()) because a store layout can have two genuinely distinct edges between
        # the same pair of checkpoints (e.g. a loop aisle) - the second one to process
        # this pair must not fail trying to remove an already-removed link.
        if b.id in a.neighbors:
            a.neighbors.remove(b.id)
        if a.id in b.neighbors:
            b.neighbors.remove(a.id)

        chains_by_point = {}  # side -> {path_point: Checkpoint}, for the ladder-rung pass below
        for side, sweep in sweeps.items():
            chain = []
            for cp_dict in sweep:
                chain.append(Checkpoint(
                    id=next_id, cell=cp_dict["cell"], world_xz=cp_dict["world_xz"], kind="shelf",
                    shelf_side=side, reading_distance_m=cp_dict["reading_distance_m"],
                    shelf_cell=cp_dict["shelf_cell"],
                ))
                next_id += 1

            for i, cp in enumerate(chain):
                if i > 0:
                    cp.neighbors.append(chain[i - 1].id)
                if i < len(chain) - 1:
                    cp.neighbors.append(chain[i + 1].id)

            first, last = chain[0], chain[-1]
            first.neighbors.append(a.id)
            a.neighbors.append(first.id)
            last.neighbors.append(b.id)
            b.neighbors.append(last.id)

            chain_nodes = [a] + chain + [b]
            for n0, n1 in zip(chain_nodes, chain_nodes[1:]):
                length_m = math.hypot(n1.world_xz[0] - n0.world_xz[0], n1.world_xz[1] - n0.world_xz[1])
                new_edges.append(TopologyEdge(a=n0.id, b=n1.id, length_m=length_m))

            new_checkpoints.extend(chain)
            chains_by_point[side] = {cp_dict["path_point"]: cp for cp_dict, cp in zip(sweep, chain)}

        if add_ladder_rungs and "left" in chains_by_point and "right" in chains_by_point:
            left_by_point, right_by_point = chains_by_point["left"], chains_by_point["right"]
            for point in left_by_point.keys() & right_by_point.keys():
                left_cp, right_cp = left_by_point[point], right_by_point[point]
                left_cp.neighbors.append(right_cp.id)
                right_cp.neighbors.append(left_cp.id)
                length_m = math.hypot(right_cp.world_xz[0] - left_cp.world_xz[0],
                                       right_cp.world_xz[1] - left_cp.world_xz[1])
                new_edges.append(TopologyEdge(a=left_cp.id, b=right_cp.id, length_m=length_m))

    graph.checkpoints.extend(new_checkpoints)
    graph.edges = kept_edges + new_edges
    return graph
