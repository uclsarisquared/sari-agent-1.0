"""
Phase 2 - shelf-hugging checkpoint generation. See slamtest/plans/
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # overhaul/slamtest
_OVERHAUL_DIR = os.path.dirname(_THIS_DIR)                     # overhaul
for _p in (_OVERHAUL_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from occupancy_grid import _bresenham_line  # noqa: E402
from frontier_planner import _inflate_occupied, _line_of_sight  # noqa: E402
from topology import Checkpoint, TopologyEdge  # noqa: E402


MIN_READING_DISTANCE_M = 1
MAX_READING_DISTANCE_M = 1.5
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
Raised from an initial 1.0m for the same reason as DEFAULT_SHELF_SEARCH_RADIUS_M above -
plain density reduction on a real store layout felt excessive at 1.0m - not yet grounded
in the sim's actual camera field of view, so revisit if Phase 3 finds real coverage gaps
between checkpoints."""

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

        n_steps = max(1, round(seg_length_m / interval_m))
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
