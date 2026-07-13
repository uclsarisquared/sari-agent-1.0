"""
Post-hoc topological map extraction from a completed OccupancyGrid.

Frontier exploration is designed to MINIMIZE travel by exploiting visibility - it
deliberately stops going somewhere the moment it can be inferred free/occupied from a
distance, which is why a Full360 LiDAR run never has to walk every aisle. That's the
right behavior for building the occupancy grid efficiently, but it means the raw grid
alone isn't a walkable route graph.

This module runs ONCE, after exploration finishes, over the grid it already produced:
thins the resolved free space into a 1-cell-wide medial skeleton (topology-preserving
centerlines), then reduces that skeleton to a small graph of discrete checkpoints -
aisle ends (skeleton degree 1), junctions (degree >=3), and inserted doorway/pinch-point
checkpoints where a corridor narrows sharply - connected by edges carrying real-world
path length. This is a simplified, LLM-consumable adjacency map ("aisle A connects to
this doorway, which connects to aisle B"), derived purely from what the agent actually
observed (free vs occupied in the grid) - no privileged access to the Unity scene's true
static layout (ShelfBuilder/AisleMarker positions).

Hand-rolled (no scipy/scikit-image dependency, neither is installed in this project's
environment) using the same padding-not-np.roll convention OccupancyGrid.frontiers()
already established, and reusing frontier_planner's connectivity/clustering helpers
rather than duplicating them.

This is a one-time end-of-run post-process, not a per-exploration-step hot path (unlike
cluster_frontiers()/astar() in frontier_planner.py) - correctness/clarity is favored over
per-call performance here.

Run standalone against a saved grid_final.npy for inspection:
    python slamtest/slamtest_frontier/topology.py path/to/grid_final.npy --resolution 0.1
"""
import argparse
import heapq
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # overhaul/slamtest/slamtest_frontier
_SLAMTEST_DIR = os.path.dirname(_THIS_DIR)                     # overhaul/slamtest
_OVERHAUL_DIR = os.path.dirname(_SLAMTEST_DIR)                 # overhaul
for _p in (_OVERHAUL_DIR, _SLAMTEST_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from occupancy_grid import OccupancyGrid  # noqa: E402
from frontier_planner import cluster_frontiers, _neighbor_offsets  # noqa: E402


DEFAULT_MIN_BRANCH_LENGTH_M = 0.5
DEFAULT_DOORWAY_MAX_WIDTH_M = 1.2
DEFAULT_DOORWAY_WIDTH_RATIO = 0.7

# Guards a topologically malformed skeleton (e.g. a pure loop with no degree!=2 pixel
# anywhere on it) from an infinite walk - real store layouts should always have plenty of
# genuine branch points, so hitting this is a sign of a skeletonization edge case, not a
# normal outcome.
_MAX_TRACE_STEPS = 200_000

# Zhang-Suen neighbor order, clockwise starting due north: p2, p3, p4, p5, p6, p7, p8, p9.
_ZS_OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


@dataclass
class Checkpoint:
    id: int
    cell: tuple          # (cx, cz)
    world_xz: tuple       # (x, z)
    kind: str             # "end" | "junction" | "doorway"


@dataclass
class TopologyEdge:
    a: int                # Checkpoint.id
    b: int                # Checkpoint.id
    length_m: float


@dataclass
class TopologyGraph:
    checkpoints: list = field(default_factory=list)
    edges: list = field(default_factory=list)


def _pad_shift(padded, dr, dc, n0, n1):
    return padded[1 + dr: 1 + dr + n0, 1 + dc: 1 + dc + n1]


def skeletonize(mask):
    """Zhang-Suen thinning: reduces a boolean free-space mask to its 1-cell-wide medial
    skeleton, via iterative parallel-per-sub-iteration boundary erosion. Fully vectorized
    with padded array shifts (constant-False padding, matching OccupancyGrid.frontiers()'
    fix) instead of a per-pixel Python loop, so a cell on the grid boundary never sees the
    opposite edge as a neighbor."""
    img = mask.copy()
    n0, n1 = img.shape
    changed = True
    while changed:
        changed = False
        for pass_variant in (1, 2):
            padded = np.pad(img, 1, mode="constant", constant_values=False)
            neighbors = [_pad_shift(padded, dr, dc, n0, n1) for dr, dc in _ZS_OFFSETS]
            b = np.zeros((n0, n1), dtype=np.int32)
            for nb in neighbors:
                b += nb
            a = np.zeros((n0, n1), dtype=np.int32)
            for i in range(8):
                a += (~neighbors[i] & neighbors[(i + 1) % 8]).astype(np.int32)
            p2, p3, p4, p5, p6, p7, p8, p9 = neighbors

            cond = img & (b >= 2) & (b <= 6) & (a == 1)
            if pass_variant == 1:
                cond &= ~(p2 & p4 & p6)
                cond &= ~(p4 & p6 & p8)
            else:
                cond &= ~(p2 & p4 & p8)
                cond &= ~(p2 & p6 & p8)

            if cond.any():
                img &= ~cond
                changed = True
    return img


def _skeleton_degree(skeleton):
    """Per-cell count of 8-connected skeleton neighbors - 1 means a dead end, 2 means a
    plain corridor pixel (not a checkpoint), >=3 means a branch point."""
    n0, n1 = skeleton.shape
    padded = np.pad(skeleton, 1, mode="constant", constant_values=False)
    degree = np.zeros((n0, n1), dtype=np.int32)
    for dr, dc in _ZS_OFFSETS:
        degree += _pad_shift(padded, dr, dc, n0, n1)
    return degree


def _find_node_clusters(skeleton, degree, connectivity):
    """Skeleton pixels with degree != 2 (dead ends and branch points), grouped into
    single logical nodes via the same BFS flood-fill frontier clustering already uses -
    a real junction is often a small clump of a few adjacent degree>=3 pixels after
    thinning, not a single unambiguous pixel. min_cluster_size=1 (unlike frontier
    clustering's noise-filtering default) since a lone endpoint pixel is still a real,
    meaningful checkpoint."""
    node_mask = skeleton & (degree != 2)
    node_cells = np.argwhere(node_mask)
    if len(node_cells) == 0:
        return []
    return cluster_frontiers(node_cells, connectivity=connectivity, min_cluster_size=1)


def _classify_cluster(cluster, degree):
    for cx, cz in cluster.cells:
        if degree[cx, cz] >= 3:
            return "junction"
    return "end"


def _build_node_index(clusters):
    node_of = {}
    for idx, cluster in enumerate(clusters):
        for cx, cz in cluster.cells:
            node_of[(int(cx), int(cz))] = idx
    return node_of


def _skeleton_cell_set(skeleton):
    return {(int(c[0]), int(c[1])) for c in np.argwhere(skeleton)}


def _path_length(path, res):
    total = 0.0
    for (x0, z0), (x1, z1) in zip(path, path[1:]):
        total += math.hypot(x1 - x0, z1 - z0) * res
    return total


def _trace_edges(skeleton_cells, node_of, connectivity):
    """Walk every node cluster's exits along the skeleton to the next node, returning
    (start_cluster_idx, target_cluster_idx, path_cells) triples. `consumed` tracks
    interior degree-2 pixels already claimed by a trace so a normal (length > 0 interior)
    edge is only walked once; a zero-interior-length edge (two clusters directly
    pixel-adjacent) gets traced independently from both ends and deduplicated afterward,
    since there's no interior pixel to mark as already-visited in that case."""
    offsets = _neighbor_offsets(connectivity)
    consumed = set()
    raw = []

    for start_cell, start_idx in list(node_of.items()):
        for dx, dz in offsets:
            first = (start_cell[0] + dx, start_cell[1] + dz)
            if first not in skeleton_cells:
                continue

            first_idx = node_of.get(first)
            if first_idx is not None:
                if first_idx == start_idx:
                    continue  # internal neighbor within the same cluster, not a real exit
                raw.append((start_idx, first_idx, [start_cell, first]))
                continue

            if first in consumed:
                continue

            path = [start_cell, first]
            prev, current = start_cell, first
            target_idx = None
            for _ in range(_MAX_TRACE_STEPS):
                node_idx = node_of.get(current)
                if node_idx is not None:
                    target_idx = node_idx
                    break
                nxt = None
                for ndx, ndz in offsets:
                    cand = (current[0] + ndx, current[1] + ndz)
                    if cand == prev or cand not in skeleton_cells:
                        continue
                    nxt = cand
                    break
                if nxt is None:
                    break  # dead end that never reached a registered node - shouldn't
                    # happen for a well-formed skeleton; drop this partial trace.
                consumed.add(current)
                path.append(nxt)
                prev, current = current, nxt

            if target_idx is not None and target_idx != start_idx:
                raw.append((start_idx, target_idx, path))
            # target_idx == start_idx: a loop back to the same cluster - not useful for an
            # adjacency graph between distinct checkpoints; dropped deliberately.

    # Zero-interior-length edges get discovered from both ends independently (no interior
    # pixel exists to mark `consumed`) - collapse those duplicate undirected pairs.
    seen = set()
    deduped = []
    for a_idx, b_idx, path in raw:
        key = frozenset((a_idx, b_idx, len(path)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((a_idx, b_idx, path))
    return deduped


def _prune_short_spurs(kinds, raw_edges, min_branch_length_m, res):
    """Drop short dead-end spurs - typically a thinning artifact from a jagged shelf
    corner rather than a real, walkable aisle stub. Uses each cluster's ORIGINAL
    degree-based classification (computed before any pruning): a junction that ends up
    with only two remaining exits after its other spurs are pruned stays classified as a
    junction here rather than being re-evaluated as a pass-through - a known, minor
    simplification, not a correctness issue for the checkpoint graph itself."""
    kept = []
    for a_idx, b_idx, path in raw_edges:
        if (kinds[a_idx] == "end" or kinds[b_idx] == "end") and _path_length(path, res) < min_branch_length_m:
            continue
        kept.append((a_idx, b_idx, path))
    return kept


def _distance_to_occupied(occupied_mask, res, connectivity):
    """Multi-source Dijkstra distance (world meters) from every cell to its nearest
    occupied cell - used as a per-cell clearance/half-width estimate for doorway
    detection. Not a hot-path operation (see module docstring): computed once per
    extract_topology() call, same heapq-based approach as frontier_planner.astar()."""
    n0, n1 = occupied_mask.shape
    dist = np.full((n0, n1), np.inf, dtype=np.float64)
    heap = []
    for cx, cz in np.argwhere(occupied_mask):
        cx, cz = int(cx), int(cz)
        dist[cx, cz] = 0.0
        heapq.heappush(heap, (0.0, (cx, cz)))

    offsets_costs = [(dx, dz, math.hypot(dx, dz) * res) for dx, dz in _neighbor_offsets(connectivity)]
    while heap:
        d, (cx, cz) = heapq.heappop(heap)
        if d > dist[cx, cz]:
            continue
        for dx, dz, cost in offsets_costs:
            nx, nz = cx + dx, cz + dz
            if not (0 <= nx < n0 and 0 <= nz < n1):
                continue
            nd = d + cost
            if nd < dist[nx, nz]:
                dist[nx, nz] = nd
                heapq.heappush(heap, (nd, (nx, nz)))
    return dist


def _maybe_split_for_doorway(path, clearance, width_ratio, max_width_m):
    """If `path`'s narrowest interior point (excluding its two node endpoints) is both
    below max_width_m and meaningfully narrower than the path's own median width, return
    the split index into `path`; else None. Only the single narrowest point per edge is
    considered - a long edge with two genuine doorways would need a second pass, not
    handled here (documented simplification, kept for a tractable v1)."""
    if len(path) < 3:
        return None
    widths = [2.0 * clearance[cx, cz] for cx, cz in path[1:-1]]
    if not widths:
        return None
    min_w = min(widths)
    med_w = float(np.median(widths))
    if min_w >= max_width_m or med_w <= 0 or min_w > med_w * width_ratio:
        return None
    return widths.index(min_w) + 1  # +1 re-offsets into path's own indexing


def _drop_orphaned_checkpoints(checkpoints, edges):
    """A checkpoint whose only edge got removed by spur pruning is left referencing
    nothing - meaningless (and confusing) for an adjacency-graph consumer. Drop any
    checkpoint with zero remaining edges and remap the rest to contiguous ids."""
    referenced = set()
    for e in edges:
        referenced.add(e.a)
        referenced.add(e.b)

    remap = {}
    kept_checkpoints = []
    for c in checkpoints:
        if c.id not in referenced:
            continue
        remap[c.id] = len(kept_checkpoints)
        kept_checkpoints.append(Checkpoint(id=remap[c.id], cell=c.cell, world_xz=c.world_xz, kind=c.kind))

    remapped_edges = [TopologyEdge(a=remap[e.a], b=remap[e.b], length_m=e.length_m) for e in edges]
    return kept_checkpoints, remapped_edges


def extract_topology(grid, connectivity=8, min_branch_length_m=DEFAULT_MIN_BRANCH_LENGTH_M,
                      doorway_max_width_m=DEFAULT_DOORWAY_MAX_WIDTH_M,
                      doorway_width_ratio=DEFAULT_DOORWAY_WIDTH_RATIO):
    """Reduce grid's resolved free space to a small topological checkpoint graph. See
    module docstring for the overall approach and its "derived purely from what's been
    observed" scope."""
    free = grid.log_odds < grid.FREE_THRESHOLD
    occupied = grid.log_odds > grid.OCCUPIED_THRESHOLD

    skeleton = skeletonize(free)
    degree = _skeleton_degree(skeleton)
    clusters = _find_node_clusters(skeleton, degree, connectivity)
    if not clusters:
        return TopologyGraph()

    kinds = [_classify_cluster(c, degree) for c in clusters]
    node_of = _build_node_index(clusters)
    skeleton_cells = _skeleton_cell_set(skeleton)

    raw_edges = _trace_edges(skeleton_cells, node_of, connectivity)
    raw_edges = _prune_short_spurs(kinds, raw_edges, min_branch_length_m, grid.res)

    clearance = _distance_to_occupied(occupied, grid.res, connectivity)

    checkpoints = []
    id_of_cluster = {}
    for idx, cluster in enumerate(clusters):
        world = grid.to_world(*cluster.centroid_cell)
        checkpoints.append(Checkpoint(id=idx, cell=tuple(int(v) for v in cluster.centroid_cell),
                                       world_xz=world, kind=kinds[idx]))
        id_of_cluster[idx] = idx
    next_id = len(clusters)

    edges = []
    for a_idx, b_idx, path in raw_edges:
        a_id, b_id = id_of_cluster[a_idx], id_of_cluster[b_idx]
        split_idx = _maybe_split_for_doorway(path, clearance, doorway_width_ratio, doorway_max_width_m)
        if split_idx is None:
            edges.append(TopologyEdge(a=a_id, b=b_id, length_m=_path_length(path, grid.res)))
            continue

        door_cell = path[split_idx]
        door_id = next_id
        next_id += 1
        checkpoints.append(Checkpoint(id=door_id, cell=door_cell, world_xz=grid.to_world(*door_cell), kind="doorway"))
        edges.append(TopologyEdge(a=a_id, b=door_id, length_m=_path_length(path[:split_idx + 1], grid.res)))
        edges.append(TopologyEdge(a=door_id, b=b_id, length_m=_path_length(path[split_idx:], grid.res)))

    checkpoints, edges = _drop_orphaned_checkpoints(checkpoints, edges)
    return TopologyGraph(checkpoints=checkpoints, edges=edges)


def save_topology(topology, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"topology_{tag}.json")
    payload = {
        "checkpoints": [
            {"id": c.id, "cell": list(c.cell), "world_xz": list(c.world_xz), "kind": c.kind}
            for c in topology.checkpoints
        ],
        "edges": [{"a": e.a, "b": e.b, "length_m": e.length_m} for e in topology.edges],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract a topological checkpoint/adjacency graph from a saved grid_*.npy."
    )
    parser.add_argument("grid_npy", help="Path to a saved OccupancyGrid.log_odds .npy file")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--min-branch-length-m", type=float, default=DEFAULT_MIN_BRANCH_LENGTH_M)
    parser.add_argument("--doorway-max-width-m", type=float, default=DEFAULT_DOORWAY_MAX_WIDTH_M)
    parser.add_argument("--doorway-width-ratio", type=float, default=DEFAULT_DOORWAY_WIDTH_RATIO)
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--tag", default="inspect")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    log_odds = np.load(args.grid_npy)
    loaded_grid = OccupancyGrid(size_m=log_odds.shape[0] * args.resolution, resolution=args.resolution)
    loaded_grid.log_odds = log_odds

    result = extract_topology(
        loaded_grid, connectivity=args.connectivity, min_branch_length_m=args.min_branch_length_m,
        doorway_max_width_m=args.doorway_max_width_m, doorway_width_ratio=args.doorway_width_ratio,
    )
    out_path = save_topology(result, args.output_dir, args.tag)
    kinds_count = {}
    for c in result.checkpoints:
        kinds_count[c.kind] = kinds_count.get(c.kind, 0) + 1
    print(f"[topology] {len(result.checkpoints)} checkpoints {kinds_count}, {len(result.edges)} edges -> {out_path}")
