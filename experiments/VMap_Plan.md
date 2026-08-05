# SariVoxeLLMap Integration into the Overhauled Agent

## Context

The overhauled agent navigates a simulated sari-sari store using a VLM (vision-language model) loop. Currently, spatial awareness is limited to:
- Single-frame visual inspection (no persistent spatial memory)
- Static hardcoded shelf coordinates in semantic memory (trusted less than visual cues)
- Monocular depth estimation used **only** to gate grab attempts (>2.0m = refuse grab)
- No obstacle map, no path planning, no knowledge of previously traversed space

SariVoxeLLMap builds a live 2D heightmap (occupancy grid with heights) from RGB frames + camera poses using Depth-Anything-3. It exposes a clean async API: `queue_frame(img, extrinsics, intrinsics)` → `get_current_map(position)` → list of `(x, y, z)` relative voxel coordinates.

The goal is to use this heightmap to close the loop on navigation — giving the agent persistent, metric spatial memory it currently lacks.

---

## Current Agent Architecture Summary

```
experiments/v1_agent_eval.py / subagent_run.py (loop)
  └── EmbodiedAgent.execute_lean()
        ├── SemanticEpisodicAssociativeLearner  → mode decision (perception/nav/manipulation)
        ├── VLMAgent.send_message()             → action sequence generation
        └── SemanticEpisodicAssociativeLearner  → episodic reflection
              ↕
        overhaul/memory.py       (BASE_SEMANTIC_MEMORY — static shelf coords)
        overhaul/semantic_memory.txt  (runtime-updated)
        overhaul/episodic_memory.txt  (runtime-updated reflections)
```

**Navigation bottlenecks identified:**
1. Blind dead-reckoning: agent moves without knowing what's ahead
2. No obstacle persistence: obstacles seen in past frames are forgotten
3. Imprecise localization: semantic memory coordinates distrusted ("trust visual > numerical")
4. No path planning: LLM generates pan+move sequences without any map
5. Exploration is reactive: agent only knows what's in current frame

---

## Integration Methods

### Method 1 — Heightmap as VLM Navigation Context (Text Injection)
**Description:** After each frame, update the heightmap and serialize it as a compact text description injected into the VLM system prompt (`SYS_INST_VLM_LEAN` and `SYS_INST_ASSOCIATIVE_SEMANTIC` in `overhaul/sys_inst.py`).

**Example injection:**
```
HEIGHTMAP (relative to agent, forward=+X, right=+Y, bin=0.75m):
Obstacles: (+1.5m, 0.0m), (+1.5m, +0.75m), (+3.0m, -0.75m)
Clear path: forward up to +2.25m center lane
```

**Implementation touch points:**
- `overhaul/experiments/v1_agent_eval.py` — initialize `SariVoxeLLMap`, `queue_frame` after each screenshot, serialize map
- `overhaul/sys_inst.py` — add heightmap context block to prompts
- `overhaul/agent.py` — pass heightmap string into `execute_lean()` request dict

**Research backing:**
- **NavGPT** (Yu et al., 2023, *arXiv:2305.16943*): Pure LLM navigation using map-augmented language descriptions outperformed previous VLN baselines by giving the model rich spatial context in text form.
- **SayPlan** (Rana et al., 2023, *arXiv:2307.06135*): Demonstrated that injecting 3D scene graph descriptions into LLM prompts dramatically improves task planning success in cluttered environments.
- **NLMap** (Chen et al., 2023, *arXiv:2301.10303*): Open-queryable natural language maps show that converting metric maps to language tokens bridges the gap between spatial representations and language models.

**Expected gains:** Fewer collisions, better forward-planning. **Low integration complexity.**

---

### Method 2 — Occupancy Grid Path Planning (A*/BFS)
**Description:** Use the heightmap as a binary occupancy grid. When the agent needs to reach a target shelf coordinate, run A* (or BFS at low resolution) to find a collision-free path. Convert waypoints to discrete `move_forward` + `pan_left/right` action sequences.

**Pipeline:**
```
target = shelf_coordinate (from semantic memory)
occupancy_grid = heightmap → binary (occupied if height > floor_thresh)
path = A_star(current_pos, target, occupancy_grid)
actions = path_to_actions(path, current_rotation)
```

**Implementation touch points:**
- New file `overhaul/planner.py` — A*/BFS implementation on 2D numpy grid
- `overhaul/actions.py` — `plan_path_to(target)` action
- `overhaul/agent.py` — call planner when mode=navigation and target is known
- `overhaul/experiments/v1_agent_eval.py` — pass heightmap grid to planner each step

**Research backing:**
- **Elfes (1989)**, *"Using Occupancy Grids for Mobile Robot Perception and Navigation"*, IEEE Computer: The foundational reference showing 2D occupancy grids + grid search enable reliable autonomous navigation in unknown environments. Cited 4,000+ times.
- **Thrun, Burgard, Fox — "Probabilistic Robotics" (2005)**, MIT Press: Chapter 9 establishes grid-based path planning as the standard backbone for mobile robot navigation.
- **SemExp** (Chaplot et al., 2020, *NeurIPS*): Extended occupancy maps with semantic labels; showed that metric path planning on learned maps reduces navigation steps by 34% vs reactive baselines in Habitat simulator.

**Expected gains:** Optimal collision-free paths, eliminates aimless wandering. **Medium integration complexity.**

---

### Method 3 — Frontier-Based Exploration
**Description:** When the target item is not found in the current view, use the heightmap to identify **frontiers** — grid cells on the boundary between explored (mapped) and unexplored space — and navigate toward them. Replaces random pan/move exploration with principled coverage.

**Pipeline:**
```
frontier_cells = cells_adjacent_to_unmapped_regions(heightmap)
nearest_frontier = min(frontier_cells, key=distance_to_agent)
navigate_to(nearest_frontier)  # via Method 2 planner
```

**Implementation touch points:**
- `overhaul/planner.py` — `get_frontiers(heightmap, explored_mask)` function
- `overhaul/agent.py` — trigger frontier navigation when `mode=navigation` and no target visible
- `overhaul/experiments/v1_agent_eval.py` — maintain `explored_mask` as agent moves

**Research backing:**
- **Yamauchi (1997)**, *"A Frontier-Based Approach for Autonomous Exploration"*, IEEE CIRA: Original frontier exploration paper. Proves completeness — robot will eventually cover all reachable space.
- **SemExp** (Chaplot et al., 2020, *NeurIPS*): Frontier exploration + semantic maps achieved state-of-the-art on ObjectNav in AI2-THOR. Their agent found objects 16% faster than reactive visual baselines.
- **FrontierNet** (Jing et al., 2023, *ICRA*): Neural frontier exploration showing that prioritizing semantically likely frontiers (toward areas resembling target objects) further reduces search time.

**Expected gains:** Systematic coverage, no repeated re-exploration of same areas. **Medium complexity (depends on Method 2).**

---

### Method 4 — Semantic Voxel Map (RGB-Labeled Heightmap)
**Description:** SariVoxeLLMap preserves RGB colors in its point cloud. Map dominant colors per voxel bin to identify semantic regions (shelf shelving=brown/grey, floor=beige, products=varied). Store semantic labels alongside heights. This creates a queryable map: "where is the blue bottle?" → find voxel clusters matching blue color.

**Pipeline:**
```
colored_points = pointcloud.points + pointcloud.colors
semantic_heightmap[bin] = {height, dominant_color, semantic_label}
label = color_cluster_to_label(dominant_color)  # "shelf", "floor", "product"
```

**Implementation touch points:**
- `SariVoxeLLMap/voxelization/core.py` — extend `Voxelizator` to track dominant color per bin
- `SariVoxeLLMap/voxellmap.py` — expose `get_semantic_map()` returning labels
- `overhaul/memory.py` — update BASE_SEMANTIC_MEMORY with discovered semantic positions
- `overhaul/agent.py` — merge semantic map updates into semantic memory each timestep

**Research backing:**
- **ConceptGraphs** (Gu et al., 2023, *arXiv:2309.16650*): Open-vocabulary 3D scene graphs built from RGB-D data; agents using them achieved 85.8% task success on object retrieval vs 52.4% without maps.
- **CLIP-Fields** (Shafiullah et al., 2022, *arXiv:2210.05663*): Neural semantic fields built from RGB views enable open-vocabulary spatial queries ("find the red can") with metric precision.
- **OpenScene** (Peng et al., 2023, *CVPR*): 3D point cloud features for open-vocabulary scene understanding, demonstrating zero-shot semantic labeling of voxelized environments.

**Expected gains:** Agent can ground semantic memory in metric space. "Shelf 3 is at (+4m, +1m)" becomes verifiable/updateable. **High complexity.**

---

### Method 5 — Pre-Movement Collision Check
**Description:** Before executing each `move_forward(N)`, query the heightmap to check if the target cells are occupied. If occupied, substitute a `pan_left/right` + `move_forward` detour. This is the lightest-weight integration — a guard layer over existing actions.

**Pipeline:**
```
def safe_move_forward(units):
    target_cells = project_forward(current_pos, current_rotation, units)
    if any(heightmap[cell] > OBSTACLE_HEIGHT_THRESH for cell in target_cells):
        pan_to_clear_lane()  # find nearest free lane
    else:
        move_forward(units)
```

**Implementation touch points:**
- `overhaul/actions.py` — wrap `move_forward` with `safe_move_forward`
- `overhaul/experiments/v1_agent_eval.py` — pass current heightmap to action wrappers
- `overhaul/sys_inst.py` — note in prompt that collision pre-checks are active

**Research backing:**
- **ANS** (Chaplot et al., 2020, *ICML*) — "Learning to Explore using Active Neural SLAM": Pre-movement collision checking on occupancy maps reduced collision rate by 62% vs reactive visual agents in Gibson simulator.
- **DD-PPO** (Wijmans et al., 2019, *ICLR*): Decentralized distributed PPO for PointNav; showed that even simple geometric collision prediction significantly boosts SPL (success weighted by path length).

**Expected gains:** Immediate collision reduction, minimal code change. **Low complexity.**

---

## Recommended Implementation Order

| Priority | Method | Complexity | Impact |
|----------|--------|------------|--------|
| 1 | Method 5 — Pre-movement collision check | Low | Immediate: fewer stuck episodes |
| 2 | Method 1 — Heightmap text injection | Low | Better VLM spatial reasoning |
| 3 | Method 2 — A* path planning | Medium | Optimal navigation to known targets |
| 4 | Method 3 — Frontier exploration | Medium | Efficient search when target not visible |
| 5 | Method 4 — Semantic voxel map | High | Grounded semantic memory |

Methods 1 and 5 together form a **minimal viable integration** with high payoff and low risk. Methods 2–4 build on top incrementally.

---

## Critical Files

| File | Role in Integration |
|------|---------------------|
| `overhaul/experiments/v1_agent_eval.py` | Initialize `SariVoxeLLMap`, call `queue_frame` each step |
| `overhaul/subagent_run.py` | Same loop integration for multi-subtask runs |
| `overhaul/agent.py` | Pass heightmap context into `execute_lean()`, call planner |
| `overhaul/actions.py` | Wrap `move_forward` with collision pre-check |
| `overhaul/sys_inst.py` | Inject heightmap description block into prompts |
| `overhaul/memory.py` | Merge semantic map discoveries into BASE_SEMANTIC_MEMORY |
| `SariVoxeLLMap/voxellmap.py` | `SariVoxeLLMap` class — primary API to use |
| `SariVoxeLLMap/voxelization/core.py` | Extend for RGB/semantic labels (Method 4 only) |

---

## Verification

1. **Run with heightmap logging**: add a step that prints heightmap bin count and agent-relative coordinates after each frame — confirm map is growing and correctly positioned
2. **Collision rate**: compare episodes with/without collision pre-check; expect visible reduction in `isColliding=True` states
3. **Navigation efficiency (SPL proxy)**: count `move_forward` steps to reach a target shelf with/without path planning; expect fewer steps with A*
4. **Exploration coverage**: after N timesteps, visualize heightmap — expect denser coverage with frontier exploration vs reactive baseline
