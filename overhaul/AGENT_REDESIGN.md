# Overhaul Agent Redesign — Reference Document

> Covers the full redesign plan discussed across sessions. Use this as the briefing document for any future chat.

---

## 1. General Plan

The core redesign has three interlocking ideas:

### 1.1 Cardinal Axis Movement Constraint
The agent's **body yaw is locked to cardinal directions only**: 0°, 90°, 180°, 270°. The agent may still look freely (pitch/tilt) for perception. This constraint is justified because:
- All shelves in the Sari Unity sandbox are axis-aligned — no diagonal shelf faces exist
- It eliminates LLM degree-arithmetic errors (e.g. "pan 4 × 2.5° = 10°")
- Navigation planning becomes a grid problem rather than a continuous 2D problem
- Shelf scanning via strafing is the natural access pattern when facing a shelf cardinally

### 1.2 Cartesian JSON Store Map
Each store layout has a companion JSON file encoding its spatial structure precisely. The JSON drives:
- Obstacle avoidance (shelf footprints, walls, counter)
- Shelf lookup by ID → approach position and products
- Aisle corridor definitions for routing and episodic labeling
- Cardinal approach headings, snapped to true 0/90/180/270 (replacing the approximate values in the old `memory.py`)

The JSON **does not replace** the LLM's conceptual knowledge of the store. Product names, adjacency descriptions, and navigational prose stay in `BASE_SEMANTIC_MEMORY`. Coordinates and geometry live in the JSON.

### 1.3 Routing Without A\*
Because movement is cardinal and shelves are axis-aligned, routing reduces to:
1. Choose X-first or Z-first L-shaped path to the target
2. Check each segment against JSON shelf footprints (bounding box intersection)
3. If a segment intersects an obstacle, insert a single detour waypoint through the nearest aisle corridor

This is arithmetic, not search. The LLM navigation reasoner decides **which waypoints to hit and in what order**; the execution layer handles step-by-step movement between them.

---

## 2. Cartesian JSON Map Format

### 2.1 Top-Level Fields

```json
{
  "layout_id": "layout_2",
  "bounds": { "x": [0.0, 9.0], "z": [0.0, 7.8] },
  "grid_resolution": 0.1,
  "nav_buffer": 0.3,
  "agent_start": { "x": 2.82, "z": 0.78, "yaw": 0 }
}
```

- `grid_resolution`: matches `move_forward` step size (0.1 m)
- `nav_buffer`: clearance maintained from any obstacle edge (0.3 m)
- `agent_start`: spawn position; yaw should already be cardinal

### 2.2 Obstacles

Each obstacle is a named bounding box. Shelf units list their accessible face IDs.

```json
"obstacles": [
  {
    "id": "unit_a",
    "type": "shelf_unit",
    "footprint": { "x_min": 2.14, "x_max": 7.86, "z_min": 2.7, "z_max": 3.6 },
    "faces": ["S1", "S2", "S3", "S4"]
  },
  {
    "id": "counter",
    "type": "counter",
    "footprint": { "x_min": 0.0, "x_max": 0.75, "z_min": 0.36, "z_max": 1.9 }
  }
]
```

Types: `shelf_unit`, `wall_shelf`, `back_shelf`, `counter`, `wall`

### 2.3 Shelves

Each shelf entry represents one **accessible face** of a physical shelf unit. The `approach` position is the canonical standing point with a true cardinal yaw.

```json
"shelves": [
  {
    "id": "S1",
    "unit": "unit_a",
    "face": "south",
    "face_span": { "x_min": 2.6, "x_max": 7.4, "z": 2.7 },
    "approach": { "x": 5.0, "z": 2.4, "yaw": 0 },
    "products": ["Pringles", "potato chips", "cheese rings"]
  }
]
```

- `face_span`: defines the strafe range the agent can use while scanning this shelf
- `approach`: derived as face_coordinate ± nav_buffer, yaw snapped to nearest cardinal
- `products`: enables product-level lookup without embedding in LLM prompt every step

### 2.4 Counter

```json
"counter": {
  "footprint": { "x_min": 0.0, "x_max": 0.75, "z_min": 0.36, "z_max": 1.9 },
  "approach":  { "x": 1.05, "z": 1.13, "yaw": 270 }
}
```

The counter approach position is used as the navigation target for the STOP condition (delivery task).

### 2.5 Aisles

Named navigable corridors. Used by the routing logic and the episodic learner for position labeling.

```json
"aisles": [
  { "id": "front_zone",    "bounds": { "x_min": 0.0,  "x_max": 9.0,  "z_min": 0.0,  "z_max": 2.4  } },
  { "id": "left_corridor", "bounds": { "x_min": 0.74, "x_max": 1.84, "z_min": 0.0,  "z_max": 7.05 } },
  { "id": "mid_aisle",     "bounds": { "x_min": 0.0,  "x_max": 9.0,  "z_min": 3.9,  "z_max": 4.53 } },
  { "id": "back_aisle",    "bounds": { "x_min": 0.0,  "x_max": 9.0,  "z_min": 6.07, "z_max": 6.75 } },
  { "id": "right_corridor","bounds": { "x_min": 8.16, "x_max": 9.0,  "z_min": 0.0,  "z_max": 7.05 } }
]
```

### 2.6 Routing Notes (optional but recommended)

```json
"routing_notes": {
  "preferred_ns_route": "left_corridor",
  "reason": "Spawn x=2.82 is 0.98 m from left_corridor vs 5.34 m from right_corridor.",
  "unit_a_blocks_direct_north": true,
  "unit_b_blocks_direct_north": true,
  "to_cross_north_of_unit_a": "Enter left_corridor or right_corridor, then traverse mid_aisle.",
  "to_cross_north_of_unit_b": "Same corridors, then traverse back_aisle."
}
```

> **Layout 2 structural note:** The two central shelf units (`unit_a`, `unit_b`) each span x∈[2.14, 7.86] — 5.72 m of the 9 m store width. North-south movement is only possible through `left_corridor` (x∈[0.74, 1.84]) or `right_corridor` (x∈[8.16, 9.0]). East-west crossings are available at `front_zone`, `mid_aisle`, and `back_aisle` only.

---

## 3. What Needs to Change

### 3.1 Action Set

| Current Action | Status | Replacement / Notes |
|---|---|---|
| `pan_left` / `pan_right` | **Replace (navigation)** | `turn_left_90` / `turn_right_90` for body rotation |
| `pan_left` / `pan_right` | **Repurpose (perception/scan)** | Free-look only if Unity supports camera/body separation |
| `center_object_on_screen` | **Modify** | Strip yaw adjustment; use strafe for horizontal centering, tilt only for vertical |

**New actions to add:**

| New Action | Purpose |
|---|---|
| `turn_left_90` | Snap body yaw to next cardinal counter-clockwise (uses `face_cardinal_direction()`) |
| `turn_right_90` | Snap body yaw to next cardinal clockwise |
| `snap_to_cardinal` | Recovery action — snaps to nearest cardinal from any drifted yaw |
| `scan_shelf_left(n)` | Strafe left N steps along shelf while checking for target item |
| `scan_shelf_right(n)` | Strafe right N steps along shelf while checking for target item |
| `get_heading` | Returns current cardinal as human-readable label — removes arithmetic burden from LLM |

> **Implementation note:** `turn_left_90`, `turn_right_90`, and `snap_to_cardinal` are built on `define_cardinal_direction()` and `face_cardinal_direction()`, which already exist in `perception.py`. They use a single absolute `TransformAgent` rotation call, not incremental pan steps.

### 3.2 Mode Structure

A **scan mode** is added between navigation and perception to handle shelf scanning without constant mode-swapping.

| Mode | Available Actions |
|---|---|
| Navigation | `move_forward`, `move_backward`, `move_left`, `move_right`, `turn_left_90`, `turn_right_90` |
| **Scan** *(new)* | `move_left`, `move_right`, `tilt_up`, `tilt_down`, `scan_shelf_left`, `scan_shelf_right` |
| Perception | `center_object_on_screen`, `tilt_up`, `tilt_down`, `move_left`, `move_right` (minor) |
| Manipulation | all hand actions, `grab_item_in_view_left/right`, `grip_left`, `grip_right` |
| STOP | — |

**Mode transition rules:**

```
navigation  → scan        : arrived at approach (x, z) within arrival radius AND cardinal yaw correct
scan        → perception  : target item visible in frame
scan        → navigation  : strafed to shelf face_span edge without spotting target
perception  → manipulation: target centered AND depth ≤ 1 m OR hoveredObject matches target
manipulation→ STOP        : item gripped AND agent within 1 m of counter
any         → navigation  : target not visible, not near target shelf
```

> **Cardinal check in mode transitions:** Navigation mode stays active until BOTH position AND yaw are correct. This prevents premature mode switches when the agent arrives at the right (x, z) but is still facing the wrong direction.

### 3.3 System Instructions

#### `SYS_INST_ASSOCIATIVE_SEMANTIC` — Semantic Learner
- **Split the dual role**: output `observation` first, then derive `mode` from it — sequential, not conflated
- **Fix `recall` contradiction**: recall must not contain routing instructions, but target coordinates must be injected from JSON lookup (not parsed from recall text)
- **Add cardinal heading to mode pre-conditions**: manipulation/scan require correct yaw
- **Add `mode_rationale` field**: forces the model to cite evidence for its mode decision
- **Cap memory growth**: `new_semantic_memory` is observations only (≤ 3 sentences); coordinates are in JSON, not memory

New output schema:
```json
{
  "observation": "What is currently visible — products, spatial relationships, obstructions.",
  "new_semantic_memory": "Net-new facts only, max 3 sentences.",
  "mode": "navigation | scan | perception | manipulation | STOP",
  "mode_rationale": "Cites state fields and cardinal alignment explicitly."
}
```

#### `SYS_INST_ASSOCIATIVE_EPISODIC` — Episodic Learner
- **Change cadence**: call once at end of episode, not every timestep
- **Expand input**: pass full trajectory log `[(timestep, x, z, yaw, mode, actions, isColliding)]` + JSON aisle labels alongside VLM history
- **Define episode**: navigation mode start → STOP or failure
- Output schema unchanged (`dense_summary`, `what_worked`, `what_to_avoid`) — the structure is correct; the problem was input quality and call frequency

#### `SYS_INST_VLM_LEAN` — VLM Agent
- **Remove**: degree arithmetic example from rule 12; pan references in rules 6, 7, 8, 12
- **Rewrite rule 12**: navigation = follow pre-computed waypoints, not self-computed heading math
- **Resolve rules 8 vs 13 conflict**: in scan mode, always strafe; panning is not used at shelf proximity
- **Promote rule 14** (waypoint following) to primary navigation strategy, not a footnote
- **Update action references**: `pan_left`/`pan_right` → `turn_left_90`/`turn_right_90` throughout
- **Add scan mode instructions**: strafe bounds from `face_span`, exit conditions

#### Navigation Reasoner (new — replaces `HumanNavigationReasoner`)
New LLM-based component. Receives JSON map context and outputs cardinal waypoints.

Input:
```
current (x, z, yaw)
target approach (x, z, yaw)  ← from JSON lookup
shelf footprints              ← from JSON
aisle corridor bounds         ← from JSON
navigation memory             ← collision records, past paths
episodic memory               ← what worked, what to avoid
```

Output:
```json
[
  { "x": 1.84, "z": 0.78, "yaw": 270, "hint": "strafe west into left_corridor" },
  { "x": 1.84, "z": 3.9,  "yaw": 0,  "hint": "move north through left_corridor to mid_aisle" },
  { "x": 5.0,  "z": 3.9,  "yaw": 180,"hint": "strafe east to S2 approach, face south" }
]
```

Fires only when replanning is needed (start of navigation, missed waypoint, collision at current position).

### 3.4 `memory.py` and `BASE_SEMANTIC_MEMORY`
- **Remove**: Fast Tracking positions (superseded by JSON `approach` fields)
- **Remove**: Shelf Footprints section (superseded by JSON `obstacles`)
- **Remove**: Aisle Corridors section (superseded by JSON `aisles`)
- **Keep**: Product descriptions per shelf, top-down prose adjacency descriptions, navigational landmarks

### 3.5 `agent.py` — `execute_lean`
- **Replace** `_extract_nav_target()` regex with a direct JSON shelf lookup by shelf ID
- **Remove** `timestep == 1` / `else` branching — logic is identical; merge into a single path
- **Move** episodic learner call to end-of-episode trigger, not per-step
- **Add** JSON map loading at init; pass shelf approach coordinates directly to VLM user message

---

## 4. What Can Be Kept

### 4.1 Actions (fully retained, no changes)
All manipulation actions are body-orientation-independent and unaffected by the cardinal constraint:

| Group | Actions |
|---|---|
| Hand extension | `extend_left/right_hand_forward`, `pull_left/right_hand_backward` |
| Hand height | `raise/lower_left/right_hand` |
| Hand rotation | `rotate_left/right_clockwise/counterclockwise` |
| Grip | `grip_left`, `grip_right` |
| Smart grab | `grab_item_in_view_right`, `grab_item_in_view_left` |
| Camera tilt | `tilt_up`, `tilt_down` |
| Translation | `move_forward`, `move_backward`, `move_left`, `move_right` |

> **Step size stays at 0.1 m.** The previous developers' instability concern is valid and unchanged. The 0.1 m increment moves from being the LLM's responsibility (counting steps) to an implementation detail inside `navigate_to_waypoint`. The LLM sees high-level waypoints; the execution layer steps at 0.1 m.

### 4.2 System Instruction Content (retained from existing instructions)

**From Semantic Learner:**
- Full state field definitions (translation, rotation, isColliding, hand states, hovered/gripped)
- Five-mode framing as cognitive scaffold
- Manipulation entry trigger: `hoveredObject` matches target OR depth ≤ 1 m
- 512-character / 3-sentence cap on `new_semantic_memory`

**From Episodic Learner:**
- `dense_summary` + `what_worked` + `what_to_avoid` output schema
- Coordinate-anchoring requirement (city-block style: "collision in left_corridor at (1.84, 2.3)")

**From VLM Agent:**
- Five-step reasoning process (analyze → reason → plan → act → reflect)
- Output schema: `reasoning`, `actions`, `times`, `notes`
- Rule 2 (keep target centered while approaching)
- Rule 3 (centering as continuous adjustment, not one-shot)
- Rule 5 (item must dominate central FOV before grabbing)
- Rule 9 (grab priority: `grab_item_in_view` first, manual only on failure)
- Rule 10 (only execute actions available in current mode)
- Rule 11 (batch repeated actions for efficiency)
- Rule 13 (strafe parallel to shelf face — now enforced by scan mode)
- Visual cues over numerical targets footer

### 4.3 Existing Helper Functions in `perception.py` (reuse as-is)

| Function | New role |
|---|---|
| `define_cardinal_direction(yaw)` | Core of `snap_to_cardinal` |
| `face_cardinal_direction(angle)` | Core of `turn_left_90` / `turn_right_90` |
| `strafe_to_center(bbox)` | Core of modified `center_object_on_screen` |
| `approach_target(target_name)` | Kept for manipulation approach |
| `detect_object_via_moondream()` | Kept for depth hint and `grab_item_in_view` |

### 4.4 Architecture (retained)
- WebSocket command interface to Unity (`env.py`) — unchanged
- VLM multi-turn conversation history (`VLMAgent.history`) — unchanged
- Semantic + episodic memory as separate stores — unchanged
- Navigation memory (collision records, past paths) — unchanged, just never compressed today (future improvement)
- `OpenRouterConfig` and model selection — unchanged

---

## 5. Agent Reasoner Flow Per Step

### Initialization (once per task)

```
1. Load JSON map for current layout
2. Identify target shelf from task description → look up approach (x, z, yaw) from JSON
3. Build obstacle list from JSON footprints + nav_buffer
4. Set BASE_SEMANTIC_MEMORY (product/conceptual content only — no coordinates)
5. Set agent_start from JSON
```

---

### Per-Step Loop

#### Step 1 — State Reader *(deterministic, no LLM)*
```
Input:  raw WebSocket state response
Output: (x, z, yaw, isColliding, leftHovered, rightHovered, leftGripped, rightGripped)

- Check isColliding → log to navigation_memory if true
- Check current waypoint arrival radius → advance waypoint index if reached
- Run get_heading() → confirm cardinal alignment
```

#### Step 2 — Semantic Observation Updater *(LLM call #1)*
```
Input:  screenshot, depth map, state, task, current semantic memory
Output: { observation, new_semantic_memory }

Scope: observations only — what is visible, product-level facts.
Does NOT decide mode. Does NOT produce routing.
Appends new_semantic_memory to running memory (bounded, observations only).
```

#### Step 3 — Mode Decider *(LLM call #2, or merged with Step 2)*
```
Input:  state, (x, z, yaw) vs target approach, grippedState, depth hint, observation
Output: { mode, mode_rationale }

Cardinal-aware switching:
- navigation  : not at approach (x,z) OR yaw not at target cardinal
- scan        : at approach (x,z) AND yaw correct AND target not visible
- perception  : target visible in frame
- manipulation: target centered AND depth ≤ 1m OR hoveredObject matches target
- STOP        : item gripped AND within 1m of counter
```

> Steps 2 and 3 may be a single LLM call with a combined output schema to save one API call per step.

#### Step 4 — Navigation Reasoner *(LLM call, on-demand only)*
```
Fires when: entering navigation mode AND no current waypoints,
            OR new target detected in recall,
            OR collision logged at current position

Input:  current (x, z, yaw)
        target approach (x, z, yaw) from JSON lookup
        shelf footprints from JSON
        aisle corridor bounds from JSON
        navigation_memory (collision records, past paths)
        episodic_memory

Output: [(x, z, yaw, hint), ...]  — ordered cardinal waypoints

Routing:
- Choose X-first or Z-first L-path to target
- Check each segment against footprints (bounding box intersection)
- If segment blocked: insert detour waypoint through nearest aisle corridor
- No A* — pure arithmetic + obstacle check
```

#### Step 5 — VLM Action Planner *(LLM call #2 or #3)*
```
Input:  screenshot, depth map
        state, mode, available actions for this mode
        current waypoint text (x, z, cardinal_yaw hint) if navigation
        recall (target shelf name + approach coords — injected from JSON, not parsed)
        episodic memory

Output: {
  reasoning: "Chain-of-thought",
  actions:   ["move_forward", "move_left", ...],
  times:     [10, 5, ...],
  notes:     { main_goal, sub_goal, key_info, status, item_name, checklist }
}

No degree arithmetic. No routing. Executes within current mode only.
```

#### Step 6 — Execution *(deterministic)*
```
For each (action, times) pair:
  - Execute action at 0.1m steps (translation) or single absolute call (turn_left/right_90)
  - After each physical move: check isColliding → log if true
  - After completing full sequence: check waypoint arrival radius → advance if reached
```

---

### End-of-Episode *(once per episode, not per step)*

```
Trigger: mode == STOP, OR task failure, OR timeout

Episode Learner Input:
  - Full trajectory log: [(timestep, x, z, yaw, mode, actions, isColliding), ...]
  - JSON aisle corridor definitions (for position labeling)
  - Task description
  - Episode outcome (success / collision-stopped / timeout)
  - Last N VLM reasoning excerpts (qualitative context)

Output: {
  dense_summary:   "with (x, z) at key moments",
  what_worked:     "named aisle + coordinate range that succeeded",
  what_to_avoid:   "(x, z) with aisle label + reason (e.g. too close to unit_a east face)"
}

Stored in episodic_memory (rolling last-5 episodes).
Used in: Step 4 (navigation reasoner), Step 5 (VLM action planner).
```

---

## 6. Reasoner Roles Summary

| Reasoner | Role | Cadence | Key Inputs | Key Outputs |
|---|---|---|---|---|
| Semantic Updater | Observe and record what's visible | Every step | Screenshot, depth, state, memory | `observation`, `new_semantic_memory` |
| Mode Decider | Gate which action set is available | Every step | State, position vs target, observation | `mode`, `mode_rationale` |
| Navigation Reasoner | Plan cardinal waypoint sequence | On-demand (replan only) | Position, JSON map, nav/episodic memory | `[(x, z, yaw, hint)]` |
| VLM Action Planner | Execute within current mode | Every step | Screenshot, mode, waypoint, recall | `actions`, `times`, `reasoning` |
| Episodic Learner | Reflect on full episode performance | End of episode | Trajectory log, JSON aisles, outcome | `dense_summary`, `what_worked`, `what_to_avoid` |

> The Semantic Updater and Mode Decider may be implemented as a single combined LLM call with a structured output schema to minimise API calls per step.

---

## 7. Layout 2 — Derived Cartesian JSON (Reference)

Bounds measured from the Unity sandbox. `agent_start` confirmed at spawn.

```json
{
  "layout_id": "layout_2",
  "bounds": { "x": [0.0, 9.0], "z": [0.0, 7.8] },
  "grid_resolution": 0.1,
  "nav_buffer": 0.3,
  "agent_start": { "x": 2.82, "z": 0.78, "yaw": 0 },

  "obstacles": [
    { "id": "unit_a",  "type": "shelf_unit",
      "footprint": { "x_min": 2.14, "x_max": 7.86, "z_min": 2.7,  "z_max": 3.6  },
      "faces": ["S1", "S2", "S3", "S4"] },
    { "id": "unit_b",  "type": "shelf_unit",
      "footprint": { "x_min": 2.14, "x_max": 7.86, "z_min": 4.83, "z_max": 5.77 },
      "faces": ["S5", "S6", "S7", "S8"] },
    { "id": "s9",      "type": "wall_shelf",
      "footprint": { "x_min": 0.0,  "x_max": 0.44, "z_min": 2.77, "z_max": 3.71 } },
    { "id": "s10",     "type": "wall_shelf",
      "footprint": { "x_min": 0.0,  "x_max": 0.44, "z_min": 4.52, "z_max": 5.46 } },
    { "id": "s11",     "type": "wall_shelf",
      "footprint": { "x_min": 0.0,  "x_max": 0.44, "z_min": 6.26, "z_max": 7.2  } },
    { "id": "s12_13",  "type": "back_shelf",
      "footprint": { "x_min": 1.86, "x_max": 3.17, "z_min": 7.05, "z_max": 7.8  } },
    { "id": "s14_15",  "type": "back_shelf",
      "footprint": { "x_min": 3.64, "x_max": 4.95, "z_min": 7.05, "z_max": 7.8  } },
    { "id": "s16",     "type": "back_shelf",
      "footprint": { "x_min": 5.54, "x_max": 6.85, "z_min": 7.05, "z_max": 7.8  } },
    { "id": "s17",     "type": "back_shelf",
      "footprint": { "x_min": 7.15, "x_max": 8.42, "z_min": 7.05, "z_max": 7.8  } },
    { "id": "counter", "type": "counter",
      "footprint": { "x_min": 0.0,  "x_max": 0.75, "z_min": 0.36, "z_max": 1.9  } },
    { "id": "north_wall", "type": "wall",
      "footprint": { "x_min": 0.0, "x_max": 9.0, "z_min": 7.8,  "z_max": 7.9  } },
    { "id": "south_wall", "type": "wall",
      "footprint": { "x_min": 0.0, "x_max": 9.0, "z_min": -0.1, "z_max": 0.0  } },
    { "id": "east_wall",  "type": "wall",
      "footprint": { "x_min": 9.0, "x_max": 9.1, "z_min": 0.0,  "z_max": 7.8  } },
    { "id": "west_wall",  "type": "wall",
      "footprint": { "x_min": -0.1,"x_max": 0.0, "z_min": 0.0,  "z_max": 7.8  } }
  ],

  "shelves": [
    { "id": "S1",    "unit": "unit_a",  "face": "south",
      "face_span": { "x_min": 2.6, "x_max": 7.4, "z": 2.7 },
      "approach": { "x": 5.0,  "z": 2.4,  "yaw": 0   }, "products": [] },
    { "id": "S2",    "unit": "unit_a",  "face": "north",
      "face_span": { "x_min": 2.6, "x_max": 7.4, "z": 3.6 },
      "approach": { "x": 5.0,  "z": 3.9,  "yaw": 180 }, "products": [] },
    { "id": "S3",    "unit": "unit_a",  "face": "west",
      "face_span": { "x": 2.14, "z_min": 2.7, "z_max": 3.6 },
      "approach": { "x": 1.84, "z": 3.15, "yaw": 90  }, "products": [] },
    { "id": "S4",    "unit": "unit_a",  "face": "east",
      "face_span": { "x": 7.86, "z_min": 2.7, "z_max": 3.6 },
      "approach": { "x": 8.16, "z": 3.15, "yaw": 270 }, "products": [] },
    { "id": "S5",    "unit": "unit_b",  "face": "south",
      "face_span": { "x_min": 2.6, "x_max": 7.4, "z": 4.83 },
      "approach": { "x": 5.0,  "z": 4.53, "yaw": 0   }, "products": [] },
    { "id": "S6",    "unit": "unit_b",  "face": "north",
      "face_span": { "x_min": 2.6, "x_max": 7.4, "z": 5.77 },
      "approach": { "x": 5.0,  "z": 6.07, "yaw": 180 }, "products": [] },
    { "id": "S7",    "unit": "unit_b",  "face": "west",
      "face_span": { "x": 2.14, "z_min": 4.83, "z_max": 5.77 },
      "approach": { "x": 1.84, "z": 5.30, "yaw": 90  }, "products": [] },
    { "id": "S8",    "unit": "unit_b",  "face": "east",
      "face_span": { "x": 7.86, "z_min": 4.83, "z_max": 5.77 },
      "approach": { "x": 8.16, "z": 5.30, "yaw": 270 }, "products": [] },
    { "id": "S9",    "unit": "s9",      "face": "east",
      "face_span": { "x": 0.44, "z_min": 2.77, "z_max": 3.71 },
      "approach": { "x": 0.74, "z": 3.24, "yaw": 270 }, "products": [] },
    { "id": "S10",   "unit": "s10",     "face": "east",
      "face_span": { "x": 0.44, "z_min": 4.52, "z_max": 5.46 },
      "approach": { "x": 0.74, "z": 4.99, "yaw": 270 }, "products": [] },
    { "id": "S11",   "unit": "s11",     "face": "east",
      "face_span": { "x": 0.44, "z_min": 6.26, "z_max": 7.2  },
      "approach": { "x": 0.74, "z": 6.73, "yaw": 270 }, "products": [] },
    { "id": "S12_13","unit": "s12_13",  "face": "south",
      "face_span": { "x_min": 1.86, "x_max": 3.17, "z": 7.05 },
      "approach": { "x": 2.52, "z": 6.75, "yaw": 0   }, "products": [] },
    { "id": "S14_15","unit": "s14_15",  "face": "south",
      "face_span": { "x_min": 3.64, "x_max": 4.95, "z": 7.05 },
      "approach": { "x": 4.30, "z": 6.75, "yaw": 0   }, "products": [] },
    { "id": "S16",   "unit": "s16",     "face": "south",
      "face_span": { "x_min": 5.54, "x_max": 6.85, "z": 7.05 },
      "approach": { "x": 6.20, "z": 6.75, "yaw": 0   }, "products": [] },
    { "id": "S17",   "unit": "s17",     "face": "south",
      "face_span": { "x_min": 7.15, "x_max": 8.42, "z": 7.05 },
      "approach": { "x": 7.79, "z": 6.75, "yaw": 0   }, "products": [] }
  ],

  "counter": {
    "footprint": { "x_min": 0.0, "x_max": 0.75, "z_min": 0.36, "z_max": 1.9 },
    "approach":  { "x": 1.05, "z": 1.13, "yaw": 270 }
  },

  "aisles": [
    { "id": "front_zone",
      "bounds": { "x_min": 0.0,  "x_max": 9.0,  "z_min": 0.0,  "z_max": 2.4  },
      "note": "Agent spawns here. Counter blocks x<0.75 for z<1.9." },
    { "id": "left_corridor",
      "bounds": { "x_min": 0.74, "x_max": 1.84, "z_min": 0.0,  "z_max": 7.05 },
      "note": "Primary N-S route from spawn. ~10 steps west from spawn x=2.82. S3 and S7 approach positions are inside this corridor." },
    { "id": "mid_aisle",
      "bounds": { "x_min": 0.0,  "x_max": 9.0,  "z_min": 3.9,  "z_max": 4.53 },
      "note": "E-W crossing between unit_a and unit_b." },
    { "id": "back_aisle",
      "bounds": { "x_min": 0.0,  "x_max": 9.0,  "z_min": 6.07, "z_max": 6.75 },
      "note": "E-W crossing behind unit_b. Approach zone for all back wall shelves (S12-S17)." },
    { "id": "right_corridor",
      "bounds": { "x_min": 8.16, "x_max": 9.0,  "z_min": 0.0,  "z_max": 7.05 },
      "note": "Alternate N-S route. S4 and S8 approach positions are inside this corridor." }
  ],

  "routing_notes": {
    "preferred_ns_route": "left_corridor",
    "reason": "Spawn x=2.82 is 0.98m from left_corridor entry vs 5.34m from right_corridor.",
    "unit_a_blocks_direct_north": true,
    "unit_b_blocks_direct_north": true,
    "to_cross_north_of_unit_a": "Route through left_corridor or right_corridor to mid_aisle, then traverse east-west.",
    "to_cross_north_of_unit_b": "Same corridors to back_aisle, then traverse east-west.",
    "s3_s7_access_warning": "S3 and S7 approach positions sit inside left_corridor — accessing these end caps partially blocks the primary N-S throughway."
  }
}
```

---

## 8. Key Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| Cardinal yaw constraint | Shelves are axis-aligned; eliminates LLM degree-arithmetic drift; makes routing deterministic |
| 0.1 m step size retained | Unity physics stability; prior developer decision; now hidden inside execution layer |
| No A\* (adviser constraint) | Axis-aligned obstacles + cardinal movement makes L-path + obstacle check sufficient |
| Episodic learner moved to end-of-episode | Per-step episodic calls produced low-quality reflections from a 4-step window; full trajectory gives meaningful arcs |
| Scan mode added | Strafing for shelf search is perceptual work that should not require navigation mode; removing constant mode-swapping reduces semantic learner load |
| Semantic Updater and Mode Decider may be merged | Saves one API call per step; acceptable if output schema is clearly ordered (observe first, mode second) |
| JSON map replaces coordinate sections of `memory.py` | Coordinates in prose are imprecise and LLM-parsed unreliably; structured lookup is exact and programmatic |
| `face_span` in shelf definitions | Gives scan mode a concrete strafe boundary; allows episodic learner to say "strafed to face edge at x=7.4" |
