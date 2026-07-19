# Navigation Reasoner — Implementation Plan

## Context

Navigation is the primary failure mode of the agent: it collides with walls, takes suboptimal paths, and the VLM wastes its reasoning budget on global path planning it fundamentally cannot do from a first-person view. The root cause is that no component has a global view of the store. The semantic memory has shelf coordinates but no path topology between them.

This plan adds a dedicated **NavigationReasoner** that outputs a 3–5 waypoint plan the VLM follows step-by-step. To validate the architecture cheaply before committing to a real LLM reasoner, **Phase 1 uses human input**: when the navigation reasoner is triggered, the console prompts the operator (you) for waypoints. If the VLM follows human-provided waypoints well, Phase 2 swaps in the real LLM-based planner.

Additional changes from design review:
- Navigation memory is separated from semantic memory
- Episodic memory accumulates rather than overwrites
- `SYS_INST_ASSOCIATIVE_SEMANTIC` is trimmed — navigation guidance removed
- `SYS_INST_ASSOCIATIVE_EPISODIC` updated to require spatial context
- Waypoint rationale format carries action hints, not just destination explanations

---

## Critical Files

| File | Change |
|---|---|
| `overhaul/sys_inst.py` | Append Rule 13 to `SYS_INST_VLM_LEAN`; trim `SYS_INST_ASSOCIATIVE_SEMANTIC`; update `SYS_INST_ASSOCIATIVE_EPISODIC`; add `SYS_INST_NAVIGATION_REASONER` (Phase 2) |
| `overhaul/agent.py` | New `HumanNavigationReasoner` + `NavigationReasoner` classes; new state on `EmbodiedAgent`; updated `execute_lean()`; accumulating `set_episodic_memory` |
| `overhaul/memory.py` | Extend `BASE_SEMANTIC_MEMORY` with static aisle topology + shelf footprints (read-only) |
| `overhaul/store_map.png` | New static Unity asset — needed for Phase 2 only |

`subtask_agents.py` — no changes needed.

---

## System Instruction Changes (`sys_inst.py`)

These changes apply in both Phase 1 and Phase 2.

### A. Append Rule 13 to `SYS_INST_VLM_LEAN`

Before the closing `\n\n"` of `SYS_INST_VLM_LEAN`, append:

```
"13. **Waypoint following**: When '## NAVIGATION WAYPOINT' appears in your input, treat it as your immediate sub-goal. Navigate to the specified (x, z) coordinate using the action hints provided before pursuing the final target. These waypoints were pre-computed by a global planner with access to the full store map.\n"
```

### B. Trim `SYS_INST_ASSOCIATIVE_SEMANTIC`

Remove the following — they are replaced by the navigation reasoner and navigation memory:

- **Rule 2** (nearby shelf routing logic — "Go to Shelf Y first, then navigate to Shelf X")
- Both **"EXPLORATION IS ENCOURAGED"** blocks
- Both **"TRUST THE VISUAL CUES MORE THAN THE NUMERICAL TARGET POSITION"** caps (keep obstacle avoidance mention only in Rule 3, remove the navigation planning version)

Replace the `recall` instruction with:

```
'recall': (string) From your updated semantic memory, recall information relevant 
to the current observation and task — specifically: item locations, what is currently 
visible, and the target shelf identity and coordinates. Do NOT include path planning 
or navigation routing; that is handled separately by the navigation reasoner.
```

Rewrite the mode-switching rules to ground them in agent state fields rather than recall:

```
**Mode-switching rules**:
- Switch to *manipulation* when: `leftHoveredObject` or `rightHoveredObject` in the 
  agent state matches the target item, OR the depth hint indicates the item is ≤1 meter away.
- Switch to *STOP* when: the gripped item matches the task target AND (if the task 
  requires placement) the agent is within 1 meter of the counter. Gripping alone is 
  NOT sufficient for STOP if the task requires bringing the item somewhere.
- Switch to *navigation* when the target item is not visible or the agent is not near 
  the target shelf.
- Switch to *perception* when the agent is near the target shelf but needs to visually 
  confirm and center on the item.
```

### C. Update `SYS_INST_ASSOCIATIVE_EPISODIC`

Add explicit requirements for spatial grounding in all three output fields:

```
'dense_summary': (string) A concise summary of the agent's actions and observations. 
Include the agent's position (x, z) at key moments — where it started, where it got 
stuck, and where it succeeded.

'what_worked': (string) A reflection on effective strategies. Reference the specific 
aisle or coordinate range that successful paths used (e.g., 'routing through central 
aisle A at x≈2.0 from z=3.9 to z=7.6 worked well').

'what_to_avoid': (string) A reflection on ineffective strategies. Reference specific 
coordinates where failures occurred (e.g., 'moving directly from x≈1.5 toward x≈4.4 
at z≈3.9 causes a collision near x=3.8 — Shelf 3 east face'). Remind the agent to 
trust visual cues over numerical targets.
```

---

## Phase 1 — Human-in-the-Loop Validation

The goal is to test whether the waypoint injection architecture actually improves VLM navigation **before** building the LLM reasoner. You manually act as the navigation reasoner.

### 1a. New `HumanNavigationReasoner` class (`agent.py`)

Add after `SemanticEpisodicAssociativeLearner`, before `VLMAgent`. Add to imports: `import json`, `import math`, `from pathlib import Path`.

```python
class NavigationPlanError(Exception):
    pass


class HumanNavigationReasoner:
    """
    Prompts the operator for waypoints via stdin.
    Used to validate the waypoint-following architecture before committing to an LLM reasoner.
    """
    ARRIVAL_RADIUS = 0.5  # world units

    def plan(self, current_x, current_z, current_yaw, target_x, target_z,
             target_shelf="", known_obstacles="", navigation_memory="") -> list[dict]:
        print("\n" + "=" * 60)
        print("[NAVIGATION REASONER] Input required.")
        print(f"  Current position : x={current_x:.2f}, z={current_z:.2f}, yaw={current_yaw:.1f}°")
        print(f"  Target position  : x={target_x:.2f}, z={target_z:.2f}  ({target_shelf})")
        if known_obstacles:
            print(f"  Known obstacles  :\n{known_obstacles}")
        if navigation_memory:
            print(f"  Past navigation records:\n{navigation_memory}")
        print()
        print("Enter 2-5 waypoints as a JSON list.")
        print("Each rationale should be an ACTION HINT — facing direction + steps, e.g.:")
        print('  [{"x": 2.0, "z": 4.5, "rationale": "Pan right to yaw≈0, move_forward ~8 steps"},')
        print('   {"x": 2.0, "z": 7.6, "rationale": "Continue move_forward ~31 steps"},')
        print('   {"x": 3.01, "z": 7.65, "rationale": "Pan right to yaw≈270, move_forward ~10 steps"}]')
        print("Then press Enter twice.")
        print("=" * 60)

        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)

        raw = " ".join(lines).strip()
        try:
            waypoints = json.loads(raw)
            if not isinstance(waypoints, list):
                raise ValueError("Expected a JSON array.")
            return waypoints
        except Exception as e:
            raise NavigationPlanError(f"Could not parse your waypoint input: {e}\nRaw: {raw[:200]}")

    @staticmethod
    def distance_2d(ax, az, bx, bz) -> float:
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)
```

### 1b. New waypoint state and navigation memory on `EmbodiedAgent` (`agent.py`)

Inside `__init__`, in the `if mode == 'lean':` block, after `self.set_semantic_memory()`:

```python
self.navigation_reasoner = HumanNavigationReasoner()
self.current_waypoints: list[dict] = []
self.current_waypoint_idx: int = 0
self.current_nav_target: Optional[tuple[float, float]] = None
self.navigation_memory: str = ""          # separate from semantic memory; read by nav reasoner and VLM (nav mode only)
```

### 1c. Update `set_episodic_memory` to accumulate (`agent.py`)

Replace the existing method:

```python
def set_episodic_memory(self, entry: str, max_entries: int = 5) -> None:
    existing = [e for e in self.vlm_agent.episodic_memory.strip().split("\n\n") if e]
    existing.append(entry)
    self.vlm_agent.episodic_memory = "\n\n".join(existing[-max_entries:])
    logger.info(f"Episodic memory updated ({len(existing[-max_entries:])} entries).")
```

### 1d. Five new helper methods on `EmbodiedAgent` (`agent.py`)

```python
def _check_waypoint_advance(self, ax: float, az: float):
    if not self.current_waypoints or self.current_waypoint_idx >= len(self.current_waypoints):
        return
    wp = self.current_waypoints[self.current_waypoint_idx]
    if self.navigation_reasoner.distance_2d(ax, az, wp['x'], wp['z']) <= self.navigation_reasoner.ARRIVAL_RADIUS:
        logger.info(f"Waypoint {self.current_waypoint_idx + 1}/{len(self.current_waypoints)} reached.")
        self.current_waypoint_idx += 1

def _get_current_waypoint_text(self) -> str:
    if not self.current_waypoints or self.current_waypoint_idx >= len(self.current_waypoints):
        return ""
    wp = self.current_waypoints[self.current_waypoint_idx]
    n, total = self.current_waypoint_idx + 1, len(self.current_waypoints)
    return (f"## NAVIGATION WAYPOINT ({n}/{total}): "
            f"Move to (x={wp['x']}, z={wp['z']}). "
            f"Action hint: {wp.get('rationale', '')}\n")

@staticmethod
def _extract_nav_target(recall: str) -> Optional[tuple[float, float]]:
    m = re.search(r'translation\s*:\s*\(\s*([0-9.]+)\s*,\s*[0-9.]+\s*,\s*([0-9.]+)\s*\)', recall)
    return (float(m.group(1)), float(m.group(2))) if m else None

@staticmethod
def _extract_target_shelf(recall: str) -> str:
    m = re.search(r'Shelf\s+\d+', recall)
    return m.group(0) if m else "unknown shelf"

def _maybe_plan_navigation(self, agent_mode: str, recall: str,
                            ax: float, az: float, yaw: float) -> str:
    if agent_mode != "navigation":
        self.current_waypoints = []
        self.current_waypoint_idx = 0
        self.current_nav_target = None
        return ""

    self._check_waypoint_advance(ax, az)
    new_target = self._extract_nav_target(recall)
    need_replan = (not self.current_waypoints) or (
        new_target is not None and new_target != self.current_nav_target
    )

    if not need_replan:
        return self._get_current_waypoint_text()

    if new_target is None:
        logger.warning("Navigation mode but no target coords in recall. Skipping planner.")
        return self._get_current_waypoint_text()

    tx, tz = new_target
    if self.navigation_reasoner.distance_2d(ax, az, tx, tz) <= self.navigation_reasoner.ARRIVAL_RADIUS:
        self.current_waypoints = [{'x': tx, 'z': tz, 'rationale': 'Already at target.'}]
        self.current_waypoint_idx = 0
        self.current_nav_target = new_target
        return self._get_current_waypoint_text()

    try:
        shelf = self._extract_target_shelf(recall)
        waypoints = self.navigation_reasoner.plan(
            ax, az, yaw, tx, tz,
            target_shelf=shelf,
            navigation_memory=self.navigation_memory,   # pass past records to reasoner
        )
        self.current_waypoints = waypoints
        self.current_waypoint_idx = 0
        self.current_nav_target = new_target
        summary = "; ".join(f"({w['x']},{w['z']})" for w in waypoints)
        # append to navigation_memory — NOT semantic memory
        self.navigation_memory += f"[PATH] start=({ax:.1f},{az:.1f}) → {shelf} → waypoints={summary} → outcome=PENDING\n"
        logger.info(f"Navigation plan: {summary}")
    except NavigationPlanError as e:
        logger.error(f"Navigation planner failed: {e}. Proceeding without waypoints.")

    return self._get_current_waypoint_text()
```

Also add a method to record collisions into navigation memory — call this anywhere `isColliding` is detected:

```python
def _record_collision(self, ax: float, az: float, timestep: int):
    self.navigation_memory += f"[COLLISION] position=({ax:.1f},{az:.1f}) → timestep={timestep}\n"
    logger.warning(f"Collision recorded at ({ax:.1f},{az:.1f}) timestep={timestep}")
```

### 1e. `execute_lean()` changes (`agent.py`)

**A) State extraction** — replace `state = str(request['state'])` with:
```python
_raw_state = request['state']
if isinstance(_raw_state, dict):
    _t = _raw_state.get('translation', (0, 0, 0))
    _r = _raw_state.get('rotation', (0, 0, 0))
    _colliding = _raw_state.get('isColliding', False)
else:
    _t = _r = (0, 0, 0)
    _colliding = False
agent_x   = float(_t[0])
agent_z   = float(_t[2])
agent_yaw = float(_r[1]) if len(_r) > 1 else 0.0
state = str(_raw_state)
```

**B) Collision recording** — in BOTH branches, immediately after state extraction:
```python
if _colliding:
    self._record_collision(agent_x, agent_z, timestep)
```

**C) Navigation planning call** — in BOTH branches, after the `agent_mode == "STOP"` early-return block and before the `depth_hint` line:
```python
waypoint_text = self._maybe_plan_navigation(
    agent_mode=agent_mode, recall=recall,
    ax=agent_x, az=agent_z, yaw=agent_yaw,
)
```

**D) VLM message construction** — in BOTH branches, build `user_msg` as follows. Navigation memory is injected only in navigation mode:
```python
nav_memory_block = (f"## NAVIGATION MEMORY:\n{self.navigation_memory}\n"
                    if agent_mode == "navigation" and self.navigation_memory else "")

user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
            f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
            f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
            f"## STATE: {state}\n"
            f"## AGENT MODE: {agent_mode}\n"
            f"## AVAILABLE ACTIONS:\n{available_actions}"
            f"{depth_hint}"
            f"{nav_memory_block}"
            f"{waypoint_text}")
```

*(At timestep 1 the recall/episodic layout differs slightly — apply the same `nav_memory_block` and `waypoint_text` injection pattern to that branch too.)*

### 1f. Extend `BASE_SEMANTIC_MEMORY` (`memory.py`) — static only

Append before the closing `\n\n"` of Layout 3's `BASE_SEMANTIC_MEMORY`. This is **static store knowledge** — it is never modified during a run:

```
**Aisle Corridors** (navigable lanes — use these to plan paths):
- Entrance corridor: x∈[0,6], z∈[0,1.0] — in front of Shelves 1 & 2, near starting point
- Central aisle A: x∈[1.8,2.5], z∈[1.0,7.5] — runs between Shelves 1/2 and Shelves 3/6
- Central aisle B: x∈[3.4,4.0], z∈[1.0,7.5] — runs between Shelves 3/6 and Shelves 4/5
- Back corridor: x∈[0,6], z∈[7.5,9.0] — behind Shelves 5/7/8/9
- Right corridor: x∈[4.5,6.0], z∈[0,9.0] — runs along right wall past Shelves 7/8/9

**Shelf Footprints** (approximate bounding boxes — avoid placing waypoints inside these):
- Shelf 1: x∈[0.8,1.8], z∈[3.0,4.0]
- Shelf 2: x∈[0.8,1.8], z∈[3.7,4.8]
- Shelf 3: x∈[3.8,4.8], z∈[3.4,4.2]
- Shelf 4: x∈[2.3,3.4], z∈[0.0,0.8]
- Shelf 5: x∈[2.5,3.5], z∈[7.2,8.1]
- Shelf 6: x∈[3.8,4.8], z∈[3.2,4.2]
- Shelves 7-8: x∈[2.8,4.0], z∈[6.7,7.6]
- Shelf 9: x∈[4.2,5.4], z∈[6.7,7.6]
- Counter: x∈[0.5,2.5], z∈[0.0,1.5]
```

> Footprint values are estimates from fast-tracking coordinates. Adjust after `store_map.png` is rendered. **Do not append to `BASE_SEMANTIC_MEMORY` at runtime** — it is read-only during a run.

### Phase 1 Verification

1. Run a navigation task (e.g., `"Pick up the milk from Shelf 9"`).
2. When the agent enters navigation mode, the console should print the `[NAVIGATION REASONER]` prompt with current position, target, and any past navigation records.
3. Enter 3 waypoints as JSON with action-hint rationales (facing direction + step count), press Enter twice.
4. Confirm `## NAVIGATION WAYPOINT (1/3): ... Action hint: ...` appears in the VLM's `user_msg` log.
5. Confirm `## NAVIGATION MEMORY:` appears in `user_msg` only during navigation mode, not during perception or manipulation.
6. Confirm `[PATH]` entry is appended to `navigation_memory` (not `base_semantic_memory`) after planning.
7. If the agent collides, confirm `[COLLISION]` appears in `navigation_memory` and in the console prompt on the next navigation plan request.
8. Run 3–5 trials. If the VLM follows action-hint waypoints reliably, Phase 2 is warranted.

---

## Phase 2 — LLM-Based Navigation Reasoner (after Phase 1 validates the concept)

**Prerequisite:** Unity top-down orthographic render saved to `overhaul/store_map.png`.
- Camera at `(5.0, 15.0, 5.0)`, pitch = 90°, 1024×1024 PNG
- Must include: shelf labels, 1.0-unit coordinate grid, axis direction labels (+X right, +Z down), scale bar, origin marker

### New `SYS_INST_NAVIGATION_REASONER` in `sys_inst.py`

Contains:
1. Role: "You are a spatial path planner for an Embodied AI Agent in a 3D convenience store simulation."
2. Verbatim coordinate system block:
```
COORDINATE SYSTEM:
- Store occupies x∈[0,10], z∈[0,10] in Unity world-space units.
- IMAGE LEFT   = low X  (west wall)
- IMAGE RIGHT  = high X (east wall)
- IMAGE TOP    = low Z  (entrance / starting point)
- IMAGE BOTTOM = high Z (back wall)
- Yaw: 0°=facing +Z (image bottom), 90°=facing -X (left),
       180°=facing -Z (image top), 270°=facing +X (right)
- Grid overlaid at 1.0-unit intervals. Agent moves 0.1 units per move_forward step.
- pan_left/pan_right rotate 2.5° per step.
```
3. Reasoning rules:
   - Visually locate agent start and target on the map
   - Identify shelves/walls blocking the direct path
   - Plan 3–5 waypoints through open aisles only — never through a shelf footprint
   - Final waypoint within 0.5 units of target
   - Check past navigation records (`## PAST NAVIGATION RECORDS`) — avoid repeating paths marked FAILED or positions marked COLLISION
4. JSON output schema (single-quoted for `ast.literal_eval`). **Rationale must be an action hint** — facing direction and step count, not a description of the destination:
```
{
  'waypoints': [
    {
      'x': <float 1 decimal>,
      'z': <float 1 decimal>,
      'rationale': '<action hint: e.g. Pan right to yaw≈0, move_forward ~8 steps>'
    },
    ...
  ],
  'overall_strategy': '<two sentences>'
}
```

### `NavigationReasoner` class (replaces `HumanNavigationReasoner`)

```python
class NavigationReasoner(BaseAgent):
    STORE_MAP_PATH = Path(__file__).parent / "store_map.png"
    ARRIVAL_RADIUS = 0.5

    def __init__(self, config=None):
        super().__init__(config)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.config.api_key)
        self._map_image: Optional[Image.Image] = None

    def _load_map(self) -> Image.Image:
        if self._map_image is None:
            if not self.STORE_MAP_PATH.exists():
                raise NavigationPlanError(
                    f"store_map.png not found at {self.STORE_MAP_PATH}. "
                    "Export a top-down orthographic render from Unity."
                )
            self._map_image = Image.open(self.STORE_MAP_PATH).convert("RGB")
        return self._map_image

    def plan(self, current_x, current_z, current_yaw, target_x, target_z,
             target_shelf="", known_obstacles="", navigation_memory="") -> list[dict]:
        map_img = self._load_map()
        user_text = (
            f"## AGENT CURRENT POSITION: x={current_x:.2f}, z={current_z:.2f}, yaw={current_yaw:.1f}°\n"
            f"## TARGET POSITION: x={target_x:.2f}, z={target_z:.2f} ({target_shelf})\n"
            f"## KNOWN OBSTACLES:\n{known_obstacles}\n"
            f"## PAST NAVIGATION RECORDS:\n{navigation_memory if navigation_memory else 'None'}\n"
        )
        content = _build_content(map_img, "## TOP-DOWN STORE MAP\n", user_text)
        resp = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "system", "content": SYS_INST_NAVIGATION_REASONER},
                      {"role": "user",   "content": content}],
            temperature=self.config.temperature,
        )
        return self._parse_waypoints(resp.choices[0].message.content)

    def _parse_waypoints(self, raw: str) -> list[dict]:
        m = re.search(self.extractable_json_structured_output, raw)
        if m:
            try:
                return ast.literal_eval(m.group(1))['waypoints']
            except Exception:
                pass
        m2 = re.search(r'\{[\s\S]*\}', raw)
        if m2:
            try:
                return json.loads(m2.group(0).replace("'", '"'))['waypoints']
            except Exception:
                pass
        raise NavigationPlanError(f"Could not parse navigation plan. Raw: {raw[:300]}")

    @staticmethod
    def distance_2d(ax, az, bx, bz) -> float:
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)
```

**Swap in `EmbodiedAgent.__init__`:**
```python
nav_config = OpenRouterConfig(model_id='google/gemini-3.1-pro-preview', temperature=0.2, mode='lean')
self.navigation_reasoner = NavigationReasoner(nav_config)  # replaces HumanNavigationReasoner()
```

All other methods (`_check_waypoint_advance`, `_get_current_waypoint_text`, `_maybe_plan_navigation`, `_record_collision`, etc.) and all `execute_lean()` changes remain **identical** between Phase 1 and Phase 2. The swap is a one-line change.

---

## Edge Cases (Both Phases)

| Case | Handling |
|---|---|
| Parse failure / bad input | `NavigationPlanError` raised, caught in `_maybe_plan_navigation`, returns `""` — agent proceeds without waypoints |
| Already at target | Pre-plan distance check, no console prompt / LLM call made |
| No coords in recall | Log WARNING, skip planner, return existing waypoint text |
| Same target re-entered | `need_replan = False`, no extra prompt/LLM call |
| All waypoints passed | `_get_current_waypoint_text` returns `""` |
| `store_map.png` missing (Phase 2) | `NavigationPlanError` with actionable message, caught gracefully |
| Collision detected | `_record_collision` appends to `navigation_memory`; visible in next planning prompt |
| Navigation memory in wrong mode | `nav_memory_block` only built when `agent_mode == "navigation"` |
| Episodic memory overflow | `set_episodic_memory` keeps last `max_entries=5` entries only |