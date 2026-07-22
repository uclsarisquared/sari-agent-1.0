# Phase 6 — checklist & results

Living tracker for `slamtest/plans/phase6/phase6_long_horizon_tasks.md`. One section per sub-phase:
the checklist is the plan's sequenced steps, **Gate** is the scripted LLM-free probe that must pass
before the next sub-phase starts, and **Results** records what was actually measured (dates, numbers,
and honest caveats — findings live here even when they're inconvenient). Update this file as each
step lands; future phases follow the same shape.

Legend: `[x]` done · `[ ]` pending · `[~]` partial / known-limited

---

## 6.1 — Hand pose state machine (rest / grab) — **DONE 2026-07-22, gate passed**

### Checklist

- [x] **Step 0 — measure first** (`step0_hand_pose_probe.py` 🎮): M1 occlusion A/B, M2 LiDAR
      self-cull at REST, M3 grip survives moves + checkpoint drive, M4 carried-item occlusion.
      *User-validated live 2026-07-22: "works well."*
- [x] **Step 1 — `set_hand_pose` primitive** (`manipulation.py`): `REST_POSE (-0.213, -0.09, 0.2)` /
      `GRAB_POSE (-0.01, 0.006, 0.33)`; closed-loop drive self-corrects the 0.5 per-component Unity
      clamp and reports `arrived=False` on a frame mismatch instead of a silent wrong pose.
- [x] **Step 2 — mode router conversion** (`agent.py`): nav/perception → `_set_hand_pose("rest")`
      (hands stay ACTIVE, fire-on-change); manipulation → `_invalidate_hand_pose()` (hand free for
      the tool, pose marked unknown so the next step re-asserts REST); `_set_hands` retained for the
      between-task hard reset only.
- [x] **Step 3 — grab tool owns its poses** (`extend_arm_until_grabbed`): full-hand guard REFUSES
      (`{'blocked': True}`) instead of force-opening (which would drop a carried item); GRAB at
      entry, REST restored on every exit (success / miss / exception).
- [x] **Step 4 — between-task reset verified**: `return_to_start` stow left intact (dropping
      leftovers between tasks is intended); three stale "hands are inactive outside manipulation"
      dispatcher comments corrected (the mode gate now stands for mode coherence).
- [x] **Unit tests** (`plan6/test_files/`, offline): 16/16 pass under pytest and standalone.
- [x] **Step 5 — Gate probe built** (`carry_probe.py`).

### Gate 6.1 — scripted carry probe (no LLM) 🎮

> Grab a reachable item → REST → drive a 3–4 checkpoint route → assert grip + arrival every leg.
> **Pass = 3/3 routes, zero drops.**

**Result: PASS (user-run, 2026-07-22).** Carried item survives navigation across checkpoint routes —
zero drops. The original failure mode (item dropped at the first mode transition) is gone.

### Findings / carried-forward caveats

1. **Reach-and-drop on release** (observed in the gate run): the grab/reach sequence can end with
   the item released rather than held in some reach geometries. Accepted for 6.1 — the carry
   contract held — and **owned by 6.2**, whose place envelope + `place_held_item` make release a
   deliberate, measured act instead of a side effect.
2. **`REACH_ENVELOPE` (0.85 m) was fit before GRAB was set at entry.** If grabs start missing at
   distances that used to work, re-run `reach_probe` / `fit_envelope` from the GRAB start pose
   (noted in code at the guard block in `manipulation.py`).
3. Stray-closed-hand recovery now lives at **task start (harness)** only — never mid-task. A hand
   that's gripping at grab-call entry is refused, not opened.
4. **`SetHandsActive(False)` was NOT purged everywhere — only from the agent's mid-task path.**
   Verified by repo-wide grep 2026-07-22. What's gone: the three `agent.py` call sites that fired
   `_set_hands(agent_mode == "manipulation")` / `_set_hands(False)` *during* a task
   (`execute_lean` both arms, `_graph_navigate`, `_metric_approach`) — those were the actual bug.
   What's **intentionally still there**:
   - `eval_pickup.return_to_start` → `agent._set_hands(False)` — the between-task hard reset the
     plan explicitly allows (dropping leftovers between tasks is intended, once per task boundary).
   - `store_map.NavSession(stow_hands=True)` default — the agent's own nav calls already pass
     `stow_hands=False`; the default only protects non-carrying callers.
   - `slamtest/capture_walk.py`, `explore.py`, `explore_vlm.py`, `passive_map.py`, `walk_map.py` —
     SLAM/mapping walks, not the live agent; they stow hands so hands don't clip the LiDAR/camera
     while mapping. Out of scope for 6.1 entirely.
   - `center_live_test.py` — standalone centering test script, unrelated to carry.

---

## 6.2 — Place-at-counter: `plan_place` + `place_held_item` — **NOT STARTED**

### Checklist

- [ ] **Measure first — `place_probe.py`** 🎮: carry an item in, release at a range of distances from
      the counter face (incl. the cp54 dock at 0.20 m standoff); record
      `(slant_distance, pitch, camera_height, landed_on_surface)`; landing judged by eye per trial.
      **Do not reuse 0.85 m by analogy** — falling to a surface ≠ reaching to a grip. Note whether
      release *height* matters; if it does, a distinct PLACE pose earns its existence.
- [ ] Fit the **place envelope** with the `fit_envelope.py` pattern.
- [ ] **`plan_place(sample, envelope)`** — pure function in `manipulation.py`, unit-tested offline
      (`test_plan_place.py`, no sim). Verdicts: `placeable | move N | recenter | unavailable`.
- [ ] **`place_held_item`** — mirror of the grab tool: refuse if NO hand grips; GRAB (or PLACE)
      pose → extend → `ToggleGrip` release → REST; report `{'placed', 'released', 'reason'}`;
      surface as `last_place` in `dispatch_action`.
- [ ] **cp54 standoff decision** — from the probe, not upfront: prefer approach-past-checkpoint
      (code) over re-docking cp54 (data migration). If re-docked anyway: surgical `world_xz` edit +
      `annotate_pass --resume` for route hints.
- [ ] **Sidequest (same sim session):** `capture_walk --kind landmark --ids 54 --angles 1` 🎮 — gives
      cp54 its capture → annotation → prose (open thread #3 in CLAUDE.md).
- [ ] Honest scoring wired: `placed_measured` (deterministic) vs `placed_verified` (screenshot
      audit) — report both, never promote the first to the second.

### Gate 6.2 — scripted grab-carry-place (no LLM) 🎮

> Grab at a shelf → carry to cp54 (6.1 machinery) → centre counter → `plan_place` →
> `place_held_item`. **Pass = 4/5 runs with the item visibly ON the counter.**

**Result:** —

---

## 6.3 — Orchestrator hardening: typed subtasks + deterministic completion — **NOT STARTED**

### Checklist

- [ ] **Typed decomposer**: `decompose_task` returns objects (`type ∈ pickup | place | compare |
      goto`, closed vocabulary) instead of free strings; parse-failure fallback →
      `{"type": "unknown"}` gets the old keyword guards, logged as `untyped`.
- [ ] **A/B the decomposer prompt OFFLINE first** on a fixed battery of ~10 task prompts (four
      families + paraphrases) before wiring in — it's one LLM call, no sim.
- [ ] **Code-side completion predicates** in `run_subtask` (the VLM's STOP becomes a *request* code
      grants or refuses): pickup = grip + name overlap; place = released under `placeable` + at the
      location's checkpoint; compare = choice named from `targets` (observation logged for audit);
      goto = nearest checkpoint matches.
- [ ] **Refusal cap**: 3 refused halts → force-continue with the reason surfaced in state.
- [ ] Compare tasks decompose into **physical inspection** legs (route to both candidates; criterion
      resolved from the camera, never the product index alone).
- [ ] **NOT doing**: a verifier LLM grading completion from screenshots — tighten predicates
      instead; a judge model is last resort and never feeds the headline number.

### Gate 6.3 — one supervised end-to-end run 🎮

> `"get the green Piattos and bring it to the checkout counter"` watched live: sane typed
> decomposition, pickup halt refused until a real grip, place halt refused until placed-at-counter.
> **Pass = one clean run + honest notes on every intervention.**

**Result:** —

---

## 6.4 — `eval_longhorizon.py`: the four families, measured — **NOT STARTED**

### Checklist

- [ ] **Pre-flight harness debt**: fix `eval_pickup.py`'s TASKS unpack bug (plain strings vs
      `(task, kw)` pairs — the no-`--task` path crashes; git history has the tuples).
- [ ] Harness modeled on `eval_pickup.py` (JSONL step logs, per-step screenshots, caps, honest
      `end_reason`) wrapping `subtask_agents.orchestrate`, with per-subtask metric rows.
- [ ] Task families wired with success specs: (1) fetch-to-counter, (2) multi-fetch,
      (3) physical comparison (ground truth from catalog; compare observation audited as
      camera-grounded), (4) budget multi-fetch (prices from `PriceData.json` ground truth; the
      agent's own price knowledge only from tags/annotations).
- [ ] Metrics per task: `t_grip_k`/`t_place_k` per leg, `subtasks_planned/completed`,
      `halts_refused`, `placed_measured` vs `placed_verified`, `success`, `timesteps`, `llm_calls`,
      `wall_s`, `end_reason`.
- [ ] Run families 1→4 in order, 3–4 prompts each (12–16 tasks; expect a multi-hour sim session).

### Result — where do long-horizon runs die?

> This eval is a diagnosis, not a leaderboard: attribute each death to decomposition / carry /
> place / completion without re-running.

**Result:** —

---

## Standing constraints (apply to every sub-phase — from CLAUDE.md + the phase 6 plan)

- **No `env.Reset()`** (duplication bug) — `return_to_start` between tasks.
- **No `isColliding`** — LiDAR + state only, for everything including landing verification.
- **Prompt changes get A/B'd on identical inputs** before claiming improvement.
- **All agent-runtime reasoners on UCL qwen** (`claude -p` is the annotator's path, not the agent's).
- **The frozen map stays frozen** — topology edits are deliberate, documented decisions.
- **Measure, don't assume** — every sub-phase opens with a scripted probe and closes with its gate.

## Template for future phases

```markdown
## <phase id> — <name> — NOT STARTED | IN PROGRESS | DONE <date>, gate <passed/failed>

### Checklist
- [ ] Measure first — <probe> 🎮: <what gets measured, and the trap to avoid>
- [ ] <pure function, unit-tested offline, no sim>
- [ ] <tool/integration step>
- [ ] <decision deferred to the measurement, not made upfront>

### Gate — <scripted, LLM-free probe> 🎮
> <exact pass criterion, as a number>
**Result:** —

### Findings / carried-forward caveats
1. <what was observed, dated; who owns the follow-up>
```
