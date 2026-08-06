# plan6 — Phase 6.1 implementation working folder (DISSOLVED 2026-07-24)

> The `plan6/` folder was dissolved in the 2026-07-24 reorg and organized again under the
> repository-level `validation/` tree. Unit tests are in `validation/tests/agent/`, the decomposer
> comparison is in `validation/evals/`, live diagnostics are split between `validation/probes/`
> and `validation/calibration/`, and selected old results are in `validation/evidence/`.
> `CHECKLIST.md` (the phase-6 progress record) was **deleted on user
> request** — recoverable from git history if the measured numbers are ever needed.

Implements **Phase 6.1** (hand-pose state machine: hands stay ACTIVE, parked at REST, so a carried
item survives navigation). Design lives in
[`phase6_long_horizon_tasks.md`](phase6_long_horizon_tasks.md).

## Step 0 — measure before building (this is what's here now)

The plan front-loads four measurements that have each burned this project when assumed. **None of the
agent code changes until these pass** — Step 0 only reads the sim.

Run with the sim in **Play mode** on `ws://localhost:8080`:

```
python validation/calibration/step0_hand_pose_probe.py
```

| # | Question | Command | Pass condition |
|---|---|---|---|
| M1 | Does REST keep the hand out of the camera frame? | `m1` (at a shelf **and** in an aisle) | resting hand doesn't intrude; else re-pick the pose |
| M2 | Does REST survive the LiDAR clearance gate? | `m2` | clearance at REST ≈ clearance hands-off (self-culled) |
| M3 | Does a gripped item survive moves + a full checkpoint drive? | `grab`, then `w/s/a/d` and `drive <cp>` | grip stays closed every leg, zero drops |
| M4 | Does a carried item occlude the camera / LiDAR centre ray? | `grab`, then `lidar` + drive around | recorded as a known cost if it does |

Every read is appended under `validation/artifacts/calibration/hand_pose/`; screenshots land beside it;
`notes.md` is seeded with the four questions to fill in. Type `?` in the probe for the full command
list.

### What Step 0 does NOT do

It does not touch `agent.py`. The `set_hand_pose` helper it carries is a **local, closed-loop copy**
used only to place the hand for measurement — the production `set_hand_pose` and the `_set_hands →
_set_hand_pose` conversion are Steps 1–2, below.

## Steps 1–4 — implemented (user validated Step 0 as working 2026-07-22)

The production changes that make the hand a REST/GRAB pose machine instead of active/inactive:

- **Step 1 — `set_hand_pose` primitive** ([manipulation.py](../manipulation.py)): `REST_POSE`/`GRAB_POSE`
  constants + a closed-loop `set_hand_pose(pose, hand)` (the step-0 approach, ported). Drives by
  `(target − reported)` each iteration, so Unity's 0.5 per-component clamp just costs an extra
  iteration, and a frame mismatch surfaces as `arrived=False` rather than a silent wrong pose.
- **Step 2 — mode router** ([agent.py](../agent.py)): `_set_hands(active)` is replaced on the
  nav/perception path by `_set_hand_pose("rest")` (hands stay ACTIVE at REST, state-tracked,
  fire-on-change). In manipulation mode the router calls `_invalidate_hand_pose()` — the hand is left
  free for the grab/place tool, but the pose is marked UNKNOWN so the next nav/perception step
  re-asserts REST (covers a manual hand poke displacing it). `_set_hands` is retained for the
  between-task hard reset only.
- **Step 3 — grab tool** ([manipulation.py](../manipulation.py) `extend_arm_until_grabbed`):
  refuses (`{'blocked': True, ...}`) when the hand is already gripping instead of force-opening it
  (which would drop a carried item); sets GRAB at entry and restores REST on **every** exit path
  (success / miss / exception) via a `finally`.
- **Step 4 — between-task reset**: `pickup_navigation.return_to_start`'s stow logic is left intact (it
  drops leftovers between tasks by design); three dispatcher comments that wrongly claimed hands are
  "inactive outside manipulation" were corrected (post-6.1 the gate is for **mode coherence**).

### Tests (offline, no sim)

```
uv run pytest validation/tests/agent -q
```

- `validation/tests/agent/test_set_hand_pose.py` — convergence, over-clamp iteration,
  named + xyz poses, right-hand routing, frame-mismatch → `arrived=False`.
- `validation/tests/agent/test_extend_arm_guard.py` — full-hand guard refuses without
  touching the pose; GRAB-then-REST on success, miss, **and** exception.
- `validation/tests/agent/test_agent_hand_pose_router.py` — activate + drive
  REST; no-spam on repeat; invalidate forces a re-drive; hard-stow → re-activate + re-drive; router
  never sets GRAB.

## Step 5 — Acceptance criterion 6.1: `validation/probes/carry_probe.py` 🎮

The scripted, no-LLM gate — run it to test the whole 6.1 chain live:

```
python validation/probes/carry_probe.py
python validation/probes/carry_probe.py --grab-cp 32 --route 15,20,32,54 --routes 3
```

Per route it drives to the grab checkpoint, secures a grip (auto-grab, with a no-VLM nudge fallback
if the item isn't under the hand), then carries through a 3–4 checkpoint route — asserting after
**every leg** that `leftGrippedState` is still True and the drive arrived (clearance never froze),
saving a screenshot per leg for the occlusion/parenting eyeball. **Pass = 3/3 routes, zero drops.**
Logs and screenshots land under `validation/artifacts/probes/carry/`.
