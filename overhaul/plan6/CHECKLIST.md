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
- [x] **Step 1 — `set_hand_pose` primitive** (`manipulation.py`): `REST_POSE (-0.213, -0.09, 0.26)` /
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

## 6.2 — Checkout: scan at the scanner, then bag in the tray — **DONE 2026-07-23 (user-accepted)**

> **Restructured 2026-07-22** — the first live probe run showed cp54 is a SELF-CHECKOUT (POS screen +
> barcode scanner + bagging tray), not a passive counter: the task is scan-then-bag, not
> drop-anywhere. Full plan (incl. what the Unity C# says about the scanner):
> `slamtest/plans/phase6/phase6.2_scan_checkout.md`. The probe machinery, `go_to_counter`,
> `center_to_counter`, and the tray-landing rows all SURVIVE the restructure.

### Findings so far (measured 2026-07-22)

1. **Counter-surface centring works**: with the re-worded target ("top surface, not the entire
   counter") `surface_height` locked to 0.583–0.585 m across pitches 32–51° (4 trials). Aim-at-edge
   failure mode found and fixed via the `center [y]` dial.
2. **cp54 dock is too far**: slant 1.64 m at arrival → drop MISSES the tray. Lands at ≤ 1.29 m
   (+5 steps in). Far boundary still loose (1.29–1.64 gap unsampled); approach-past-checkpoint now
   forced regardless (scan needs hand-over-pad range).
3. **Scanner mechanics read from source** (BarcodeScanner.cs / RetailItemRuntimeService.cs): scan =
   Barcode-tagged MeshCollider entering a trigger box; held items KEEP their MeshColliders as live
   triggers → an in-hand sweep scans WITHOUT dropping (M1 verifies live). Easy/Medium/Hard
   zones differ hugely (inspector box / 4 cm slab / raycast); no checkout-state websocket command
   exists → verification = region-OCR of the POS-screen receipt (measured) + the human's beep/eye
   (verified), with an agent-readable channel awaiting a filed `RequestCheckoutState` sim ask.
4. **Simplifying decisions (user, 2026-07-22):** the Easy trigger box was RESIZED to cover the
   ENTIRE scan region, and the drop-scan fallback is DELETED (scanning never releases the item).
   Net effect: no stop-distance / sweep-height calibration at all — the one geometric rule is
   "pad's centre-LiDAR slant inside `SCAN_REACH_M` (seed 0.85 m), then extend fully"; M3 shrinks
   to a yes/no check of that seed, and M6 (drop-scan reliability) is gone with the fallback. Scan
   verification KEEPS OCR, but region-cropped: `center_to_screen` locks the POS monitor and its
   final bbox scopes the OCR (`read_text_in_box`) to just the receipt, diffed against a running
   baseline — no full-frame read, no stop-condition role (it runs after the sweep retracts).
5. **M1 / M3 / M5 PASS — scan sweep works, measured (2026-07-22, 2 datapoints, full
   `scanner→align→adv→sweep` chain each, `scan_probe.csv`):** both runs `scanned_measured=True`,
   `scanned_verified=True`, `still_holding=True` — the in-hand sweep scans and never drops the
   item (**M1**). Scanned at slant **0.889 m** and **0.844 m** (19 extend steps each), bracketing
   the `SCAN_REACH_M=0.85` seed with one datapoint slightly OVER it — the seed is comfortable, not
   tight; keep 0.85 (**M3**). OCR read real catalog IDs off the POS screen (`JIN_RAMEN_MILD_120`,
   `LESLIES_CLOVERCHIP`) + subtotal/tax; run 2 correctly showed the receipt ACCUMULATING (the
   running-baseline diff works across scans) — user confirms "very accurate" (**M5**). M2/M4
   worked implicitly (the chain centred + aligned well enough to scan twice); their residual
   numbers weren't separately logged. **Caveat for the production tool:** the `new_lines` diff is
   EXACT-STRING, so OCR char jitter across frames (`JIN_RAKEN`→`JIN_RAMEN`) re-counted an
   already-present line as new. Harmless to `scanned_measured` (digit-line test), but if the
   production `scan_held_item` needs "a SPECIFIC new item," fuzzy/normalize the diff instead of raw
   compare.
6. **M7 — tray far bound bracketed; Option A chosen (measured 2026-07-23, `place_envelope.csv`):**
   three drops at slant **1.36 / 1.44 / 1.48 m** all MISSED. With the first run's LANDS-at-≤1.29 m,
   the far edge sits in **(1.29, 1.36)** → `place_max = 1.28 m` (midpoint 1.325 − 0.04 margin). User
   chose **Option A** (build on the confirmed bound) over Option B (probe the aim confound). Two
   honest caveats: (a) the current CSV holds ONLY the misses — the ≤1.29 land row is from the
   earlier session, so `fit_place_envelope.py` can't re-derive `place_max` until a land row is
   re-logged; (b) the misses could be "fell short" OR "rolled off the front edge" (center_to_counter's
   un-dialed aim) — if the gate shows roll-off, dial `COUNTER_AIM_NORM` deeper and re-measure, the
   envelope may widen. No near bound was seen (every miss was farther than every land).
7. **Orchestration decision (user Q + doctrine, 2026-07-23):** the ~7 manual probe steps
   (`goto/g/counter/scanner/align/adv/sweep`) are the DEBUG interface, not the agent's job. Per
   CLAUDE.md ("geometry is deterministic; the VLM judges only what is in front of it"), the checkout
   mechanical chain becomes ONE deterministic tool — `checkout_held_item()` = `align_to_scanner` →
   `scan_held_item` → `center_to_tray` → `place_held_item`, no LLM inside. The VLM's only role is
   emitting a typed `checkout`/`place` subtask (6.3); it never sequences align/sweep/place itself.
8. **First live place run dropped on the SCANNER PAD — two root causes, both fixed (2026-07-23):**
   the run did the whole chain but `place_held_item` released the item onto the scan pad, not the
   tray. (a) **Release-gate bug:** `center_to_counter` STALLED (residual (57.6,-18.9)px — the counter
   surface's bbox is huge/unstable up close) yet `place_held_item` released anyway (it only aborted
   on `not_detected`). FIX: release now requires a `success` centre; stall→re-centre, repeated
   stalls→abort HOLDING (never a blind drop). (b) **Wrong target:** centring "the counter surface"
   lands near the scanner, not the recessed bin. FIX: dedicated **`center_to_tray`** canned target
   (perception.py), and `place_held_item` now targets it. **OPEN:** `place_max=1.28 m` was measured
   centring the COUNTER SURFACE — it must be RE-MEASURED against the tray (the bin is recessed, so
   the slant differs); `center_to_tray`'s wording/aim are provisional until a live probe/gate pins
   them. The next smoke run is the first test of both.
9. **Hand clips THROUGH the counter — release-depth rule (user-measured, 2026-07-23):** the sim lets
   the hand pass through the counter (a physics limit we cannot fix), so a full-extension drop
   releases the item INSIDE/beyond the counter. Agent-side rule (user personally measured): release
   when the hand's forward translation z reaches `centre-LiDAR distance − 0.1 m` — i.e. stop 0.1 m
   SHORT of the surface so the item falls ONTO it. Implemented in `place_held_item`
   (`PLACE_CLIP_STANDOFF_M = 0.1`, stops on that depth OR a stall, whichever first; reports
   `release_z`). Applies to the DROP only — the scan sweep still extends fully (an item passing
   through the scan zone is fine, and is never released there). **Note:** when the tray envelope is
   re-measured (finding 8), the probe's extend-release must use THIS same clip-standoff, or the
   measured envelope won't match production behaviour.
10. **Two full-chain smoke runs — both scanned + bagged by eye; align verdict CALIBRATED
    (2026-07-23):** run 1 aligned=True (slant 0.87, yaw −2.2°); run 2 the sweep scanned and the item
    landed in the tray, but the chain reported `aligned=False` at yaw +3.2°. Cause: the verdict tested
    `abs(yaw) <= 2.5°`, stricter than the align loop's OWN convergence (it stops when the strafe rounds
    to 0 steps, i.e. lateral < 0.05 m) and stricter than reality (scanned fine at +3.2°). FIX: the
    verdict now judges the **lateral offset** (`slant·sin(yaw)`), re-measured from the final pose,
    against `ALIGN_LATERAL_TOL_M = 0.05 m` (the strafe's own resolution AND above both measured-OK
    offsets: 0.033 m @ −2.2°, 0.047 m @ +3.2°). Both runs now read aligned=True. `align_to_scanner`
    also returns `lateral` now. **M2/M4 effectively PASS** — align converges and the pad centres well
    enough to scan across runs. **`place_held_item` released cleanly** (clip-standoff working); the
    tray envelope still wants the finding-8 re-measure, but the drop mechanics are sound.

### Checklist

- [x] **`scanningDifficulty` confirmed Easy** (user, Unity inspector, 2026-07-22) — and the Easy
      box resized same day to cover the whole scan region; Medium/Hard are follow-ons, not 6.2.
- [x] **M1 / M3 / M5 scan probes — PASS** (2026-07-22, 2 datapoints, finding 5): sweep scans +
      item held (M1); `SCAN_REACH_M=0.85` comfortable (M3); POS screen locks + OCR very accurate
      (M5). M2 (pad centring) / M4 (align convergence) worked implicitly across both chain runs but
      their residuals weren't separately logged — capture them if a number is wanted before the
      production `align_to_scanner`. (M6 deleted with the drop-scan fallback.)
- [x] **`go_to_counter` tool** (`store_map.go_to_counter` + `agent._navigate_to_counter`) — built;
      exercised live by the `gp` runs. 6.3 dispatch wiring still deferred.
- [~] **`center_to_counter`** — built + measured stable on the counter SURFACE (finding 1). NOT the
      bag target: up close its bbox stalls and lands near the scanner (finding 8) → the tray got its
      own sibling (`center_to_tray`) below. Still usable for a generic counter-surface centre.
- [~] **`center_to_tray`** — built (`perception.py`, canned target for the recessed bagging bin +
      `TRAY_AIM_NORM`), split off `center_to_counter` after finding 8. Wording/aim PROVISIONAL until
      a live probe/gate pins them; `place_held_item` targets it.
- [~] **`center_to_scanner`** — built (`perception.py`, canned target + `SCANNER_AIM_NORM`);
      wording/aim PROVISIONAL until M2 pins them (same A/B rule as every prompt).
- [~] **`center_to_screen` + `read_text_in_box`** — built (`perception.py`): centre the POS
      screen, region-OCR its bbox (`center_object_on_screen` now returns its final `box` for the
      crop). Wording/aim PROVISIONAL until M5.
- [x] **`align_to_scanner`** — BUILT + CALIBRATED (`store_map.align_to_scanner(nav)`, mirrors
      `go_to_counter`): Option A lateral strafe (vs cp54 perpendicular) + advance-to-`target_slant`
      loop, all existing primitives, frozen map untouched. Returns `{aligned, slant, residual_yaw,
      lateral, reason}` for `last_align`. The `aligned` verdict now judges the **lateral offset**
      (`ALIGN_LATERAL_TOL_M = 0.05 m`), calibrated from 2 live runs (finding 10) — not raw yaw, which
      wrongly failed a run that scanned fine. Converged both runs; splice a scan-dock node ONLY if it
      later measures flaky. Agent mode-machine dispatch is 6.3.
- [x] **`scan_held_item`** — BUILT (`perception.scan_held_item`): refuse if NOT holding; GRAB pose
      → FULL extension (stall/clamp stop, no distance calibration) → retract → REST →
      `center_to_screen` + region-OCR **fuzzy** delta → `scanned`. Returns
      `{scanned, still_holding, receipt, new_lines, reason}` for `last_scan`. Grip NEVER opens (no
      drop-scan variant). The fuzzy diff (`_fuzzy_new_lines`, absorbs OCR jitter) is unit-tested
      offline (`plan6/test_files/test_scan_diff.py`, 5/5, incl. the real run1→run2 delta). The OCR
      delta is the interim measured signal until `RequestCheckoutState` lands.
- [~] **Bagging tools — BUILT, envelope re-measure REOPENED** (M7 + finding 8): **`plan_place`**
      (pure, mirrors `plan_reach`) + **`test_plan_place.py`** (offline, 5/5) + **`place_held_item`**
      (`perception.py`: guard → **center_to_tray** → require a `success` centre → depth-gated approach
      loop → GRAB-extend to `centre_dist − 0.1 m` (clip-standoff, finding 9) → open grip → REST;
      release NEVER fires on a stalled centre — finding 8 fix). `place_max=1.28 m` is in
      `manipulation.PLACE_ENVELOPE` but was measured on the COUNTER SURFACE, so it is **unverified for
      the tray** and must be re-measured with `center_to_tray` + the clip-standoff before the bag
      numbers are trusted (the next smoke run is the first test).
- [x] **`checkout_held_item()` — the dedicated deterministic macro — BUILT** (finding 7,
      `store_map.checkout_held_item(nav)`): `[go_to_counter] → align_to_scanner → (baseline receipt) →
      scan_held_item → place_held_item`, no LLM inside — the single tool the 6.3 typed `checkout`
      subtask dispatches to. Reads the POS baseline AFTER aligning (screen legible) then re-acquires
      the pad, so `scanned` is an honest delta. `bag_if_unscanned=False` — won't bag an unscanned item
      (hides the failure). Returns `{success, scanned, placed, aligned, steps, reason}`; success =
      scanned AND placed (both measured). Composes the four smoke-validated primitives; needs its own
      one-call live run (the Gate is that run).
- [x] Honest scoring wired: `checkout_held_item` emits the two MEASURED numbers (`scanned` OCR delta,
      `placed` under a `placeable` verdict); `gate_checkout.py` records the two VERIFIED ones (human
      beep + in-tray), never promoting. On the 2 accepted runs measured≈verified (both scanned + in
      tray by eye). At RUNTIME the agent uses only the measured signals (no human); the verified pass
      is what earns that trust.
- [ ] **CARRIED FORWARD (non-blocking — 6.2 accepted without these):** cp54 capture sidequest
      (`capture_walk --kind landmark --ids 54 --angles 1` 🎮 → annotation → prose, open thread #3);
      and the SariSandboxV2 asks (`RequestCheckoutState` itemOrder/qty/subtotal; item-world-position).
      These improve prose / give a durable ground-truth scan+place signal, but do not affect the
      shipped checkout capability.

### Gate 6.2 (revised) — scripted checkout chain (no LLM) 🎮

> Grab at a shelf → carry to the checkout (6.1) → `align_to_scanner` → scan (region-OCR of the
> POS-screen receipt + user beep/eye) → bag in the tray (`plan_place` + `place_held_item`).
> **Pass = 4/5 runs with the item scanned AND visibly in the tray.**

**Driver:** `python gate_checkout.py` (default 5 runs, pass 4) — grab in place each run (you align on
an item), then `checkout_held_item(nav)` runs the deterministic chain; you confirm beep + in-tray by
eye. Logs the four honest numbers per run to `slamtest/output/placetests/gate_checkout.csv`, which
**doubles as the finding-8 tray-envelope re-measure** (`place_slant` + `placed_verified` per drop).

**Result: PASS — user-accepted 2026-07-23.** Honest record: the formal 5-run `gate_checkout.py` was
NOT run; the user accepted 6.2 on the basis of **two full-chain live runs** (finding 10) that both
scanned AND bagged the item in the tray by eye, with the drop bugs (finding 8) and clip-through
(finding 9) fixed and the align verdict calibrated. `gate_checkout.py` remains available for the full
4/5 number and the tray-envelope re-measure if a harder sign-off is ever wanted. Carried forward, NOT
blocking (they do not affect the checkout capability): the cp54 capture sidequest and the two
SariSandboxV2 asks below; the tray envelope stays at `place_max=1.28 m` (works in both runs, formally
unverified-at-edge — re-measure via the gate CSV if a drop ever misses).

---

## 6.3 — Orchestrator hardening: typed subtasks + deterministic completion — **PLANNED 2026-07-23**

> **Replanned 2026-07-23** to absorb the 6.2 restructure — full plan:
> `slamtest/plans/phase6/phase6.3_orchestrator_hardening.md`. Key change vs the master plan's 6.3:
> the type vocabulary is `pickup | checkout | compare | goto` — **`checkout` replaces `place`**,
> because the whole scan-then-bag chain is now ONE deterministic macro
> (`store_map.checkout_held_item(nav)`, 6.2 finding 7); the VLM emits the typed subtask and calls
> the one tool, never sequencing align/sweep/place. Pre-flight (Gate 6.2 recorded): SATISFIED
> 2026-07-23 — 6.3 is unblocked end-to-end.

### Checklist

- [ ] **Offline first — typed decomposer A/B** (`plan6/test_files/ab_decompose.py` +
      `decompose_battery.json`, ~10 prompts: four families + anti-keyword paraphrases like
      "obtain"/"deposit"/"leave it at the till"); eyeball the JSON before wiring in. One LLM call
      per prompt, no sim.
- [ ] **Offline — completion predicates as pure functions** +
      `plan6/test_files/test_completion_predicates.py` (grant / refuse / the old guards' known
      misses: released-in-aisle, paraphrased pickup, scanned-but-not-bagged). Refusal cap included.
- [ ] **Typed decomposer wired**: `decompose_task` returns objects (`type ∈ pickup | checkout |
      compare | goto`, closed vocabulary); parse-failure fallback → `{"type": "unknown"}` keeps the
      old keyword guards, logged as `untyped`. Decomposer prompt teaches checkout = scan + bag as
      ONE subtask (the macro owns the navigation too).
- [ ] **Dispatch wiring (the deferred 6.2 debt)**: `checkout_held_item` + `go_to_counter` exposed
      as dispatchable actions (mode-gated like the rest); macro result surfaced as `last_checkout`
      (the `last_reach`/`last_scan` state-channel pattern). The checkout internals
      (`align_to_scanner` / `scan_held_item` / `place_held_item`) are NOT individually exposed.
- [ ] **Code-side completion predicates** in `run_subtask` (the VLM's STOP becomes a *request*
      code grants or refuses): pickup = grip + name overlap (`eval_pickup.name_matches`);
      checkout = `last_checkout.scanned` AND `.placed` (both measured) AND no grip; compare =
      choice named from `targets` (observation logged for 6.4 audit); goto = nearest checkpoint
      matches; unknown = old keyword guards.
- [ ] **Refusal cap**: 3 refused halts on a leg → force-continue with the reason in state
      (`last_halt_refused`); leg marked `halt_forced` (6.4 counts these). The cap forces
      continuation of the LOOP, never a fake grant.
- [ ] Compare tasks decompose into **physical inspection** legs (route to both candidates; criterion
      resolved from the camera, never the product index alone).
- [ ] **NOT doing**: a verifier LLM grading completion from screenshots — tighten predicates
      instead; a judge model is last resort and never feeds the headline number. Also NOT
      re-exposing the checkout internals as agent actions "for flexibility".

### Gate 6.3 — one supervised end-to-end run 🎮

> `"get the green Piattos and bring it to the checkout counter"` watched live: sane typed
> decomposition (a `pickup` leg + a `checkout` leg, no invented locations), pickup halt refused
> until a real grip on a name-matching item, checkout halt refused until the macro reports
> scanned AND placed, findings summary between legs, refusal cap never silently fires.
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
