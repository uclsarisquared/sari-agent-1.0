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
11. **Drop-centring tolerance LOOSENED (user, 2026-07-23, post-acceptance refinement):** the finding-8
    fix required a dead-centre (`outcome=='success'`, residual ≤ 20 px) tray lock before releasing —
    stricter than needed; the drop only has to land ON the tray, not at its centre. `place_held_item`
    now releases when the LiDAR ray (frame centre = the drop point) falls within the inner
    `PLACE_TRAY_EDGE_MARGIN = 0.6` of the **tray's own bbox** (`_ray_in_box`), so an off-centre but
    not-near-the-lip lock is accepted; ray out near the lip → re-centre → abort holding. Still safe
    from finding-8 (the check is against the TRAY box, so an off-centre accept can't drift onto the
    scanner). Tunable via `edge_margin`; bigger = more permissive but riskier at the lip.

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

## 6.3 — Orchestrator hardening: typed subtasks + deterministic completion — **BUILT 2026-07-23 (offline + wiring + restructure + map planning); GATE PENDING**

> **Replanned 2026-07-23** to absorb the 6.2 restructure — full plan:
> `slamtest/plans/phase6/phase6.3_orchestrator_hardening.md`. Key change vs the master plan's 6.3:
> the type vocabulary is `pickup | checkout | compare | goto` — **`checkout` replaces `place`**,
> because the whole scan-then-bag chain is now ONE deterministic macro
> (`store_map.checkout_held_item(nav)`, 6.2 finding 7); the VLM emits the typed subtask and calls
> the one tool, never sequencing align/sweep/place. Pre-flight (Gate 6.2 recorded): SATISFIED
> 2026-07-23 — 6.3 is unblocked end-to-end.
>
> **Extended 2026-07-23 (user request):** the orchestrator was restructured to be eval_pickup-shaped
> (`run_subtask` → `run_leg` = `run_one` generalised: `--arm {vlm,graph}`, per-leg caps, crash-proof
> JSONL, honest `end_reason`; eval_pickup FROZEN), and made MAP-AWARE (plan-time resolver, leg
> ordering, `nearest_checkpoint` feed, compare visited-both). Everything except the live supervised
> gate is built and offline-verified — **66/66 offline tests pass**; typed decomposer A/B 11/11 clean;
> a live decompose→plan→order on the gate task resolves the pickup leg to real checkpoints
> ([26,45,52]) in one resolver call.

### Checklist

- [x] **Offline — typed decomposer A/B** (`plan6/test_files/ab_decompose.py` + `decompose_battery.json`,
      11 prompts: four families + anti-keyword paraphrases obtain/deposit/till/pay/fetch). RAN live vs
      qwen 2026-07-23 → **11/11 typed decompositions parsed with NO `unknown` degradation**, every type
      sequence matched expectation (dump: `plan6/test_files/ab_decompose_out.json`). See finding 1.
- [x] **Offline — completion predicates as pure functions** (`subtask_completion.py`) +
      `plan6/test_files/test_completion_predicates.py` — **28/28 pass**, incl. the three failures the
      old keyword guards got wrong (wrong-item grab, scanned-but-not-bagged, paraphrased pickup) and
      the parser's typed/legacy/garbage fallbacks. Runs in ms (the logic is in a sim-free module, NOT
      subtask_agents which pulls the model stack).
- [x] **Typed decomposer wired**: `decompose_task` now returns typed dicts via
      `subtask_completion.parse_decomposition`; parse-failure / untypeable element → `{"type":
      "unknown"}` handled by the old keyword guards (warned as `untyped`). Prompt teaches checkout =
      scan + bag as ONE subtask. `run_subtask`/`orchestrate` thread the dict (`text` is agent-facing).
- [x] **Dispatch wiring (the deferred 6.2 debt)**: `checkout_held_item` exposed as a dispatchable,
      manipulation-gated macro action — advertised in `actions_str.MANIPULATION_ACTIONS`, dispatched
      via new `agent._checkout_held_item()` (runs `store_map.checkout_held_item(nav)` on the cached
      carry-safe nav session), result surfaced as `last_checkout`. Wrong-mode emit blocks and
      self-corrects via `last_action_blocked` (same loop as `extend_arm_until_grabbed`). The checkout
      internals (`align_to_scanner`/`scan_held_item`/`place_held_item`) are NOT individually exposed.
      (`go_to_counter` primitive `agent._navigate_to_counter` stays available; the macro drives itself,
      so a separate goto-counter dispatch wasn't needed for the fetch-to-counter path.)
- [x] **Code-side completion predicates** in `run_subtask` (STOP is now a *request* code grants or
      refuses via `completion_predicate`): pickup = grip + `name_overlap` (re-homed `name_matches`,
      now shared with eval_pickup); checkout = `last_checkout.scanned` AND `.placed` AND no grip;
      compare = choice named from `targets` (observation logged, not judged); goto = nearest == target
      checkpoint; unknown = old keyword guards verbatim.
- [x] **Refusal cap**: `HALT_REFUSAL_CAP = 3` refused halts on a leg → force-END the leg
      (`halt_forced=True` in the leg result, reason left in `last_halt_refused`), never a fake grant.
- [x] Compare decomposes into **physical inspection** legs — validated by the A/B (family 3 →
      `goto → compare(criterion) → pickup → checkout`); the predicate checks the *choice* in code and
      logs the observation for the 6.4 camera-grounded audit.
- [x] **NOT done, on purpose**: no verifier LLM grading completion from screenshots; no re-exposing
      the checkout internals as separate agent actions.

#### Restructure + map-aware planning (2026-07-23, user request — finding 4)

- [x] **`run_subtask` → `run_leg`** = `eval_pickup.run_one` generalised: per-leg caps
      (`--max-steps`/`--max-minutes`), crash-proof per-leg JSONL + one-dir-per-leg screenshots,
      `try/except` execute_lean + 3-error cap, parse-error tolerance, honest `end_reason ∈
      {halt_granted, halt_forced, step_cap, time_cap, errors}`. Differences from run_one: predicate
      stop (not auto-grip), semantic/episodic memory PERSISTS across legs, map-aware. **eval_pickup.py
      FROZEN** (4.2 baseline; harness duplication is the accepted cost). `--match` dropped (the typed
      pickup predicate grounds success). A failed leg ABORTS remaining legs.
- [x] **CLI** now eval_pickup-shaped: `--arm {vlm,graph}` (default **graph**), `--task`,
      `--max-steps`, `--max-minutes`, `--out`, `--run-dir` (+ bare positional task still works). One
      run-dir per task: `leg<NN>.jsonl` + `leg<NN>/step<NN>.png` + `summary.json` (kills the old
      SIM_RUNS2/SIM_RUNS3 split).
- [x] **#1 Plan-time map resolution** (`subtask_planning.plan_legs`, sim-free module +
      `test_subtask_planning.py` 12/12): resolver runs ONCE per pickup/goto/compare target before any
      sim motion → `candidates` / `target_checkpoint` / `candidate_sets` on the leg; `feasible=False`
      flags a doomed leg up front. `agent.seed_nav_candidates()` makes `_graph_navigate` reuse the
      plan-time candidates instead of re-resolving (no plan/execution disagreement). Location naming
      stays semantic (decomposer never emits ids).
- [x] **`nearest_checkpoint` state feed** each step (from the live pose the refresh already fetched +
      `StoreMap.nearest_checkpoint`, no extra round-trip) → **closes finding 2's goto `[unverified]`
      gap**; also grows the task-level visit trace.
- [x] **#3 `order_legs`**: reorders independent (pickup→checkout) pairs nearest-first by hops;
      conservative (clean-pairs only, else unchanged). **#4 compare visited-both**: compare predicate
      requires each candidate's checkpoint in the visit trace before granting (defensive `[unverified]`
      if unresolved).
- [ ] **Deferred (own step, 6.4 prep)**: decomposer store-index (enhancement #2 — lets budget family
      fan out to N items + gives compare distinct SKUs; a prompt change needing its own A/B); grounded
      findings summary (#5). Noted, not built.

### Findings (2026-07-23)

1. **Typed decomposer A/B — 11/11 clean, all sequences sane.** The anti-keyword paraphrases that
   defeat the old guard ("...deposit them at the till", "pay", "checked out", "fetch...leave it at
   the self-checkout") ALL produced `pickup → checkout` with checkout as ONE leg — never a goto+place
   pair. Family 3 produced `goto → compare → pickup → checkout` with a criterion. Two honest limits to
   carry into 6.4: (a) **family 4 (budget) decomposes to a SINGLE item**, not an accumulate-to-budget
   loop — the budget rides in the text but the plan doesn't fan out to N items; (b) **compare
   `targets` are sometimes non-distinct** (`["Pik Nik","Pik Nik"]`, or collapsed to one string) when
   the prompt doesn't name the two variants — the choice check still works (token overlap), but 6.4's
   audit should watch it.
2. **goto predicate feed — NOW WIRED (was the open item; closed by the restructure).** `run_leg`
   localises `state['nearest_checkpoint']` each step from the live pose + the map, and `plan_legs`
   resolves `target_checkpoint` at leg start — so a `goto` leg is checked against real checkpoints,
   not granted `[unverified]`. (The `[unverified]` path survives only as the honest fallback when a
   target genuinely doesn't resolve.)
3. **New files**: `subtask_completion.py` + `subtask_planning.py` (light modules),
   `plan6/test_files/test_completion_predicates.py`, `test_subtask_planning.py`, `decompose_battery.json`,
   `ab_decompose.py`. **Edited**: `subtask_agents.py` (typed decompose, `run_leg` restructure, plan/order,
   predicate halt-gate + cap, checkout dispatch, map feed, eval_pickup-shaped `main`), `agent.py`
   (`_checkout_held_item`, `seed_nav_candidates`, `_graph_navigate` seed-reuse), `actions_str.py`
   (advertise checkout), `eval_pickup.py` (`name_matches` re-homed — otherwise FROZEN).
4. **Restructure + map planning — offline-validated (2026-07-23).** `run_leg` is `run_one`
   generalised (per-leg caps, JSONL, error tolerance, predicate stop, persistent memory, map-aware);
   `--arm graph` gives the orchestrator the measured-better navigator (long-horizon tasks multiply nav
   legs). Live check on the gate task: decompose → `plan_legs` resolves `pickup(green Piattos)` to
   `[26,45,52]` in ONE resolver call, `checkout` resolves nothing (drives itself), `order_legs` no-op
   (non-pair structure), all feasible. Two A/B limits from finding 1 still stand and are now the honest
   frontier for the map work: **family-4 budget → single item** (enhancement #2, deferred, would let it
   fan out) and **compare non-distinct `targets`** (which weakens #4's `candidate_sets` resolution when
   the two variants aren't named — #2 would fix this too). Both are 6.4-prep, not gate blockers.
5. **First live pickup leg — resolver+graph WORK; two leg-boundary leaks found + fixed (2026-07-23).**
   A live run got the full pickup leg right end-to-end: plan-seeded cp45 (after cp45 path-blocked on an
   earlier run, the graph correctly cycled candidates), center → measured MOVE (slant 1.13 > 0.85) →
   close → re-center → **grip on the real SKU** `JACK_AND_JILL_PIATTOS_..._40G`. But at the grip it did
   NOT stop: (a) it kept going after the pickup goal was met — the semantic learner had written leg-2's
   *"transport to the self-checkout"* intent into PERSISTENT memory, which the router then recalled and
   pursued ("the task" ambiguity: mission vs leg); and (b) it emitted `checkout_held_item` ON THE PICKUP
   LEG, and the old mode-gate replied *"route to manipulation first"* — actively steering it to run the
   whole checkout early (which would empty the hand, then the pickup predicate refuses STOP forever →
   cap). **Fixes (all built, offline-verified):**
   - **Leg-scoped tool gate** — `checkout_held_item` is dispatchable ONLY on a `checkout` leg
     (`dispatch_action(leg_type=...)`, checked BEFORE the mode-gate); a wrong-leg emit is redirected to
     STOP, not to manipulation. Advertisement text + `MANIPULATION_ACTIONS` updated to say "only during
     a checkout subtask".
   - **Measured STOP nudge** — `run_leg` runs the completion predicate SILENTLY each step; when the
     CURRENT GOAL measurably holds it injects `state['goal_check']` (a first-class state field, doc'd in
     `sys_inst`) telling the router the leg is done → emit STOP. The agent still chooses (not an
     auto-end — preserves the stop-condition measurement).
   - **Completion backstop** — the symmetric twin of the refusal cap: goal met for `COMPLETION_BACKSTOP
     = 3` consecutive steps without a STOP ends the leg `success=True`, `end_reason="completed_no_stop"`
     (counted, so 6.4 reports self-stop vs backstop as the completion-detection health signal).
   - **Prompt sentence** — the `CURRENT GOAL` line now says future goals belong to a different agent;
     this one is leg-task text (not the decomposer), so its real test is the live gate, not an offline
     battery.
6. **False STOP refusal on a real grip — hovered clears after retract; FIXED with a durable
   `gripped_name` (measured live 2026-07-23).** The next run gripped the right SKU but the pickup
   halt was refused: *"gripping, but the held item ('null'/'null') does not match target"*. Cause: the
   predicate matched against the LIVE `<side>HoveredObject` fields, which clear to 'null' once the
   hand retracts from the shelf — eval_pickup's own comment documents this false-negative (it dodged
   it by checking at the instant of grip; our halt comes steps later). Fix: `run_leg` captures the
   grab tool's reported name AT the grip into `state['gripped_name']` (sticky while a hand grips,
   cleared the moment nothing grips so a stale name can never vouch for an empty hand);
   `name_matches` now includes it in the match blob (empty for eval_pickup — backward-safe). The
   check is not loosened: a remembered WRONG item is still refused, and the refusal now names what is
   actually held. Regression-tested offline (the exact 'null'/'null' scenario + the wrong-item case;
   35/35).
7. **Consistency audit of subtask_agents.py + cross-check vs overhaul/slamtest/sim (2026-07-23) — 7
   issues found, all fixed; 68/68 offline after.** The load-bearing one: **checkout legs were UNSEEDED**,
   so a navigation-mode step on a checkout leg would make the runtime resolver re-resolve the leg's
   augmented text — which NAMES the carried product — and could drive the carrying agent BACK to the
   shelf it just left. Fix: `plan_legs` seeds checkout legs with the counter checkpoint (verified [54]
   on the real map, zero LLM). Also fixed: stale `goal_check`/backstop streak surviving a halt refusal
   (agent could see a refusal and a "you're done" nudge simultaneously); the resolver `llm_calls`
   count (step-1 heuristic → exact `_nav_task`-transition detection); the leg's STARTING checkpoint
   missing from the visit trace (a compare leg starting at a candidate got no credit); dead code
   (`reset_hands_in_front2` import, `EXTRACTABLE_JSON`, `sys`); four stale `run_subtask` docstring refs;
   and `sys_inst` state-doc entries r/s/t for `last_halt_refused` / `last_checkout` /
   `nearest_checkpoint` (the agent saw these fields raw with no explanation of how to react). Cross-check
   found nothing missing: perception exports all six functions the checkout macro imports; store_map has
   the macro + align; the sim side needs nothing new beyond the two already-filed asks
   (`RequestCheckoutState`, item-world-position). One honest note: `_fresh_agent_state` is shared with
   eval_pickup, so the new state fields (all `None`/inert there) do appear in the frozen baseline's
   prompt surface — its loop and metrics are untouched, but a future strict A/B rerun should know the
   state dict grew.
8. **`return_to_start` made opt-in (default OFF) — user request (2026-07-23).** It returned to the
   spawn pose before every run, which is awkward + inefficient for interactive use and adds no
   capability (it's eval-reproducibility machinery: makes a batteried run start each task identically;
   pose-only, never `env.Reset()`). Now `reset_start=False` by default; CLI flag flipped from
   `--no-reset` to `--reset-start` (opt-in). 6.4's `eval_longhorizon` will pass `reset_start=True` for
   comparable metrics. Bonus: leg ordering now uses the agent's ACTUAL current pose
   (`_current_nearest_cp`, a zero-delta pose read) instead of an assumed spawn corner, so multi-fetch
   ordering is correct whether or not the reset ran.
9. **Saved logging images capped at 1080p; functional images stay native — user request
   (2026-07-23).** The rule: on-disk debug frames are capped at 1920×1080; anything a model/algorithm
   consumes (VLM input, OCR crop, verifier frame, depth map, zoom tile) stays native. Most of this
   already existed — `env.downscale_for_storage` + `MAX_SAVE 1920×1080`, and env's `save_image=True`
   path already caps on disk while returning full-res bytes (so every `RequestScreenshot(save_image=
   True)` and `capture_walk` was already handled; depth maps deliberately excluded). Newly routed
   through it: the harness step frames (`subtask_agents.run_leg`, `eval_pickup.run_one` — the VLM gets
   the NATIVE bytes via base64; only the `stepNN.png` on disk is capped) and perception's two
   debug-only annotated frames (`annotated_target.png`, `_draw_debug_frame`) via a new PIL sibling
   `env.downscale_pil_for_storage`. Left native by design: `store_map.NavSession.screenshot`
   (locate/verify VLM input), `locate_task` zoom tiles, OCR crops, depth. The probe crosshair frames
   (`reach_probe`/`place_probe`) already comply transitively — they load from the already-capped
   `ClientScreenshot.png`. Verified: 4K→1920×1080 (aspect kept), ≤1080p passes byte-for-byte, native
   return preserved; offline suite green.

### Gate 6.3 — one supervised end-to-end run 🎮

> `python subtask_agents.py "get the green Piattos and bring it to the checkout counter"` (defaults to
> `--arm graph`) watched live: sane typed decomposition (a `pickup` leg + a `checkout` leg, no invented
> locations — CONFIRMED offline, resolves the pickup to `[26,45,52]`), pickup halt refused until a real
> grip on a name-matching item, checkout halt refused until the macro reports scanned AND placed,
> findings summary between legs, refusal cap never silently fires; `summary.json` + per-leg JSONL land
> in the run-dir (`overhaul/subtask_run_outputs/<ts>_graph/`).
> **Pass = one clean run + honest notes on every intervention.**
>
> **Result: PENDING** — needs the sim in Play mode + a human watching. All code built and
> offline-verified (66/66); this is the only remaining 6.3 step.

**Result: PENDING** — needs the sim in Play mode + a human watching. All code is built and
offline-verified; this is the only remaining 6.3 step.

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
