"""Execution loop for one typed subtask leg."""

import base64
import itertools
import json
import os
import re
import time
from datetime import datetime

from sim.env import (
    _GRIP_LEFT_,
    _GRIP_RIGHT_,
    _REQUEST_SCREENSHOT_,
    TransformAgent,
    TransformHands,
    downscale_for_storage,
)
from orchestrator.action_dispatch import (
    _GRAB_ACTIONS,
    _INSPECT_MACRO_ACTIONS,
    _INSPECT_MOVE_BUDGET_STEPS,
    _MACRO_ACTIONS,
    _grab_ready,
    dispatch_action,
    parse_actor_response,
)
from orchestrator.held_item_inspection import (
    _inspection_action_batch,
    _inspection_macro_summary,
)
from orchestrator.pickup_vlm_guard import (
    cache_compare_candidate_frames,
    evaluate_hands,
    make_compare_guard,
    make_inspect_guard,
    make_unknown_guard,
)
from orchestrator.subtask_completion import (
    COMPLETION_BACKSTOP,
    HALT_REFUSAL_CAP,
    WRONG_ITEM_RELEASE_AFTER,
    blob_matches_target,
    completion_predicate,
    held_item_inspection_active,
    inspect_scope_violation,
    mismatched_hands,
    pickup_has_target,
    reported_completion_answer,
)

def _fresh_agent_state() -> dict:
    """Read simulator state and initialize the per-leg execution state record."""
    agent_pos = TransformAgent((0, 0, 0), (0, 0, 0))
    hands_pos = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    state = {
        "translation": (0, 0, 0),
        "rotation": (0, 0, 0),
        "isColliding": False,
        "leftTranslation": (0, 0, 0),
        "leftRotation": (0, 0, 0),
        "rightTranslation": (0, 0, 0),
        "rightRotation": (0, 0, 0),
        "leftHoveredObject": "None",
        "leftGrippedState": False,
        "rightHoveredObject": "None",
        "rightGrippedState": False,
        "last_grab_failed": False,
        "last_action_blocked": False,
        "last_center": None,
        # 6.3 completion-predicate channels (mirror the last_reach/last_scan pattern):
        "last_checkout": None,       # the checkout macro's {scanned, placed, ...} verdict (checkout predicate)
        "last_inspection": None,     # restricted held-item inspection macro summary
        "last_halt_refused": None,   # the reason the most recent STOP was refused (surfaced to the actor)
        "nearest_checkpoint": None,  # nav's nearest checkpoint id (goto predicate); None until wired live
        "goal_check": None,          # measured STOP nudge: set when the CURRENT GOAL predicate would grant
        "gripped_name": None,        # the name the grab tool reported AT the grip (hovered clears after)
        "mode": "perception",
    }
    for k, v in {**agent_pos, **hands_pos}.items():
        state[k] = v
    return state


# State keys that ONLY code consumes - never rendered into the LLM prompt (execute_lean stringifies
# the whole state dict into all 3 calls, so anything here would cost input tokens on every step for no
# benefit). `visited_checkpoints` is the worst offender: a set the compare predicate reads in code,
# and it GROWS every step, so leaving it in makes each step progressively slower.
_MODEL_STATE_DROP = {"visited_checkpoints"}


def _model_facing_state(state: dict) -> dict:
    """The state the LLM sees: the full state MINUS fields only code reads (`_MODEL_STATE_DROP`), and
    with `last_checkout` trimmed to the verdict the agent acts on (scanned/placed/aligned/reason) - its
    `steps` sub-dict is per-primitive logging the model never uses. The completion PREDICATE always
    gets the FULL `state`, so nothing here weakens a halt check; this only shrinks the prompt."""
    view = {k: v for k, v in state.items() if k not in _MODEL_STATE_DROP}
    lc = view.get("last_checkout")
    if isinstance(lc, dict) and "steps" in lc:
        view["last_checkout"] = {k: v for k, v in lc.items() if k != "steps"}
    li = view.get("last_inspection")
    if isinstance(li, dict):
        # `frame_b64` MUST be dropped here, not merely trimmed for size: it is a full-resolution
        # base64 screenshot, and the actor already sees that frame as its own image input.
        view["last_inspection"] = _inspection_macro_summary(li)
    return view


def write_step_output(out_dir, step, response, stamp=""):
    """Dump a step's FULL agent output to out_dir/step<NN><STAMP>.txt (untruncated, unlike the JSONL
    fields) so it pairs with the step screenshot for debugging: the mode router's decision, the
    VLM actor's output, the episodic reflection (what_worked / what_to_avoid), and any nav note.
    `stamp` is the caller's per-step `_MMDD_HHMMSS` timestamp, woven into the name so the dump sorts
    by step index yet still records WHEN it ran (same stamp as its .png / _center dir). No-op if
    out_dir is falsy."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"step{step:02d}{stamp}.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"=== STEP {step} | mode={response.get('agent_mode')} "
                 f"| halt={response.get('halt')} ===\n\n")
        if response.get("nav_note"):
            fh.write(f"--- NAV NOTE ---\n{response['nav_note']}\n\n")
        fh.write(f"--- MODE ROUTER (semantic) ---\n{response.get('semantic') or '(n/a)'}\n\n")
        fh.write(f"--- VLM ACTOR OUTPUT ---\n{response.get('text') or ''}\n\n")
        fh.write(f"--- EPISODIC REFLECTION ---\n{response.get('episodic') or '(n/a)'}\n")


# Location gate (2026-07-24) — the graph, not the VLM, owns "am I at the target yet". A checkpoint
# counts as "at the target" if it is one of the leg's resolved candidates OR within this many graph
# hops of one. The graph navigator parks the agent ON a candidate, so this margin only tolerates the
# fine-approach creep nudging localisation to an immediate neighbour (node spacing 2.0 m, reading
# distance 1.0 m). MEASURED basis: in run 0724_145735 the agent that hallucinated its target onto a
# just-checked-out item sat 8–12 hops from every candidate, so any small margin fires the gate; 1
# keeps it from fighting a legitimate arrival.
_AT_TARGET_HOP_MARGIN = 1

# Only legs whose in-place work REQUIRES standing at the target's resolved checkpoints are gated:
# pickup (centre/grab must happen at the shelf) and goto (arrival IS the goal). checkout drives to
# the counter via its own macro; compare sweeps its candidate sets — neither is force-navigated here.
_LOCATION_GATED_TYPES = {"pickup", "goto"}


def _deterministic_guard_details(leg, state):
    """Per-hand diagnostic detail for deterministic targeted-pickup evaluations."""
    if leg.get("type") != "pickup" or not pickup_has_target(leg):
        return None
    names = state.get("gripped_names")
    names = names if isinstance(names, dict) else {}
    out = {}
    for side in ("left", "right"):
        if not state.get(f"{side}GrippedState"):
            continue
        sku = names.get(side)
        if not sku:
            hovered = state.get(f"{side}HoveredObject")
            sku = hovered if hovered and str(hovered).lower() not in ("none", "null") else None
        match = bool(sku and blob_matches_target(sku, leg.get("target") or ""))
        out[side] = {
            "match": match,
            "reason": ("deterministic SKU/target match" if match else
                       "deterministic SKU/target mismatch or unidentified held item"),
            "conclusive": bool(sku),
            "latency_ms": 0.0,
            "sku": sku,
            "reused": False,
        }
    return out


def _off_target(sm, leg, near_cp) -> bool:
    """True iff this leg's work must happen AT its resolved candidate checkpoints and the agent is not
    there yet — more than `_AT_TARGET_HOP_MARGIN` hops from EVERY candidate. Lets the graph, not the
    mode-router VLM, own "am I in the right place": the fix for the router hallucinating the target
    onto whatever is in front of it (e.g. a product left on the checkout counter after a checkout leg)
    and centre-grabbing in place forever instead of navigating to the shelf (run 0724_145735, leg 3).

    Returns False (gate OFF) whenever it cannot ground the check — wrong leg type, no resolved
    candidates, no localisation, or a disconnected graph — never a silent force-navigate the wiring
    can't justify."""
    if leg.get("type") not in _LOCATION_GATED_TYPES:
        return False
    cands = [c for c in (leg.get("candidates") or []) if c in sm.by_id]
    if not cands or near_cp is None:
        return False
    dists = [h for h in (sm.hops(near_cp, c) for c in cands) if h is not None]
    if not dists:
        return False
    return min(dists) > _AT_TARGET_HOP_MARGIN


def _run_leg_impl(agent, leg, sm, caps, log_path=None, context="", future_legs=None,
                  visited=None, leg_idx=0, completion_guard="deterministic", carried_names=None):
    """Run ONE typed subtask leg as a self-contained embodied-agent loop - pickup_navigation.run_one
    generalised for a leg of a long-horizon task (see the module docstring for the three differences).

    `leg` is a TYPED dict ({"type", "text", ...(+plan-resolved candidates)}); a bare string degrades to
    an `unknown` leg. `caps` = (max_steps, max_minutes) PER LEG; either cap set to 0 means NO LIMIT for
    that dimension (an unbounded step loop / no wall-clock check). `sm` is the StoreMap (localisation +
    the leg's seeded candidates). `visited` is the TASK-level checkpoint set the orchestrator threads
    through every leg (the compare predicate's visit trace); this leg keeps growing it.

    Returns a metrics dict (mirroring run_one's, plus halts_refused / halt_forced / t_checkout and a
    per-leg `end_reason` in {halt_granted, halt_forced, step_cap, time_cap, errors}) that ALSO carries
    `final_state` and `new_semantic_entries` for the orchestrator's findings summary + abort decision."""
    leg = leg if isinstance(leg, dict) else {"type": "unknown", "text": str(leg)}
    leg_text = leg.get("text") or ""
    future_legs = future_legs or []
    visited = visited if visited is not None else set()
    max_steps, max_minutes = caps

    parts = [f"CURRENT GOAL: {leg_text}\n"
             "(This is ONE subtask of a larger task. When THIS goal's end state holds, emit STOP to "
             "hand off — the FUTURE GOALS below belong to a different agent, not you. Do not pursue "
             "them, and do not check out an item unless THIS goal is the checkout.)"]
    parts.append(f"CONTEXT FROM PREVIOUS SUBTASKS:\n{context}" if context
                 else "CONTEXT FROM PREVIOUS SUBTASKS: None — this is the first subtask.")
    if leg.get("type") == "inspect":
        parts.append(
            "INSPECTING FOR EXPIRATION/INGREDIENTS: the expiration date is printed somewhere on "
            "the item's surface, format DD/MM/YY. If an item is already held, present and safely "
            "reposition that held item until the requested expiration date, nutritional facts, or "
            "ingredient label DIRECTLY FACES THE CAMERA and is clearly legible in the CURRENT screen. "
            "A glimpse at an angle is not enough: do not emit STOP until the printed surface is "
            "front-facing. Never grab, release, or check out during this inspect leg. If no item is "
            "held, use camera pan/tilt and visual centering, plus - only if the target is not "
            "visible from where you stand - a few small steps toward it (move_forward/backward/"
            "left/right). That stepping allowance is about 2 metres for the WHOLE subtask and is "
            "not a substitute for travelling: you were already routed here. If the target is still "
            "not visible once the allowance is spent, it is NOT at this location - emit STOP and say "
            "exactly that in `reported_answer` (for example 'no Choco Mallows are on this shelf'). "
            "Reporting a definite absence is a valid answer; spinning in place is not."
        )
    if future_legs:
        numbered = "\n".join(f"  {i+1}. {s.get('text') if isinstance(s, dict) else s}"
                             for i, s in enumerate(future_legs))
        parts.append("FUTURE GOALS (for awareness only — do NOT pursue these yet; record any "
                     "observations that would help future agents accomplish them):\n" + numbered)
    else:
        parts.append("FUTURE GOALS: None — this is the final subtask.")
    augmented_task = "\n\n".join(parts)

    print(f"\n[LEG {leg_idx}] ({leg.get('type')}) {leg_text}")
    if context:
        print(f"[CONTEXT] {context}")

    # Per-leg agent setup: reset CONVERSATION history only (semantic + episodic persist across legs -
    # the orchestrator's shared-memory contract), and seed the graph navigator with THIS leg's
    # plan-time candidates so it does not re-resolve at runtime (6.3 #1). NOTE: no return_to_start here
    # - that is the orchestrator's per-TASK job; calling it mid-task stows the hands and drops a carry.
    semantic_before = agent.begin_leg(
        leg.get("candidates"), leg.get("target_name"), leg_idx
    )

    # Logging + per-step screenshots: ONE dir per leg (replaces the old SIM_RUNS2/SIM_RUNS3 split).
    shots_dir = os.path.splitext(log_path)[0] if log_path else None
    if shots_dir:
        os.makedirs(shots_dir, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8") if log_path else None
    t0 = time.time()

    def log(rec):
        """Append one crash-safe, timestamped event to this leg's JSONL log."""
        if log_fh:
            rec["wall"] = round(time.time() - t0, 1)
            log_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            log_fh.flush()   # crash-proof: a dead run still leaves its trail

    log({"event": "leg_start", "leg": leg_idx, "type": leg.get("type"), "text": leg_text,
         "candidates": leg.get("candidates"), "arm": agent.nav_mode,
         "completion_guard": completion_guard,
         "ts": datetime.now().isoformat(timespec="seconds")})

    m = {"type": leg.get("type"), "text": leg_text, "t_manip": None, "t_grip": None,
         "t_checkout": None, "success": False, "timesteps": 0, "llm_calls": 0, "errors": 0,
         "completion_guard": completion_guard,
         "halts_refused": 0, "halt_forced": False, "corrective_release": None, "end_reason": None,
         "completion_evidence": None, "reported_answer": None}

    state = _fresh_agent_state()
    # The leg's STARTING checkpoint counts as visited (the post-action refresh only records positions
    # AFTER a step, so without this a compare leg that starts at a candidate gets no credit for it).
    # Also store it as `nearest_checkpoint` so STEP 1 already has a localisation — the per-step refresh
    # below only fills that field from step 2 on, but the location gate must judge off-target on the
    # very first step, which is exactly when the mode router latches onto whatever is in front of it.
    start_near = sm.nearest_checkpoint((state["translation"][0], state["translation"][2]))
    visited.add(start_near)
    state["nearest_checkpoint"] = start_near
    halt_refusals = 0
    corrective_release_done = False   # one wrong-item auto-release allowance per leg (see the refusal branch)
    goal_met_streak = 0        # consecutive steps the completion predicate would grant (backstop)
    last_actor_text = ""       # actor's last REAL output (compare predicate's choice check)
    # SKU the grab tool reported AT the grip, PER HAND - the durable record the pickup predicate
    # matches on (live hovered clears to 'null' once the hand retracts from the shelf; measured live
    # 2026-07-23: a held Piattos read 'null'/'null' at the halt step and the STOP was wrongly
    # refused). Per-hand (dual-hand 2026-07-23): one slot per side, so carrying an item in each hand
    # keeps BOTH names - the old single slot lost the first item's name at the second grab.
    # Seeded from the PREVIOUS leg's names (2026-07-29): a hand that grabbed in an earlier leg and is
    # STILL gripping now carries its name forward - without this, a leg boundary alone wiped the name
    # to None while the sim kept gripping (measured: VLM guard then skipped that hand entirely -
    # "no valid VLM guard verdict" - for an item that was visibly, verifiably held; TransformHands
    # cannot recover the identity after the fact since *HoveredObject clears once retracted from the
    # shelf, confirmed live 2026-07-29). Only seeded for hands gripping RIGHT NOW - a name from a hand
    # that has since released must never leak onto a new, different grip.
    _carried = carried_names if isinstance(carried_names, dict) else {}
    gripped_names = {
        side: (_carried.get(side) if state.get(f"{side}GrippedState") else None)
        for side in ("left", "right")
    }
    # Dual-hand (2026-07-23): sides already gripping at LEG START. A pickup leg must produce a NEW
    # grip - without this, an item carried in from a previous leg would satisfy an untargeted pickup
    # predicate the moment the leg begins (any-hand _gripping is True the whole time).
    start_grips = {side for side in ("left", "right") if state.get(f"{side}GrippedState")}
    state["gripped_names"] = dict(gripped_names)
    state["gripped_name"] = None
    last_guard_skus = None
    step_guard_verdicts = None
    compare_frames = {}
    compare_guard = None
    last_inspection_result = None
    # Per-leg inspection evidence ledger, keyed by hand (2026-07-29). One entry per successful
    # inspect_held_item run: the frame whose visibility gate passed, plus the SKU it showed. A
    # multi-item inspect ("which of these two has less sugar") reads one label per turn, so the
    # completion guard must judge the accumulated frames TOGETHER - a single frame can never show
    # both labels front-facing, and the old one-frame guard therefore refused every such STOP.
    inspection_evidence = {}
    # Remaining 0.1 m steps of body repositioning for an UNHELD inspect leg (see
    # _INSPECT_MOVE_BUDGET_STEPS). Non-inspect legs keep 0, which leaves the dispatch gate inert.
    inspect_move_left = _INSPECT_MOVE_BUDGET_STEPS if leg.get("type") == "inspect" else 0
    targeted_vlm_pickup = (
        completion_guard == "vlm"
        and leg.get("type") == "pickup"
        and pickup_has_target(leg)
    )
    targeted_vlm_compare = (
        completion_guard == "vlm"
        and leg.get("type") == "compare"
        and bool(leg.get("targets"))
    )
    targeted_vlm_unknown = (
        completion_guard == "vlm"
        and leg.get("type") not in (
            "pickup", "checkout", "compare", "goto", "inspect")
    )

    # A cap of 0 means NO LIMIT (now the default): max_steps 0 -> an unbounded step loop
    # (itertools.count), max_minutes 0 -> the wall-clock check is skipped. So a leg ends only on a
    # real terminal reason (halt_granted / halt_forced / errors / completed_no_stop). A positive cap
    # restores the old bounded behaviour, including the `step_cap` end_reason after the loop.
    step_iter = itertools.count(1) if not max_steps else range(1, max_steps + 1)
    for step in step_iter:
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            m["end_reason"] = "time_cap"
            break
        m["timesteps"] = step
        # Per-step timestamp woven into every artifact name (the screenshot, the step<NN>.txt dump,
        # the center-debug dir) so a step's files record WHEN it ran, not just its index — handy
        # across retries and long unbounded runs. `_` prefix keeps step<NN> as the leading sort key.
        step_stamp = f"_{datetime.now():%m%d_%H%M%S}"

        img_bytes = _REQUEST_SCREENSHOT_()["image"]
        if shots_dir:
            # Cap the SAVED debug frame at 1080p; the VLM below gets the NATIVE bytes (imageb64).
            with open(os.path.join(shots_dir, f"step{step:02d}{step_stamp}.png"), "wb") as fh:
                fh.write(downscale_for_storage(img_bytes))
        imageb64 = base64.b64encode(img_bytes).decode("utf-8")
        state["visited_checkpoints"] = set(visited)   # compare predicate reads the task visit trace (code only)
        step_inspect_guard = None
        step_unknown_guard = None
        if leg.get("type") == "inspect":
            # This closure is bound to the exact frame passed to execute_lean below, PLUS every
            # earlier frame this leg proved a label in (the evidence ledger - a two-item comparison
            # is only verifiable across frames). Its cache makes identical STOP/backstop checks
            # within this step one VLM call, and inspection ignores --completion-guard because
            # inspection is mandatory-VLM in both modes.
            evidence_frames = [
                {"label": (f"{inspection_evidence[side]['sku'] or 'held item'} "
                           f"({side} hand, step {inspection_evidence[side]['step']})"),
                 "image_b64": inspection_evidence[side]["image_b64"]}
                for side in ("left", "right") if side in inspection_evidence
            ]

            def _log_inspect_verdict(query, auxiliary_context, verdict, reused):
                """Record one held-item inspection guard decision and its VLM cost."""
                if not reused:
                    m["llm_calls"] += 1
                row = verdict if isinstance(verdict, dict) else {}
                log({"event": "completion_guard", "step": step, "backend": "vlm",
                     "guard": "inspect", "match": row.get("match"),
                     "reason": row.get("reason"), "conclusive": row.get("conclusive"),
                     "latency_ms": row.get("latency_ms"), "query": query,
                     "auxiliary_context": auxiliary_context, "reused": reused,
                     "evidence_frames": [frame["label"] for frame in evidence_frames]})

            step_inspect_guard = make_inspect_guard(
                agent.vlm_agent.client,
                agent.vlm_agent.config.model_id,
                agent.vlm_agent.config,
                imageb64,
                on_verdict=_log_inspect_verdict,
                evidence_frames=evidence_frames,
            )
        if targeted_vlm_unknown:
            def _log_unknown_verdict(task, auxiliary_context, verdict, reused):
                """Record one unknown-task completion guard decision and its VLM cost."""
                if not reused:
                    m["llm_calls"] += 1
                row = verdict if isinstance(verdict, dict) else {}
                log({"event": "completion_guard", "step": step, "backend": "vlm",
                     "guard": "unknown", "match": row.get("match"),
                     "reason": row.get("reason"), "conclusive": row.get("conclusive"),
                     "latency_ms": row.get("latency_ms"), "query": task,
                     "auxiliary_context": auxiliary_context, "reused": reused})

            step_unknown_guard = make_unknown_guard(
                agent.vlm_agent.client,
                agent.vlm_agent.config.model_id,
                agent.vlm_agent.config,
                imageb64,
                on_verdict=_log_unknown_verdict,
            )
        if targeted_vlm_compare and compare_guard is None:
            targets = list(leg.get("targets") or [])
            candidate_sets = leg.get("candidate_sets")
            if isinstance(candidate_sets, list) and len(candidate_sets) == len(targets):
                near = state.get("nearest_checkpoint")
                cache_compare_candidate_frames(
                    compare_frames, targets, candidate_sets, near, imageb64, step)
                if len(compare_frames) == len(targets) and len(targets) >= 2:
                    ordered_frames = [compare_frames[index] for index in range(len(targets))]

                    def _log_compare_verdict(criterion, auxiliary_context, verdict, reused):
                        """Record one comparison guard decision and the frames it evaluated."""
                        if not reused:
                            m["llm_calls"] += 1
                        row = verdict if isinstance(verdict, dict) else {}
                        log({"event": "completion_guard", "step": step, "backend": "vlm",
                             "guard": "compare", "match": row.get("match"),
                             "reason": row.get("reason"),
                             "conclusive": row.get("conclusive"),
                             "latency_ms": row.get("latency_ms"),
                             "criterion": criterion, "targets": targets,
                             "candidate_frames": [
                                 {"target": frame["target"],
                                  "checkpoint": frame["checkpoint"],
                                  "step": frame["step"]}
                                 for frame in ordered_frames
                             ],
                             "auxiliary_context": auxiliary_context, "reused": reused})

                    compare_guard = make_compare_guard(
                        agent.vlm_agent.client,
                        agent.vlm_agent.config.model_id,
                        agent.vlm_agent.config,
                        ordered_frames,
                        on_verdict=_log_compare_verdict,
                    )
        # VLM pickup completion guard: evaluate the CURRENT screenshot against the
        # held state produced by the preceding action. The direct client helper is stateless and does
        # not touch actor history. Untargeted/empty/unidentified grips make no call and fail closed.
        step_guard_verdicts = (_deterministic_guard_details(leg, state)
                               if completion_guard == "deterministic" else None)
        if targeted_vlm_pickup:
            held_skus = {
                side: gripped_names.get(side)
                for side in ("left", "right")
                if state.get(f"{side}GrippedState") and gripped_names.get(side)
            }
            guard_skus = tuple((side, held_skus.get(side)) for side in ("left", "right")
                               if held_skus.get(side))
            if guard_skus != last_guard_skus:
                goal_met_streak = 0
                last_guard_skus = guard_skus
            if held_skus:
                step_guard_verdicts, guard_calls = evaluate_hands(
                    agent.vlm_agent.client,
                    agent.vlm_agent.config.model_id,
                    agent.vlm_agent.config,
                    imageb64,
                    leg.get("target") or "",
                    held_skus,
                )
                m["llm_calls"] += guard_calls
                for side, verdict in step_guard_verdicts.items():
                    log({"event": "completion_guard", "step": step, "backend": "vlm",
                         "guard": "pickup",
                         "side": side, "sku": verdict.get("sku"),
                         "match": verdict.get("match"), "reason": verdict.get("reason"),
                         "conclusive": verdict.get("conclusive"),
                         "latency_ms": verdict.get("latency_ms"),
                         "reused": verdict.get("reused", False)})
            early_met, early_reason = completion_predicate(
                leg, state, final_text=last_actor_text, guard_backend=completion_guard,
                guard_verdicts=step_guard_verdicts)
            if early_met:
                goal_met_streak += 1
                state["goal_check"] = (
                    f"MEASURED: your CURRENT GOAL appears complete — {early_reason}. "
                    "Emit STOP to finish THIS subtask; a fresh agent handles any future goals. "
                    "Do NOT keep going.")
            else:
                goal_met_streak = 0
                state["goal_check"] = None
            if goal_met_streak >= COMPLETION_BACKSTOP:
                m["success"] = True
                m["end_reason"] = "completed_no_stop"
                print(f"[LEG {leg_idx} DONE] VLM completion backstop: positive guard held for "
                      f"{goal_met_streak} steps — ending leg (success). {early_reason}")
                log({"event": "completed_no_stop", "step": step, "streak": goal_met_streak,
                     "reason": early_reason, "backend": completion_guard,
                     "guard_verdicts": step_guard_verdicts})
                break
        # The LLM gets a LEAN view (drops code-only bookkeeping like the growing visit set); the FULL
        # `state` still backs every predicate call, so no halt check is weakened.
        # nav_goal = the BARE per-leg goal (pre-augmentation), handed to the advised navigator's per-hop
        # GOAL line so it isn't reading the cumulative context / future-goals blob when it only has to
        # pick the next hop or judge the product visible HERE (see execute_lean's nav_goal note). The
        # actor/learner still get the full augmented_task; only the graph-advised advisor narrows.
        # Location gate (2026-07-24): the graph, not the VLM, owns "am I at the target yet". Surface
        # the leg's resolved checkpoint(s) to the model (target_checkpoints), and flag force_navigate
        # whenever the agent is not there — the fix for the mode router hallucinating the target onto
        # whatever is in front of it (a just-checked-out item on the counter) and centre-grabbing in
        # place forever instead of driving to the shelf. execute_lean honours force_navigate by
        # overriding the mode to *navigation* (a candidate hop). Scoped to pickup/goto legs.
        gated_cps = ([c for c in (leg.get("candidates") or []) if c in sm.by_id]
                     if leg.get("type") in _LOCATION_GATED_TYPES else [])
        state["target_checkpoints"] = gated_cps or None
        off_target = _off_target(sm, leg, state.get("nearest_checkpoint"))
        # Held-item inspection is reevaluated from live grip state every timestep. It ends
        # immediately when both hands are empty or this runner moves to a different leg.
        force_manipulate = held_item_inspection_active(leg, state)
        inspect_mode = (
            ("held" if force_manipulate else "visual")
            if leg.get("type") == "inspect" else None
        )
        if off_target:
            print(f"[GATE] off-target: at cp{state.get('nearest_checkpoint')}, target at "
                  f"{gated_cps} — forcing navigation this step.")
        request = {"task": augmented_task, "nav_goal": leg_text, "force_navigate": off_target,
                   "force_manipulate": force_manipulate,
                   "inspect_mode": inspect_mode,
                   "image": imageb64, "state": _model_facing_state(state)}

        try:
            prev_nav_task = getattr(agent, "_nav_task", None)
            adv0 = getattr(agent, "_advised_llm_calls", 0)
            response = agent.execute_lean(request, step)
            m["llm_calls"] += 3  # semantic + VLM + episodic per execute_lean
            # graph-advised arm: the per-hop advisor VLM's calls, counted not hidden.
            m["llm_calls"] += getattr(agent, "_advised_llm_calls", 0) - adv0
            if (agent.nav_mode in ("graph", "graph-advised")
                    and getattr(agent, "_nav_task", None) != prev_nav_task
                    and not getattr(agent, "_nav_seeded", None)):
                # _graph_navigate initialised for this leg WITHOUT a plan-time seed -> the runtime
                # resolver LLM actually ran this step (counted exactly, not by the step-1 heuristic:
                # it fires at the first NAVIGATION-mode step, which can be any step or never).
                m["llm_calls"] += 1
        except Exception as e:  # noqa: BLE001 - one bad step shouldn't kill the leg (run_one parity)
            m["errors"] += 1
            print(f"    [leg {leg_idx} step {step}] execute_lean error: {type(e).__name__}: {e}")
            log({"event": "error", "step": step, "error": f"{type(e).__name__}: {e}"})
            if m["errors"] >= 3:
                m["end_reason"] = "errors"
                break
            continue

        if shots_dir:
            write_step_output(shots_dir, step, response, stamp=step_stamp)

        mode = response.get("agent_mode")
        if mode == "manipulation" and m["t_manip"] is None:
            m["t_manip"] = round(time.time() - t0, 1)

        # ---- STOP is a REQUEST: the typed predicate grants or refuses (6.3) --------------------
        if response.get("halt"):
            # Inspection consumes ONLY the structured answer emitted with this frame's STOP. Never
            # feed the termination placeholder or a previous action reply to the verifier.
            reported_answer = reported_completion_answer(response)
            if leg.get("type") == "inspect":
                final_text = reported_answer
            elif leg.get("type") == "compare" or targeted_vlm_unknown:
                final_text = reported_answer or last_actor_text or ""
            else:
                final_text = last_actor_text or response.get("text") or ""
            granted, reason = completion_predicate(
                leg, state, final_text=final_text, guard_backend=completion_guard,
                guard_verdicts=step_guard_verdicts, inspect_guard=step_inspect_guard,
                compare_guard=compare_guard, unknown_guard=step_unknown_guard)
            log({"event": "halt_request", "step": step, "granted": granted, "reason": reason,
                 "completion_guard": completion_guard, "guard_verdicts": step_guard_verdicts,
                 "reported_answer": (
                     final_text if leg.get("type") in ("inspect", "compare")
                     or targeted_vlm_unknown else None)})
            if granted:
                m["success"] = True
                m["end_reason"] = "halt_granted"
                m["completion_evidence"] = reason
                # This is safe to surface only after the same predicate that judges completion has
                # granted the STOP. Observation/inspection answers used to disappear at this point,
                # leaving the task-level responder nothing verified to tell the user.
                m["reported_answer"] = reported_answer or None
                print(f"[LEG {leg_idx} DONE] halt granted: {reason}")
                break
            halt_refusals += 1
            m["halts_refused"] = halt_refusals
            state["last_halt_refused"] = reason
            # A refusal means the goal does NOT hold right now - clear any stale "you're done" nudge
            # (set on an earlier step) and reset the backstop streak, so the agent never sees a
            # refusal and a completion nudge at the same time.
            state["goal_check"] = None
            goal_met_streak = 0
            print(f"[GUARD] STOP refused ({halt_refusals}/{HALT_REFUSAL_CAP}): {reason}")
            # ---- mid-leg SELF-CORRECTION (2026-07-23): drop a verifiably-wrong item ------------
            # A pickup STOP refused repeatedly while a hand holds the WRONG item (its grip-time name
            # fails the target match) means the agent is not acting on the refusal guidance - the
            # measured failure mode was spinning the remaining refusals into halt_forced and taking
            # the whole task down. Code releases the mismatched hand(s) - never an item carried in
            # from a previous leg (start_grips), never a hand with no recorded name - and resets the
            # refusal budget ONCE per leg so the grab gets a real second attempt. The predicate stays
            # the sole completion truth: nothing is granted here, the leg just keeps going.
            if (leg.get("type") == "pickup" and not corrective_release_done
                    and halt_refusals >= WRONG_ITEM_RELEASE_AFTER):
                released = []
                for side in mismatched_hands(
                        leg, state, start_grips, guard_backend=completion_guard,
                        guard_verdicts=step_guard_verdicts):
                    try:
                        (_GRIP_LEFT_ if side == "left" else _GRIP_RIGHT_)()  # toggle: gripping -> open
                        released.append(f"{side}:{gripped_names.get(side)}")
                    except Exception as e:  # noqa: BLE001 - a failed release just leaves the cap to fire
                        print(f"[CORRECT] release toggle failed on {side}: {type(e).__name__}: {e}")
                if released:
                    corrective_release_done = True
                    halt_refusals = 0
                    m["corrective_release"] = released
                    # Re-read the live grip state so the predicate/actor see the empty hand(s) now.
                    fresh = _fresh_agent_state()
                    for key in ("leftGrippedState", "rightGrippedState",
                                "leftHoveredObject", "rightHoveredObject"):
                        state[key] = fresh[key]
                    for side in ("left", "right"):
                        if not state.get(f"{side}GrippedState"):
                            gripped_names[side] = None
                    state["gripped_names"] = dict(gripped_names)
                    state["gripped_name"] = " ".join(n for n in gripped_names.values() if n) or None
                    state["new_grip_this_leg"] = any(
                        state.get(f"{s}GrippedState") and s not in start_grips
                        for s in ("left", "right"))
                    sides = "/".join(r.split(":", 1)[0] for r in released)
                    state["last_halt_refused"] = (
                        f"{reason} | SELF-CORRECTION: the wrong item was auto-released from your "
                        f"{sides} hand. Do NOT stop yet - find and grab the actual target "
                        f"({leg.get('target')!r}), then STOP.")
                    print(f"[CORRECT] auto-released wrong item(s) {released}; refusal budget reset.")
                    log({"event": "corrective_release", "step": step, "released": released})
                    continue
            if halt_refusals >= HALT_REFUSAL_CAP:
                # Escape hatch so the agent can't spin forever on STOP - NOT a grant. The leg ends
                # halt_forced with the reason in state; the orchestrator counts it as a non-clean leg.
                m["halt_forced"] = True
                m["end_reason"] = "halt_forced"
                print(f"[GUARD] refusal cap reached — force-ending leg (halt_forced): {reason}")
                break
            continue

        # ---- parse the actor's action JSON (a bad parse skips the step, run_one parity) --------
        # Tolerant of the single-quote-apostrophe break (see parse_actor_response): a value like
        # "Kellogg's" no longer wastes the step - actions/times are salvaged so it still executes.
        parsed = parse_actor_response(response.get("text") or "",
                                      agent.vlm_agent.extractable_json_structured_output)
        if parsed is None:
            m["errors"] += 1
            log({"event": "parse_error", "step": step, "raw": (response.get("text") or "")[:400]})
            if m["errors"] >= 3:
                m["end_reason"] = "errors"
                break
            continue
        last_actor_text = response.get("text") or ""
        notes = parsed.get("notes", {})

        center_dir = os.path.join(shots_dir, f"step{step:02d}{step_stamp}_center") if shots_dir else None
        acted, blocked_reason, center_msg, last_reach = [], False, None, None
        grab_failed, checkout_result, inspection_result = False, None, None
        action_batch = list(zip(parsed.get("actions", []), parsed.get("times", [])))
        if leg.get("type") == "inspect" and force_manipulate:
            action_batch = _inspection_action_batch(
                parsed.get("actions", []), parsed.get("times", []))
        for action, tt in action_batch:
            raw_action = action.strip()
            inline = None
            im = re.match(r'^(\w+)\([\'"]?(.*?)[\'"]?\)$', raw_action)
            if im:
                raw_action, inline = im.group(1), im.group(2)
            step_center = None
            if center_dir and raw_action == "center_object_on_screen":
                os.makedirs(center_dir, exist_ok=True)
                step_center = center_dir
            # Option 2 (2026-07-23, UNMEASURED - A/B before trusting): perception->manipulation grab
            # auto-promotion. The learner picks the step's MODE one call BEFORE the actor picks the
            # ACTION, off a multi-step `recall` plan; the actor, reading the same plan, routinely emits
            # the grab a step early. Today that grab is BLOCKED (the dispatch mode-gate), the step is
            # wasted, and the router only flips to manipulation the step after (the handshake edit's
            # self-correcting loop). When a MEASURED readiness signal holds (_grab_ready reads the
            # PREVIOUS step's last_center/last_reach on `state`), run the grab in *manipulation* NOW
            # instead of eating the blocked step. SCOPED on purpose: only the self-posing grab family
            # (_GRAB_ACTIONS drives GRAB->REST itself, so the REST pose execute_lean parked for this
            # perception step stays valid); NOT raw grip_*/hand nudges (they don't self-restore and would
            # desync the pose tracker) and NOT navigation (the graph arm exists to keep the VLM out of
            # open-ended nav - doctrine). Behaviour-neutral beyond saving the step: the grab does exactly
            # what it would next step (a FREE hand, or refuse if both hands are full).
            eff_mode, promoted = mode, False
            if raw_action in _GRAB_ACTIONS and mode not in (None, "manipulation") and _grab_ready(state):
                eff_mode, promoted = "manipulation", True
            scope_pre_state = state if leg.get("type") == "inspect" else None
            inspection_frames_dir = (
                os.path.join(shots_dir, f"step{step:02d}_inspection")
                if shots_dir and raw_action in _INSPECT_MACRO_ACTIONS else None
            )
            res = dispatch_action(raw_action, int(tt), notes, inline_arg=inline, mode=eff_mode,
                                  debug_dir=step_center, agent=agent, leg_type=leg.get("type"),
                                  state=state,
                                  inspection_query=(leg.get("query") or leg.get("text") or ""),
                                  inspection_log=(
                                      (lambda row: log({"step": step, **row}))
                                      if raw_action in _INSPECT_MACRO_ACTIONS else None
                                  ),
                                  inspection_frames_dir=inspection_frames_dir,
                                  inspect_move_allowance=inspect_move_left) or {}
            if res.get("inspect_move_steps"):
                inspect_move_left = max(0, inspect_move_left - int(res["inspect_move_steps"]))
                log({"event": "inspect_approach_step", "step": step, "action": raw_action,
                     "steps": int(res["inspect_move_steps"]), "budget_left": inspect_move_left})
            if scope_pre_state is not None:
                scope_event = inspect_scope_violation(raw_action, step, scope_pre_state, res)
                if scope_event:
                    log(scope_event)
            if promoted and not res.get("blocked"):
                if m["t_manip"] is None:
                    m["t_manip"] = round(time.time() - t0, 1)
                log({"event": "grab_promoted", "step": step, "action": raw_action,
                     "router_mode": mode, "gripped": res.get("gripped")})
            if res.get("blocked"):
                blocked_reason = res.get("reason", True)
            if res.get("center_message"):
                center_msg = res["center_message"]
            if res.get("last_reach"):
                last_reach = res["last_reach"]
            if raw_action in _MACRO_ACTIONS and not res.get("blocked"):
                checkout_result = res
            if raw_action in _INSPECT_MACRO_ACTIONS:
                inspection_result = res
                last_inspection_result = res
                m["llm_calls"] += int(res.get("vlm_calls") or 0)
                # File the winning frame under the hand it read. `label_visible` covers BOTH useful
                # outcomes - a legible read and a locked-but-illegible best-effort read - because the
                # actor is told to answer from either; a sweep that never found the label has no
                # frame worth replaying to the guard.
                ev_hand = res.get("hand")
                if (not res.get("blocked") and res.get("label_visible")
                        and res.get("frame_b64") and ev_hand in ("left", "right")):
                    inspection_evidence[ev_hand] = {
                        "hand": ev_hand,
                        "sku": gripped_names.get(ev_hand),
                        "step": step,
                        "label_legible": bool(res.get("label_legible")),
                        "best_effort_read": bool(res.get("best_effort_read")),
                        "image_b64": res["frame_b64"],
                    }
                    log({"event": "inspection_evidence_recorded", "step": step, "hand": ev_hand,
                         "sku": gripped_names.get(ev_hand),
                         "label_legible": bool(res.get("label_legible")),
                         "best_effort_read": bool(res.get("best_effort_read")),
                         "hands_covered": sorted(inspection_evidence)})
            if raw_action in _GRAB_ACTIONS and not res.get("blocked"):
                # blocked = wrong-mode, not a distance failure; a measured move/crouch/bail/recenter
                # carries its own recovery in last_reach - only a reachable-but-missed grab is a failure.
                verdict = res.get("reach_verdict")
                if verdict in (None, "reachable") and not res.get("gripped", False):
                    grab_failed = True
                if res.get("gripped") and res.get("hovered"):
                    # Capture the SKU AT the grip, filed under the hand that grabbed it - the durable
                    # name record (hovered clears later). The grab result names its hand since the
                    # dual-hand change; default left covers a hypothetical old-style result.
                    gripped_names[res.get("hand") or "left"] = res["hovered"]
            acted.append([raw_action, int(tt)])

        # ---- refresh state from the sim, localise on the map, grow the visit trace -------------
        state = _fresh_agent_state()
        state["mode"] = mode
        state["last_action_blocked"] = blocked_reason
        state["last_center"] = center_msg
        state["last_reach"] = last_reach
        state["last_grab_failed"] = grab_failed
        if checkout_result is not None:
            state["last_checkout"] = checkout_result   # sticky verdict the checkout predicate reads
            if m["t_checkout"] is None:
                m["t_checkout"] = round(time.time() - t0, 1)
        if last_inspection_result is not None:
            state["last_inspection"] = last_inspection_result
        # Evidence for a hand that has since let go (or swapped items) is no longer evidence about
        # what it holds now, so drop it before the predicate reads the ledger.
        for ev_hand in [side for side in inspection_evidence
                        if not state.get(f"{side}GrippedState")]:
            inspection_evidence.pop(ev_hand, None)
        # The predicate + the actor see the ledger WITHOUT the frames - just which held item was read
        # when. Only the next step's inspect-guard closure is handed the images themselves.
        state["inspection_evidence"] = [
            {k: v for k, v in inspection_evidence[side].items() if k != "image_b64"}
            for side in ("left", "right") if side in inspection_evidence
        ]
        # 6.3 #1: localise from the live pose the state refresh just fetched (no extra sim round-trip) +
        # the frozen map. Feeds the goto predicate and grows the task-level visit trace (compare).
        near = sm.nearest_checkpoint((state["translation"][0], state["translation"][2]))
        state["nearest_checkpoint"] = near
        visited.add(near)
        state["visited_checkpoints"] = set(visited)

        gripping_now = bool(state.get("leftGrippedState") or state.get("rightGrippedState"))
        if gripping_now and m["t_grip"] is None:
            m["t_grip"] = round(time.time() - t0, 1)
        # Sticky while THAT hand grips; cleared the moment it releases (checkout/drop), so a stale
        # name can never vouch for an empty hand - per hand, so the other hand's record survives.
        for side in ("left", "right"):
            if not state.get(f"{side}GrippedState"):
                gripped_names[side] = None
        state["gripped_names"] = dict(gripped_names)
        # Back-compat blob: name_matches folds this into its search text; joining both names means a
        # targeted pickup grants when EITHER held item is the target (the right item in hand is the
        # right item, whichever hand holds it).
        state["gripped_name"] = " ".join(n for n in gripped_names.values() if n) or None
        # True once a hand that was EMPTY at leg start grips - the pickup predicate's "this leg
        # actually grabbed something" signal under dual-hand carry.
        state["new_grip_this_leg"] = any(
            state.get(f"{side}GrippedState") and side not in start_grips
            for side in ("left", "right"))
        # True once a hand that was GRIPPING at leg start released - the untyped drop guard's "this
        # leg actually put something down" signal (a second carried item no longer stalls the STOP).
        state["released_grip_this_leg"] = any(
            side in start_grips and not state.get(f"{side}GrippedState")
            for side in ("left", "right"))

        # 6.3 completion nudge + backstop: run the SAME predicate the STOP request will face, SILENTLY,
        # each step. When the CURRENT GOAL measurably holds, put that in front of the router (goal_check)
        # so it stops THIS leg instead of drifting into future goals it recalled from persistent memory
        # (the leg-overrun failure). The agent still chooses STOP - this is a nudge, not an auto-end. But
        # if it stays satisfied for COMPLETION_BACKSTOP steps without ever proposing STOP, end the leg
        # anyway (success=True, the goal holds) - the symmetric twin of the refusal cap.
        if targeted_vlm_pickup:
            # The step guard describes the screenshot/held state from BEFORE this step's action.
            # Do not reuse it against a newly changed grip here; the next fresh screenshot evaluates
            # that state. Its result already drove this step's nudge/streak above.
            met, met_reason = early_met, early_reason
        elif leg.get("type") == "inspect":
            # Inspection has no valid answer until the actor emits structured `reported_answer` with
            # STOP. `last_actor_text` is action JSON and may merely say that a label needs rotation;
            # treating it as an answer produced false-positive goal_check nudges. Also, this step's
            # guard is bound to the screenshot from BEFORE the action above. Defer inspection
            # verification to a later STOP, whose guard is bound to that iteration's fresh frame.
            goal_met_streak = 0
            state["goal_check"] = None
            met = False
            met_reason = "inspection awaits a structured STOP answer on a fresh frame"
        else:
            predicate_started = time.monotonic()
            met, met_reason = completion_predicate(
                leg, state, final_text=last_actor_text, guard_backend=completion_guard,
                inspect_guard=step_inspect_guard, compare_guard=compare_guard,
                unknown_guard=step_unknown_guard)
            predicate_latency = round((time.monotonic() - predicate_started) * 1000, 1)
            step_guard_verdicts = _deterministic_guard_details(leg, state)
            for side, verdict in (step_guard_verdicts or {}).items():
                verdict["latency_ms"] = predicate_latency
                log({"event": "completion_guard", "step": step, "backend": "deterministic",
                     "side": side, "sku": verdict.get("sku"), "match": verdict.get("match"),
                     "reason": verdict.get("reason"), "conclusive": verdict.get("conclusive"),
                     "latency_ms": verdict.get("latency_ms"), "reused": False})
            if met:
                goal_met_streak += 1
                state["goal_check"] = (f"MEASURED: your CURRENT GOAL appears complete — {met_reason}. "
                                       "Emit STOP to finish THIS subtask; a fresh agent handles any "
                                       "future goals. Do NOT keep going.")
            else:
                goal_met_streak = 0
                state["goal_check"] = None

        log({"event": "step", "step": step, "mode": mode,
             "nav_note": (response.get("nav_note") or "")[:200] or None,
             "actions": acted, "blocked": blocked_reason or None, "center": center_msg,
             "reach": last_reach, "near_cp": near, "pos": state.get("translation"),
             "hovered": [state.get("leftHoveredObject"), state.get("rightHoveredObject")],
             "gripped": [state.get("leftGrippedState"), state.get("rightGrippedState")],
             "gripped_names": dict(gripped_names), "off_target": off_target or None,
             "checkout": checkout_result,
             "inspection": (
                 {k: v for k, v in inspection_result.items() if k != "steps"}
                 if isinstance(inspection_result, dict) else None),
             "goal_met": met, "status": notes.get("status")})

        if not targeted_vlm_pickup and goal_met_streak >= COMPLETION_BACKSTOP:
            m["success"] = True
            m["end_reason"] = "completed_no_stop"
            m["completion_evidence"] = met_reason
            print(f"[LEG {leg_idx} DONE] completion backstop: goal measurably held for "
                  f"{goal_met_streak} steps without a STOP — ending leg (success). {met_reason}")
            log({"event": "completed_no_stop", "step": step, "streak": goal_met_streak,
                 "reason": met_reason})
            break

    if m["end_reason"] is None:
        m["end_reason"] = "step_cap"
    m["wall_s"] = round(time.time() - t0, 1)
    m["final_state"] = state
    m["new_semantic_entries"] = agent.vlm_agent.semantic_log.since(semantic_before)
    log({"event": "leg_end", **{k: v for k, v in m.items() if k != "final_state"}})
    if log_fh:
        log_fh.close()
    return m


def run_leg(agent, leg, sm, caps, log_path=None, context="", future_legs=None,
            visited=None, leg_idx=0, completion_guard="deterministic", carried_names=None):
    """Run a leg and unconditionally restore canonical hand transforms after inspection.

    Cleanup deliberately lives outside the main loop so it covers granted STOP, completion
    backstops, caps, repeated errors, retries, and exceptions without changing their original result.

    `carried_names` (2026-07-29): the previous leg's ending `gripped_names`, so a hand still
    gripping across the leg boundary keeps its recorded SKU instead of losing it to the per-leg
    reset - see the seeding comment in `_run_leg_impl`.
    """
    typed_leg = leg if isinstance(leg, dict) else {"type": "unknown", "text": str(leg)}
    result = None
    try:
        result = _run_leg_impl(
            agent, leg, sm, caps, log_path=log_path, context=context,
            future_legs=future_legs, visited=visited, leg_idx=leg_idx,
            completion_guard=completion_guard, carried_names=carried_names)
        return result
    finally:
        if typed_leg.get("type") == "inspect":
            cleanup = None
            try:
                restore = getattr(agent, "restore_hands_after_inspection", None)
                restore = restore or getattr(agent, "_restore_hands_after_inspection")
                cleanup = restore()
                if not cleanup.get("restored"):
                    agent._hand_pose = None
                if result is not None and cleanup.get("restored"):
                    refreshed = _fresh_agent_state()
                    prior = result.get("final_state") or {}
                    prior.update(refreshed)
                    result["final_state"] = prior
            except Exception as cleanup_error:  # noqa: BLE001 - never mask the leg result/error
                try:
                    agent._hand_pose = None
                except Exception:
                    pass
                cleanup = {
                    "restored": False,
                    "error": f"{type(cleanup_error).__name__}: {cleanup_error}",
                }
                print(f"[WARN] inspect cleanup failed: {cleanup['error']}")
            if result is not None:
                result["inspection_cleanup"] = cleanup
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as cleanup_log:
                        cleanup_log.write(json.dumps({
                            "event": "inspect_cleanup",
                            "leg": leg_idx,
                            **(cleanup or {"restored": False}),
                        }, ensure_ascii=False, default=str) + "\n")
                except Exception as log_error:  # noqa: BLE001 - logging cannot mask the leg outcome
                    print(f"[WARN] could not log inspect cleanup: "
                          f"{type(log_error).__name__}: {log_error}")
