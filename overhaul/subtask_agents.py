"""
subtask_agents.py

Long-horizon orchestrator: decompose a big task into typed subtask LEGS, run each leg as a
self-contained embodied-agent loop (the eval_pickup.run_one harness, generalised), and let the AGENT
decide when a leg is done - a STOP that a code-side completion predicate grants or refuses (Phase 6.3),
not eval_pickup's automatic end-on-grip. Shared semantic/episodic memory carries across legs; a
findings summary is generated between legs and fed forward.

Relationship to eval_pickup.py (deliberate, documented):
  - `run_leg` is `eval_pickup.run_one` re-shaped for a LEG of a longer task: same caps / crash-proof
    JSONL / per-step screenshots / error tolerance / honest end_reason, but (1) the stop condition is
    the typed predicate, (2) semantic+episodic memory PERSIST across legs, (3) it is map-aware
    (plan-seeded candidates, nearest_checkpoint localised into state, a cumulative visit trace).
  - eval_pickup.py is left FROZEN - it is the measured Phase-4.2 A/B baseline; editing its loop would
    invalidate past-vs-future comparisons. The duplication of the harness pattern is the accepted cost
    (see the phase6.3 doc's decision 1). Its `TASKS` unpack bug is 6.4 pre-flight debt, fixed THERE.

Map awareness (Phase 6.3 #1/#3/#4):
  - plan_legs() resolves each pickup/goto/compare target ONCE at PLAN time (the map resolver), so the
    completion predicates get real checkpoints to check, a doomed plan is caught before burning sim
    steps, and the graph navigator reuses the candidates instead of re-resolving at runtime.
  - order_legs() reorders independent (pickup->checkout) pairs nearest-first by graph hops.
  - the per-leg visit trace (nearest checkpoint each step) feeds the goto/compare predicates.

Usage:
    python subtask_agents.py "get the green Piattos and bring it to the checkout counter"
    python subtask_agents.py --task "..." --arm graph --max-steps 150 --max-minutes 40
    python subtask_agents.py --task "..." --arm vlm      # control arm (old VLM navigation)
    python subtask_agents.py --task "..." --arm graph-advised  # per-hop advisor-VLM drive
    python subtask_agents.py --task "..." --reset-start   # eval-reproducibility: start from spawn

The agent starts from wherever it is by default; `--reset-start` is opt-in machinery for a batteried
eval (6.4), not needed for an interactive run.
"""

import argparse
import ast
import base64
import json
import os
import re
import time
import chime  # cross-platform run-completion beep (was winsound: Windows-only)
from datetime import datetime

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Repo-root api.env, resolved from __file__ so it loads regardless of CWD.
load_dotenv(Path(__file__).resolve().parent.parent / 'api.env')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from env import (
    _REQUEST_SCREENSHOT_,
    RequestLidarCenter,
    TransformAgent,
    TransformHands,
    downscale_for_storage,
    init_logger,
)
from manipulation import plan_reach
from actions import (
    NAVIGATION_ACTIONS_REF,
    PERCEPTION_ACTIONS_REF,
    MANIPULATION_ACTIONS_REF,
)
from agent import EmbodiedAgent, ucl_qwen_config
# Phase 6.3: the typed-subtask contract + deterministic completion predicates live in a lightweight,
# sim-free module (importing THIS module pulls the whole model stack) so they stay offline-unit-testable.
from subtask_completion import (
    TYPED_DECOMPOSER_SYSTEM,
    parse_decomposition,
    completion_predicate,
    HALT_REFUSAL_CAP,
    COMPLETION_BACKSTOP,
)
# Plan-time map planning (also sim-free, so it stays offline-unit-testable - test_subtask_planning.py).
from subtask_planning import (
    SPAWN_XZ,
    make_resolve_call,
    plan_legs,
    order_legs,
)

ORCHESTRATOR_MODEL = "Qwen/Qwen3.6-27B"  # UCL qwen (OpenRouter retired 2026-07-19)

# Every reasoner runs on the UCL qwen vLLM server (OpenRouter fully retired 2026-07-21).
# ucl_qwen_config carries the load-bearing enable_thinking=False + max_tokens cap for this
# server - see agent.ucl_qwen_config. Mirrors eval_pickup.py / env_simulation.py. The
# orchestrator LLM below (_llm_client) already targets the same server.
VLM_CONFIG = ucl_qwen_config(temperature=0.5)
ASSOCIATIVE_CONFIG = ucl_qwen_config(temperature=0.3)


# ---------------------------------------------------------------------------
# Orchestrator LLM helpers
# ---------------------------------------------------------------------------

def _llm_client() -> OpenAI:
    from agent import _ucl_creds
    host, key = _ucl_creds()
    return OpenAI(base_url=f"http://{host}:8000/v1", api_key=key)


def _llm_call(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=ORCHESTRATOR_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
        timeout=120,
    )
    return resp.choices[0].message.content


def decompose_task(client: OpenAI, task: str) -> list:
    """Phase 6.3: returns a list of TYPED subtask dicts ({"type", "text", ...}), not free strings, so
    each leg's completion is checked by a code-side predicate keyed on its type instead of by grepping
    its prose (the pre-6.3 keyword guards). The type vocabulary is closed (pickup|checkout|compare|
    goto); any untypeable element degrades to `{"type": "unknown"}` inside parse_decomposition, which
    run_leg then handles with the OLD keyword guards. The A/B that validated this prompt lives in
    plan6/test_files/ab_decompose.py (11/11 clean on the four-family battery, 2026-07-23)."""
    raw = _llm_call(client, TYPED_DECOMPOSER_SYSTEM, f"Task: {task}")
    subtasks = parse_decomposition(raw, task)
    if any(s.get("type") == "unknown" for s in subtasks):
        print("[WARN] Decomposition had untypeable element(s) — those legs fall back to keyword guards "
              "(logged as `untyped`).")
    return subtasks


def generate_findings_summary(
    client: OpenAI,
    completed_subtask: str,
    final_state: dict,
    new_semantic_entries: str,
) -> str:
    """
    Comprehensive summary of everything the agent found/learned during a subtask.
    Passed to the orchestrator so all future subtask agents receive accumulated context.
    """
    system = (
        "You are a findings reporter for an Embodied AI Agent in a 3D convenience "
        "store simulation. After a subtask completes, produce a comprehensive findings "
        "summary for future agent instances. Include ALL of the following:\n"
        "  1. POSITION: Current agent position in plain English (near which shelf/counter).\n"
        "  2. HANDS: What each hand is holding (gripped items, or empty).\n"
        "  3. OBJECTS LOCATED: Every object/item seen and its approximate shelf or position.\n"
        "  4. NAVIGATION INSIGHTS: Which paths/routes worked; where the agent got stuck or lost.\n"
        "  5. SEMANTIC LEARNINGS: Key facts about the store environment learned this subtask.\n"
        "  6. WHAT TO AVOID: Any approaches that failed or cost unnecessary time.\n"
        "  7. UPCOMING TASK PREP: Specific observations that will help with future subtasks.\n"
        "Be comprehensive and factual. Future agents cannot re-explore what you already found, "
        "so document every useful detail."
    )
    user = (
        f"Completed subtask: {completed_subtask}\n\n"
        f"Final agent state:\n{json.dumps(final_state, indent=2, default=str)}\n\n"
        f"New semantic memory entries learned during this subtask:\n{new_semantic_entries}"
    )
    return _llm_call(client, system, user)


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

def _last_reach_line(plan, gripped=None):
    """Human-readable `last_reach` string the actor/learner reads (AGENT_STATE_DOC p). Carries the
    measured Phase-D verdict + distance so the recovery is specific, not a guess."""
    v = plan["verdict"]
    if v == "reachable":
        if gripped:
            return f"REACHABLE and GRABBED - {plan['reason']}"
        return (f"REACHABLE by measure but the grab MISSED - {plan['reason']}; move a little closer, "
                f"re-center, and retry (the reach envelope may be slightly off)")
    if v == "move":
        return f"MOVE - {plan['reason']}; then re-center and retry the grab"
    if v == "crouch":
        return f"CROUCH (too low) - {plan['reason']}"
    if v == "bail":
        return f"UNREACHABLE (too high) - {plan['reason']}"
    if v == "recenter":
        return f"RE-CENTER - {plan['reason']}"
    return f"{v.upper()} - {plan.get('reason', '')}"


# Phase 6.3 macro actions: dispatched via the AGENT (they need the live NavSession), not via the
# free-function action refs. Gated to manipulation mode alongside the raw hand actions (mode
# coherence, 6.1) - a wrong-mode emit is blocked and the router flips to manipulation next step, the
# same self-correcting loop extend_arm_until_grabbed already relies on.
_MACRO_ACTIONS = {"checkout_held_item"}


def dispatch_action(action: str, time_units: int, notes: dict, inline_arg: str = None,
                    mode: str = None, debug_dir: str = None, agent=None, leg_type: str = None) -> dict:
    """Execute one action. Returns a result dict; grab actions include a 'gripped' key, and the
    checkout macro returns its {scanned, placed, ...} verdict (surfaced by run_leg as `last_checkout`).

    Two gates guard the checkout macro:
      - LEG-SCOPED (6.3): `checkout_held_item` belongs ONLY to a `checkout` leg. On any other leg
        (pickup/goto/compare) the agent must NOT run the whole checkout chain - the item hands off to
        the next (checkout) leg. A wrong-leg emit is redirected to STOP, not to manipulation, because
        the fix is to FINISH this leg, not to run checkout early. Checked FIRST so it wins over the
        mode-gate's 'route to manipulation' message (which was correct for grabs, harmful here).
      - MODE COHERENCE (6.1): manipulation actions + the checkout macro run only in *manipulation*
        mode (hands stay ACTIVE at REST elsewhere, so firing a hand action off-mode perturbs a carried
        item). A wrong-mode emit blocks and the router flips to manipulation next step.

    `leg_type=None` disables the leg gate (eval_pickup, which has no legs, calls it that way)."""
    if action == "checkout_held_item" and leg_type is not None and leg_type != "checkout":
        print(f"[BLOCKED] 'checkout_held_item' belongs to the *checkout* subtask, not this "
              f"'{leg_type}' leg. If the CURRENT GOAL is complete, choose STOP to hand off.")
        return {"blocked": True, "reason": ("checkout_held_item belongs to the checkout subtask; "
                "if your CURRENT GOAL is complete choose STOP to hand off (do not check out here)")}
    if (action in MANIPULATION_ACTIONS_REF or action in _MACRO_ACTIONS) \
            and mode is not None and mode != "manipulation":
        print(f"[BLOCKED] '{action}' only works in *manipulation* mode (current mode: {mode}); "
              f"the hands are inactive otherwise. Skipped - route to manipulation first.")
        return {"blocked": True, "reason": f"{action} requires manipulation mode (was {mode})"}
    if action == "checkout_held_item":
        # The 6.3 deterministic checkout macro - drive to the counter, align, scan, bag - one call.
        # Needs the agent for its live nav session; run_leg passes it through.
        if agent is None:
            print("[WARN] checkout_held_item dispatched without an agent - cannot reach the nav "
                  "session; skipped.")
            return {"blocked": True, "reason": "checkout_held_item needs the agent (no nav session)"}
        return agent._checkout_held_item() or {}
    if action in NAVIGATION_ACTIONS_REF:
        action_ref = NAVIGATION_ACTIONS_REF[action]
    elif action in PERCEPTION_ACTIONS_REF:
        action_ref = PERCEPTION_ACTIONS_REF[action]
    elif action in MANIPULATION_ACTIONS_REF:
        action_ref = MANIPULATION_ACTIONS_REF[action]
    else:
        print(f"[WARN] Unknown action skipped: {action}")
        return {}

    main_goal = notes.get('main_goal', '')
    sub_goals = notes.get('sub_goal', '')
    key_info  = notes.get('key_info', '')
    checklist = notes.get('checklist', '')

    if action == "center_object_on_screen":
        target_info = f"main_goal={main_goal}\nsub_goals={sub_goals}\nkey_info={key_info}\nchecklist={checklist}"
        # debug_dir (when a runner passes one) makes center_object_on_screen drop its per-look
        # candidate/locked/aim frames there - see the runners' per-step screenshot logging.
        return action_ref(target_info, debug_dir=debug_dir) or {}
    elif action in ("retrieve_item", "approach_object"):
        return action_ref(main_goal) or {}
    elif action == "extend_arm_until_grabbed":
        # Phase D: MEASURE before the blind reach. RequestLidarCenter reads depth along the pitched
        # gaze (hands are LiDAR self-culled, so an active hand does not occlude it); plan_reach turns
        # it into a verdict. The tool still never moves the body - a move/crouch/bail/recenter verdict
        # is surfaced as `last_reach` for the router/actor to act on next step (measured, not a guess).
        # Any failure (transport error, or a pre-Phase-D sim build with no pose in the sample) falls
        # back to the exact prior behaviour: one blind reach.
        plan = None
        try:
            plan = plan_reach(RequestLidarCenter())
        except Exception as e:
            print(f"[REACH] RequestLidarCenter/plan_reach failed ({type(e).__name__}: {e}); "
                  f"falling back to a blind reach.")
        if plan is None or plan["verdict"] == "unavailable":
            result = action_ref(time_units) or {}
            if not result.get('gripped', False):
                print("[GRAB] extend_arm_until_grabbed did not grip — item out of reach, reposition.")
            return result
        if plan["verdict"] == "reachable":
            result = action_ref(time_units) or {}
            result["reach_verdict"] = "reachable"
            result["last_reach"] = _last_reach_line(plan, gripped=result.get("gripped", False))
            if not result.get('gripped', False):
                print(f"[GRAB] plan_reach said reachable but the grab missed - {plan['reason']}; "
                      f"the reach envelope may need retuning.")
            return result
        # move / crouch / bail / recenter: skip the blind reach, report the measured verdict.
        print(f"[REACH] {plan['verdict']}: {plan['reason']}")
        return {"gripped": False, "reach_verdict": plan["verdict"],
                "move_steps": plan["move_steps"], "last_reach": _last_reach_line(plan)}
    else:
        return action_ref(time_units) or {}


# ---------------------------------------------------------------------------
# Per-leg runner (eval_pickup.run_one, generalised)
# ---------------------------------------------------------------------------

def _fresh_agent_state() -> dict:
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
    return view


def write_step_output(out_dir, step, response):
    """Dump a step's FULL agent output to out_dir/step<NN>.txt (untruncated, unlike the JSONL
    fields) so it pairs with the step screenshot for debugging: the mode router's decision, the
    VLM actor's output, the episodic reflection (what_worked / what_to_avoid), and any nav note.
    No-op if out_dir is falsy."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"step{step:02d}.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"=== STEP {step} | mode={response.get('agent_mode')} "
                 f"| halt={response.get('halt')} ===\n\n")
        if response.get("nav_note"):
            fh.write(f"--- NAV NOTE ---\n{response['nav_note']}\n\n")
        fh.write(f"--- MODE ROUTER (semantic) ---\n{response.get('semantic') or '(n/a)'}\n\n")
        fh.write(f"--- VLM ACTOR OUTPUT ---\n{response.get('text') or ''}\n\n")
        fh.write(f"--- EPISODIC REFLECTION ---\n{response.get('episodic') or '(n/a)'}\n")


def run_leg(agent, leg, sm, caps, log_path=None, context="", future_legs=None,
            visited=None, leg_idx=0):
    """Run ONE typed subtask leg as a self-contained embodied-agent loop - eval_pickup.run_one
    generalised for a leg of a long-horizon task (see the module docstring for the three differences).

    `leg` is a TYPED dict ({"type", "text", ...(+plan-resolved candidates)}); a bare string degrades to
    an `unknown` leg. `caps` = (max_steps, max_minutes) PER LEG. `sm` is the StoreMap (localisation +
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
        print(f"[CONTEXT] {context[:120]}{'...' if len(context) > 120 else ''}")

    # Per-leg agent setup: reset CONVERSATION history only (semantic + episodic persist across legs -
    # the orchestrator's shared-memory contract), and seed the graph navigator with THIS leg's
    # plan-time candidates so it does not re-resolve at runtime (6.3 #1). NOTE: no return_to_start here
    # - that is the orchestrator's per-TASK job; calling it mid-task stows the hands and drops a carry.
    agent.vlm_agent.reset_history()
    agent.seed_nav_candidates(leg.get("candidates"), leg.get("target_name"))
    semantic_before = agent.vlm_agent.base_semantic_memory

    # Logging + per-step screenshots: ONE dir per leg (replaces the old SIM_RUNS2/SIM_RUNS3 split).
    shots_dir = os.path.splitext(log_path)[0] if log_path else None
    if shots_dir:
        os.makedirs(shots_dir, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8") if log_path else None
    t0 = time.time()

    def log(rec):
        if log_fh:
            rec["wall"] = round(time.time() - t0, 1)
            log_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            log_fh.flush()   # crash-proof: a dead run still leaves its trail

    log({"event": "leg_start", "leg": leg_idx, "type": leg.get("type"), "text": leg_text,
         "candidates": leg.get("candidates"), "arm": agent.nav_mode,
         "ts": datetime.now().isoformat(timespec="seconds")})

    m = {"type": leg.get("type"), "text": leg_text, "t_manip": None, "t_grip": None,
         "t_checkout": None, "success": False, "timesteps": 0, "llm_calls": 0, "errors": 0,
         "halts_refused": 0, "halt_forced": False, "end_reason": None}

    state = _fresh_agent_state()
    # The leg's STARTING checkpoint counts as visited (the post-action refresh only records positions
    # AFTER a step, so without this a compare leg that starts at a candidate gets no credit for it).
    visited.add(sm.nearest_checkpoint((state["translation"][0], state["translation"][2])))
    halt_refusals = 0
    goal_met_streak = 0        # consecutive steps the completion predicate would grant (backstop)
    last_actor_text = ""       # actor's last REAL output (compare predicate's choice check)
    gripped_name = None        # SKU the grab tool reported AT the grip - the durable record the pickup
                               # predicate matches on (live hovered clears to 'null' once the hand
                               # retracts from the shelf; measured live 2026-07-23: a held Piattos read
                               # 'null'/'null' at the halt step and the STOP was wrongly refused)

    for step in range(1, max_steps + 1):
        if (time.time() - t0) / 60 > max_minutes:
            m["end_reason"] = "time_cap"
            break
        m["timesteps"] = step

        img_bytes = _REQUEST_SCREENSHOT_()["image"]
        if shots_dir:
            # Cap the SAVED debug frame at 1080p; the VLM below gets the NATIVE bytes (imageb64).
            with open(os.path.join(shots_dir, f"step{step:02d}.png"), "wb") as fh:
                fh.write(downscale_for_storage(img_bytes))
        imageb64 = base64.b64encode(img_bytes).decode("utf-8")
        state["visited_checkpoints"] = set(visited)   # compare predicate reads the task visit trace (code only)
        # The LLM gets a LEAN view (drops code-only bookkeeping like the growing visit set); the FULL
        # `state` still backs every predicate call, so no halt check is weakened.
        request = {"task": augmented_task, "image": imageb64, "state": _model_facing_state(state)}

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
            write_step_output(shots_dir, step, response)

        mode = response.get("agent_mode")
        if mode == "manipulation" and m["t_manip"] is None:
            m["t_manip"] = round(time.time() - t0, 1)

        # ---- STOP is a REQUEST: the typed predicate grants or refuses (6.3) --------------------
        if response.get("halt"):
            final_text = last_actor_text or response.get("text") or ""
            granted, reason = completion_predicate(leg, state, final_text=final_text)
            log({"event": "halt_request", "step": step, "granted": granted, "reason": reason})
            if granted:
                m["success"] = True
                m["end_reason"] = "halt_granted"
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
            if halt_refusals >= HALT_REFUSAL_CAP:
                # Escape hatch so the agent can't spin forever on STOP - NOT a grant. The leg ends
                # halt_forced with the reason in state; the orchestrator counts it as a non-clean leg.
                m["halt_forced"] = True
                m["end_reason"] = "halt_forced"
                print(f"[GUARD] refusal cap reached — force-ending leg (halt_forced): {reason}")
                break
            continue

        # ---- parse the actor's action JSON (a bad parse skips the step, run_one parity) --------
        match = re.search(agent.vlm_agent.extractable_json_structured_output, response.get("text") or "")
        parsed = None
        if match:
            try:
                parsed = ast.literal_eval(match.group(1))
            except Exception:  # noqa: BLE001
                parsed = None
        if parsed is None:
            m["errors"] += 1
            log({"event": "parse_error", "step": step, "raw": (response.get("text") or "")[:400]})
            if m["errors"] >= 3:
                m["end_reason"] = "errors"
                break
            continue
        last_actor_text = response.get("text") or ""
        notes = parsed.get("notes", {})

        center_dir = os.path.join(shots_dir, f"step{step:02d}_center") if shots_dir else None
        acted, blocked_reason, center_msg, last_reach = [], False, None, None
        grab_failed, checkout_result = False, None
        for action, tt in zip(parsed.get("actions", []), parsed.get("times", [])):
            raw_action = action.strip()
            inline = None
            im = re.match(r'^(\w+)\([\'"]?(.*?)[\'"]?\)$', raw_action)
            if im:
                raw_action, inline = im.group(1), im.group(2)
            step_center = None
            if center_dir and raw_action == "center_object_on_screen":
                os.makedirs(center_dir, exist_ok=True)
                step_center = center_dir
            res = dispatch_action(raw_action, int(tt), notes, inline_arg=inline, mode=mode,
                                  debug_dir=step_center, agent=agent, leg_type=leg.get("type")) or {}
            if res.get("blocked"):
                blocked_reason = res.get("reason", True)
            if res.get("center_message"):
                center_msg = res["center_message"]
            if res.get("last_reach"):
                last_reach = res["last_reach"]
            if raw_action == "checkout_held_item" and not res.get("blocked"):
                checkout_result = res
            if raw_action == "extend_arm_until_grabbed" and not res.get("blocked"):
                # blocked = wrong-mode, not a distance failure; a measured move/crouch/bail/recenter
                # carries its own recovery in last_reach - only a reachable-but-missed grab is a failure.
                verdict = res.get("reach_verdict")
                if verdict in (None, "reachable") and not res.get("gripped", False):
                    grab_failed = True
                if res.get("gripped") and res.get("hovered"):
                    # Capture the SKU AT the grip - the durable name record (hovered clears later).
                    gripped_name = res["hovered"]
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
        # 6.3 #1: localise from the live pose the state refresh just fetched (no extra sim round-trip) +
        # the frozen map. Feeds the goto predicate and grows the task-level visit trace (compare).
        near = sm.nearest_checkpoint((state["translation"][0], state["translation"][2]))
        state["nearest_checkpoint"] = near
        visited.add(near)
        state["visited_checkpoints"] = set(visited)

        gripping_now = bool(state.get("leftGrippedState") or state.get("rightGrippedState"))
        if gripping_now and m["t_grip"] is None:
            m["t_grip"] = round(time.time() - t0, 1)
        # Sticky while a hand grips; cleared the moment nothing grips (release/checkout/drop), so a
        # stale name can never vouch for an empty hand.
        if not gripping_now:
            gripped_name = None
        state["gripped_name"] = gripped_name

        # 6.3 completion nudge + backstop: run the SAME predicate the STOP request will face, SILENTLY,
        # each step. When the CURRENT GOAL measurably holds, put that in front of the router (goal_check)
        # so it stops THIS leg instead of drifting into future goals it recalled from persistent memory
        # (the leg-overrun failure). The agent still chooses STOP - this is a nudge, not an auto-end. But
        # if it stays satisfied for COMPLETION_BACKSTOP steps without ever proposing STOP, end the leg
        # anyway (success=True, the goal holds) - the symmetric twin of the refusal cap.
        met, met_reason = completion_predicate(leg, state, final_text=last_actor_text)
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
             "gripped_name": gripped_name,
             "checkout": checkout_result, "goal_met": met, "status": notes.get("status")})

        if goal_met_streak >= COMPLETION_BACKSTOP:
            m["success"] = True
            m["end_reason"] = "completed_no_stop"
            print(f"[LEG {leg_idx} DONE] completion backstop: goal measurably held for "
                  f"{goal_met_streak} steps without a STOP — ending leg (success). {met_reason}")
            log({"event": "completed_no_stop", "step": step, "streak": goal_met_streak,
                 "reason": met_reason})
            break

    if m["end_reason"] is None:
        m["end_reason"] = "step_cap"
    m["wall_s"] = round(time.time() - t0, 1)
    m["final_state"] = state
    m["new_semantic_entries"] = agent.vlm_agent.base_semantic_memory[len(semantic_before):]
    log({"event": "leg_end", **{k: v for k, v in m.items() if k != "final_state"}})
    if log_fh:
        log_fh.close()
    return m


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _load_store_map():
    from store_map import StoreMap
    return StoreMap()


def _current_nearest_cp(sm):
    """The checkpoint nearest the agent's LIVE pose (a zero-delta TransformAgent is a read, not a
    move). Used to order legs from where the agent ACTUALLY is - spawn if we just reset, else wherever
    it happens to be. Falls back to the spawn corner if the pose read fails."""
    try:
        p = TransformAgent((0, 0, 0), (0, 0, 0))["translation"]
        return sm.nearest_checkpoint((p[0], p[2]))
    except Exception:  # noqa: BLE001
        return sm.nearest_checkpoint(SPAWN_XZ)


def orchestrate(task, arm="graph", caps=(150, 40.0), out=None, run_dir=None,
                resolver_backend="qwen", reset_start=False):
    """Decompose `task` -> typed legs, resolve each leg on the map (plan time), order the legs, then
    run each with run_leg until the AGENT stops (predicate-granted) or a per-leg cap fires. Shared
    semantic/episodic memory + a between-leg findings summary carry context forward. A leg that does
    not complete ABORTS the remaining legs (a failed pickup shouldn't burn a checkout leg). Writes a
    summary.json + per-leg JSONL to run_dir (eval_pickup layout).

    arm: 'graph' (default - the measured-better navigator, right for long-horizon), 'vlm'
    (control), or 'graph-advised' (graph targets, per-hop advisor-VLM drive - see
    agent._advised_goto; adds one advisor call per graph hop, counted in llm_calls).
    caps: (max_steps, max_minutes) PER LEG.

    reset_start (default FALSE): drive to the fixed spawn checkpoint ONCE before the first leg
    (return_to_start; pose-only, never between legs - it stows hands, which would drop a carry). This
    is EVAL-reproducibility machinery, not an agent capability: it makes a batteried run (6.4's
    eval_longhorizon) start every task from the identical pose so metrics compare. A plain interactive
    run leaves it OFF - the agent starts from wherever it is (returning to spawn every time is awkward
    and adds nothing the resolver + graph nav don't already handle)."""
    client = _llm_client()
    init_logger(run_name=f"subtask-{datetime.now():%m%d_%H%M%S}")
    agent = EmbodiedAgent(vlm_config=VLM_CONFIG, associative_config=ASSOCIATIVE_CONFIG,
                          mode='lean', nav_mode=arm, resolver_backend=resolver_backend)

    # Run outputs (per-leg JSONL + screenshots + summary.json) land under overhaul/subtask_run_outputs/
    # (_THIS_DIR is overhaul/), one timestamped dir per task run.
    run_dir = run_dir or os.path.join(_THIS_DIR, "subtask_run_outputs",
                                      f"{datetime.now():%m%d_%H%M%S}_{arm}")
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"[ORCHESTRATOR] task: {task!r}")
    print(f"[ORCHESTRATOR] arm={arm}  caps={caps[0]} steps / {caps[1]} min per leg  run dir: {run_dir}")

    # -- decompose (1 LLM) + resolve each leg on the map (N LLM, plan time) --
    subtasks = decompose_task(client, task)
    task_llm = 1
    sm = _load_store_map()
    resolve_call = make_resolve_call(resolver_backend)
    legs, n_resolves = plan_legs(sm, resolve_call, subtasks)
    task_llm += n_resolves

    # -- per-TASK reset (OPT-IN, default off): eval-reproducibility only; see the docstring. Done
    #    BEFORE ordering so legs order from the true post-reset start. Pose-only, never between legs.
    if reset_start:
        try:
            from eval_pickup import return_to_start
            return_to_start(agent)
        except Exception as e:  # noqa: BLE001 - a reset hiccup shouldn't abort the whole task
            print(f"[ORCHESTRATOR] return_to_start skipped ({type(e).__name__}: {e})")

    # Order independent pickup->checkout pairs from where the agent ACTUALLY is (spawn if we reset,
    # else its current pose) - not an assumed spawn corner.
    legs = order_legs(sm, legs, _current_nearest_cp(sm))

    print(f"[ORCHESTRATOR] {len(legs)} leg(s) (resolver calls: {n_resolves}):")
    for i, lg in enumerate(legs, 1):
        feas = "" if lg.get("feasible", True) else "  [INFEASIBLE: target resolved to no checkpoint]"
        cps = lg.get("candidates")
        print(f"  {i}. [{lg.get('type')}] {lg.get('text')}"
              + (f"  -> cps {cps}" if cps else "") + feas)
    infeasible = [i + 1 for i, lg in enumerate(legs) if not lg.get("feasible", True)]
    if infeasible:
        print(f"[ORCHESTRATOR] WARNING: leg(s) {infeasible} resolved to no checkpoint - the plan may "
              f"be doomed, but running so the failure is measured, not assumed.")

    cumulative_context = ""
    visited = set()                 # task-level visit trace (compare predicate), grown by every leg
    leg_rows = []
    task_success = True
    try:
        for i, leg in enumerate(legs):
            future = legs[i + 1:]
            print(f"\n[ORCHESTRATOR] ── Leg {i + 1}/{len(legs)} ──")
            m = run_leg(agent, leg, sm, caps,
                        log_path=os.path.join(run_dir, f"leg{i:02d}.jsonl"),
                        context=cumulative_context, future_legs=future,
                        visited=visited, leg_idx=i + 1)
            task_llm += m["llm_calls"]
            leg_rows.append({k: v for k, v in m.items()
                             if k not in ("final_state", "new_semantic_entries")})
            print(f"### leg {i+1} {m['end_reason']}: success={m['success']} "
                  f"t_grip={m['t_grip']} t_checkout={m['t_checkout']} steps={m['timesteps']} "
                  f"halts_refused={m['halts_refused']} wall={m['wall_s']}s")

            if not m["success"]:
                task_success = False
                print(f"[ORCHESTRATOR] leg {i+1} did not complete ({m['end_reason']}) — "
                      f"aborting the remaining {len(legs) - i - 1} leg(s).")
                break

            if i + 1 < len(legs):
                print("[ORCHESTRATOR] Generating findings summary...")
                findings = generate_findings_summary(
                    client, completed_subtask=leg.get("text", ""),
                    final_state=m["final_state"], new_semantic_entries=m["new_semantic_entries"])
                task_llm += 1
                print(f"[FINDINGS SUMMARY]\n{findings}\n")
                cumulative_context += f"\n\n--- LEG {i + 1} FINDINGS ---\n{findings}"
    finally:
        summary = {"task": task, "arm": arm, "success": task_success,
                   "legs_planned": len(legs),
                   "legs_completed": sum(1 for r in leg_rows if r.get("success")),
                   "resolver_calls": n_resolves, "llm_calls": task_llm,
                   "wall_s": round(time.time() - t0, 1), "legs": leg_rows}
        if arm == "graph-advised":
            # Whole-task advisor attribution (per-hop detail rides the agent's logger lines):
            # agree ~= hops means the graph arm with a per-hop tax; deviations/stops are the
            # rows the arm exists to surface (VLMAdvisedPlanner's read-together rule).
            st = getattr(agent, "_advised_stats", [])
            summary["advised"] = {"hops": len(st),
                                  "agreed": sum(1 for s in st if s["agreed"]),
                                  "invalid": sum(1 for s in st if s["invalid"]),
                                  "stops": sum(1 for s in st if s["stop_here"])}
        out_path = out or os.path.join(run_dir, "summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("-" * 40)
        print(f"[ORCHESTRATOR] task success={task_success}  "
              f"legs {summary['legs_completed']}/{summary['legs_planned']}  "
              f"llm={task_llm}  wall={summary['wall_s']}s  -> {out_path}")
        print("-" * 40)
        try:
            if agent._graph_nav:
                agent._graph_nav[1].close()
        except Exception:  # noqa: BLE001
            pass
        chime.beep()
    return summary


def main():
    ap = argparse.ArgumentParser(description="Long-horizon typed-subtask orchestrator.")
    ap.add_argument("task", nargs="?", default=None,
                    help="the long-horizon task (or use --task)")
    ap.add_argument("--task", dest="task_opt", default=None, help="the long-horizon task")
    ap.add_argument("--arm", choices=["vlm", "graph", "graph-advised"], default="graph-advised",
                    help="navigation arm (default graph - the measured-better navigator; "
                         "graph-advised drives each graph hop through a per-hop advisor VLM)")
    ap.add_argument("--max-steps", type=int, default=150, help="per-leg step cap")
    ap.add_argument("--max-minutes", type=float, default=40.0, help="per-leg wall-clock cap")
    ap.add_argument("--out", default=None, help="summary.json path (default: <run-dir>/summary.json)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resolver-backend", choices=["qwen", "claude-cli"], default="qwen")
    ap.add_argument("--reset-start", action="store_true",
                    help="drive to the fixed spawn pose once before starting (eval-reproducibility; "
                         "OFF by default - a plain run starts from the agent's current pose)")
    args = ap.parse_args()

    task = args.task_opt or args.task or input("Task: ")
    orchestrate(task, arm=args.arm, caps=(args.max_steps, args.max_minutes), out=args.out,
                run_dir=args.run_dir, resolver_backend=args.resolver_backend,
                reset_start=args.reset_start)


if __name__ == "__main__":
    main()
