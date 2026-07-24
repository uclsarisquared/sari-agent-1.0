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

Self-correction (2026-07-23, two levels - added after run 0723_061651_graph spun a pickup leg's
refusals into halt_forced and took the task down):
  - MID-LEG: a pickup STOP refused WRONG_ITEM_RELEASE_AFTER times while a hand verifiably holds the
    wrong item auto-releases that hand (never a carried-in item) and resets the refusal budget once,
    so the agent re-attempts the grab instead of spinning to the cap (run_leg's refusal branch).
  - ORCHESTRATOR: a failed leg is retried up to --leg-retries times (default 1) with the failure
    reason in the retry's context, before the task aborts.
(The same run also exposed a predicate false-refusal - a category target like 'Biscuits' never
substring-matched any SKU; fixed by catalog-category grounding in subtask_completion.)

The agent starts from wherever it is by default; `--reset-start` is opt-in machinery for a batteried
eval (6.4), not needed for an interactive run. `--restart-env` is a separate opt-in that hard-resets
the STORE (items back on shelves, prior checkouts undone) so a fresh task doesn't inherit the last
run's displaced items:

    python subtask_agents.py "pick up 2 Jin Ramen" --restart-env
"""

import argparse
import ast
import base64
import json
import os
import re
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OVERHAUL_DIR = os.path.dirname(_THIS_DIR)
# Entry point: `python orchestrator/subtask_agents.py` puts orchestrator/ (not overhaul/) on
# sys.path, so the package imports below need the root added explicitly — before the first one.
if _OVERHAUL_DIR not in sys.path:
    sys.path.insert(0, _OVERHAUL_DIR)

from sim import chime  # cross-platform run-completion beep (was winsound: Windows-only)
from datetime import datetime

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Repo-root api.env (overhaul/orchestrator/ -> repo root is three parents up), resolved from
# __file__ so it loads regardless of CWD.
load_dotenv(Path(__file__).resolve().parent.parent.parent / 'api.env')

from sim.env import (
    _GRIP_LEFT_,
    _GRIP_RIGHT_,
    _REQUEST_SCREENSHOT_,
    RequestLidarCenter,
    TransformAgent,
    TransformHands,
    downscale_for_storage,
    init_logger,
)
from manip.manipulation import plan_reach
from toolset.actions import (
    NAVIGATION_ACTIONS_REF,
    PERCEPTION_ACTIONS_REF,
    MANIPULATION_ACTIONS_REF,
)
from agent_core.agent import EmbodiedAgent, ucl_qwen_config
# Phase 6.3: the typed-subtask contract + deterministic completion predicates live in a lightweight,
# sim-free module (importing THIS module pulls the whole model stack) so they stay offline-unit-testable.
from orchestrator.subtask_completion import (
    TYPED_DECOMPOSER_SYSTEM,
    parse_decomposition,
    completion_predicate,
    mismatched_hands,
    HALT_REFUSAL_CAP,
    COMPLETION_BACKSTOP,
    WRONG_ITEM_RELEASE_AFTER,
)
# Plan-time map planning (also sim-free, so it stays offline-unit-testable - test_subtask_planning.py).
from orchestrator.subtask_planning import (
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
    from agent_core.agent import _ucl_creds
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
    tests/ab_decompose.py (11/11 clean on the four-family battery, 2026-07-23)."""
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

def _crouched_grab(action_ref, time_units, target_info, debug_dir, plan):
    """AUTO-CROUCH (Phase D, 2026-07-23): resolve a `crouch` verdict INSIDE this one dispatch call -
    crouch, re-center, re-measure, grab if now reachable, and ALWAYS stand back up (finally-guarded).

    Posture is deliberately NOT the VLM's job: if standing up were left to the actor, a run could end
    with the agent navigating crouched (or the learner drawing lessons from a stale posture). The
    invariant this enforces: **the agent is only ever crouched inside a single grab call** - the same
    scoping the hand-retract already uses. Measured basis (envelope.csv, 2026-07-22): L1 items missed
    standing at slant 1.1-1.3 m and grabbed crouched at 0.6-0.7 m - crouching shortens the slant
    distance below max_reach.

    The re-center is MANDATORY: crouching drops the camera ~0.7 m, so the just-centred target is now
    well off-centre; grabbing without re-centering is the grab-the-neighbour failure. Hands stay
    ACTIVE throughout (Phase 6.1: hands are always-on at REST/GRAB; centring already runs with resting
    hands everywhere else). If after crouching the target is STILL too far, we stand up and report the
    measured move - the tool never moves the body (2026-07-21 directive); the router moves and the
    next grab re-crouches (one spare cycle, rare)."""
    from sim.env import SetCrouch
    from vision.perception import center_object_on_screen
    base = {"gripped": False, "reach_verdict": "crouch", "move_steps": 0}
    try:
        SetCrouch(True)
        c = center_object_on_screen(target_info, debug_dir=debug_dir) or {}
        if not c.get("centered"):
            base["last_reach"] = ("CROUCHED but could not RE-CENTER the target from the lower view "
                                  f"({c.get('outcome', 'no result')}) - stood back up; bring the target "
                                  "into view and retry the grab")
            return base
        plan2 = plan_reach(RequestLidarCenter())
        if plan2["verdict"] == "reachable":
            result = action_ref(time_units) or {}
            result["reach_verdict"] = "crouch"
            result["last_reach"] = (
                f"CROUCHED and GRABBED - {plan2['reason']}" if result.get("gripped") else
                f"CROUCHED into range but the grab MISSED - {plan2['reason']}; stood back up - "
                f"move a little closer, re-center, and retry")
            return result
        if plan2["verdict"] == "move":
            base["move_steps"] = plan2["move_steps"]
            base["last_reach"] = (f"CROUCHED but still too far - {plan2['reason']}; stood back up - "
                                  f"move that distance, re-center, and retry (it will crouch again)")
            return base
        base["last_reach"] = (f"CROUCHED but the target is still out of reach "
                              f"({plan2['verdict']}: {plan2['reason']}) - stood back up")
        return base
    finally:
        SetCrouch(False)   # ALWAYS stand back up - crouch must never leak past this call


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
# Dual-hand (2026-07-23): each macro/grab family is {bare-name: 'auto', _left: 'left', _right: 'right'};
# 'auto' resolves deterministically from grip state (manipulation.resolve_grab_hand/_release_hand).
_MACRO_ACTIONS = {"checkout_held_item": "auto",
                  "checkout_held_item_left": "left",
                  "checkout_held_item_right": "right"}
_GRAB_ACTIONS = {"extend_arm_until_grabbed",
                 "extend_arm_until_grabbed_left",
                 "extend_arm_until_grabbed_right"}


def _grab_ready(state):
    """MEASURED readiness gate for run_leg's perception->manipulation grab auto-promotion (Option 2,
    2026-07-23). True iff a MEASURED signal says the target is centred or in reach: last_center's
    "SUCCESS ..." (center_object_on_screen verified the target on the aim point) or last_reach's
    "REACHABLE ..." (plan_reach measured the slant distance inside the envelope). Both are CODE reading
    the sim, never the model's eyeballed "looks centred" - so promoting a grab on this signal cannot
    encourage the blind grab-at-air the centring/grab handshake edit fought (sys_inst.py header). An
    absent/None field (e.g. step 1, before anything has been centred) reads as NOT ready, so a premature
    grab still blocks and the router flips to manipulation next step, exactly as before."""
    lc = state.get("last_center") or ""
    lr = state.get("last_reach") or ""
    return lc.startswith("SUCCESS") or lr.startswith("REACHABLE")


# The actor emits a single-quoted Python-literal dict (sys_inst OUTPUT FORMAT). ast.literal_eval
# parses that - UNTIL a value carries an apostrophe ("Kellogg's", "it's"), which terminates the
# single-quoted string early and raises SyntaxError. The free-text fields (reasoning, notes) are
# where apostrophes live; the ACTIONABLE fields (actions, times) are apostrophe-free flat lists.
# So when the whole-dict parse fails we recover just those two, letting the step execute instead of
# being wasted. Measured 2026-07-24 on the "Get a cereal" run, where every cereal-brand mention
# crashed the actor parse and burned the leg's 3-error budget.
_ACTOR_LIST_RE = lambda key: re.compile(r"['\"]%s['\"]\s*:\s*\[([^\[\]]*)\]" % key, re.DOTALL)
_ACTOR_ITEM_RE = re.compile(r"""['"]([^'"]+)['"]""")


def _salvage_actions_times(blob: str):
    """Recover (actions, times) by regex from an actor blob whose full literal parse failed. Both
    are flat lists of short, apostrophe-free tokens, so a bracket-scoped match is robust to broken
    quoting anywhere else in the dict. Returns None if either list is missing or their lengths
    disagree (a mismatch means the salvage is unreliable - fall through to the error path)."""
    am, tm = _ACTOR_LIST_RE("actions").search(blob), _ACTOR_LIST_RE("times").search(blob)
    if not (am and tm):
        return None
    actions = [a.strip() for a in _ACTOR_ITEM_RE.findall(am.group(1)) if a.strip()]
    times = [int(x) for x in re.findall(r"-?\d+", tm.group(1))]
    if actions and len(actions) == len(times):
        return actions, times
    return None


def parse_actor_response(text: str, pattern) -> dict:
    """Parse the actor's fenced dict, tolerant of the apostrophe break above. Tiers, cheapest first:
    ast.literal_eval (the contract) -> json.loads (a model that emitted real JSON) -> regex salvage
    of actions/times with notes defaulted to {} (dispatch_action reads every notes field via .get,
    so an empty dict is safe). Returns None only when even the actions/times can't be recovered."""
    m = re.search(pattern, text or "")
    blob = m.group(1) if m else (text or "").strip()
    if not blob:
        return None
    for parse in (ast.literal_eval, json.loads):
        try:
            d = parse(blob)
        except Exception:  # noqa: BLE001 - any parse failure just falls through to the next tier
            continue
        if isinstance(d, dict) and "actions" in d:
            return d
    salvaged = _salvage_actions_times(blob)
    if salvaged:
        actions, times = salvaged
        return {"actions": actions, "times": times, "notes": {}}
    return None


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
    if action in _MACRO_ACTIONS and leg_type is not None and leg_type != "checkout":
        print(f"[BLOCKED] '{action}' belongs to the *checkout* subtask, not this "
              f"'{leg_type}' leg. If the CURRENT GOAL is complete, choose STOP to hand off.")
        return {"blocked": True, "reason": (f"{action} belongs to the checkout subtask; "
                "if your CURRENT GOAL is complete choose STOP to hand off (do not check out here)")}
    if (action in MANIPULATION_ACTIONS_REF or action in _MACRO_ACTIONS) \
            and mode is not None and mode != "manipulation":
        print(f"[BLOCKED] '{action}' only works in *manipulation* mode (current mode: {mode}); "
              f"the hands are inactive otherwise. Skipped - route to manipulation first.")
        return {"blocked": True, "reason": f"{action} requires manipulation mode (was {mode})"}
    if action in _MACRO_ACTIONS:
        # The 6.3 deterministic checkout macro - drive to the counter, align, scan, bag - one call.
        # Needs the agent for its live nav session; run_leg passes it through. The variant name pins
        # the hand: 'auto' checks out EVERY held item in one fused pass (both hands = scan-scan-bag-bag
        # off one drive+align, degrading to a single item when one hand holds); _left/_right pin one.
        if agent is None:
            print(f"[WARN] {action} dispatched without an agent - cannot reach the nav "
                  f"session; skipped.")
            return {"blocked": True, "reason": f"{action} needs the agent (no nav session)"}
        return agent._checkout_held_item(hand=_MACRO_ACTIONS[action]) or {}
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
    elif action in _GRAB_ACTIONS:
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
                print(f"[GRAB] {action} (hand={result.get('hand', '?')}) did not grip — "
                      f"item out of reach, reposition.")
            return result
        if plan["verdict"] == "reachable":
            result = action_ref(time_units) or {}
            result["reach_verdict"] = "reachable"
            result["last_reach"] = _last_reach_line(plan, gripped=result.get("gripped", False))
            if not result.get('gripped', False):
                print(f"[GRAB] plan_reach said reachable but the grab (hand="
                      f"{result.get('hand', '?')}) missed - {plan['reason']}; "
                      f"the reach envelope may need retuning.")
            return result
        if plan["verdict"] == "crouch":
            # AUTO-CROUCH: resolved inside this call (crouch -> re-center -> re-measure -> grab ->
            # ALWAYS stand). See _crouched_grab - posture never leaks to the router/VLM.
            print(f"[REACH] crouch: {plan['reason']} - auto-crouching")
            target_info = (f"main_goal={main_goal}\nsub_goals={sub_goals}\n"
                           f"key_info={key_info}\nchecklist={checklist}")
            result = _crouched_grab(action_ref, time_units, target_info, debug_dir, plan)
            print(f"[REACH] {result.get('last_reach')}")
            return result
        # move / bail / recenter: skip the blind reach, report the measured verdict.
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
    # Tag this leg's semantic-memory entries with the leg number so the accumulated blob keeps clean
    # provenance across legs: each leg restarts `timestep` at 1, so without this the entries collide
    # under duplicate `@ timestep N` keys describing different places (see EmbodiedAgent._semantic_tag).
    agent._mem_leg = leg_idx
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
         "halts_refused": 0, "halt_forced": False, "corrective_release": None, "end_reason": None}

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
    gripped_names = {"left": None, "right": None}
    # Dual-hand (2026-07-23): sides already gripping at LEG START. A pickup leg must produce a NEW
    # grip - without this, an item carried in from a previous leg would satisfy an untargeted pickup
    # predicate the moment the leg begins (any-hand _gripping is True the whole time).
    start_grips = {side for side in ("left", "right") if state.get(f"{side}GrippedState")}

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
        if off_target:
            print(f"[GATE] off-target: at cp{state.get('nearest_checkpoint')}, target at "
                  f"{gated_cps} — forcing navigation this step.")
        request = {"task": augmented_task, "nav_goal": leg_text, "force_navigate": off_target,
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
                for side in mismatched_hands(leg, state, start_grips):
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
            res = dispatch_action(raw_action, int(tt), notes, inline_arg=inline, mode=eff_mode,
                                  debug_dir=step_center, agent=agent, leg_type=leg.get("type")) or {}
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
             "gripped_names": dict(gripped_names), "off_target": off_target or None,
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

def _load_store_map(output_dir=None):
    from nav.store_map import StoreMap
    return StoreMap(output_dir=output_dir) if output_dir else StoreMap()


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
                resolver_backend="qwen", reset_start=False, restart_env=False, leg_retries=1,
                output_dir=None):
    """Decompose `task` -> typed legs, resolve each leg on the map (plan time), order the legs, then
    run each with run_leg until the AGENT stops (predicate-granted) or a per-leg cap fires. Shared
    semantic/episodic memory + a between-leg findings summary carry context forward. A failed leg is
    RETRIED up to `leg_retries` times (default 1) with the failure reason fed into the retry's
    context (orchestrator-level self-correction, 2026-07-23 - a halt_forced leg used to abort the
    task outright); only when the retries are also exhausted does it ABORT the remaining legs (a
    failed pickup shouldn't burn a checkout leg). Writes a summary.json + per-leg JSONL to run_dir
    (eval_pickup layout; a retry attempt logs to leg<NN>_retry<K>.jsonl).

    arm: 'graph' (default - the measured-better navigator, right for long-horizon), 'vlm'
    (control), or 'graph-advised' (graph targets, per-hop advisor-VLM drive - see
    agent._advised_goto; adds one advisor call per graph hop, counted in llm_calls).
    caps: (max_steps, max_minutes) PER LEG.

    reset_start (default FALSE): drive to the fixed spawn checkpoint ONCE before the first leg
    (return_to_start; pose-only, never between legs - it stows hands, which would drop a carry). This
    is EVAL-reproducibility machinery, not an agent capability: it makes a batteried run (6.4's
    eval_longhorizon) start every task from the identical pose so metrics compare. A plain interactive
    run leaves it OFF - the agent starts from wherever it is (returning to spawn every time is awkward
    and adds nothing the resolver + graph nav don't already handle).

    restart_env (default FALSE): hard-reset the STORE to its pristine initial state before the first
    leg via env.Reset() (Unity's ResetEnvironment) - items back on shelves, prior checkouts undone,
    agent teleported to spawn. Distinct from reset_start, which only MOVES the agent and leaves the
    shelf state a previous run displaced. Use it when a fresh task must not inherit the last run's
    grabbed/checked-out items (e.g. re-running 'pick up 2 Jin Ramen' after a run that already removed
    two). NOTE: eval_pickup's docstrings say 'never call Reset()' because ResetEnvironment used to
    DOUBLE every non-RetailItem object (price tags, cans); that warning PREDATES the C# fix -
    ResetEnvironment now calls ItemPoolingManager.ClearPool() + ShelfBuilder.DeleteAllPriceTags()
    before reloading (DataHandler.cs:617). Verify the duplication is gone on your build before relying
    on this in a batteried eval; it stays OFF by default."""
    client = _llm_client()
    init_logger(run_name=f"subtask-{datetime.now():%m%d_%H%M%S}")
    agent = EmbodiedAgent(vlm_config=VLM_CONFIG, associative_config=ASSOCIATIVE_CONFIG,
                          mode='lean', nav_mode=arm, resolver_backend=resolver_backend,
                          map_output_dir=output_dir)

    # Run outputs (per-leg JSONL + screenshots + summary.json) land under overhaul/subtask_run_outputs/
    # (_OVERHAUL_DIR is overhaul/), one timestamped dir per task run.
    run_dir = run_dir or os.path.join(_OVERHAUL_DIR, "subtask_run_outputs",
                                      f"{datetime.now():%m%d_%H%M%S}_{arm}")
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"[ORCHESTRATOR] task: {task!r}")
    print(f"[ORCHESTRATOR] arm={arm}  caps={caps[0]} steps / {caps[1]} min per leg  run dir: {run_dir}")

    # -- decompose (1 LLM) + resolve each leg on the map (N LLM, plan time) --
    subtasks = decompose_task(client, task)
    task_llm = 1
    sm = _load_store_map(output_dir)
    resolve_call = make_resolve_call(resolver_backend)
    legs, n_resolves = plan_legs(sm, resolve_call, subtasks)
    task_llm += n_resolves

    # -- hard STORE reset (OPT-IN, default off): put the shelves back before the task starts, so a
    #    fresh run doesn't inherit items a previous run grabbed/checked out. Done FIRST (before the
    #    pose reset and before ordering) so return_to_start re-syncs the nav pose to the post-reset
    #    spawn and order_legs reads the true start. See the docstring re: the (now-fixed) duplication
    #    warning in eval_pickup.
    if restart_env:
        try:
            from sim.env import Reset as _reset_env
            _reset_env()
            time.sleep(1.5)   # let Unity destroy + LoadStore() re-instantiate before the first frame
            print("[ORCHESTRATOR] hard env reset (ResetEnvironment): store restored to initial state.")
        except Exception as e:  # noqa: BLE001 - a reset hiccup shouldn't abort the whole task
            print(f"[ORCHESTRATOR] restart_env skipped ({type(e).__name__}: {e})")

    # -- per-TASK reset (OPT-IN, default off): eval-reproducibility only; see the docstring. Done
    #    BEFORE ordering so legs order from the true post-reset start. Pose-only, never between legs.
    if reset_start:
        try:
            from evals.eval_pickup import return_to_start
            return_to_start(agent, output_dir=output_dir)
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
            attempt, m = 0, None
            while True:
                attempt += 1
                leg_context = cumulative_context
                if attempt > 1:
                    # Orchestrator-level self-correction (2026-07-23): re-run the leg with WHY the
                    # last attempt was not accepted in front of the fresh agent. Semantic/episodic
                    # memory already persists, so the retry keeps everything the failure learned.
                    fail_reason = ((m.get("final_state") or {}).get("last_halt_refused")
                                   or m["end_reason"])
                    leg_context = cumulative_context + (
                        f"\n\n--- YOUR PREVIOUS ATTEMPT AT THIS EXACT SUBTASK FAILED "
                        f"({m['end_reason']}) ---\n"
                        f"Why it was not accepted: {fail_reason}\n"
                        f"Fix that specifically this time; everything you learned is still in memory.")
                    print(f"[ORCHESTRATOR] retrying leg {i + 1} "
                          f"(attempt {attempt}/{1 + leg_retries}): {fail_reason}")
                suffix = "" if attempt == 1 else f"_retry{attempt - 1}"
                m = run_leg(agent, leg, sm, caps,
                            log_path=os.path.join(run_dir, f"leg{i:02d}{suffix}.jsonl"),
                            context=leg_context, future_legs=future,
                            visited=visited, leg_idx=i + 1)
                task_llm += m["llm_calls"]
                leg_rows.append({**{k: v for k, v in m.items()
                                    if k not in ("final_state", "new_semantic_entries")},
                                 "attempt": attempt})
                print(f"### leg {i+1} attempt {attempt} {m['end_reason']}: success={m['success']} "
                      f"t_grip={m['t_grip']} t_checkout={m['t_checkout']} steps={m['timesteps']} "
                      f"halts_refused={m['halts_refused']} wall={m['wall_s']}s")
                if m["success"] or attempt > leg_retries:
                    break

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
    ap.add_argument("--arm", choices=["vlm", "graph", "graph-advised"], default="graph",
                    help="navigation arm (default graph - the measured-better navigator; "
                         "graph-advised drives each graph hop through a per-hop advisor VLM)")
    ap.add_argument("--max-steps", type=int, default=150, help="per-leg step cap")
    ap.add_argument("--max-minutes", type=float, default=40.0, help="per-leg wall-clock cap")
    ap.add_argument("--out", default=None, help="summary.json path (default: <run-dir>/summary.json)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resolver-backend", choices=["qwen", "claude-cli"], default="qwen")
    ap.add_argument("--output-dir", default=None,
                    help="slamtest output dir to load the map from (topology/annotations/grid). "
                         "Default: slamtest/output (StoreMap's DEFAULT_OUTPUT_DIR).")
    ap.add_argument("--leg-retries", type=int, default=1,
                    help="how many times to RETRY a failed leg with the failure reason in context "
                         "before aborting the task (orchestrator-level self-correction; 0 restores "
                         "the old abort-on-first-failure behaviour)")
    ap.add_argument("--reset-start", action="store_true",
                    help="drive to the fixed spawn pose once before starting (eval-reproducibility; "
                         "OFF by default - a plain run starts from the agent's current pose)")
    ap.add_argument("--restart-env", action="store_true",
                    help="hard-reset the STORE to its initial state before starting (Unity's "
                         "ResetEnvironment: items back on shelves, prior checkouts undone, agent to "
                         "spawn). OFF by default - use it so a fresh task doesn't inherit the last "
                         "run's grabbed/checked-out items. (Unlike --reset-start, which only moves "
                         "the agent.)")
    args = ap.parse_args()

    task = args.task_opt or args.task or input("Task: ")
    orchestrate(task, arm=args.arm, caps=(args.max_steps, args.max_minutes), out=args.out,
                run_dir=args.run_dir, resolver_backend=args.resolver_backend,
                reset_start=args.reset_start, restart_env=args.restart_env,
                leg_retries=max(0, args.leg_retries), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
