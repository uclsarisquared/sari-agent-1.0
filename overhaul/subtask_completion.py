"""subtask_completion.py - Phase 6.3 typed-subtask contract + deterministic completion predicates.

Lightweight and sim-free BY DESIGN: no torch/agent/env/openai imports, so the offline predicate
tests and the decomposer A/B harness load in milliseconds (importing subtask_agents pulls the whole
model stack, ~25 s). subtask_agents.py imports the prompt, the parser, and the predicates FROM HERE
so there is ONE home for the task-completion truth table.

The principle (phase6.3 plan): the sim state and the measured tool verdicts own the truth about
completion; the VLM only PROPOSES it. A VLM `STOP` becomes a *request* these predicates grant or
refuse - exactly what the pre-6.3 keyword guards gestured at, minus the fragility of grepping the
subtask's free text.

Type vocabulary is CLOSED: pickup | checkout | compare | goto. An untypeable decomposition degrades
to `unknown`, which falls back to the pre-6.3 keyword guards (behaviour preserved, logged as
`untyped`). `checkout` REPLACES the master plan's `place`: 6.2 collapsed scan-then-bag into the one
deterministic macro `store_map.checkout_held_item`, whose measured `{scanned, placed}` verdict the
checkout predicate reads rather than re-deriving.
"""

import json
import re

# Closed type vocabulary. `checkout` (not `place`) - see the module docstring.
SUBTASK_TYPES = ("pickup", "checkout", "compare", "goto")

# A leg whose halt is refused this many times force-continues the LOOP (never a fake grant) with the
# last refusal reason left in state, so a confused agent can't spin forever on STOP. The leg is
# marked `halt_forced` so 6.4 can count it. (phase6.3 plan, "Refusal cap".)
HALT_REFUSAL_CAP = 3

# The symmetric twin of the refusal cap: if the completion predicate stays satisfied for this many
# CONSECUTIVE steps and the agent still never proposes STOP (it keeps pursuing future goals from
# persistent memory), the leg ends success=True with end_reason='completed_no_stop'. One cap stops
# infinite STOP-spam; this stops infinite not-stopping. 6.4 reports halt_granted vs completed_no_stop
# as the completion-detection health signal (did the agent know it was done, or need the backstop?).
COMPLETION_BACKSTOP = 3


# ---------------------------------------------------------------------------
# Typed decomposer contract
# ---------------------------------------------------------------------------

TYPED_DECOMPOSER_SYSTEM = (
    "You are a task planner for an Embodied AI Agent in a 3D convenience store simulation. The agent "
    "can navigate, locate items on shelves, pick them up, carry them, scan them at the self-checkout, "
    "and bag them.\n\n"
    "Decompose the given task into a short ordered list of self-contained subtasks. Return ONLY a "
    "JSON array of subtask OBJECTS - no prose, no markdown fences around anything else.\n\n"
    "Each object MUST have a \"type\" (one of exactly: pickup, checkout, compare, goto) and a \"text\" "
    "(the natural-language instruction the agent will act on). Type-specific fields:\n"
    "  - pickup:   \"target\"   = the product to grab, as specifically as the task names it.\n"
    "  - checkout: (no extra field required) - take the CURRENTLY HELD item to the self-checkout, "
    "scan it, and bag it. This is ONE subtask: the agent has a single tool that drives to the "
    "counter, scans, and bags. NEVER split checkout into a separate 'go to the counter' + 'place it'.\n"
    "  - compare:  \"targets\" = a list of the candidate products to compare; \"criterion\" = what "
    "decides it (e.g. size, price, weight). The agent must decide by LOOKING at the items, not from "
    "any list.\n"
    "  - goto:     \"location\" = where to go, named ONLY as the task or store memory names it "
    "(e.g. 'Checkpoint 32', 'the checkout counter'). Never invent shelf numbers or location names.\n\n"
    "Rules:\n"
    "  - Each subtask ends in a clear, verifiable physical state change.\n"
    "  - 'Bring/carry/take X to the counter' after a pickup is a single `checkout` subtask, not a "
    "goto+place pair.\n"
    "  - A bare 'pick up X' is a single-element array with one pickup subtask.\n"
    "  - For a comparison ('the larger of...', 'the cheaper...'), route the agent to the candidates "
    "with goto/pickup as needed and use a compare subtask to make the visual decision.\n\n"
    "Example input: \"pick up the milk and bring it to the counter\"\n"
    "Example output: "
    "[{\"type\": \"pickup\", \"target\": \"milk\", \"text\": \"Pick up the milk.\"}, "
    "{\"type\": \"checkout\", \"text\": \"Take the held milk to the self-checkout: scan it and bag it.\"}]"
    "\n\nExample input: \"pick up the orange Pringles\"\n"
    "Example output: "
    "[{\"type\": \"pickup\", \"target\": \"orange Pringles\", \"text\": \"Pick up the orange Pringles.\"}]"
)


def parse_decomposition(raw: str, original_task: str) -> list:
    """Turn the decomposer LLM's raw output into a list of typed subtask dicts.

    Every returned item is a dict with at least {"type", "text"}, "type" in SUBTASK_TYPES or
    "unknown". Robust by construction - ANY failure degrades to a single `unknown` subtask carrying
    the original task text (which run_leg then handles with the pre-6.3 keyword guards), never an
    exception. Old-style bare strings (a legacy or half-migrated decomposer) are wrapped as `unknown`
    too, so a mixed response never crashes.
    """
    array_match = re.search(r'\[[\s\S]*\]', raw or "")
    if not array_match:
        return [{"type": "unknown", "text": original_task}]
    try:
        items = json.loads(array_match.group(0))
    except (json.JSONDecodeError, ValueError):
        return [{"type": "unknown", "text": original_task}]
    if not isinstance(items, list) or not items:
        return [{"type": "unknown", "text": original_task}]

    out = []
    for item in items:
        if isinstance(item, str):
            out.append({"type": "unknown", "text": item})
            continue
        if not isinstance(item, dict):
            continue  # skip a stray non-object rather than poison the whole plan
        text = str(item.get("text") or "").strip()
        t = str(item.get("type") or "").strip().lower()
        if t not in SUBTASK_TYPES:
            # Untypeable element: preserve its instruction, fall back to keyword guards.
            out.append({"type": "unknown", "text": text or original_task})
            continue
        norm = {"type": t, "text": text or original_task}
        # Carry the type-specific structured fields through untouched when present.
        for field in ("target", "location", "targets", "criterion"):
            if item.get(field) is not None:
                norm[field] = item[field]
        out.append(norm)
    return out or [{"type": "unknown", "text": original_task}]


# ---------------------------------------------------------------------------
# Name matching (the pickup / compare grounding primitive)
# ---------------------------------------------------------------------------

# Tokens that carry no product identity - dropped before overlap so a target like "the milk" doesn't
# match on "the". Kept deliberately small; colour/qualifier words (green, large, orange) are content.
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "with", "from", "for",
    "some", "any", "one", "it", "its", "that", "this", "these", "those", "item", "items",
    "product", "pick", "up", "grab", "get", "take", "lift", "bring", "carry", "please",
}


def _tokens(text: str) -> list:
    """Content tokens of a product name/phrase, lowercased, stopwords + <=2-char noise dropped."""
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [t for t in toks if len(t) > 2 and t not in _STOPWORDS]


def name_matches(state: dict, keywords) -> bool:
    """True iff any keyword appears in the agent's hovered/gripped object blob. Re-homed here from
    eval_pickup (its single caller now imports it) so the pickup predicate and the 6.4 eval share ONE
    implementation. Same substring-containment semantics as before - deliberately loose; the phase6.3
    plan flags tightening as a 6.4-audit follow-up, not a guess to make now.

    The blob includes `gripped_name` - the name the grab tool reported AT THE GRIP - because the live
    `<side>HoveredObject` fields clear to 'null' once the hand retracts from the shelf (a measured
    false-negative: the item is still held, but hovered no longer names it). `gripped_name` is the
    durable record run_leg keeps while a hand is gripping; it is '' for callers (eval_pickup) that
    don't set it, so this stays backward-safe."""
    fields = [state.get("leftHoveredObject") or "", state.get("rightHoveredObject") or "",
              state.get("gripped_name") or ""]
    blob = " ".join(str(f) for f in fields).lower()
    return any(str(k).lower() in blob for k in keywords)


def name_overlap(state: dict, target: str) -> bool:
    """True iff the gripped/hovered object name overlaps the decomposer's free-text `target`. Bridges
    a natural-language target ('Piattos (green)') to name_matches by tokenizing it into content
    keywords. An empty/degenerate target (no content tokens) returns True - we can't ground it, so we
    don't block on it (the grip itself is still required by the pickup predicate)."""
    kws = _tokens(target)
    if not kws:
        return True
    return name_matches(state, kws)


# ---------------------------------------------------------------------------
# Completion predicates - each: (subtask_dict, state[, final_text]) -> (granted: bool, reason: str)
# ---------------------------------------------------------------------------

def _gripping(state: dict) -> bool:
    return bool(state.get("leftGrippedState") or state.get("rightGrippedState"))


# Pre-6.3 keyword sets, preserved verbatim for the `unknown` fallback so an untypeable subtask keeps
# exactly the old behaviour (no better, no worse - just no longer the ONLY guard).
_PICKUP_KW = ("pick up", "grab", "get", "take", "lift")
_DROP_KW = ("drop", "place", "put down", "set down", "release", "leave")


def predicate_pickup(sub: dict, state: dict) -> tuple:
    """Grant iff a hand grips AND (when a target names a product) the gripped/hovered name overlaps
    it. Refuses the exact failure the old drop-agnostic guard could not see on a paraphrase, and the
    wrong-item grab a bare grip-check would pass."""
    if not _gripping(state):
        return False, "pickup not complete: no hand is gripping anything yet"
    target = sub.get("target") or ""
    if target and not name_overlap(state, target):
        held = (state.get("gripped_name") or state.get("leftHoveredObject")
                or state.get("rightHoveredObject"))
        return False, (f"gripping, but the held item ({held!r}) does not match target {target!r} - "
                       "grab the right item")
    return True, "pickup complete: gripping the target item"


def predicate_checkout(sub: dict, state: dict) -> tuple:
    """Grant iff the checkout macro reported scanned AND placed (both MEASURED - OCR receipt delta /
    released-under-a-placeable-verdict) and no hand is still gripping. Reads `last_checkout` (the
    macro's own result dict surfaced into state); it does NOT re-derive the verdict."""
    lc = state.get("last_checkout")
    if not lc:
        return False, ("checkout not attempted yet: dispatch the checkout tool "
                       "(it drives to the counter, scans, and bags in one call)")
    if not lc.get("scanned"):
        return False, f"checkout incomplete: item not scanned - {lc.get('reason', 'retry / re-align')}"
    if not lc.get("placed"):
        return False, f"checkout incomplete: scanned but not bagged - {lc.get('reason', 'retry the bag')}"
    if _gripping(state):
        return False, "checkout reported placed but a hand is still gripping - inconsistent, retry"
    return True, "checkout complete: item scanned and bagged (both measured)"


def predicate_goto(sub: dict, state: dict) -> tuple:
    """Grant iff the agent's nearest checkpoint is (one of) the leg's resolved target checkpoint(s).
    Both are injected at leg start (Phase 6.3 #1): `sub['target_checkpoint']` - resolved ONCE at plan
    time by the map resolver, an int OR a list (a product/area can live on several shelves) - and
    `state['nearest_checkpoint']` (localised each step from the live pose + the map). If either is
    unavailable the predicate cannot verify, so it grants on the VLM's judgment and says so - honest
    [unverified], never a silent strictness the wiring can't feed."""
    target_cp = sub.get("target_checkpoint")
    near_cp = state.get("nearest_checkpoint")
    if target_cp is None or near_cp is None:
        return True, (f"goto granted [unverified]: checkpoint info unavailable "
                      f"(target={target_cp}, nearest={near_cp})")
    targets = set(target_cp) if isinstance(target_cp, (list, tuple, set)) else {target_cp}
    if not targets:
        return True, "goto granted [unverified]: target resolved to no checkpoint"
    if near_cp in targets:
        return True, f"goto complete: at checkpoint {near_cp} (target {sorted(targets)})"
    return False, f"goto not complete: nearest checkpoint is {near_cp}, target is {sorted(targets)}"


def predicate_compare(sub: dict, state: dict, final_text: str = "") -> tuple:
    """Grant iff (a) the agent's final output names a choice from `targets`, AND (b) it actually went
    and LOOKED at both candidates - it visited each candidate's checkpoint. The CHOICE is checked in
    code; the stated observation is only LOGGED (6.4 audits it as camera-grounded) - never judged here
    (no verifier LLM - phase6.3 plan).

    The visited-both check (Phase 6.3 #4) turns "decide from the camera, not the product index" from a
    prompt instruction into a deterministic gate: `sub['candidate_sets']` (one resolved checkpoint list
    per target, from plan_legs) must each intersect `state['visited_checkpoints']` (the cumulative
    task-level visit trace). It is DEFENSIVE - if the candidates didn't resolve (no `candidate_sets`),
    the visit half is skipped and flagged [unverified] rather than wrongly blocking; a physical-
    inspection eval should ensure they resolve. With no declared targets we can't check the choice
    either, so we grant [unverified]."""
    targets = sub.get("targets") or []
    if not targets:
        return True, "compare granted [unverified]: no candidate targets to check the choice against"
    blob = str(final_text or "").lower()
    named = next((cand for cand in targets if any(tok in blob for tok in _tokens(cand))), None)
    if named is None:
        return False, ("compare not complete: final output must name your choice from "
                       f"{targets} with what you observed")
    # (b) visited-both, when the candidates resolved to checkpoints.
    cand_sets = sub.get("candidate_sets")
    if cand_sets:
        visited = state.get("visited_checkpoints") or set()
        visited = set(visited)
        unseen = [s for s in cand_sets if s and not (set(s) & visited)]
        if unseen:
            return False, (f"compare not complete: chose {named!r} but never stood at "
                           f"candidate checkpoint(s) {unseen} - go LOOK at both before deciding")
        return True, f"compare complete: chose {named!r} after visiting both candidates"
    return True, f"compare complete: chose {named!r} [unverified: candidates had no resolved checkpoints]"


def predicate_unknown(sub: dict, state: dict) -> tuple:
    """The pre-6.3 keyword guards, preserved verbatim, for an untypeable subtask. Grep the subtask
    TEXT for pickup/drop intent; refuse a pickup with nothing gripped or a drop with a hand still
    gripping. Deliberately no better than before - it is the fallback, and it is logged as `untyped`
    so 6.4 can see how often it fires."""
    text = str(sub.get("text") or "").lower()
    grip = _gripping(state)
    is_pickup = any(kw in text for kw in _PICKUP_KW)
    is_drop = any(kw in text for kw in _DROP_KW)
    if is_pickup and not grip:
        return False, "STOP blocked (untyped): pick-up task but nothing is gripped"
    if is_drop and grip:
        return False, "STOP blocked (untyped): drop task but a hand is still gripping"
    return True, "halt granted (untyped): no keyword guard blocks it"


def completion_predicate(sub: dict, state: dict, final_text: str = "") -> tuple:
    """Dispatch a subtask's halt request to its typed predicate. Returns (granted, reason). Any
    unrecognized type routes to the `unknown` keyword fallback, so this never raises on a malformed
    subtask."""
    t = (sub or {}).get("type")
    if t == "pickup":
        return predicate_pickup(sub, state)
    if t == "checkout":
        return predicate_checkout(sub, state)
    if t == "goto":
        return predicate_goto(sub, state)
    if t == "compare":
        return predicate_compare(sub, state, final_text)
    return predicate_unknown(sub, state)
