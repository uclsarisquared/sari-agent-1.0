"""subtask_planning.py - Phase 6.3 plan-time map planning for the long-horizon orchestrator.

Lightweight and sim-free (like subtask_completion.py): no torch/agent/env imports, so the planning
logic is offline-unit-testable in milliseconds (importing subtask_agents pulls the whole model stack).
The map resolver (locate_task) is imported LAZILY inside the resolve helpers, and every function takes
the StoreMap `sm` as a parameter, so this module itself pulls nothing heavy at import.

What lives here (Phase 6.3 #1/#3):
  - plan_legs():  resolve each pickup/goto/compare target to checkpoint(s) ONCE, at plan time, so the
                  completion predicates get real checkpoints, a doomed plan is caught early, and the
                  graph navigator can reuse the candidates instead of re-resolving at runtime.
  - order_legs(): reorder INDEPENDENT (pickup -> checkout) pairs nearest-first by graph hops.

Location naming stays SEMANTIC end to end: the decomposer emits names ("the checkout counter", product
names), never checkpoint ids; the resolver owns name->checkpoint. One source of truth for spatial facts.
"""

import re

# The frozen-map spawn corner (mirrors eval_pickup.return_to_start's start point) - the node the
# leg-ordering greedy measures the first pair's hops from.
SPAWN_XZ = (-3.0, -5.0)

_CP_RE = re.compile(r'(?:checkpoint|cp)\s*#?\s*(\d+)', re.IGNORECASE)


def make_resolve_call(backend: str = "qwen"):
    """A resolver callable matching agent._graph_navigate's shape: (system, prompt, schema, images)
    -> (dict, envelope). Default qwen (the agent's default backend); claude-cli is the annotator path,
    not the agent's, so only pass it deliberately."""
    from nav import locate_task
    if backend == "claude-cli":
        return lambda s, p, sc, im=(): locate_task.claude_json(s, p, sc, im)
    return lambda s, p, sc, im=(): locate_task.qwen_json(s, p, sc, im)


def _resolve_target(sm, resolve_call, target):
    """Resolve a product/target phrase to candidate checkpoints via the map resolver (one LLM call).
    Returns {'candidates': [ids], 'target_name': str, 'tier': str}; empty candidates on any failure
    (honest - a doomed leg the orchestrator can flag)."""
    from nav import locate_task
    try:
        resolution, _ = locate_task.resolve(resolve_call, sm, str(target))
    except Exception as e:  # noqa: BLE001 - a dead resolver shouldn't crash planning
        print(f"[PLAN] resolve failed for {target!r}: {type(e).__name__}: {e}")
        return {"candidates": [], "target_name": str(target), "tier": "unresolvable"}
    cands = [c for c in (resolution.get("candidates") or []) if c in sm.by_id]
    return {"candidates": cands, "target_name": resolution.get("target_name") or str(target),
            "tier": resolution.get("tier", "unresolvable")}


def _resolve_location(sm, resolve_call, text):
    """Resolve a goto LOCATION to checkpoint(s). The counter is a known singleton (no LLM); an
    explicit 'Checkpoint N'/'cpN' is literal; anything else routes through the product resolver.
    Returns (checkpoints, n_llm_calls)."""
    low = (text or "").lower()
    if any(w in low for w in ("counter", "checkout", "self-checkout", "till", "register", "cashier")):
        cp = sm.counter_checkpoint()
        return ([cp] if cp is not None else []), 0
    m = _CP_RE.search(text or "")
    if m:
        cid = int(m.group(1))
        return ([cid] if cid in sm.by_id else []), 0
    res = _resolve_target(sm, resolve_call, text)
    return res["candidates"], 1


def plan_legs(sm, resolve_call, legs):
    """Phase 6.3 #1: enrich each leg with map-resolved checkpoints, ONCE, at plan time (before any sim
    motion). Returns (enriched_legs, n_resolver_calls). Attaches, per type:
      - pickup:  `candidates` (+ `target_name`); `target_checkpoint` = the candidate list (a product can
                 sit on several shelves, so being at ANY counts for a preceding goto-style check).
      - goto:    `candidates` + `target_checkpoint` (the location's checkpoint(s)).
      - compare: `candidate_sets` (one resolved checkpoint list PER target) so the compare predicate can
                 require the agent visited BOTH before deciding (#4); `candidates` = their union (for nav
                 seeding).
      - checkout: `candidates` = [the counter checkpoint] (known singleton, NO resolver call). The
                 macro drives itself, but the router may pick *navigation* mode first - and an UNSEEDED
                 leg would make the runtime resolver re-resolve the leg's augmented text, which names
                 the CARRIED product, driving the agent back to the shelf it just left. Seeding pins
                 navigation-mode steps to the counter.
      - inspect: no plan-time resolution. It remains feasible and its natural-language instruction
                 is available to the runtime navigator, which can resolve what it needs from view.
      - unknown: nothing (no target to resolve; the runtime resolver is its documented fallback).
    Also sets `feasible=False` on a leg whose target resolved to zero candidates - a doomed plan the
    orchestrator surfaces instead of discovering twenty sim-steps in."""
    n_calls = 0
    for leg in legs:
        t = leg.get("type")
        if t == "checkout":
            counter = sm.counter_checkpoint()
            leg["candidates"] = [counter] if counter is not None else []
            leg["feasible"] = counter is not None
        elif t == "pickup":
            res = _resolve_target(sm, resolve_call, leg.get("target") or leg.get("text"))
            n_calls += 1
            leg["candidates"] = res["candidates"]
            leg["target_checkpoint"] = res["candidates"]
            leg["target_name"] = res["target_name"]
            leg["feasible"] = bool(res["candidates"])
        elif t == "goto":
            cps, c = _resolve_location(sm, resolve_call, leg.get("location") or leg.get("text"))
            n_calls += c
            leg["candidates"] = cps
            leg["target_checkpoint"] = cps
            leg["feasible"] = bool(cps)
        elif t == "compare":
            sets = []
            for name in (leg.get("targets") or []):
                r = _resolve_target(sm, resolve_call, name)
                n_calls += 1
                sets.append(r["candidates"])
            leg["candidate_sets"] = sets
            leg["candidates"] = [c for s in sets for c in s]  # union, for nav seeding only
            leg["feasible"] = all(bool(s) for s in sets) if sets else True
        elif t == "inspect":
            leg["feasible"] = True
        else:
            leg.setdefault("feasible", True)  # unknown: nothing to resolve
    return legs, n_calls


def order_legs(sm, legs, start_cp):
    """Phase 6.3 #3: reorder INDEPENDENT (pickup -> checkout) pairs nearest-first by graph hops, so a
    multi-fetch task doesn't zig-zag the store. CONSERVATIVE by design: it only reorders when the plan
    is a clean, even sequence of pickup/checkout pairs (family 2); ANY goto/compare/unknown leg or an
    odd structure means legs may carry ordering dependencies we must not break, so the plan is returned
    UNCHANGED. Greedy nearest-neighbour: each checkout ends at the counter, so after the first pair the
    'from' node is the counter; the first pair is chosen from `start_cp`. Pure code over the map - a
    spatial judgment, which is code's job, never the resolver's."""
    if not legs or len(legs) % 2 != 0:
        return legs
    pairs = []
    for i in range(0, len(legs), 2):
        a, b = legs[i], legs[i + 1]
        if a.get("type") != "pickup" or b.get("type") != "checkout":
            return legs
        pairs.append((a, b))
    if len(pairs) < 2:
        return legs
    counter = sm.counter_checkpoint()

    def pair_dist(pair, frm):
        ds = [sm.hops(frm, c) for c in (pair[0].get("candidates") or []) if c in sm.by_id]
        ds = [d for d in ds if d is not None]
        return min(ds) if ds else 10 ** 6

    remaining, ordered, frm = list(pairs), [], start_cp
    while remaining:
        nxt = min(remaining, key=lambda p: pair_dist(p, frm))
        ordered.append(nxt)
        remaining.remove(nxt)
        frm = counter if counter is not None else frm
    out = []
    for a, b in ordered:
        out.extend([a, b])
    return out
