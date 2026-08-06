"""Phase 6.3 - OFFLINE A/B of the task decomposer: OLD free-string prompt vs NEW typed prompt, on a
fixed battery (decompose_battery.json). One LLM call per prompt per arm - NO sim. The plan's rule:
A/B the decomposer prompt on identical inputs and EYEBALL the JSON before wiring the typed version
into subtask_agents.

    python validation/evals/decomposition.py                 # all prompts, both arms
    python validation/evals/decomposition.py --arm new
    python validation/evals/decomposition.py --only f3-size

It needs the configured OpenAI-compatible endpoint (same reasoner the orchestrator uses) -
importing agent/subtask_agents pulls the model stack, so first import is slow; the LLM calls need
the endpoint reachable. Writes
validation/artifacts/evals/decomposition/results.json and prints a per-prompt diff + a summary of how
many typed outputs parse cleanly (no `unknown` degradation) per family.

The OLD prompt is embedded here VERBATIM (copied from the pre-6.3 subtask_agents.decompose_task) so
this A/B stays valid as a baseline AFTER decompose_task is switched to the typed prompt.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_core import token_meter  # lightweight - no model stack
from orchestrator.subtask_completion import TYPED_DECOMPOSER_SYSTEM, parse_decomposition, SUBTASK_TYPES

_REPO_ROOT = os.path.dirname(_ROOT)
_BATTERY = os.path.join(
    _REPO_ROOT, "validation", "fixtures", "decomposition", "decompose_battery.json")
_OUT = os.path.join(
    _REPO_ROOT, "validation", "artifacts", "evals", "decomposition", "results.json")

# --- OLD arm: the pre-6.3 free-string decomposer system prompt, verbatim -----------------------
OLD_FREESTRING_SYSTEM = (
    "You are a task planner for an Embodied AI Agent in a 3D convenience "
    "store simulation. The agent can navigate, locate items on shelves, "
    "pick them up, carry them, and bring them to locations like the checkout counter. "
    "Given a complex multi-step task, decompose it into a short ordered list "
    "of simple, self-contained subtasks. Each subtask should:\n"
    "  - Be completable in a single continuous agent run.\n"
    "  - End in a clear, verifiable physical state change.\n"
    "  - Reference what the agent is currently holding when relevant.\n"
    "  - Name locations only as the task or the store memory names them (e.g. 'Checkpoint 32', "
    "'the checkout counter'). Never invent shelf numbers or location names.\n"
    "Return ONLY a JSON array of subtask strings — no other text.\n\n"
    "Example input: \"pick up the milk and bring it to the counter\"\n"
    "Example output: "
    "[\"Pick up the milk.\", "
    "\"Carry the held milk to the checkout counter and place it down.\"]"
    "\n\nIf a task is already simple (e.g. 'pick up the milk'), just return it as a single-item array."
)


def _run_arm(client, llm_call, system, task):
    # `role` bills the call to the decomposer row of the per-role token accounting; both arms ARE the
    # decomposer, so both bill there (the harness itself never reads the meter).
    try:
        raw = llm_call(client, system, f"Task: {task}", token_meter.ROLE_DECOMPOSER)
        return {"raw": raw, "error": None}
    except Exception as e:  # noqa: BLE001 - a dead server / timeout shouldn't abort the whole battery
        return {"raw": None, "error": f"{type(e).__name__}: {e}"}


def _summarize_typed(parsed):
    """Flags for eyeballing: did any element degrade to `unknown`, and the type sequence."""
    types = [s.get("type") for s in parsed]
    return {"types": types, "has_unknown": any(t == "unknown" for t in types),
            "n": len(parsed)}


def _inspect_adoption(prompt, parsed):
    """Report the explicit inspection battery expectation without guessing for older rows."""
    expected = prompt.get("expected_types")
    if expected is None:
        return None
    actual = [s.get("type") for s in parsed]
    return {"expected_types": expected, "actual_types": actual, "passed": actual == expected,
            "became_inspect_not_unknown": "inspect" in actual and "unknown" not in actual}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["old", "new", "both"], default="both")
    ap.add_argument("--only", default=None, help="run a single battery id")
    args = ap.parse_args()

    with open(_BATTERY, encoding="utf-8") as fh:
        battery = json.load(fh)
    prompts = battery["prompts"]
    if args.only:
        prompts = [p for p in prompts if p["id"] == args.only]
        if not prompts:
            print(f"no battery prompt with id {args.only!r}")
            return 1

    # Lazy heavy import - only when actually calling the LLM (keeps --help instant).
    from orchestrator.orchestrator_llm import _llm_call, _llm_client
    client = _llm_client()

    results = []
    clean_typed = 0
    for p in prompts:
        print("\n" + "=" * 88)
        print(f"[{p['id']}]  family={p['family']}")
        print(f"  PROMPT: {p['prompt']}")
        print(f"  LOOKING FOR: {p['looking_for']}")
        row = {"id": p["id"], "family": p["family"], "prompt": p["prompt"]}

        if args.arm in ("old", "both"):
            old = _run_arm(client, _llm_call, OLD_FREESTRING_SYSTEM, p["prompt"])
            row["old"] = old
            print("\n  --- OLD (free strings) ---")
            print("  " + (old["error"] or (old["raw"] or "").strip().replace("\n", "\n  ")))

        if args.arm in ("new", "both"):
            new = _run_arm(client, _llm_call, TYPED_DECOMPOSER_SYSTEM, p["prompt"])
            row["new"] = new
            if new["raw"] is not None:
                parsed = parse_decomposition(new["raw"], p["prompt"])
                summ = _summarize_typed(parsed)
                adoption = _inspect_adoption(p, parsed)
                row["new_parsed"] = parsed
                row["new_summary"] = summ
                if adoption is not None:
                    row["inspect_adoption"] = adoption
                clean = not summ["has_unknown"]
                clean_typed += int(clean)
                print("\n  --- NEW (typed) ---")
                print("  " + (new["raw"] or "").strip().replace("\n", "\n  "))
                print(f"  PARSED types: {summ['types']}  "
                      f"{'[CLEAN]' if clean else '[HAS UNKNOWN - inspect]'}")
                if adoption is not None:
                    print(f"  INSPECT ADOPTION: {'PASS' if adoption['passed'] else 'FAIL'} "
                          f"expected={adoption['expected_types']} actual={adoption['actual_types']}")
            else:
                print("\n  --- NEW (typed) ---")
                print("  " + new["error"])
        results.append(row)

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump({"battery": os.path.basename(_BATTERY), "results": results}, fh, indent=2)

    if args.arm in ("new", "both"):
        n = len(prompts)
        print("\n" + "=" * 88)
        print(f"SUMMARY: {clean_typed}/{n} typed decompositions parsed with NO `unknown` degradation.")
        print(f"Vocabulary in play: {SUBTASK_TYPES}. Full raw+parsed dump: {_OUT}")
        print("Eyeball the type sequences above against LOOKING FOR before wiring the typed prompt in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
