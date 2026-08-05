"""Offline unit tests for subtask_planning.py - the Phase 6.3 plan-time map planning
(plan_legs #1, order_legs #3). Pure logic over a FAKE StoreMap + a stubbed resolver, so no sim, no
network, no model stack (importing subtask_agents would pull torch; the logic lives in a light module
exactly so this runs in ms).

    uv run pytest validation/tests/agent/test_subtask_planning.py   # or: pytest ...
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from orchestrator import subtask_planning as P
from orchestrator.subtask_planning import plan_legs, order_legs, _resolve_location


class FakeSM:
    """Just the StoreMap surface planning touches: by_id, counter_checkpoint, hops, nearest_checkpoint.
    hops is a toy metric (|a-b|) so 'nearer' is deterministic and readable in the ordering test."""
    def __init__(self):
        self.by_id = {c: {} for c in (10, 20, 30, 32, 40, 54)}
        self._counter = 54

    def counter_checkpoint(self):
        return self._counter

    def hops(self, a, b):
        if a not in self.by_id or b not in self.by_id:
            return None
        return abs(a - b)

    def nearest_checkpoint(self, xz):
        return 10


# ---- _resolve_location deterministic branches (no LLM) --------------------

def test_resolve_location_counter_no_llm():
    cps, calls = _resolve_location(FakeSM(), None, "the checkout counter")
    assert cps == [54] and calls == 0


def test_resolve_location_till_synonym():
    cps, calls = _resolve_location(FakeSM(), None, "leave it at the till")
    assert cps == [54] and calls == 0


def test_resolve_location_literal_cp():
    cps, calls = _resolve_location(FakeSM(), None, "Checkpoint 32")
    assert cps == [32] and calls == 0


def test_resolve_location_literal_cp_not_in_map():
    cps, calls = _resolve_location(FakeSM(), None, "cp999")
    assert cps == [] and calls == 0


# ---- plan_legs enrichment (stub the resolver so no LLM/sim) ----------------

def _stub_resolver(mapping):
    """Return a _resolve_target replacement that maps a target substring -> candidate list."""
    def fake(sm, resolve_call, target):
        for key, cands in mapping.items():
            if key.lower() in str(target).lower():
                return {"candidates": list(cands), "target_name": key, "tier": "name"}
        return {"candidates": [], "target_name": str(target), "tier": "unresolvable"}
    return fake


def test_plan_legs_pickup_and_checkout(monkeypatch=None):
    P._resolve_target = _stub_resolver({"Piattos": [30, 32], "Coke": [20]})
    legs = [{"type": "pickup", "target": "green Piattos", "text": "Pick up the green Piattos."},
            {"type": "checkout", "text": "scan and bag it"}]
    out, n = plan_legs(FakeSM(), None, legs)
    assert n == 1                                   # one resolver call (pickup only; counter is free)
    assert out[0]["candidates"] == [30, 32]
    assert out[0]["target_checkpoint"] == [30, 32]  # list: any shelf counts
    assert out[0]["feasible"] is True
    # Checkout seeds the COUNTER checkpoint (no LLM) so a navigation-mode step drives to the counter
    # instead of the runtime resolver re-resolving the carried product's name back to its shelf.
    assert out[1]["candidates"] == [54]
    assert out[1]["feasible"] is True


def test_plan_legs_infeasible_when_unresolved():
    P._resolve_target = _stub_resolver({"Piattos": [30]})
    legs = [{"type": "pickup", "target": "unobtainium", "text": "Pick up the unobtainium."}]
    out, n = plan_legs(FakeSM(), None, legs)
    assert out[0]["candidates"] == [] and out[0]["feasible"] is False


def test_plan_legs_compare_candidate_sets():
    P._resolve_target = _stub_resolver({"Pik Nik large": [30], "Pik Nik small": [32]})
    legs = [{"type": "compare", "targets": ["Pik Nik large", "Pik Nik small"], "criterion": "size",
             "text": "compare them"}]
    out, n = plan_legs(FakeSM(), None, legs)
    assert n == 2                                   # one resolver call per target
    assert out[0]["candidate_sets"] == [[30], [32]]
    assert sorted(out[0]["candidates"]) == [30, 32]  # union, for nav seeding
    assert out[0]["feasible"] is True


def test_plan_legs_goto_literal_no_llm():
    legs = [{"type": "goto", "location": "Checkpoint 40", "text": "go to cp40"}]
    out, n = plan_legs(FakeSM(), None, legs)
    assert n == 0                                    # literal cp needs no resolver
    assert out[0]["candidates"] == [40] and out[0]["target_checkpoint"] == [40]


def test_plan_legs_inspect_is_feasible_without_checkpoint_resolution():
    leg = {"type": "inspect", "query": "How many Piattos?", "text": "Count the Piattos."}
    out, n = plan_legs(FakeSM(), None, [leg])
    assert n == 0
    assert out[0]["feasible"] is True
    assert "candidates" not in out[0]
    assert out[0]["text"] == "Count the Piattos."


# ---- order_legs (#3): reorder clean pickup/checkout pairs, else no-op ------

def _pair(pickup_cands):
    return [{"type": "pickup", "candidates": pickup_cands, "text": "pick"},
            {"type": "checkout", "text": "checkout"}]


def test_order_legs_reorders_nearest_first():
    sm = FakeSM()   # start_cp=10; hops=|a-b|; nearer pickup goes first
    legs = _pair([40]) + _pair([20])         # far pair first in the plan
    out = order_legs(sm, legs, start_cp=10)
    # from 10, cand 20 (hops 10) beats cand 40 (hops 30) -> the [20] pair leads
    assert out[0]["candidates"] == [20] and out[1]["type"] == "checkout"
    assert out[2]["candidates"] == [40] and out[3]["type"] == "checkout"


def test_order_legs_noop_when_goto_present():
    sm = FakeSM()
    legs = [{"type": "goto", "candidates": [30], "text": "go"}] + _pair([20]) + _pair([40])
    out = order_legs(sm, legs, start_cp=10)
    assert out is legs                       # dependency risk -> untouched (same object)


def test_order_legs_noop_single_pair():
    sm = FakeSM()
    legs = _pair([20])
    assert order_legs(sm, legs, start_cp=10) is legs


def test_order_legs_noop_odd_length():
    sm = FakeSM()
    legs = _pair([20]) + [{"type": "pickup", "candidates": [40], "text": "pick"}]
    assert order_legs(sm, legs, start_cp=10) is legs


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
