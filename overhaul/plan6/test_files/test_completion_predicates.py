"""Offline unit tests for the Phase 6.3 completion predicates + typed-subtask parser
(subtask_completion.py). Pure functions over plain dicts - no sim, no network, no model stack, so
this loads and runs in milliseconds (the whole reason the logic lives in a lightweight module rather
than in subtask_agents, which pulls torch/agent on import).

What each block pins down, in the plan's words: the VLM's STOP is a REQUEST code grants or refuses.
The load-bearing cases are the three the pre-6.3 keyword guards got WRONG and this replaces:
  - a PARAPHRASED pickup ("obtain the...") that the keyword guard never guarded, halting with an
    empty hand;
  - a drop/checkout declared DONE while the item was released in the wrong place (here: not scanned,
    or not bagged);
  - a WRONG-ITEM grab that a bare grip-check would pass.

    python plan6/test_files/test_completion_predicates.py   # or: pytest plan6/test_files/test_completion_predicates.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # overhaul/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from subtask_completion import (
    parse_decomposition,
    completion_predicate,
    name_overlap,
    SUBTASK_TYPES,
    HALT_REFUSAL_CAP,
)


# --- state helpers ---------------------------------------------------------

def _state(**over):
    s = {
        "leftGrippedState": False, "rightGrippedState": False,
        "leftHoveredObject": "None", "rightHoveredObject": "None",
        "last_checkout": None, "nearest_checkpoint": None,
    }
    s.update(over)
    return s


def _granted(sub, state, final_text=""):
    return completion_predicate(sub, state, final_text)[0]


# --- parse_decomposition ---------------------------------------------------

def test_parse_typed_array():
    raw = ('[{"type": "pickup", "target": "green Piattos", "text": "Pick up the green Piattos."}, '
           '{"type": "checkout", "text": "Scan and bag it."}]')
    out = parse_decomposition(raw, "orig")
    assert [s["type"] for s in out] == ["pickup", "checkout"]
    assert out[0]["target"] == "green Piattos"
    assert all("text" in s for s in out)


def test_parse_wrapped_in_prose_and_fence():
    raw = 'Sure! Here is the plan:\n```json\n[{"type":"goto","location":"cp32","text":"Go to cp32."}]\n```\n'
    out = parse_decomposition(raw, "orig")
    assert out[0]["type"] == "goto" and out[0]["location"] == "cp32"


def test_parse_unknown_type_degrades_but_keeps_text():
    raw = '[{"type": "frobnicate", "text": "Do the thing."}]'
    out = parse_decomposition(raw, "orig")
    assert out[0]["type"] == "unknown"
    assert out[0]["text"] == "Do the thing."   # instruction preserved, only the type degraded


def test_parse_legacy_bare_strings():
    raw = '["Pick up the milk.", "Carry it to the counter."]'
    out = parse_decomposition(raw, "orig")
    assert all(s["type"] == "unknown" for s in out)
    assert out[0]["text"] == "Pick up the milk."


def test_parse_garbage_falls_back_to_single_unknown():
    for raw in ["not json at all", "", "[unterminated", "{}", "[]"]:
        out = parse_decomposition(raw, "ORIGINAL TASK")
        assert out == [{"type": "unknown", "text": "ORIGINAL TASK"}], raw


def test_parse_never_emits_type_outside_vocab():
    raw = '[{"type":"pickup","text":"a"},{"type":"weird","text":"b"},"c"]'
    out = parse_decomposition(raw, "orig")
    assert all(s["type"] in SUBTASK_TYPES or s["type"] == "unknown" for s in out)


# --- name_overlap ----------------------------------------------------------

def test_name_overlap_matches_product_token():
    st = _state(leftHoveredObject="PIATTOS_CHEESE_40G")
    assert name_overlap(st, "Piattos (green)") is True


def test_name_overlap_rejects_unrelated():
    st = _state(leftHoveredObject="COKE_ZERO_330")
    assert name_overlap(st, "Piattos") is False


def test_name_overlap_empty_target_is_permissive():
    # No content tokens to ground - don't block on it (the grip is still required upstream).
    assert name_overlap(_state(leftHoveredObject="ANYTHING"), "the") is True


# --- pickup predicate ------------------------------------------------------

def test_pickup_granted_on_matching_grip():
    st = _state(leftGrippedState=True, leftHoveredObject="PIATTOS_CHEESE_40G")
    assert _granted({"type": "pickup", "target": "Piattos"}, st) is True


def test_pickup_granted_via_gripped_name_when_hovered_cleared():
    # The measured live failure (2026-07-23): after the hand retracts, hovered clears to 'null' even
    # though the item is still held - the STOP was wrongly refused. The durable `gripped_name`
    # (captured AT the grip from the grab tool's result) must carry the match.
    st = _state(leftGrippedState=True, leftHoveredObject="null", rightHoveredObject="null",
                gripped_name="JACK_AND_JILL_PIATTOS_SOUR_CREAM_FLAVORED_POTATO_40G")
    assert _granted({"type": "pickup", "target": "green Piattos"}, st) is True


def test_pickup_wrong_item_still_refused_with_gripped_name():
    # The durable record must not LOOSEN the check: a remembered wrong item is still refused,
    # and the refusal names what is actually held.
    st = _state(leftGrippedState=True, leftHoveredObject="null",
                gripped_name="COKE_ZERO_330")
    ok, reason = completion_predicate({"type": "pickup", "target": "Piattos"}, st)
    assert ok is False and "COKE_ZERO_330" in reason


def test_pickup_refused_empty_hand():
    st = _state(leftGrippedState=False)
    assert _granted({"type": "pickup", "target": "Piattos"}, st) is False


def test_pickup_refused_wrong_item():
    # The bare-grip-check failure: gripping SOMETHING, but not the target.
    st = _state(leftGrippedState=True, leftHoveredObject="COKE_ZERO_330")
    ok, reason = completion_predicate({"type": "pickup", "target": "Piattos"}, st)
    assert ok is False and "does not match" in reason


def test_pickup_no_target_grants_on_any_grip():
    st = _state(rightGrippedState=True, rightHoveredObject="WHATEVER")
    assert _granted({"type": "pickup"}, st) is True


# --- checkout predicate ----------------------------------------------------

def test_checkout_granted_when_scanned_and_placed():
    st = _state(last_checkout={"scanned": True, "placed": True, "reason": "ok"})
    assert _granted({"type": "checkout"}, st) is True


def test_checkout_refused_before_any_attempt():
    ok, reason = completion_predicate({"type": "checkout"}, _state())
    assert ok is False and "not attempted" in reason


def test_checkout_refused_scanned_but_not_bagged():
    # The "declared done in the wrong place" failure: scan fired, item never landed in the tray.
    st = _state(last_checkout={"scanned": True, "placed": False, "reason": "bag missed"})
    ok, reason = completion_predicate({"type": "checkout"}, st)
    assert ok is False and "not bagged" in reason


def test_checkout_refused_not_scanned():
    st = _state(last_checkout={"scanned": False, "placed": False, "reason": "no scan"})
    assert _granted({"type": "checkout"}, st) is False


def test_checkout_refused_if_still_gripping():
    st = _state(leftGrippedState=True,
                last_checkout={"scanned": True, "placed": True, "reason": "ok"})
    assert _granted({"type": "checkout"}, st) is False


# --- goto predicate --------------------------------------------------------

def test_goto_granted_at_target():
    st = _state(nearest_checkpoint="cp32")
    assert _granted({"type": "goto", "location": "cp32", "target_checkpoint": "cp32"}, st) is True


def test_goto_refused_elsewhere():
    st = _state(nearest_checkpoint="cp10")
    assert _granted({"type": "goto", "target_checkpoint": "cp32"}, st) is False


def test_goto_unverified_grants_when_checkpoint_unknown():
    # Wiring couldn't feed a resolved/near checkpoint - grant on the VLM but say [unverified].
    ok, reason = completion_predicate({"type": "goto", "location": "the counter"}, _state())
    assert ok is True and "unverified" in reason


def test_goto_granted_when_nearest_in_target_list():
    # #1: a product/area resolves to several candidate checkpoints - being at ANY of them counts.
    st = _state(nearest_checkpoint=45)
    assert _granted({"type": "goto", "target_checkpoint": [32, 45, 52]}, st) is True


def test_goto_refused_when_nearest_not_in_target_list():
    st = _state(nearest_checkpoint=10)
    assert _granted({"type": "goto", "target_checkpoint": [32, 45, 52]}, st) is False


# --- compare predicate -----------------------------------------------------

def test_compare_granted_when_choice_named():
    sub = {"type": "compare", "targets": ["Pik Nik large", "Pik Nik small"], "criterion": "size"}
    ok = _granted(sub, _state(), final_text="I choose the large Pik Nik - its bag is visibly wider.")
    assert ok is True


def test_compare_refused_when_no_choice():
    sub = {"type": "compare", "targets": ["Pik Nik large", "Pik Nik small"]}
    ok, reason = completion_predicate(sub, _state(), final_text="Both are on the shelf.")
    assert ok is False and "name your choice" in reason


def test_compare_refused_when_chose_but_didnt_visit_both():
    # #4: named a choice but never stood at candidate B's checkpoint - refuse (didn't LOOK).
    sub = {"type": "compare", "targets": ["Pik Nik large", "Pik Nik small"],
           "candidate_sets": [[30], [31]]}
    st = _state(visited_checkpoints={30})   # only visited A's shelf
    ok, reason = completion_predicate(sub, st, final_text="I choose the large one.")
    assert ok is False and "never stood at" in reason


def test_compare_granted_when_chose_and_visited_both():
    sub = {"type": "compare", "targets": ["Pik Nik large", "Pik Nik small"],
           "candidate_sets": [[30], [31]]}
    st = _state(visited_checkpoints={30, 31, 54})
    assert _granted(sub, st, final_text="The large one - its bag is visibly wider.") is True


def test_compare_visit_check_defensive_when_unresolved():
    # No candidate_sets (resolve failed) - don't wrongly block; grant on the choice, flag unverified.
    sub = {"type": "compare", "targets": ["Pik Nik large", "Pik Nik small"]}
    ok, reason = completion_predicate(sub, _state(), final_text="I pick the large Pik Nik.")
    assert ok is True and "unverified" in reason


# --- unknown fallback (pre-6.3 keyword guards, preserved) ------------------

def test_unknown_paraphrased_pickup_still_guarded_by_get_keyword():
    # 'get' is in the legacy keyword set, so even the fallback catches this empty-hand halt.
    sub = {"type": "unknown", "text": "Get the green Piattos."}
    assert _granted(sub, _state(leftGrippedState=False)) is False


def test_unknown_paraphrase_outside_keywords_is_the_known_gap():
    # 'obtain' is NOT a legacy keyword - the fallback CANNOT guard it (documents why typing matters:
    # a `pickup` type would refuse this; `unknown` cannot). This is the gap, asserted honestly.
    sub = {"type": "unknown", "text": "Obtain the green Piattos."}
    assert _granted(sub, _state(leftGrippedState=False)) is True
    # ...whereas the SAME instruction typed as pickup is correctly refused:
    assert _granted({"type": "pickup", "target": "green Piattos"}, _state(leftGrippedState=False)) is False


def test_unknown_drop_still_gripping_refused():
    sub = {"type": "unknown", "text": "Leave it at the counter."}
    assert _granted(sub, _state(leftGrippedState=True)) is False


def test_unknown_unrecognized_type_routes_to_fallback():
    # A type that somehow isn't in the vocab (defensive) uses the keyword fallback, never raises.
    sub = {"type": "frobnicate", "text": "Grab the milk."}
    assert _granted(sub, _state(leftGrippedState=False)) is False


def test_cap_constant_sane():
    assert isinstance(HALT_REFUSAL_CAP, int) and HALT_REFUSAL_CAP >= 1


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
        except Exception as e:  # noqa: BLE001 - a predicate raising is itself a failure worth surfacing
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
