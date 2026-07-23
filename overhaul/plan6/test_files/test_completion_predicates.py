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

import subtask_completion as sc
from subtask_completion import (
    parse_decomposition,
    completion_predicate,
    name_overlap,
    mismatched_hands,
    SUBTASK_TYPES,
    HALT_REFUSAL_CAP,
    WRONG_ITEM_RELEASE_AFTER,
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


def test_parse_count_normalized():
    # count survives the parse as a clamped int; a string count from the LLM is coerced.
    raw = '[{"type": "pickup", "target": "Jin Ramen", "count": "2", "text": "Pick up 2 Jin Ramen."}]'
    out = parse_decomposition(raw, "orig")
    assert out[0]["count"] == 2


def test_parse_count_garbage_dropped_and_zero_clamped():
    raw = ('[{"type": "pickup", "target": "a", "count": "lots", "text": "t"}, '
           '{"type": "pickup", "target": "b", "count": 0, "text": "t"}]')
    out = parse_decomposition(raw, "orig")
    assert "count" not in out[0]        # unparseable -> default-1 behaviour, not a crash
    assert out[1]["count"] == 1         # clamped to at least 1


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


# --- pickup predicate: count (dual-hand quantity, 2026-07-23) ---------------

def test_pickup_count2_refused_with_one_held():
    # The 'pick up 2 X' leg: one matching item in hand is NOT done - the refusal points the agent
    # at its free hand instead of letting a single grab satisfy the quantity.
    st = _state(leftGrippedState=True,
                gripped_names={"left": "JIN_RAMEN_MILD_120G", "right": None})
    ok, reason = completion_predicate({"type": "pickup", "target": "Jin Ramen", "count": 2}, st)
    assert ok is False and "1 of 2" in reason and "free hand" in reason


def test_pickup_count2_granted_with_both_hands_matching():
    st = _state(leftGrippedState=True, rightGrippedState=True,
                gripped_names={"left": "JIN_RAMEN_MILD_120G", "right": "JIN_RAMEN_SPICY_120G"})
    assert _granted({"type": "pickup", "target": "Jin Ramen", "count": 2}, st) is True


def test_pickup_count2_wrong_second_item_refused():
    # Two hands gripping but only one holds the target - the quantity is of MATCHING items.
    st = _state(leftGrippedState=True, rightGrippedState=True,
                gripped_names={"left": "JIN_RAMEN_MILD_120G", "right": "COKE_ZERO_330"})
    ok, reason = completion_predicate({"type": "pickup", "target": "Jin Ramen", "count": 2}, st)
    assert ok is False and "1 of 2" in reason


def test_pickup_count2_untargeted_counts_any_grips():
    st = _state(leftGrippedState=True, rightGrippedState=True,
                gripped_names={"left": "ANYTHING", "right": "WHATEVER"})
    assert _granted({"type": "pickup", "count": 2}, st) is True


def test_pickup_count2_degrades_without_per_hand_names():
    # A runner that never sets gripped_names (eval_pickup's flat loop) cannot COUNT items - the
    # predicate degrades to the single-item check and says so, rather than blocking on wiring it
    # can't feed (the goto/compare [unverified] pattern).
    st = _state(leftGrippedState=True, gripped_name="JIN_RAMEN_MILD_120G")
    ok, reason = completion_predicate({"type": "pickup", "target": "Jin Ramen", "count": 2}, st)
    assert ok is True and "unverified count" in reason
    # ...but a wrong-item grip is still refused even on the degraded path.
    st = _state(leftGrippedState=True, gripped_name="COKE_ZERO_330")
    assert _granted({"type": "pickup", "target": "Jin Ramen", "count": 2}, st) is False


# --- category grounding + mismatched_hands (self-correction, 2026-07-23) ---

class _lexicon:
    """Pin the catalog state hermetically (no files): category lexicon, the generic-name set (defaults
    to the lexicon's category keys, exactly what the real loader derives), and the reconciled SKU->text
    index (defaults empty)."""
    def __init__(self, lex, generic=None, reconciled=None):
        self.lex = lex or {}
        self.generic = set(self.lex.keys()) if generic is None else set(generic)
        self.reconciled = reconciled or {}
    def __enter__(self):
        self.saved = (sc._CATEGORY_LEXICON, sc._GENERIC_NAMES, sc._RECONCILED_INDEX)
        sc._CATEGORY_LEXICON = self.lex
        sc._GENERIC_NAMES = self.generic
        sc._RECONCILED_INDEX = self.reconciled
    def __exit__(self, *a):
        sc._CATEGORY_LEXICON, sc._GENERIC_NAMES, sc._RECONCILED_INDEX = self.saved


_BISCUIT_LEX = {"biscuit": ["lemonsquare_mamon_cheesy_264g", "fibisco_jolly_27g"]}
# A catalog fragment where 'chips' is a category (so it's a GENERIC type-word), used for the
# cross-brand false-grant cases below.
_CHIPS_LEX = {"chips": ["chipsy_nacho_crispies_bbq_70g", "leslie_s_clover_chips_cheese_24g"],
              "biscuit": ["lemonsquare_mamon_cheesy_264g"]}


def test_category_target_matches_member_sku():
    # THE run-0723_061651 false refusal: 'Biscuits' never substring-matches the (correct) Mamon SKU.
    with _lexicon(_BISCUIT_LEX):
        st = _state(leftGrippedState=True, gripped_name="LEMONSQUARE_MAMON_CHEESY_264G")
        ok, reason = completion_predicate({"type": "pickup", "target": "Biscuits"}, st)
        assert ok is True, reason


def test_category_target_still_refuses_non_member():
    with _lexicon(_BISCUIT_LEX):
        st = _state(leftGrippedState=True, gripped_name="COKE_ZERO_330")
        assert _granted({"type": "pickup", "target": "Biscuits"}, st) is False


def test_category_grounding_absent_degrades_to_substring():
    # No catalog on this machine -> empty lexicon -> exactly the old substring behaviour.
    with _lexicon({}):
        st = _state(leftGrippedState=True, gripped_name="LEMONSQUARE_MAMON_CHEESY_264G")
        assert _granted({"type": "pickup", "target": "Biscuits"}, st) is False
        assert _granted({"type": "pickup", "target": "Mamon"}, st) is True


def test_category_real_catalog_if_present():
    # Integration against the live Unity catalog; a no-op (with a note) where the sim repo is absent.
    sc._CATEGORY_LEXICON = None      # force a real (re)load
    try:
        if not sc._category_lexicon():
            print("(catalog absent - category integration check skipped)")
            return
        st = _state(leftGrippedState=True, gripped_name="LEMONSQUARE_MAMON_CHEESY_264G")
        assert _granted({"type": "pickup", "target": "Biscuits"}, st) is True
    finally:
        sc._CATEGORY_LEXICON = None  # don't leak the real lexicon into hermetic tests


# --- distinctive-token matching: the cross-brand false GRANT (2026-07-23) ---

def test_false_grant_cross_brand_refused():
    # THE bug: target the SPECIFIC 'Chipsy Corn Chips Nacho Crispies BBQ', agent held a DIFFERENT
    # product (Leslie's Clover Chips). The only shared token is the category word 'chips' - it must
    # NOT grant. Reconciled text for the wrong item is present to prove appearance can't rescue it.
    recon = {"leslie_s_clover_chips_cheese_24g": "clover chips cheeser orange bag cheese graphic"}
    with _lexicon(_CHIPS_LEX, reconciled=recon):
        st = _state(rightGrippedState=True, gripped_name="LESLIE_S_CLOVER_CHIPS_CHEESE_24G")
        ok, reason = completion_predicate(
            {"type": "pickup", "target": "Chipsy Corn Chips Nacho Crispies BBQ"}, st)
        assert ok is False and "does not match" in reason, reason


def test_correct_same_category_item_granted_even_if_unannotated():
    # The RIGHT item (Chipsy BBQ) shares chipsy/nacho/crispies/bbq - grants on the SKU string alone,
    # even though it is NOT in the reconciled file (78% of SKUs aren't - coverage must not block).
    with _lexicon(_CHIPS_LEX, reconciled={}):
        st = _state(rightGrippedState=True, gripped_name="CHIPSY_NACHO_CRISPIES_BBQ_70G")
        assert _granted({"type": "pickup",
                         "target": "Chipsy Corn Chips Nacho Crispies BBQ"}, st) is True


def test_appearance_enrichment_matches_colour():
    # Appearance as a SOFT tie-breaker: a colour token absent from the SKU string but present in the
    # reconciled appearance still counts. Without the reconciled record it wouldn't (and that's fine -
    # soft, additive, never a block).
    sku = "jack_and_jill_piattos_sour_cream_flavored_potato_40g"
    recon = {sku: "piattos sour cream & onion green bag with diamond logo"}
    with _lexicon({"chips": [sku]}, reconciled=recon):
        st = _state(leftGrippedState=True, gripped_name=sku.upper())
        # 'green' is only in the appearance; still grants.
        assert _granted({"type": "pickup", "target": "green Piattos"}, st) is True
    with _lexicon({"chips": [sku]}, reconciled={}):
        st = _state(leftGrippedState=True, gripped_name=sku.upper())
        # No appearance record, but 'piattos' is still in the SKU -> still grants (soft, not required).
        assert _granted({"type": "pickup", "target": "green Piattos"}, st) is True


def test_size_token_alone_does_not_match():
    # A shared size spec ('155g') is not identity - two unrelated 155g items must not match.
    with _lexicon({}, generic=set(), reconciled={}):
        st = _state(leftGrippedState=True, gripped_name="SOME_OTHER_BRAND_155G")
        assert _granted({"type": "pickup", "target": "Century Tuna 155g"}, st) is False


def test_mismatched_hands_flags_wrong_new_grip():
    st = _state(leftGrippedState=True, gripped_names={"left": "COKE_ZERO_330", "right": None})
    assert mismatched_hands({"type": "pickup", "target": "Piattos"}, st, start_grips=()) == ["left"]


def test_mismatched_hands_protects_carried_in_item():
    # The hand was already gripping at leg start - its item belongs to a previous subtask; never drop it.
    st = _state(leftGrippedState=True, gripped_names={"left": "COKE_ZERO_330", "right": None})
    assert mismatched_hands({"type": "pickup", "target": "Piattos"}, st, start_grips={"left"}) == []


def test_mismatched_hands_never_releases_unnamed_or_matching():
    # No recorded grip-time name -> we can't identify it -> never released. A matching item stays.
    st = _state(leftGrippedState=True, rightGrippedState=True,
                gripped_names={"left": None, "right": "PIATTOS_CHEESE_40G"})
    assert mismatched_hands({"type": "pickup", "target": "Piattos"}, st) == []


def test_mismatched_hands_untargeted_or_ungrounded_is_empty():
    st = _state(leftGrippedState=True, gripped_names={"left": "COKE_ZERO_330", "right": None})
    assert mismatched_hands({"type": "pickup"}, st) == []
    assert mismatched_hands({"type": "pickup", "target": "the"}, st) == []


def test_mismatched_hands_respects_category_membership():
    # A category-correct item is NOT a mismatch - the exact wrong-drop the 0723 run would have made.
    with _lexicon(_BISCUIT_LEX):
        st = _state(leftGrippedState=True,
                    gripped_names={"left": "LEMONSQUARE_MAMON_CHEESY_264G", "right": None})
        assert mismatched_hands({"type": "pickup", "target": "Biscuits"}, st) == []


def test_release_threshold_below_refusal_cap():
    # The auto-release must get a chance to fire BEFORE the cap force-ends the leg.
    assert 1 <= WRONG_ITEM_RELEASE_AFTER < HALT_REFUSAL_CAP


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


def test_checkout_refused_when_second_carried_item_still_held():
    # The dual-hand carry bug (run 0723_094628_graph): the macro bagged the LEFT item (scanned+placed,
    # hand=left, left now empty) but the RIGHT hand still holds the second item. ONE checkout leg must
    # scan BOTH - refuse and point the agent at the still-holding hand, don't end the leg.
    st = _state(leftGrippedState=False, rightGrippedState=True,
                last_checkout={"scanned": True, "placed": True, "hand": "left", "reason": "ok"})
    ok, reason = completion_predicate({"type": "checkout"}, st)
    assert ok is False and "right" in reason and "checkout tool again" in reason


def test_checkout_granted_when_both_hands_emptied():
    # Second item now bagged too (hand=right, both hands empty) -> the leg is finally complete.
    st = _state(leftGrippedState=False, rightGrippedState=False,
                last_checkout={"scanned": True, "placed": True, "hand": "right", "reason": "ok"})
    assert _granted({"type": "checkout"}, st) is True


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
