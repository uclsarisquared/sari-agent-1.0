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

    uv run pytest validation/tests/agent/test_completion_predicates.py   # or: pytest validation/tests/agent/test_completion_predicates.py
"""
import os
import sys
import logging
from unittest.mock import patch

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from orchestrator import subtask_completion as sc
from orchestrator.subtask_completion import (
    parse_decomposition,
    completion_predicate,
    name_overlap,
    mismatched_hands,
    predicate_inspect,
    reported_inspection_answer,
    held_item_inspection_active,
    planned_subtask_metrics,
    inspect_scope_violation,
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


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


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


def test_bad_count_fallback_is_logged():
    capture = _LogCapture()
    sc.logger.addHandler(capture)
    try:
        parse_decomposition(
            '[{"type":"pickup","target":"milk","count":"many","text":"Pick it up."}]',
            "orig",
        )
    finally:
        sc.logger.removeHandler(capture)
    assert any("invalid pickup count" in message and "defaulting to 1" in message
               for message in capture.messages)


def test_parse_inspect_retains_required_query():
    raw = ('[{"type":"inspect","query":"How many Piattos are visible?",'
           '"text":"Count and report the Piattos without touching them."}]')
    out = parse_decomposition(raw, "orig")
    assert out == [{"type": "inspect",
                    "query": "How many Piattos are visible?",
                    "text": "Count and report the Piattos without touching them."}]


def test_parse_inspect_without_query_degrades_safely():
    out = parse_decomposition(
        '[{"type":"inspect","text":"Look at the shelf and report."}]', "orig")
    assert out == [{"type": "unknown", "text": "Look at the shelf and report."}]


def test_parse_never_emits_type_outside_vocab():
    raw = '[{"type":"pickup","text":"a"},{"type":"weird","text":"b"},"c"]'
    out = parse_decomposition(raw, "orig")
    assert all(s["type"] in SUBTASK_TYPES or s["type"] == "unknown" for s in out)
    assert "inspect" in SUBTASK_TYPES


def test_decomposer_prompt_defines_read_only_inspection_and_separate_pickup():
    prompt = sc.TYPED_DECOMPOSER_SYSTEM.lower()
    assert "inspect" in prompt and "read-only observation" in prompt
    assert "must not pick up" in prompt
    assert "pick up the one that is not expired" in prompt
    assert '"type": "inspect"' in prompt and '"type": "pickup"' in prompt


def test_decomposer_prompt_requires_a_goto_before_a_product_inspect():
    """An inspect leg gets NO plan-time checkpoints (plan_legs) and cannot travel (the inspect scope
    gate), so a bare inspect naming a product answers from wherever the agent spawned - measured
    2026-07-30 on 'find choco mallows and check its price', which inspected from spawn and never saw
    the product. The prompt must both require the leading goto AND permit a product as a goto
    location; the location resolver already falls through to the product resolver."""
    prompt = sc.TYPED_DECOMPOSER_SYSTEM.lower()
    assert "name of a product" in prompt          # goto accepts a product destination
    assert "preceded by a goto" in prompt         # the rule itself
    assert "on this shelf" in prompt              # ... and its already-there exemption
    assert '"type": "goto", "location": "choco mallows"' in prompt


def test_inspection_answer_uses_only_structured_report_not_stop_placeholder():
    response = {
        "halt": True,
        "text": "STOP action received, terminating execution...",
        "reported_answer": "14 unique products",
    }
    assert reported_inspection_answer(response) == "14 unique products"
    assert reported_inspection_answer({
        "halt": True, "text": "STOP action received, terminating execution..."
    }) == ""
    assert reported_inspection_answer({"reported_answer": 14}) == ""


# --- inspect predicate -----------------------------------------------------

def test_inspect_blank_answer_refuses_without_calling_guard():
    calls = []
    guard = lambda *args: calls.append(args)  # noqa: E731
    for answer in ("", "   \n"):
        assert predicate_inspect(
            {"type": "inspect", "query": "How many?"}, _state(), answer, guard)[0] is False
    assert calls == []


def test_inspect_positive_and_negative_verdicts_receive_auxiliary_context():
    seen = []

    def guard(query, answer, auxiliary_context):
        seen.append((query, answer, auxiliary_context))
        return {"match": answer == "Three.", "reason": "visible count", "conclusive": True}

    st = _state(gripped_name="COKE", gripped_names={"left": "COKE", "right": None},
                nearest_checkpoint=32)
    sub = {"type": "inspect", "query": "How many Piattos?"}
    assert predicate_inspect(sub, st, "Three.", guard)[0] is True
    assert predicate_inspect(sub, st, "Four.", guard)[0] is False
    assert seen[0][2] == {
        "gripped_name": "COKE",
        "gripped_names": {"left": "COKE", "right": None},
        "nearest_checkpoint": 32,
        "inspection_evidence": None,
    }


def _two_held(evidence=None):
    return _state(leftGrippedState=True, rightGrippedState=True,
                  gripped_names={"left": "PEPERO_DOUBLE_CHOCO", "right": "HELLO_PANDA"},
                  inspection_evidence=evidence if evidence is not None else [])


def test_inspect_two_held_items_refuses_the_unread_hand_without_a_vlm_call():
    """The MEASURED 2026-07-29 failure: one label read, the guard could only see one item.

    The refusal must now name the item still to inspect - advice the agent can act on - instead of
    spending a VLM call to say the single frame does not show both labels.
    """
    calls = []
    guard = lambda *args: calls.append(args) or {  # noqa: E731
        "match": True, "reason": "ok", "conclusive": True}
    sub = {"type": "inspect", "query": "Which has less sugar?"}
    read_left = [{"hand": "left", "sku": "PEPERO_DOUBLE_CHOCO", "step": 12}]
    granted, reason = predicate_inspect(
        sub, _two_held(read_left), "Hello Panda has less sugar.", guard)
    assert granted is False
    assert calls == []
    assert "right hand (HELLO_PANDA)" in reason
    assert "inspect_held_item_right" in reason
    # Both read -> the gate is silent and the multi-frame VLM verdict decides.
    read_both = read_left + [{"hand": "right", "sku": "HELLO_PANDA", "step": 21}]
    assert predicate_inspect(
        sub, _two_held(read_both), "Hello Panda has less sugar.", guard)[0] is True
    assert len(calls) == 1
    assert calls[0][2]["inspection_evidence"] == read_both


def test_inspect_evidence_gap_ignores_stale_skus_and_leaves_single_item_legs_alone():
    sub = {"type": "inspect", "query": "What date?"}
    guard = lambda *_: {"match": True, "reason": "ok", "conclusive": True}  # noqa: E731
    # A ledger entry for an item that hand no longer holds is not evidence about what it holds now.
    swapped = [{"hand": "left", "sku": "SOME_OTHER_SKU", "step": 3},
               {"hand": "right", "sku": "HELLO_PANDA", "step": 21}]
    assert predicate_inspect(sub, _two_held(swapped), "2027-01-01", guard)[0] is False
    # One hand (or none) held: unchanged pre-2026-07-29 behaviour - the leg may legitimately be
    # answered from the shelf view without the held-item macro ever running.
    one_hand = _state(leftGrippedState=True, gripped_names={"left": "PEPERO_DOUBLE_CHOCO"})
    assert predicate_inspect(sub, one_hand, "2027-01-01", guard)[0] is True
    assert predicate_inspect(sub, _state(), "2027-01-01", guard)[0] is True
    # No ledger at all with two hands full still refuses, naming the first uncovered hand.
    assert sc.inspection_evidence_gap(_two_held(None)) == [
        {"hand": "left", "sku": "PEPERO_DOUBLE_CHOCO"},
        {"hand": "right", "sku": "HELLO_PANDA"},
    ]


def test_inspect_missing_inputs_guard_failures_and_bad_verdicts_fail_closed():
    sub = {"type": "inspect", "query": "What date?"}
    assert predicate_inspect(sub, _state(), "2027-01-01", None)[0] is False
    assert predicate_inspect({"type": "inspect"}, _state(), "answer", lambda *_: {})[0] is False

    def boom(*_):
        raise TimeoutError("late")

    assert predicate_inspect(sub, _state(), "answer", boom)[0] is False
    malformed = [
        None, True, {}, {"match": True}, {"match": 1, "reason": "x", "conclusive": True},
        {"match": True, "reason": "", "conclusive": True},
        {"match": True, "reason": "maybe", "conclusive": False},
    ]
    for verdict in malformed:
        assert predicate_inspect(sub, _state(), "answer", lambda *_, v=verdict: v)[0] is False


def test_completion_dispatch_routes_inspect_without_changing_other_types():
    called = []

    def guard(*args):
        called.append(args)
        return {"match": True, "reason": "visible", "conclusive": True}

    assert completion_predicate(
        {"type": "inspect", "query": "Count them"}, _state(), "There are two.",
        inspect_guard=guard)[0] is True
    assert called
    assert completion_predicate({"type": "goto"}, _state())[0] is True
    assert completion_predicate(
        {"type": "checkout"}, _state(last_checkout={"scanned": True, "placed": True}))[0] is True


def test_planned_subtask_metrics_count_inspect_unknown_and_rate():
    metrics = planned_subtask_metrics([
        {"type": "inspect"}, {"type": "pickup"}, {"type": "unknown"}, {"type": "bogus"},
    ])
    assert metrics["planned_leg_type_counts"]["inspect"] == 1
    assert metrics["planned_leg_type_counts"]["pickup"] == 1
    assert metrics["planned_leg_type_counts"]["unknown"] == 2
    assert metrics["unknown_subtask_rate"] == 0.5


def test_inspect_scope_violation_records_only_hard_blocked_attempts_as_unexecuted():
    empty = _state()
    held = _state(leftGrippedState=True)
    blocked = {"blocked": True, "executed": False, "inspect_scope_violation": True,
               "reason": "outside inspect scope"}
    grab = inspect_scope_violation("extend_arm_until_grabbed", 3, empty, blocked)
    release = inspect_scope_violation("grip_left", 4, held, blocked)
    checkout = inspect_scope_violation("checkout_held_item", 5, held, blocked)
    body = inspect_scope_violation("move_forward", 6, held, blocked)
    assert grab["kind"] == "grip_or_grab" and grab["executed"] is False
    assert release["kind"] == "grip_toggle_release"
    assert release["pre_action_grip_state"]["left"] is True
    assert checkout["kind"] == "checkout_macro" and checkout["blocked"] is True
    assert body["kind"] == "body_translation" and body["executed"] is False
    assert inspect_scope_violation("turn_left", 6, empty, {}) is None
    assert inspect_scope_violation("rotate_left_clockwise", 7, held, {}) is None


def test_held_item_inspection_force_tracks_both_hands_and_empty_state():
    inspect = {"type": "inspect"}
    assert held_item_inspection_active(inspect, _state(leftGrippedState=True)) is True
    assert held_item_inspection_active(inspect, _state(rightGrippedState=True)) is True
    assert held_item_inspection_active(inspect, _state()) is False
    assert held_item_inspection_active(
        {"type": "pickup"}, _state(leftGrippedState=True)) is False


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


# --- pluggable VLM pickup guard (injected verdicts; never network) ----------

def _verdict(match, reason="test verdict", conclusive=True):
    return {"match": match, "reason": reason, "conclusive": conclusive}


def test_default_completion_guard_remains_deterministic():
    st = _state(leftGrippedState=True, gripped_name="PIATTOS_CHEESE_40G")
    assert completion_predicate({"type": "pickup", "target": "Piattos"}, st)[0] is True


def test_vlm_backend_leaves_untargeted_and_nonpickup_predicates_deterministic():
    untargeted = _state(leftGrippedState=True, new_grip_this_leg=True)
    assert completion_predicate(
        {"type": "pickup"}, untargeted, guard_backend="vlm")[0] is True
    checkout = _state(last_checkout={"scanned": True, "placed": True, "reason": "ok"})
    assert completion_predicate(
        {"type": "checkout"}, checkout, guard_backend="vlm")[0] is True


def test_none_completion_guard_accepts_stop_without_verification():
    """Disabling the guard bypasses every typed predicate, including inspection verification."""
    for sub in (
        {"type": "pickup", "target": "Piattos"},
        {"type": "checkout"},
        {"type": "goto", "target_checkpoint": [99]},
        {"type": "compare", "targets": ["A", "B"]},
        {"type": "inspect", "query": "expiration date"},
        {"type": "unknown", "text": "drop the item"},
    ):
        granted, reason = completion_predicate(sub, _state(), guard_backend="none")
        assert granted is True
        assert "disabled" in reason


def test_vlm_attribute_target_match_and_mismatch():
    st = _state(leftGrippedState=True,
                gripped_names={"left": "CDO_HOME_STYLE_CORNED_BEEF_150G", "right": None})
    sub = {"type": "pickup", "target": "can of food that is good for breakfast"}
    assert completion_predicate(
        sub, st, guard_backend="vlm",
        guard_verdicts={"left": _verdict(True, "Corned beef is a canned breakfast food.")})[0] is True
    assert completion_predicate(
        sub, st, guard_backend="vlm",
        guard_verdicts={"left": _verdict(False, "This held item does not fit the description.")})[0] is False


def test_vlm_refusal_overrides_deterministic_name_match():
    st = _state(leftGrippedState=True,
                gripped_name="PIATTOS_CHEESE_40G",
                gripped_names={"left": "PIATTOS_CHEESE_40G", "right": None})
    ok, reason = completion_predicate(
        {"type": "pickup", "target": "Piattos"}, st, guard_backend="vlm",
        guard_verdicts={"left": _verdict(False, "visual evidence is inconsistent")})
    assert ok is False and "visual evidence" in reason


def test_vlm_missing_or_failed_verdict_fails_closed_without_name_fallback():
    st = _state(leftGrippedState=True,
                gripped_name="PIATTOS_CHEESE_40G",
                gripped_names={"left": "PIATTOS_CHEESE_40G", "right": None})
    sub = {"type": "pickup", "target": "Piattos"}
    assert completion_predicate(sub, st, guard_backend="vlm")[0] is False
    assert completion_predicate(
        sub, st, guard_backend="vlm",
        guard_verdicts={"left": _verdict(False, "timeout", conclusive=False)})[0] is False


def test_vlm_failed_verdict_does_not_drive_corrective_release():
    st = _state(leftGrippedState=True,
                gripped_names={"left": "COKE_ZERO_330", "right": None})
    assert mismatched_hands(
        {"type": "pickup", "target": "Piattos"}, st, guard_backend="vlm",
        guard_verdicts={"left": _verdict(False, "API timeout", conclusive=False)}) == []


def test_vlm_conclusive_wrong_item_drives_corrective_release():
    st = _state(leftGrippedState=True,
                gripped_names={"left": "COKE_ZERO_330", "right": None})
    assert mismatched_hands(
        {"type": "pickup", "target": "Piattos"}, st, guard_backend="vlm",
        guard_verdicts={"left": _verdict(False, "Coke is not Piattos")}) == ["left"]


def test_vlm_quantity_two_is_matched_per_hand():
    st = _state(leftGrippedState=True, rightGrippedState=True,
                gripped_names={"left": "JIN_RAMEN_MILD_120G", "right": "COKE_ZERO_330"})
    sub = {"type": "pickup", "target": "Jin Ramen", "count": 2}
    verdicts = {"left": _verdict(True), "right": _verdict(False, "wrong product")}
    ok, reason = completion_predicate(
        sub, st, guard_backend="vlm", guard_verdicts=verdicts)
    assert ok is False and "1 of 2" in reason
    verdicts["right"] = _verdict(True)
    assert completion_predicate(
        sub, st, guard_backend="vlm", guard_verdicts=verdicts)[0] is True


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
    # A runner that never sets gripped_names (pickup_navigation's flat loop) cannot COUNT items - the
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
    # Integration against the live Unity catalog (SARI_SANDBOX_DIR in config.env); a no-op (with a
    # note) where the sim repo is absent. Checks _CATALOG_LOADED, NOT lexicon truthiness -
    # _CATEGORY_ALIASES always merges in 5 pseudo-categories, so the lexicon is truthy even with
    # zero sim repo access; that used to make this "skip if absent" guard never actually skip.
    sc._CATEGORY_LEXICON = None      # force a real (re)load
    try:
        sc._category_lexicon()
        if not sc._CATALOG_LOADED:
            print("(catalog absent - category integration check skipped)")
            return
        st = _state(leftGrippedState=True, gripped_name="LEMONSQUARE_MAMON_CHEESY_264G")
        assert _granted({"type": "pickup", "target": "Biscuits"}, st) is True
    finally:
        sc._CATEGORY_LEXICON = None  # don't leak the real lexicon into hermetic tests
        sc._CATALOG_LOADED = False


def test_category_catalog_load_failure_is_logged():
    old_lex, old_generic, old_loaded = (
        sc._CATEGORY_LEXICON, sc._GENERIC_NAMES, sc._CATALOG_LOADED
    )
    capture = _LogCapture()
    sc.logger.addHandler(capture)
    try:
        sc._CATEGORY_LEXICON = None
        sc._GENERIC_NAMES = None
        with patch.object(sc, "categories_json", return_value="/definitely/missing/Categories.json"):
            sc._category_lexicon()
        assert sc._CATALOG_LOADED is False
    finally:
        sc.logger.removeHandler(capture)
        sc._CATEGORY_LEXICON, sc._GENERIC_NAMES, sc._CATALOG_LOADED = (
            old_lex, old_generic, old_loaded
        )
    assert any("failed to load Categories.json" in message for message in capture.messages)


# --- cereal alias: catalog has NO 'Cereal' category (all filed under Biscuit), CLAUDE.md thread #5 ---

def test_cereal_alias_grants_real_cereal():
    # THE MEASURED 2026-07-24 refusal: target 'cereal', held KELLOGG'S_COCO_POPS_30G (a real cereal) was
    # refused because 'cereal' is no catalog category, so it was a distinctive token demanding the
    # literal substring "cereal" in the SKU. The alias is self-contained (needs no sim files) and must
    # now GRANT. Force a real merge; reset in finally so hermetic tests don't inherit it.
    sc._CATEGORY_LEXICON = None
    try:
        st = _state(leftGrippedState=True, gripped_name="KELLOGG'S_COCO_POPS_30G")
        ok, reason = completion_predicate({"type": "pickup", "target": "cereal"}, st)
        assert ok is True, reason
    finally:
        sc._CATEGORY_LEXICON = None


def test_cereal_alias_refuses_non_cereal_biscuit():
    # The alias SKU list is TIGHT: a Pocky (also under Biscuit, but NOT a cereal) must still refuse -
    # proving the alias did not loosen 'cereal' to the whole Biscuit category.
    sc._CATEGORY_LEXICON = None
    try:
        st = _state(rightGrippedState=True, gripped_name="GLICO_POCKY_CHOCOLATE_20G")
        assert _granted({"type": "pickup", "target": "cereal"}, st) is False
    finally:
        sc._CATEGORY_LEXICON = None


def test_cereal_alias_covers_all_six_skus():
    # Completeness: every cereal that hides in Biscuit resolves, not just Coco Pops.
    sc._CATEGORY_LEXICON = None
    try:
        for sku in ("KELLOGG'S_COCO_POPS_30G", "KELLOGG'S_FROOT_LOOPS_25G", "KELLOGG'S_FROSTIES_30G",
                    "NESTLE_GOLD_CORN_FLAKES_275G", "NESTLE_HONEYSTARS_ORIGINAL_300G",
                    "NESTLE_KOKOKRUNCH_CHOCOLATE_330G"):
            st = _state(leftGrippedState=True, gripped_name=sku)
            assert _granted({"type": "pickup", "target": "cereal"}, st) is True, sku
    finally:
        sc._CATEGORY_LEXICON = None


def test_named_cereal_still_matches_distinctively():
    # A NAMED cereal must still match on its own tokens (the alias must not have broken the normal path).
    sc._CATEGORY_LEXICON = None
    try:
        st = _state(leftGrippedState=True, gripped_name="KELLOGG'S_COCO_POPS_30G")
        assert _granted({"type": "pickup", "target": "Coco Pops"}, st) is True
    finally:
        sc._CATEGORY_LEXICON = None


# --- more aliases (2026-07-24): candy/chocolate bars, dairy cheese/yogurt, phrase aliases ---

def _real_lex():
    """Force the real merge; the aliases are self-contained (no sim files needed). Callers reset in a
    finally so hermetic tests don't inherit the real lexicon."""
    sc._CATEGORY_LEXICON = None


def test_candy_and_chocolate_alias_the_bars():
    _real_lex()
    try:
        for sku in ("SCHOGETTEN_ALPINE_MILK_100G", "LINDT_HELLO_CRUNCHY_NOUGAT_100G",
                    "M&M'S_MILK_CHOCOLATE_87.9G", "KINDER_TRONKY_5x18G"):
            st = _state(leftGrippedState=True, gripped_name=sku)
            assert _granted({"type": "pickup", "target": "candy"}, st) is True, sku
            assert _granted({"type": "pickup", "target": "chocolate"}, st) is True, sku
    finally:
        sc._CATEGORY_LEXICON = None


def test_chocolate_alias_excludes_choco_flavored_biscuit():
    # 'chocolate' as a bare token used to match 21 items across 4 categories (a flavor word). As a
    # tight alias a choco-FLAVORED biscuit (Pocky) is NOT a chocolate bar and must refuse...
    _real_lex()
    try:
        st = _state(leftGrippedState=True, gripped_name="GLICO_POCKY_CHOCOLATE_20G")
        assert _granted({"type": "pickup", "target": "chocolate"}, st) is False
        # ...but naming that biscuit still works via its distinctive brand token.
        assert _granted({"type": "pickup", "target": "Pocky chocolate"}, st) is True
    finally:
        sc._CATEGORY_LEXICON = None


def test_cheese_alias_dairy_only():
    _real_lex()
    try:
        for sku in ("EDEN_ORIGINAL_CHEESE_160G", "MAGNOLIA_CHEEZEE_160G", "MAGNOLIA_DAILYQUEZO_160g"):
            st = _state(leftGrippedState=True, gripped_name=sku)
            assert _granted({"type": "pickup", "target": "cheese"}, st) is True, sku
        # a cheese-FLAVORED chip is not dairy cheese
        st = _state(leftGrippedState=True, gripped_name="LESLIES_CLOVERCHIPS_CHEESE_85G")
        assert _granted({"type": "pickup", "target": "cheese"}, st) is False
    finally:
        sc._CATEGORY_LEXICON = None


def test_yogurt_alias():
    _real_lex()
    try:
        for sku in ("CIMORY_YOGURT_ORIGINAL_100G", "DUTCHMILL_PROYO_STRAWBERRY_400ML",
                    "PASCUAL_GREEK_MANGO_250ML", "BAUER_FRUFRU_150G"):
            st = _state(leftGrippedState=True, gripped_name=sku)
            assert _granted({"type": "pickup", "target": "yogurt"}, st) is True, sku
    finally:
        sc._CATEGORY_LEXICON = None


def test_phrase_alias_energy_drink_includes_sting():
    # The whole point of the phrase alias: Sting is an energy drink but its SKU has no 'energy'.
    _real_lex()
    try:
        for sku in ("COBRA_ENERGY_DRINK_350ML", "REDBULL_ENERGY_DRINK_150ML", "STING_330ML"):
            st = _state(leftGrippedState=True, gripped_name=sku)
            assert _granted({"type": "pickup", "target": "energy drink"}, st) is True, sku
        st = _state(leftGrippedState=True, gripped_name="SPRITE_LEMON_LIME_DRINK_500ML")
        assert _granted({"type": "pickup", "target": "energy drink"}, st) is False
    finally:
        sc._CATEGORY_LEXICON = None


def test_phrase_alias_specific_brand_still_wins():
    # 'Cobra energy drink' names Cobra - the phrase must NOT short-circuit past the distinctive 'cobra'
    # and grant a Red Bull.
    _real_lex()
    try:
        st = _state(leftGrippedState=True, gripped_name="REDBULL_ENERGY_DRINK_150ML")
        assert _granted({"type": "pickup", "target": "Cobra energy drink"}, st) is False
        st2 = _state(leftGrippedState=True, gripped_name="COBRA_ENERGY_DRINK_350ML")
        assert _granted({"type": "pickup", "target": "Cobra energy drink"}, st2) is True
    finally:
        sc._CATEGORY_LEXICON = None


def test_phrase_alias_soy_and_oat_milk():
    _real_lex()
    try:
        st = _state(leftGrippedState=True, gripped_name="VITAMILK_300ML")
        assert _granted({"type": "pickup", "target": "soy milk"}, st) is True
        st = _state(leftGrippedState=True, gripped_name="OATSIDE_CHOCOLATE_1L")
        assert _granted({"type": "pickup", "target": "oat milk"}, st) is True
    finally:
        sc._CATEGORY_LEXICON = None


def test_bare_milk_unaffected_by_phrase_aliases():
    # Regression guard: the soy/oat-milk phrases must not neutralize a BARE 'milk' target - 'pick up
    # milk' still matches a real milk (distinctive token, no phrase present).
    _real_lex()
    try:
        st = _state(leftGrippedState=True, gripped_name="COWHEAD_PURE_MILK_1L")
        assert _granted({"type": "pickup", "target": "milk"}, st) is True
    finally:
        sc._CATEGORY_LEXICON = None


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


def test_compare_vlm_is_additive_and_receives_choice_context():
    sub = {"type": "compare", "targets": ["Pocky", "Hello Panda"],
           "criterion": "less sugar", "candidate_sets": [[30], [31]]}
    st = _state(visited_checkpoints={30, 31}, nearest_checkpoint=31)
    seen = []

    def guard(criterion, answer, auxiliary_context):
        seen.append((criterion, answer, auxiliary_context))
        return _verdict(True, "both nutrition labels support Pocky")

    ok, reason = completion_predicate(
        sub, st, final_text="Pocky has less sugar.", guard_backend="vlm",
        compare_guard=guard)
    assert ok is True and "VLM verified" in reason
    assert seen[0][0] == "less sugar"
    assert seen[0][2]["named_choice"] == "Pocky"

    # The VLM cannot override the deterministic visited-both prerequisite.
    assert completion_predicate(
        sub, _state(visited_checkpoints={30}), final_text="Pocky has less sugar.",
        guard_backend="vlm", compare_guard=guard)[0] is False


def test_compare_vlm_missing_failed_and_malformed_guards_fail_closed():
    sub = {"type": "compare", "targets": ["A", "B"], "criterion": "size",
           "candidate_sets": [[1], [2]]}
    st = _state(visited_checkpoints={1, 2})
    assert completion_predicate(
        sub, st, final_text="A is larger", guard_backend="vlm")[0] is False

    def boom(*_):
        raise TimeoutError("late")

    assert completion_predicate(
        sub, st, final_text="A is larger", guard_backend="vlm",
        compare_guard=boom)[0] is False
    for verdict in (None, {}, _verdict(True, "", True), _verdict(True, "maybe", False)):
        assert completion_predicate(
            sub, st, final_text="A is larger", guard_backend="vlm",
            compare_guard=lambda *_, v=verdict: v)[0] is False


def test_compare_vlm_without_targets_fails_closed_but_default_is_unchanged():
    sub = {"type": "compare", "text": "Compare them"}
    assert completion_predicate(sub, _state())[0] is True
    assert completion_predicate(sub, _state(), guard_backend="vlm")[0] is False


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


def test_unknown_vlm_is_additive_and_receives_free_text_claim():
    sub = {"type": "unknown", "text": "Obtain the green Piattos."}
    st = _state(leftGrippedState=True, gripped_name="PIATTOS_GREEN",
                gripped_names={"left": "PIATTOS_GREEN", "right": None},
                nearest_checkpoint=32)
    seen = []

    def guard(task, claim, auxiliary_context):
        seen.append((task, claim, auxiliary_context))
        return _verdict(True, "green Piattos is visibly held")

    ok, reason = completion_predicate(
        sub, st, final_text="I obtained it.", guard_backend="vlm",
        unknown_guard=guard)
    assert ok is True and "VLM verified" in reason
    assert seen[0][0] == sub["text"] and seen[0][1] == "I obtained it."
    assert seen[0][2]["left_gripped"] is True

    # A deterministic keyword/grip failure blocks before the VLM can grant.
    calls = []
    assert completion_predicate(
        {"type": "unknown", "text": "Get the green Piattos."},
        _state(leftGrippedState=False), guard_backend="vlm",
        unknown_guard=lambda *args: calls.append(args) or _verdict(True))[0] is False
    assert calls == []


def test_unknown_vlm_missing_failed_and_malformed_guards_fail_closed():
    sub = {"type": "unknown", "text": "Observe the shelf."}
    assert completion_predicate(sub, _state(), guard_backend="vlm")[0] is False

    def boom(*_):
        raise RuntimeError("down")

    assert completion_predicate(
        sub, _state(), guard_backend="vlm", unknown_guard=boom)[0] is False
    for verdict in (None, {}, _verdict(True, "", True), _verdict(False, "timeout", False)):
        assert completion_predicate(
            sub, _state(), guard_backend="vlm",
            unknown_guard=lambda *_, v=verdict: v)[0] is False


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
