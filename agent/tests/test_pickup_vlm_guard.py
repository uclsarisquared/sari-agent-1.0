"""Offline tests for the stateless pickup VLM adapter (fake client only)."""

import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from orchestrator.pickup_vlm_guard import (
    cache_compare_candidate_frames,
    classify_compare,
    classify_pickup,
    classify_inspection,
    classify_inspection_label_presence,
    classify_inspection_visibility,
    classify_unknown,
    evaluate_hands,
    make_compare_guard,
    make_inspect_guard,
    make_unknown_guard,
)


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Client:
    def __init__(self, *outcomes):
        self.completions = _Completions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


CONFIG = SimpleNamespace(temperature=0.2, max_tokens=1536,
                         extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def test_valid_json_and_runtime_config():
    client = _Client('{"match": true, "reason": "the SKU names Piattos"}')
    result = classify_pickup(client, "runtime-model", CONFIG, "abc", "PIATTOS_40G", "Piattos")
    assert result["match"] is True and result["conclusive"] is True
    call = client.completions.calls[0]
    assert call["model"] == "runtime-model"
    assert call["timeout"] == 30
    assert call["max_tokens"] == 256
    assert call["extra_body"] == CONFIG.extra_body


def test_malformed_json_fails_closed():
    result = classify_pickup(_Client("not json"), "m", CONFIG, "abc", "SKU", "target")
    assert result["match"] is False and result["conclusive"] is False
    assert "JSONDecodeError" in result["reason"]


def test_timeout_and_api_failure_fail_closed_without_retry():
    for exc in (TimeoutError("slow"), RuntimeError("server down")):
        client = _Client(exc)
        result = classify_pickup(client, "m", CONFIG, "abc", "SKU", "target")
        assert result["match"] is False and result["conclusive"] is False
        assert len(client.completions.calls) == 1


def test_strict_boolean_parsing_rejects_string_and_integer():
    for raw in ('{"match": "true", "reason": "x"}', '{"match": 1, "reason": "x"}'):
        result = classify_pickup(_Client(raw), "m", CONFIG, "abc", "SKU", "target")
        assert result["match"] is False and result["conclusive"] is False


def test_each_call_uses_fresh_isolated_messages():
    client = _Client('{"match": true, "reason": "one"}',
                     '{"match": false, "reason": "two"}')
    classify_pickup(client, "m", CONFIG, "img1", "SKU1", "target1")
    classify_pickup(client, "m", CONFIG, "img2", "SKU2", "target2")
    first, second = client.completions.calls
    assert len(first["messages"]) == len(second["messages"]) == 2
    assert first["messages"] is not second["messages"]
    assert "SKU1" in first["messages"][1]["content"][1]["text"]
    assert "SKU1" not in second["messages"][1]["content"][1]["text"]


def test_unique_sku_is_reused_across_hands():
    client = _Client('{"match": true, "reason": "same product"}')
    verdicts, calls = evaluate_hands(
        client, "m", CONFIG, "abc", "Jin Ramen",
        {"left": "JIN_RAMEN_120G", "right": "JIN_RAMEN_120G"})
    assert calls == 1 and len(client.completions.calls) == 1
    assert verdicts["left"]["reused"] is False
    assert verdicts["right"]["reused"] is True


def test_inspection_uses_bound_frame_query_answer_and_auxiliary_context():
    client = _Client('{"match": true, "reason": "three bags are visible"}')
    aux = {"gripped_name": None, "gripped_names": {}, "nearest_checkpoint": 32}
    result = classify_inspection(
        client, "m", CONFIG, "current-frame", "How many Piattos?", "Three.", aux)
    assert result["match"] is True and result["conclusive"] is True
    content = client.completions.calls[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].endswith("current-frame")
    assert "How many Piattos?" in content[1]["text"]
    assert "Three." in content[1]["text"]
    assert "nearest_checkpoint" in content[1]["text"]
    system = client.completions.calls[0]["messages"][0]["content"]
    assert "directly face the camera" in system
    assert "oblique" in system and "upside-down" in system
    assert "must directly and concretely answer the query" in system
    assert "needs rotation" in system and "match=false" in system


def test_inspection_sends_earlier_evidence_frames_before_the_current_frame():
    """A two-item label comparison is only verifiable ACROSS frames (2026-07-29 regression)."""
    client = _Client('{"match": true, "reason": "Hello Panda\'s panel reads 12g"}')
    evidence = [
        {"label": "PEPERO_DOUBLE_CHOCO (left hand, step 12)", "image_b64": "pepero-frame"},
        {"label": "HELLO_PANDA (right hand, step 21)", "image_b64": "panda-frame"},
    ]
    result = classify_inspection(
        client, "m", CONFIG, "current-frame", "Which has less sugar?",
        "Pepero: 24g; Hello Panda: 12g; Hello Panda has less.", {"nearest_checkpoint": 17},
        evidence_frames=evidence)
    assert result["match"] is True and result["conclusive"] is True
    content = client.completions.calls[0]["messages"][1]["content"]
    assert content[0]["text"] == "EVIDENCE 1: PEPERO_DOUBLE_CHOCO (left hand, step 12)"
    assert content[1]["image_url"]["url"].endswith("pepero-frame")
    assert content[2]["text"] == "EVIDENCE 2: HELLO_PANDA (right hand, step 21)"
    assert content[3]["image_url"]["url"].endswith("panda-frame")
    assert content[4]["text"] == "CURRENT FRAME:"
    assert content[5]["image_url"]["url"].endswith("current-frame")
    assert "Which has less sugar?" in content[6]["text"]
    system = client.completions.calls[0]["messages"][0]["content"]
    assert "CONSIDERED TOGETHER" in system
    assert "never reject an answer merely because no single image shows every item" in system


def test_inspection_without_evidence_keeps_the_single_image_request_shape():
    """No ledger yet must reproduce the pre-2026-07-29 request exactly - no empty evidence slots."""
    client = _Client('{"match": true, "reason": "three bags are visible"}')
    classify_inspection(
        client, "m", CONFIG, "current-frame", "How many?", "Three.", {},
        evidence_frames=[{"label": "no frame captured", "image_b64": ""}])
    content = client.completions.calls[0]["messages"][1]["content"]
    assert len(content) == 2
    assert content[0]["image_url"]["url"].endswith("current-frame")
    assert "How many?" in content[1]["text"]


def test_inspect_guard_replays_its_bound_evidence_frames_on_every_call():
    client = _Client('{"match": true, "reason": "supported"}',
                     '{"match": true, "reason": "supported"}')
    guard = make_inspect_guard(
        client, "m", CONFIG, "actor-frame",
        evidence_frames=[{"label": "COKE (left hand, step 4)", "image_b64": "coke-frame"}])
    guard("Which is cheaper?", "Coke", {})
    guard("Which is cheaper?", "Sprite", {})
    for call in client.completions.calls:
        content = call["messages"][1]["content"]
        assert content[0]["text"] == "EVIDENCE 1: COKE (left hand, step 4)"
        assert content[1]["image_url"]["url"].endswith("coke-frame")


def test_inspection_failure_is_inconclusive_and_fails_closed():
    result = classify_inspection(
        _Client(TimeoutError("slow")), "m", CONFIG, "frame", "What date?", "2027-01-01", {})
    assert result["match"] is False and result["conclusive"] is False
    assert "TimeoutError" in result["reason"]


def test_inspection_visibility_uses_fresh_minimal_context_and_request():
    client = _Client(
        '{"match": false, "reason": "nutrition panel is angled"}',
        '{"match": true, "reason": "expiration date is directly legible"}',
    )
    first = classify_inspection_visibility(
        client, "m", CONFIG, "frame-one", "Read the nutritional facts.")
    second = classify_inspection_visibility(
        client, "m", CONFIG, "frame-two", "Read the expiration date.")

    assert first["match"] is False and second["match"] is True
    assert len(client.completions.calls) == 2
    first_call, second_call = client.completions.calls
    assert first_call["messages"] is not second_call["messages"]
    assert len(first_call["messages"]) == len(second_call["messages"]) == 2
    assert first_call["temperature"] == second_call["temperature"] == 0
    assert "frame-one" in first_call["messages"][1]["content"][0]["image_url"]["url"]
    assert "frame-two" in second_call["messages"][1]["content"][0]["image_url"]["url"]
    assert "nutritional facts" in first_call["messages"][1]["content"][1]["text"]
    assert "expiration date" in second_call["messages"][1]["content"][1]["text"]
    system = first_call["messages"][0]["content"]
    assert "fresh-frame visibility gate" in system
    assert "LEGIBILITY IS MANDATORY" in system
    assert "panel outline" in system and "relevant value rows must be readable" in system
    assert "every character of the complete date must be readable" in system
    assert "If uncertain" in system and "match=false" in system
    assert "Do not extract or report the values yourself" in system


def test_inspection_label_presence_locks_recognizable_unreadable_label_side():
    client = _Client(
        '{"match": true, "reason": "Nutrition Facts panel is recognizable but too small to read"}')
    result = classify_inspection_label_presence(
        client, "m", CONFIG, "label-frame", "Read the nutritional facts.")

    assert result["match"] is True and result["conclusive"] is True
    call = client.completions.calls[0]
    assert call["temperature"] == 0
    assert "label-frame" in call["messages"][1]["content"][0]["image_url"]["url"]
    assert "nutritional facts" in call["messages"][1]["content"][1]["text"]
    system = call["messages"][0]["content"]
    assert "less strict than the separate legibility gate" in system
    assert "STOP ROTATING" in system
    assert "DIRECTLY FACING THE CAMERA" in system
    assert "perspective foreshortening" in system
    assert "lines seen obliquely" in system and "match=false" in system
    assert "barcode" in system and "unrelated text" in system


def test_inspection_legibility_receives_untrusted_paddleocr_auxiliary_text():
    client = _Client('{"match": true, "reason": "front-facing values are readable"}')
    result = classify_inspection_visibility(
        client, "m", CONFIG, "frame", "Read calories.",
        ocr_lines=["Nutrition Facts", "Calories 140"])

    assert result["match"] is True
    call = client.completions.calls[0]
    user_text = call["messages"][1]["content"][1]["text"]
    assert 'PADDLEOCR AUXILIARY TEXT: ["Nutrition Facts", "Calories 140"]' in user_text
    system = call["messages"][0]["content"]
    assert "UNTRUSTED AUXILIARY EVIDENCE" in system
    assert "OCR never relaxes the front-facing requirement" in system


def test_image_bound_inspection_guard_caches_identical_checks_within_step():
    client = _Client('{"match": true, "reason": "visible"}',
                     '{"match": false, "reason": "different answer"}')
    events = []
    guard = make_inspect_guard(
        client, "m", CONFIG, "actor-frame", on_verdict=lambda *row: events.append(row))
    aux = {"gripped_name": None, "gripped_names": {}, "nearest_checkpoint": 10}
    first = guard("Count them", "Three", aux)
    second = guard("Count them", "Three", aux)
    third = guard("Count them", "Four", aux)
    assert first == second
    assert third["match"] is False
    assert guard.call_count == 2
    assert len(client.completions.calls) == 2
    assert events[0][-1] is False and events[1][-1] is True


def test_compare_uses_one_request_with_ordered_labeled_candidate_images():
    client = _Client('{"match": true, "reason": "candidate 1 shows less sugar"}')
    frames = [
        {"target": "Pocky", "image_b64": "pocky-frame"},
        {"target": "Hello Panda", "image_b64": "panda-frame"},
    ]
    result = classify_compare(
        client, "m", CONFIG, frames, "less sugar", "Pocky has less sugar",
        {"named_choice": "Pocky"})
    assert result["match"] is True and result["conclusive"] is True
    content = client.completions.calls[0]["messages"][1]["content"]
    assert content[0]["text"] == "CANDIDATE 1: Pocky"
    assert content[1]["image_url"]["url"].endswith("pocky-frame")
    assert content[2]["text"] == "CANDIDATE 2: Hello Panda"
    assert content[3]["image_url"]["url"].endswith("panda-frame")
    assert "CRITERION: less sugar" in content[4]["text"]


def test_compare_requires_complete_frames_and_fails_closed_without_call():
    client = _Client()
    result = classify_compare(
        client, "m", CONFIG, [("Pocky", "frame")], "less sugar", "Pocky", {})
    assert result["match"] is False and result["conclusive"] is False
    assert client.completions.calls == []


def test_compare_guard_reuses_identical_multi_image_verdict():
    client = _Client('{"match": true, "reason": "visible comparison"}')
    events = []
    guard = make_compare_guard(
        client, "m", CONFIG, [("A", "frame-a"), ("B", "frame-b")],
        on_verdict=lambda *row: events.append(row))
    aux = {"named_choice": "A"}
    assert guard("larger", "A is larger", aux)["match"] is True
    assert guard("larger", "A is larger", aux)["match"] is True
    assert guard.call_count == 1 and len(client.completions.calls) == 1
    assert events[0][-1] is False and events[1][-1] is True


def test_compare_frame_cache_is_first_seen_and_positionally_labeled():
    cache = {}
    targets = ["A", "B"]
    candidate_sets = [[30], [30]]
    assert cache_compare_candidate_frames(
        cache, targets, candidate_sets, 30, "first-frame", 2) == 2
    assert cache == {
        0: {"target": "A", "image_b64": "first-frame", "checkpoint": 30, "step": 2},
        1: {"target": "B", "image_b64": "first-frame", "checkpoint": 30, "step": 2},
    }
    assert cache_compare_candidate_frames(
        cache, targets, candidate_sets, 30, "later-frame", 3) == 0
    assert cache[0]["image_b64"] == cache[1]["image_b64"] == "first-frame"


def test_compare_frame_cache_ignores_unresolved_or_misaligned_candidates():
    cache = {}
    assert cache_compare_candidate_frames(
        cache, ["A", "B"], [[30]], 30, "frame", 1) == 0
    assert cache_compare_candidate_frames(
        cache, ["A", "B"], [[30], []], 31, "frame", 1) == 0
    assert cache == {}


def test_unknown_uses_current_frame_task_claim_and_context():
    client = _Client('{"match": true, "reason": "the requested bag is visibly held"}')
    result = classify_unknown(
        client, "m", CONFIG, "current-frame", "Obtain the green Piattos.",
        "The task is complete.", {"left_gripped": True})
    assert result["match"] is True and result["conclusive"] is True
    content = client.completions.calls[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].endswith("current-frame")
    assert "TASK: Obtain the green Piattos." in content[1]["text"]
    assert "COMPLETION CLAIM: The task is complete." in content[1]["text"]
    assert "left_gripped" in content[1]["text"]


def test_unknown_failure_and_guard_cache_fail_closed():
    failed = classify_unknown(
        _Client(TimeoutError("slow")), "m", CONFIG, "frame", "task", "done", {})
    assert failed["match"] is False and failed["conclusive"] is False
    client = _Client('{"match": false, "reason": "not visible"}')
    guard = make_unknown_guard(client, "m", CONFIG, "frame")
    assert guard("task", "done", {})["match"] is False
    assert guard("task", "done", {})["match"] is False
    assert guard.call_count == 1 and len(client.completions.calls) == 1


def test_every_classifier_bills_its_call_to_the_guard_role():
    """The guards must be one line item in the per-role token accounting, or an ablation that drops
    them cannot read its own saving. Asserted at the moment of the request - the meter itself reads
    the role from inside the SDK patch, which a fake client never reaches."""
    from agent_core import token_meter

    seen = []

    class _RoleSpy(_Completions):
        def create(self, **kwargs):
            seen.append(token_meter.current_role())
            return super().create(**kwargs)

    def _client(payload):
        client = _Client(payload)
        client.completions = _RoleSpy([payload])
        client.chat = SimpleNamespace(completions=client.completions)
        return client

    ok = '{"match": true, "reason": "visible"}'
    classify_pickup(_client(ok), "m", CONFIG, "f", "SKU", "target")
    classify_inspection(_client(ok), "m", CONFIG, "f", "query", "answer", {})
    classify_inspection_visibility(_client(ok), "m", CONFIG, "f", "query")
    classify_inspection_label_presence(_client(ok), "m", CONFIG, "f", "query")
    classify_compare(_client(ok), "m", CONFIG, [("A", "fa"), ("B", "fb")], "cheaper", "A", {})
    classify_unknown(_client(ok), "m", CONFIG, "f", "task", "claim", {})

    assert seen == [token_meter.ROLE_GUARD] * 6, seen
    # ...and the role does not leak past the call: the next thing the runtime does is not a guard.
    assert token_meter.current_role() == token_meter.UNATTRIBUTED
