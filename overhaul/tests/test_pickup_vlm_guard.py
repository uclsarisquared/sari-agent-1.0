"""Offline tests for the stateless pickup VLM adapter (fake client only)."""

import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from orchestrator.pickup_vlm_guard import (
    classify_pickup,
    classify_inspection,
    evaluate_hands,
    make_inspect_guard,
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


def test_inspection_failure_is_inconclusive_and_fails_closed():
    result = classify_inspection(
        _Client(TimeoutError("slow")), "m", CONFIG, "frame", "What date?", "2027-01-01", {})
    assert result["match"] is False and result["conclusive"] is False
    assert "TimeoutError" in result["reason"]


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
