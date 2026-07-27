"""Offline tests for the agent's bounded, quiet LLM retry policy."""

import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_core.agent import BaseAgent, OpenRouterConfig


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Client:
    def __init__(self, outcomes):
        self.completions = _Completions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class _Agent(BaseAgent):
    def __init__(self):
        self.config = OpenRouterConfig(model_id="test", api_key="test")


def test_api_call_recovers_quietly(monkeypatch, caplog):
    sleeps = []
    monkeypatch.setattr("agent_core.agent.time.sleep", sleeps.append)
    client = _Client([TimeoutError("down"), TimeoutError("still down"), "recovered"])

    assert _Agent()._api_call_with_retry(client, []) == "recovered"
    assert client.completions.calls == 3
    assert sleeps == [1, 2]
    assert caplog.records == []


def test_api_call_raises_original_error_after_ten_attempts(monkeypatch, caplog):
    monkeypatch.setattr("agent_core.agent.time.sleep", lambda _delay: None)
    final_error = TimeoutError("server stayed down")
    failures = [TimeoutError(f"timeout {n}") for n in range(9)] + [final_error]
    client = _Client(failures)

    with pytest.raises(TimeoutError) as raised:
        _Agent()._api_call_with_retry(client, [])

    assert raised.value is final_error
    assert client.completions.calls == 10
    assert caplog.records == []


def test_non_transient_programming_error_fails_immediately(monkeypatch):
    sleeps = []
    monkeypatch.setattr("agent_core.agent.time.sleep", sleeps.append)
    error = ValueError("bad response handling")
    client = _Client([error])

    with pytest.raises(ValueError) as raised:
        _Agent()._api_call_with_retry(client, [])

    assert raised.value is error
    assert client.completions.calls == 1
    assert sleeps == []
