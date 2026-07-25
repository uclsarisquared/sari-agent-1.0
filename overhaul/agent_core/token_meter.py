"""Process-wide token accounting for every LLM call the agent makes.

WHY A PATCH AND NOT PER-CALL-SITE COUNTERS: the token cost of one benchmark attempt is spread over
call sites in agent_core.agent (actor + semantic/episodic learner + advisor), vision.perception
(bbox/centering/OCR reasoning), vision.md_tools' qwen fallback, orchestrator.subtask_agents
(decomposer + findings summary), slamtest.drivers.vlm_planner and the map resolver. Every one of
them goes through the OpenAI SDK's ``chat.completions.create`` against the same UCL vLLM server, so
patching that single method counts ALL of them - including the retries inside
``BaseAgent._api_call_with_retry``, which a per-call-site counter would silently miss even though
the server was really billed for them.

NOT COUNTED, deliberately: moondream (vision/md_tools) is a different API that reports no token
usage at all, so there is nothing to add up; its qwen ERROR-fallback path does go through the SDK
and is counted. Streamed responses carry no ``usage`` either and land in ``untracked_calls`` rather
than being guessed at - the repo streams nowhere today, so that counter should stay 0.

Usage (see orchestrator/subtask_agents.py):

    from agent_core import token_meter
    token_meter.install(run_dir)      # once, before any LLM traffic
    before = token_meter.snapshot()
    ...
    token_meter.delta(before)         # {"tokens_in": .., "tokens_out": .., "calls": ..}
    token_meter.dump()                # tokens.json, also auto-dumped every DUMP_INTERVAL_S

``install`` is idempotent, so importing this from two places cannot double-count.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

# tokens.json is rewritten at most this often from inside the patch. The file exists so an attempt
# that is SIGKILLed (harness timeout, operator kill) still leaves its token cost behind - summary.json
# is only written at exit, and those attempts never reach it. Cheap: one small atomic write.
DUMP_INTERVAL_S = 10.0

TOKENS_FILE = "tokens.json"

_lock = threading.Lock()
_installed = False
_run_dir: Optional[str] = None
_last_dump = 0.0

_totals: dict[str, Any] = {
    "tokens_in": 0,          # prompt tokens billed to the server
    "tokens_out": 0,         # completion tokens (reasoning tokens included, per the server's count)
    "calls": 0,              # SDK calls that reported usage
    "untracked_calls": 0,    # SDK calls with no usage on the response (streaming); see module docstring
    "by_model": {},          # model_id -> {"tokens_in", "tokens_out", "calls"}
}


def install(run_dir: Optional[str] = None) -> None:
    """Patches the OpenAI SDK so every chat completion is counted. Safe to call more than once."""
    global _installed, _run_dir, _totals

    if run_dir:
        _run_dir = run_dir

    if _installed:
        return

    try:
        from openai.resources.chat import completions as _completions
    except ImportError:  # noqa: BLE001 - counting is bookkeeping, never a reason to fail a run
        return

    original_create = _completions.Completions.create
    if getattr(original_create, "_sari_token_meter", False):
        # Already patched by another copy of this module (subtask_agents imports it as
        # `agent_core.token_meter`, a different sys.path root would import it again) - wrapping the
        # wrapper would double-count every call. Adopt the counters that copy is already filling, so
        # reading the totals through EITHER copy sees the same run.
        _totals = original_create._sari_totals
        _installed = True
        return

    def create(self, *args, **kwargs):
        response = original_create(self, *args, **kwargs)
        try:
            _record(kwargs.get("model"), getattr(response, "usage", None))
        except Exception:  # noqa: BLE001 - never let accounting break a call that succeeded
            pass
        return response

    create.__wrapped__ = original_create
    create._sari_token_meter = True  # the marker the double-install guard above reads
    create._sari_totals = _totals    # ...and the counters it adopts
    _completions.Completions.create = create
    _installed = True


def _record(model: Any, usage: Any) -> None:
    global _last_dump

    with _lock:
        if usage is None:
            _totals["untracked_calls"] += 1
            return

        # The vLLM server speaks the OpenAI schema, so usage is an object; a dict shows up when a
        # response has been round-tripped through JSON (annotate_qwen's envelope replay).
        if isinstance(usage, dict):
            tokens_in = usage.get("prompt_tokens") or 0
            tokens_out = usage.get("completion_tokens") or 0
        else:
            tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(usage, "completion_tokens", 0) or 0

        _totals["tokens_in"] += int(tokens_in)
        _totals["tokens_out"] += int(tokens_out)
        _totals["calls"] += 1

        key = str(model or "unknown")
        row = _totals["by_model"].setdefault(key, {"tokens_in": 0, "tokens_out": 0, "calls": 0})
        row["tokens_in"] += int(tokens_in)
        row["tokens_out"] += int(tokens_out)
        row["calls"] += 1

        due = time.monotonic() - _last_dump >= DUMP_INTERVAL_S
        totals = _copy_locked()

    if due and _run_dir:
        _last_dump = time.monotonic()
        _write(totals)


def _copy_locked() -> dict[str, Any]:
    """A deep-enough copy of the totals. Caller holds the lock."""
    snapshot = dict(_totals)
    snapshot["by_model"] = {model: dict(row) for model, row in _totals["by_model"].items()}
    return snapshot


def snapshot() -> dict[str, Any]:
    """Totals so far. Take one before a leg and pass it to ``delta`` after."""
    with _lock:
        return _copy_locked()


def delta(before: dict[str, Any]) -> dict[str, int]:
    """How many tokens were spent since ``before`` was taken."""
    now = snapshot()
    return {
        "tokens_in": now["tokens_in"] - int(before.get("tokens_in", 0)),
        "tokens_out": now["tokens_out"] - int(before.get("tokens_out", 0)),
        "calls": now["calls"] - int(before.get("calls", 0)),
    }


def totals() -> dict[str, Any]:
    """Totals with the derived ``tokens_total``, shaped for summary.json."""
    payload = snapshot()
    payload["tokens_total"] = payload["tokens_in"] + payload["tokens_out"]
    return payload


def dump(run_dir: Optional[str] = None) -> None:
    """Writes tokens.json now. No-op until a run dir is known."""
    global _run_dir, _last_dump
    if run_dir:
        _run_dir = run_dir
    if not _run_dir:
        return
    _last_dump = time.monotonic()
    _write(totals())


def _write(payload: dict[str, Any]) -> None:
    """Atomic so the benchmark runner, which may read this while the agent is still running, never
    sees a half-written file."""
    if not _run_dir:
        return
    payload = dict(payload)
    payload.setdefault("tokens_total", payload["tokens_in"] + payload["tokens_out"])
    path = os.path.join(_run_dir, TOKENS_FILE)
    temp = path + ".tmp"
    try:
        os.makedirs(_run_dir, exist_ok=True)
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temp, path)
    except OSError:
        pass  # a run dir that cannot be written to is not worth killing the attempt over


def reset() -> None:
    """Zeroes the counters (tests only - a real process meters one run from start to finish)."""
    with _lock:
        _totals["tokens_in"] = 0
        _totals["tokens_out"] = 0
        _totals["calls"] = 0
        _totals["untracked_calls"] = 0
        _totals["by_model"] = {}
