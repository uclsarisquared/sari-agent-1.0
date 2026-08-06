"""Compact task journal and final user-facing response synthesis.

This module deliberately has no simulator or model imports.  The orchestrator supplies the one
completion callable, while these helpers own the crash-diagnostic journal, its bounded responder
view, and the deterministic fallback used when that completion fails.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Callable

from agent_core.prompt_loader import load_prompt

RESPONSE_MEMORY_FILE = "response_memory.json"
RESPONSE_FILE = "response.txt"
RESPONDER_MAX_CHARS = 24_000

RESPONDER_SYSTEM = load_prompt("orchestrator/responder")


def new_response_memory(prompt: str) -> dict[str, Any]:
    """Create the small, incrementally persisted journal for one original user request."""
    return {
        "version": 1,
        "prompt": str(prompt),
        "planned_subtasks": [],
        "attempts": [],
        "latest_episodic_reflection": "",
        "final": {
            "success": False,
            "completed_subtasks": [],
            "incomplete_subtasks": [],
            "location": None,
            "held_items": {},
            "checkout": None,
            "inspection": None,
            "failure_reason": "",
        },
    }


def _compact_subtask(subtask: Any) -> dict[str, Any]:
    """Keep only stable, response-relevant planning fields from a subtask."""
    if not isinstance(subtask, dict):
        return {"type": "unknown", "text": str(subtask)}
    # Resolver candidates are useful final-response evidence; arbitrary planner/debug additions are
    # not. Copy only stable task contract fields so the journal never becomes a planner dump.
    fields = (
        "type", "text", "target", "targets", "query", "destination", "candidates", "feasible",
    )
    return {key: copy.deepcopy(subtask[key]) for key in fields if key in subtask}


def set_planned_subtasks(memory: dict[str, Any], subtasks: list[Any]) -> None:
    """Store a compact snapshot of the task plan in the response journal."""
    memory["planned_subtasks"] = [_compact_subtask(subtask) for subtask in subtasks]


def concise_final_state(state: Any) -> dict[str, Any]:
    """Retain user-relevant state while excluding screenshots and large action/debug structures."""
    state = state if isinstance(state, dict) else {}
    location = state.get("nearest_checkpoint")
    if location is None:
        translation = state.get("translation")
        if isinstance(translation, (list, tuple)) and len(translation) >= 3:
            location = [translation[0], translation[2]]

    names = state.get("gripped_names")
    names = names if isinstance(names, dict) else {}
    held_items: dict[str, Any] = {}
    for side in ("left", "right"):
        if state.get(f"{side}GrippedState"):
            held_items[side] = names.get(side) or "unidentified item"

    checkout = state.get("last_checkout")
    if isinstance(checkout, dict):
        checkout = {
            key: copy.deepcopy(checkout[key])
            for key in ("scanned", "placed", "aligned", "reason")
            if key in checkout
        }
    else:
        checkout = None

    inspection = state.get("last_inspection")
    if isinstance(inspection, dict):
        inspection = {
            key: copy.deepcopy(inspection[key])
            for key in (
                "hand", "label_visible", "label_legible", "best_effort_read", "reason", "outcome",
            )
            if key in inspection
        }
    else:
        inspection = None

    return {
        "location": copy.deepcopy(location),
        "held_items": held_items,
        "checkout": checkout,
        "inspection": inspection,
    }


def _failure_reason(metrics: dict[str, Any]) -> str:
    """Choose the most actionable recorded reason for an incomplete attempt."""
    if metrics.get("success"):
        return ""
    state = metrics.get("final_state")
    state = state if isinstance(state, dict) else {}
    reason = state.get("last_halt_refused")
    return str(reason or metrics.get("failure_reason") or metrics.get("end_reason") or "incomplete")


def record_attempt(
    memory: dict[str, Any],
    *,
    leg_number: int,
    attempt_number: int,
    subtask: Any,
    metrics: dict[str, Any],
    episodic_reflection: str = "",
) -> dict[str, Any]:
    """Append one completed attempt immediately, including evidence omitted from summary.json."""
    entry = {
        "subtask_number": int(leg_number),
        "attempt": int(attempt_number),
        "subtask": _compact_subtask(subtask),
        "success": bool(metrics.get("success")),
        "end_reason": str(metrics.get("end_reason") or ""),
        "completion_evidence": str(metrics.get("completion_evidence") or ""),
        "verified_reported_answer": str(metrics.get("reported_answer") or ""),
        "failure_reason": _failure_reason(metrics),
        "final_state": concise_final_state(metrics.get("final_state")),
        "findings": "",
        "semantic_memory_delta": str(metrics.get("new_semantic_entries") or ""),
    }
    memory.setdefault("attempts", []).append(entry)
    if episodic_reflection:
        memory["latest_episodic_reflection"] = str(episodic_reflection)
    return entry


def attach_findings(
    memory: dict[str, Any], leg_number: int, attempt_number: int, findings: str
) -> None:
    """Attach a between-subtask findings report to the exact successful attempt it describes."""
    for entry in reversed(memory.get("attempts") or []):
        if (
            entry.get("subtask_number") == leg_number
            and entry.get("attempt") == attempt_number
        ):
            entry["findings"] = str(findings or "")
            return


def finalize_response_memory(
    memory: dict[str, Any], *, success: bool, planned_subtasks: list[Any] | None = None
) -> None:
    """Write the concise terminal state used by both the responder and deterministic fallback."""
    if planned_subtasks is not None:
        set_planned_subtasks(memory, planned_subtasks)

    attempts = memory.get("attempts") or []
    successful_numbers = {
        int(entry.get("subtask_number") or 0)
        for entry in attempts
        if entry.get("success")
    }
    planned = memory.get("planned_subtasks") or []
    completed = [
        subtask.get("text") or subtask.get("type") or f"subtask {index}"
        for index, subtask in enumerate(planned, 1)
        if index in successful_numbers
    ]
    incomplete = [
        subtask.get("text") or subtask.get("type") or f"subtask {index}"
        for index, subtask in enumerate(planned, 1)
        if index not in successful_numbers
    ]

    terminal = attempts[-1].get("final_state") if attempts else {}
    terminal = terminal if isinstance(terminal, dict) else {}
    latest_checkout = next(
        (
            entry.get("final_state", {}).get("checkout")
            for entry in reversed(attempts)
            if isinstance(entry.get("final_state"), dict)
            and entry["final_state"].get("checkout") is not None
        ),
        None,
    )
    latest_inspection = next(
        (
            entry.get("final_state", {}).get("inspection")
            for entry in reversed(attempts)
            if isinstance(entry.get("final_state"), dict)
            and entry["final_state"].get("inspection") is not None
        ),
        None,
    )
    failed = (
        next((entry for entry in reversed(attempts) if not entry.get("success")), None)
        if not success
        else None
    )
    memory["final"] = {
        "success": bool(success),
        "completed_subtasks": completed,
        "incomplete_subtasks": incomplete,
        "location": copy.deepcopy(terminal.get("location")),
        "held_items": copy.deepcopy(terminal.get("held_items") or {}),
        "checkout": copy.deepcopy(latest_checkout),
        "inspection": copy.deepcopy(latest_inspection),
        "failure_reason": str((failed or {}).get("failure_reason") or ""),
    }


def save_response_memory(run_dir: str | os.PathLike[str], memory: dict[str, Any]) -> Path:
    """Atomically publish the journal so a killed process leaves either old or complete JSON."""
    path = Path(run_dir) / RESPONSE_MEMORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temp, path)
    return path


def write_response_artifact(run_dir: str | os.PathLike[str], response: str) -> Path:
    """Atomically publish the final user-facing response text for a run."""
    path = Path(run_dir) / RESPONSE_FILE
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(str(response).strip() + "\n", encoding="utf-8")
    os.replace(temp, path)
    return path


def _serialized(view: dict[str, Any]) -> str:
    """Serialize a response-memory view compactly for its size-budget check."""
    return json.dumps(view, ensure_ascii=False, default=str, separators=(",", ":"))


def bounded_response_view(
    memory: dict[str, Any], max_chars: int = RESPONDER_MAX_CHARS
) -> dict[str, Any]:
    """Drop older narrative memory first while preserving task outcomes and verified answers."""
    view = copy.deepcopy(memory)
    view.pop("response", None)
    view.pop("response_source", None)
    if len(_serialized(view)) <= max_chars:
        return view

    # Findings and learned prose are useful context but never outrank outcomes. Remove them oldest
    # first; the most recent narrative survives longest.
    for entry in view.get("attempts") or []:
        for field in ("findings", "semantic_memory_delta"):
            if entry.get(field):
                entry[field] = ""
                if len(_serialized(view)) <= max_chars:
                    return view
    if view.get("latest_episodic_reflection"):
        view["latest_episodic_reflection"] = ""
        if len(_serialized(view)) <= max_chars:
            return view

    # Candidate lists and completion prose can be reconstructed neither as an outcome nor an answer,
    # so they are the next tier. The prompt, subtask text, success/end reason, verified answer, and
    # failure reason remain present and unmodified.
    for subtask in view.get("planned_subtasks") or []:
        subtask.pop("candidates", None)
    for entry in view.get("attempts") or []:
        subtask = entry.get("subtask")
        if isinstance(subtask, dict):
            subtask.pop("candidates", None)
        entry["completion_evidence"] = ""
        entry["final_state"] = {}
        if len(_serialized(view)) <= max_chars:
            return view
    return view


def deterministic_fallback(memory: dict[str, Any]) -> str:
    """Return a non-empty, user-facing answer using only verified journal facts."""
    attempts = memory.get("attempts") or []
    answers: list[str] = []
    for entry in attempts:
        answer = str(entry.get("verified_reported_answer") or "").strip()
        if entry.get("success") and answer and answer not in answers:
            answers.append(answer)
    if answers:
        # Answers are already model-authored natural language verified by the completion predicate.
        return " ".join(answers).strip()

    final = memory.get("final") if isinstance(memory.get("final"), dict) else {}
    completed = [str(text) for text in final.get("completed_subtasks") or [] if text]
    incomplete = [str(text) for text in final.get("incomplete_subtasks") or [] if text]
    if final.get("success"):
        return "Done — I completed the requested task."

    failed_task = incomplete[0] if incomplete else "the requested task"
    reason = str(final.get("failure_reason") or "").strip()
    friendly_reasons = {
        "time_cap": "the available time ran out",
        "step_cap": "I could not finish within the available actions",
        "errors": "repeated execution errors prevented completion",
        "halt_forced": "I could not verify that it was complete",
        "incomplete": "I could not verify that it was complete",
    }
    reason = friendly_reasons.get(reason, reason)
    if completed:
        prefix = f"I completed {', '.join(completed)}, but could not complete {failed_task}"
    else:
        prefix = f"I could not complete {failed_task}"
    return f"{prefix}: {reason}." if reason else f"{prefix}."


def synthesize_response(
    memory: dict[str, Any],
    completion: Callable[[str, str], str],
    *,
    max_chars: int = RESPONDER_MAX_CHARS,
) -> tuple[str, str]:
    """Make exactly one logical responder call, falling back on any exception or empty reply."""
    view = bounded_response_view(memory, max_chars=max_chars)
    user = "TASK JOURNAL (facts only):\n" + json.dumps(
        view, indent=2, ensure_ascii=False, default=str
    )
    try:
        response = str(completion(RESPONDER_SYSTEM, user) or "").strip()
    except Exception:  # noqa: BLE001 - the final response must survive model/provider failures
        response = ""
    if response:
        return response, "model"
    return deterministic_fallback(memory), "fallback"
