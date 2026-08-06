"""Actor invocation and action-batch execution for a leg step."""

import os
import re
import time
from dataclasses import dataclass, field

from orchestrator.action_dispatch import (
    _GRAB_ACTIONS,
    _INSPECT_MACRO_ACTIONS,
    _MACRO_ACTIONS,
    _grab_ready,
    dispatch_action,
)
from orchestrator.held_item_inspection import _inspection_action_batch
from orchestrator.subtask_completion import inspect_scope_violation


@dataclass
class ActionOutcome:
    """Aggregated effects of every primitive dispatched during one model step."""

    acted: list = field(default_factory=list)
    blocked_reason: object = False
    center_message: object = None
    last_reach: object = None
    grab_failed: bool = False
    checkout_result: dict | None = None
    inspection_result: dict | None = None
    inspect_move_left: int = 0


def invoke_actor(agent, request, step):
    """Execute the three-part actor turn and return its response plus exact call count."""
    previous_nav_task = getattr(agent, "_nav_task", None)
    advised_before = getattr(agent, "_advised_llm_calls", 0)
    response = agent.execute_lean(request, step)
    calls = 3 + getattr(agent, "_advised_llm_calls", 0) - advised_before
    if (
        agent.nav_mode in ("graph", "graph-advised")
        and getattr(agent, "_nav_task", None) != previous_nav_task
        and not getattr(agent, "_nav_seeded", None)
    ):
        calls += 1
    return response, calls


def _parse_inline_action(action):
    raw_action = action.strip()
    match = re.match(r'^(\w+)\([\'"]?(.*?)[\'"]?\)$', raw_action)
    if not match:
        return raw_action, None
    return match.group(1), match.group(2)


def execute_action_batch(
    *,
    agent,
    leg,
    state,
    parsed,
    mode,
    force_manipulate,
    inspect_move_left,
    step,
    stamp,
    artifacts,
    events,
    metrics,
    started_at,
    grip_tracker,
    inspection_evidence,
):
    """Dispatch a parsed actor response and aggregate its observable effects."""
    outcome = ActionOutcome(inspect_move_left=inspect_move_left)
    notes = parsed.get("notes", {})
    actions = list(zip(parsed.get("actions", []), parsed.get("times", [])))
    if leg.get("type") == "inspect" and force_manipulate:
        actions = _inspection_action_batch(
            parsed.get("actions", []), parsed.get("times", [])
        )

    center_dir = artifacts.center_dir(step, stamp)
    for action, duration in actions:
        raw_action, inline = _parse_inline_action(action)
        step_center = None
        if center_dir and raw_action == "center_object_on_screen":
            os.makedirs(center_dir, exist_ok=True)
            step_center = center_dir

        effective_mode = mode
        promoted = False
        if (
            raw_action in _GRAB_ACTIONS
            and mode not in (None, "manipulation")
            and _grab_ready(state)
        ):
            effective_mode = "manipulation"
            promoted = True

        scope_pre_state = state if leg.get("type") == "inspect" else None
        inspection_frames_dir = (
            artifacts.inspection_frames_dir(step)
            if raw_action in _INSPECT_MACRO_ACTIONS
            else None
        )
        result = dispatch_action(
            raw_action,
            int(duration),
            notes,
            inline_arg=inline,
            mode=effective_mode,
            debug_dir=step_center,
            agent=agent,
            leg_type=leg.get("type"),
            state=state,
            inspection_query=leg.get("query") or leg.get("text") or "",
            inspection_log=(
                (lambda row: events({"step": step, **row}))
                if raw_action in _INSPECT_MACRO_ACTIONS
                else None
            ),
            inspection_frames_dir=inspection_frames_dir,
            inspect_move_allowance=outcome.inspect_move_left,
        ) or {}

        moved = int(result.get("inspect_move_steps") or 0)
        if moved:
            outcome.inspect_move_left = max(0, outcome.inspect_move_left - moved)
            events.emit(
                "inspect_approach_step",
                step=step,
                action=raw_action,
                steps=moved,
                budget_left=outcome.inspect_move_left,
            )

        if scope_pre_state is not None:
            scope_event = inspect_scope_violation(raw_action, step, scope_pre_state, result)
            if scope_event:
                events(scope_event)

        if promoted and not result.get("blocked"):
            if metrics["t_manip"] is None:
                metrics["t_manip"] = round(time.time() - started_at, 1)
            events.emit(
                "grab_promoted",
                step=step,
                action=raw_action,
                router_mode=mode,
                gripped=result.get("gripped"),
            )

        if result.get("blocked"):
            outcome.blocked_reason = result.get("reason", True)
        if result.get("center_message"):
            outcome.center_message = result["center_message"]
        if result.get("last_reach"):
            outcome.last_reach = result["last_reach"]
        if raw_action in _MACRO_ACTIONS and not result.get("blocked"):
            outcome.checkout_result = result

        if raw_action in _INSPECT_MACRO_ACTIONS:
            outcome.inspection_result = result
            metrics["llm_calls"] += int(result.get("vlm_calls") or 0)
            evidence = inspection_evidence.record(result, grip_tracker, step)
            if evidence:
                events.emit(
                    "inspection_evidence_recorded",
                    step=step,
                    hand=evidence["hand"],
                    sku=evidence["sku"],
                    label_legible=evidence["label_legible"],
                    best_effort_read=evidence["best_effort_read"],
                    hands_covered=sorted(inspection_evidence.by_hand),
                )

        if raw_action in _GRAB_ACTIONS and not result.get("blocked"):
            verdict = result.get("reach_verdict")
            if verdict in (None, "reachable") and not result.get("gripped", False):
                outcome.grab_failed = True
            grip_tracker.record_grab(result)

        outcome.acted.append([raw_action, int(duration)])

    return outcome
