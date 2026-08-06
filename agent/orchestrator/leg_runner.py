"""Execution loop for one typed subtask leg."""

import base64
import itertools
import time
from datetime import datetime

from sim.env import _REQUEST_SCREENSHOT_
from orchestrator.action_dispatch import (
    _INSPECT_MOVE_BUDGET_STEPS,
    parse_actor_response,
)
from orchestrator.leg_actions import execute_action_batch, invoke_actor
from orchestrator.leg_artifacts import LegArtifacts, write_step_output
from orchestrator.leg_completion import (
    CompletionController,
    deterministic_guard_details,
)
from orchestrator.leg_runtime import (
    GripTracker,
    InspectionEvidence,
    build_augmented_task,
    fresh_agent_state,
    initial_metrics,
    model_facing_state,
    normalize_leg,
    reconcile_after_actions,
)
from orchestrator.leg_session import LegSession, StepContext, StepDisposition
from orchestrator.subtask_completion import (
    held_item_inspection_active,
)

# Compatibility exports retained for existing evaluation scripts and tests.
_fresh_agent_state = fresh_agent_state
_model_facing_state = model_facing_state
_deterministic_guard_details = deterministic_guard_details


# Allow one hop for fine approach motion near a resolved target checkpoint.
_AT_TARGET_HOP_MARGIN = 1

_LOCATION_GATED_TYPES = {"pickup", "goto"}


def _off_target(sm, leg, near_cp) -> bool:
    """Return whether graph-grounded work is outside every resolved target checkpoint."""
    # Only pickup and goto require work at one resolved destination.
    if leg.get("type") not in _LOCATION_GATED_TYPES:
        return False

    # Missing candidates or localization cannot justify forced navigation.
    cands = [c for c in (leg.get("candidates") or []) if c in sm.by_id]
    if not cands or near_cp is None:
        return False

    # Gate only when every connected candidate is beyond the fine-approach margin.
    dists = [h for h in (sm.hops(near_cp, c) for c in cands) if h is not None]
    if not dists:
        return False
    return min(dists) > _AT_TARGET_HOP_MARGIN


def _run_leg_impl(agent, leg, sm, caps, log_path=None, context="", future_legs=None,
                  visited=None, leg_idx=0, completion_guard="deterministic", carried_names=None):
    """Run one leg while guaranteeing that its artifact handles are closed."""
    # Artifact ownership wraps every terminal path, including unexpected exceptions.
    with LegArtifacts(log_path) as artifacts:
        return _run_leg_loop(
            agent,
            leg,
            sm,
            caps,
            context=context,
            future_legs=future_legs,
            visited=visited,
            leg_idx=leg_idx,
            completion_guard=completion_guard,
            carried_names=carried_names,
            artifacts=artifacts,
        )


def _run_leg_loop(agent, leg, sm, caps, context="", future_legs=None,
                  visited=None, leg_idx=0, completion_guard="deterministic", carried_names=None,
                  artifacts=None):
    """Run the capture, reason, dispatch, reconcile, and completion phases."""
    # Build the durable context once; each iteration then owns only a StepContext.
    session = _start_leg_session(
        agent, leg, sm, caps, context, future_legs, visited, leg_idx,
        completion_guard, carried_names, artifacts,
    )

    # A zero step cap deliberately means an unbounded iterator.
    step_numbers = (
        itertools.count(1)
        if not session.max_steps
        else range(1, session.max_steps + 1)
    )
    for step_number in step_numbers:
        # Capture the frame and bind completion guards before asking the actor to reason.
        if _time_cap_reached(session):
            session.metrics["end_reason"] = "time_cap"
            break
        step = _prepare_step(session, step_number)
        if step.guards.terminal:
            break

        # Actor failures consume a step; repeated failures terminate the leg.
        if not _invoke_step(session, step):
            if session.metrics["end_reason"] == "errors":
                break
            continue

        # STOP and parse failures do not enter action dispatch.
        disposition = _classify_response(session, step)
        if disposition is StepDisposition.STOP:
            break
        if disposition is StepDisposition.NEXT:
            continue

        # Apply actions, reconcile live simulator state, then evaluate completion again.
        _dispatch_and_observe(session, step)
        if step.observation.terminal:
            break

    # Normalize the terminal reason and attach final state/memory to the public result.
    return _finish_leg_session(session)


def _start_leg_session(agent, leg, sm, caps, context, future_legs, visited,
                       leg_idx, completion_guard, carried_names, artifacts):
    # Normalize caller inputs and reset only the agent's per-leg conversation state.
    leg = normalize_leg(leg)
    visited = visited if visited is not None else set()
    print(f"\n[LEG {leg_idx}] ({leg.get('type')}) {leg.get('text') or ''}")
    if context:
        print(f"[CONTEXT] {context}")
    semantic_before = agent.begin_leg(
        leg.get("candidates"), leg.get("target_name"), leg_idx
    )

    # Start metrics and structured logging from the same wall-clock origin.
    started_at = time.time()
    artifacts.started_at = started_at
    events = artifacts.event_logger(leg_idx)
    metrics = initial_metrics(leg, completion_guard)

    # Seed localization, visit tracking, and durable evidence from the live simulator snapshot.
    state = _fresh_agent_state()
    nearest = sm.nearest_checkpoint(
        (state["translation"][0], state["translation"][2])
    )
    visited.add(nearest)
    state["nearest_checkpoint"] = nearest
    grip_tracker = GripTracker.from_state(state, carried_names)
    inspection_evidence = InspectionEvidence()

    # Completion policy shares the same metrics, event stream, and state reader as the session.
    completion = CompletionController(
        agent, leg, completion_guard, metrics, events, leg_idx,
        state_reader=_fresh_agent_state,
    )

    # Collect every value that must survive across step boundaries in one mutable context.
    session = LegSession(
        agent=agent,
        leg=leg,
        store_map=sm,
        artifacts=artifacts,
        events=events,
        metrics=metrics,
        state=state,
        visited=visited,
        grip_tracker=grip_tracker,
        inspection_evidence=inspection_evidence,
        completion=completion,
        augmented_task=build_augmented_task(leg, context, future_legs or []),
        semantic_before=semantic_before,
        started_at=started_at,
        max_steps=caps[0],
        max_minutes=caps[1],
        leg_idx=leg_idx,
        inspect_move_left=(
            _INSPECT_MOVE_BUDGET_STEPS if leg.get("type") == "inspect" else 0
        ),
    )

    # Emit lifecycle metadata only after the session is fully initialized.
    events.emit(
        "leg_start",
        type=leg.get("type"),
        text=session.text,
        candidates=leg.get("candidates"),
        arm=agent.nav_mode,
        completion_guard=completion_guard,
        ts=datetime.now().isoformat(timespec="seconds"),
    )
    return session


def _time_cap_reached(session):
    # A zero time cap disables the wall-clock limit.
    return bool(
        session.max_minutes
        and (time.time() - session.started_at) / 60 > session.max_minutes
    )


def _prepare_step(session, number):
    # Capture one native frame and save its bounded debug copy under the same timestamp.
    session.metrics["timesteps"] = number
    stamp = f"_{datetime.now():%m%d_%H%M%S}"
    image_bytes = _REQUEST_SCREENSHOT_()["image"]
    session.artifacts.save_frame(number, stamp, image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Bind completion guards to this exact pre-action frame and durable visit state.
    session.state["visited_checkpoints"] = set(session.visited)
    guards = session.completion.prepare(
        session.state,
        image_b64,
        number,
        session.inspection_evidence,
        session.last_actor_text,
    )
    step = StepContext(number=number, stamp=stamp, image_b64=image_b64, guards=guards)
    if guards.terminal:
        return step

    # Resolve location and inspection mode before constructing the actor request.
    candidates = (
        [
            candidate
            for candidate in (session.leg.get("candidates") or [])
            if candidate in session.store_map.by_id
        ]
        if session.leg.get("type") in _LOCATION_GATED_TYPES
        else []
    )
    step.target_checkpoints = candidates or None
    session.state["target_checkpoints"] = step.target_checkpoints
    step.off_target = _off_target(
        session.store_map, session.leg, session.state.get("nearest_checkpoint")
    )
    step.force_manipulate = held_item_inspection_active(session.leg, session.state)
    inspect_mode = (
        "held" if step.force_manipulate else "visual"
    ) if session.leg.get("type") == "inspect" else None
    if step.off_target:
        print(
            f"[GATE] off-target: at cp{session.state.get('nearest_checkpoint')}, "
            f"target at {candidates} — forcing navigation this step."
        )

    # Navigation sees the bare goal; the actor receives the augmented task and lean state.
    step.request = {
        "task": session.augmented_task,
        "nav_goal": session.text,
        "force_navigate": step.off_target,
        "force_manipulate": step.force_manipulate,
        "inspect_mode": inspect_mode,
        "image": image_b64,
        "state": _model_facing_state(session.state),
    }
    return step


def _invoke_step(session, step):
    # Treat actor failures as recoverable step errors until the shared error cap is reached.
    try:
        step.response, calls = invoke_actor(
            session.agent, step.request, step.number
        )
        session.metrics["llm_calls"] += calls
    except Exception as error:  # noqa: BLE001
        session.metrics["errors"] += 1
        print(
            f"    [leg {session.leg_idx} step {step.number}] execute_lean error: "
            f"{type(error).__name__}: {error}"
        )
        session.events.failure("error", step.number, error)
        if session.metrics["errors"] >= 3:
            session.metrics["end_reason"] = "errors"
        return False

    # Persist the successful response and record the first manipulation transition.
    session.artifacts.save_response(step.number, step.stamp, step.response)
    step.mode = step.response.get("agent_mode")
    if step.mode == "manipulation" and session.metrics["t_manip"] is None:
        session.metrics["t_manip"] = round(time.time() - session.started_at, 1)
    return True


def _classify_response(session, step):
    # STOP requests go through completion policy and never dispatch actor actions.
    if step.response.get("halt"):
        decision = session.completion.handle_stop(
            step.response,
            session.state,
            session.last_actor_text,
            step.guards,
            step.number,
            session.grip_tracker,
        )
        return (
            StepDisposition.STOP
            if decision in ("granted", "forced")
            else StepDisposition.NEXT
        )

    # A valid action payload advances to dispatch and becomes the next completion context.
    step.parsed = parse_actor_response(
        step.response.get("text") or "",
        session.agent.vlm_agent.extractable_json_structured_output,
    )
    if step.parsed is not None:
        session.last_actor_text = step.response.get("text") or ""
        return StepDisposition.DISPATCH

    # Invalid payloads consume the step and eventually terminate through the shared error cap.
    session.metrics["errors"] += 1
    session.events.emit(
        "parse_error",
        step=step.number,
        raw=(step.response.get("text") or "")[:400],
    )
    if session.metrics["errors"] >= 3:
        session.metrics["end_reason"] = "errors"
        return StepDisposition.STOP
    return StepDisposition.NEXT


def _dispatch_and_observe(session, step):
    # Dispatch the complete batch while keeping its effects grouped in ActionOutcome.
    step.outcome = execute_action_batch(
        agent=session.agent,
        leg=session.leg,
        state=session.state,
        parsed=step.parsed,
        mode=step.mode,
        force_manipulate=step.force_manipulate,
        inspect_move_left=session.inspect_move_left,
        step=step.number,
        stamp=step.stamp,
        artifacts=session.artifacts,
        events=session.events,
        metrics=session.metrics,
        started_at=session.started_at,
        grip_tracker=session.grip_tracker,
        inspection_evidence=session.inspection_evidence,
    )

    # Carry inspection-specific budget and evidence into subsequent steps.
    session.inspect_move_left = step.outcome.inspect_move_left
    if step.outcome.inspection_result is not None:
        session.last_inspection_result = step.outcome.inspection_result

    # Refresh from the simulator, then merge durable action and evidence channels.
    session.state, step.near_checkpoint = reconcile_after_actions(
        outcome=step.outcome,
        mode=step.mode,
        last_inspection_result=session.last_inspection_result,
        store_map=session.store_map,
        visited=session.visited,
        grip_tracker=session.grip_tracker,
        inspection_evidence=session.inspection_evidence,
        metrics=session.metrics,
        started_at=session.started_at,
        read_state=_fresh_agent_state,
    )

    # Evaluate the refreshed state and emit one assembled record for the completed step.
    step.observation = session.completion.observe_after_action(
        session.state, session.last_actor_text, step.guards, step.number
    )
    session.events.step(step, session)


def _finish_leg_session(session):
    # Falling through a finite iterator is the only path that defaults to step_cap.
    metrics = session.metrics
    if metrics["end_reason"] is None:
        metrics["end_reason"] = "step_cap"
    metrics["wall_s"] = round(time.time() - session.started_at, 1)
    metrics["final_state"] = session.state

    # Attach memory learned during this leg before writing the terminal lifecycle event.
    metrics["new_semantic_entries"] = (
        session.agent.vlm_agent.semantic_log.since(session.semantic_before)
    )
    session.events.emit(
        "leg_end", **{key: value for key, value in metrics.items() if key != "final_state"}
    )
    return metrics


def run_leg(agent, leg, sm, caps, log_path=None, context="", future_legs=None,
            visited=None, leg_idx=0, completion_guard="deterministic", carried_names=None):
    """Run one typed leg and restore inspection hand poses on every exit path."""
    # Normalize only for cleanup routing; the implementation retains its compatibility input.
    typed_leg = leg if isinstance(leg, dict) else {"type": "unknown", "text": str(leg)}
    result = None

    # Keep inspection cleanup outside the execution loop so every exit path reaches it.
    try:
        result = _run_leg_impl(
            agent, leg, sm, caps, log_path=log_path, context=context,
            future_legs=future_legs, visited=visited, leg_idx=leg_idx,
            completion_guard=completion_guard, carried_names=carried_names)
        return result
    finally:
        if typed_leg.get("type") == "inspect":
            cleanup = None

            # Restore canonical hand transforms and refresh the returned physical snapshot.
            try:
                restore = getattr(agent, "restore_hands_after_inspection", None)
                restore = restore or getattr(agent, "_restore_hands_after_inspection")
                cleanup = restore()
                if not cleanup.get("restored"):
                    agent._hand_pose = None
                if result is not None and cleanup.get("restored"):
                    refreshed = _fresh_agent_state()
                    prior = result.get("final_state") or {}
                    prior.update(refreshed)
                    result["final_state"] = prior
            except Exception as cleanup_error:  # noqa: BLE001 - never mask the leg result/error
                # Reset pose tracking if restoration fails, without masking the leg outcome.
                try:
                    agent._hand_pose = None
                except Exception:
                    pass
                cleanup = {
                    "restored": False,
                    "error": f"{type(cleanup_error).__name__}: {cleanup_error}",
                }
                print(f"[WARN] inspect cleanup failed: {cleanup['error']}")

            # Surface cleanup status to both the caller and the structured event trail.
            if result is not None:
                result["inspection_cleanup"] = cleanup
            if log_path:
                try:
                    with LegArtifacts(log_path) as cleanup_artifacts:
                        cleanup_artifacts.event_logger(leg_idx).emit(
                            "inspect_cleanup",
                            **(cleanup or {"restored": False}),
                        )
                except Exception as log_error:  # noqa: BLE001 - logging cannot mask the leg outcome
                    print(f"[WARN] could not log inspect cleanup: "
                          f"{type(log_error).__name__}: {log_error}")
