"""Whole-task orchestration, retries, runtime setup, and finalization."""

import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from sim import chime
from sim.env import TransformAgent, init_logger
from agent_core import token_meter
from agent_core.runtime import EmbodiedAgent
from agent_core.context_policy import resolve_context_policy
from orchestrator.leg_runner import run_leg
from orchestrator.orchestrator_llm import (
    ASSOCIATIVE_CONFIG,
    VLM_CONFIG,
    _generate_findings_if_enabled,
    _llm_call,
    _llm_client,
    decompose_task,
)
from orchestrator.subtask_completion import planned_subtask_metrics
from orchestrator.subtask_planning import (
    SPAWN_XZ,
    make_resolve_call,
    order_legs,
    plan_legs,
)
from orchestrator.task_response import (
    attach_findings,
    finalize_response_memory,
    new_response_memory,
    record_attempt,
    save_response_memory,
    set_planned_subtasks,
    synthesize_response,
    write_response_artifact,
)

_AGENT_DIR = str(Path(__file__).resolve().parent.parent)

def _load_store_map(output_dir=None):
    """Load the configured store map, using its default artifacts when unspecified."""
    from nav.store_map import StoreMap
    return StoreMap(output_dir=output_dir) if output_dir else StoreMap()


def _current_nearest_cp(sm):
    """The checkpoint nearest the agent's LIVE pose (a zero-delta TransformAgent is a read, not a
    move). Used to order legs from where the agent ACTUALLY is - spawn if we just reset, else wherever
    it happens to be. Falls back to the spawn corner if the pose read fails."""
    try:
        p = TransformAgent((0, 0, 0), (0, 0, 0))["translation"]
        return sm.nearest_checkpoint((p[0], p[2]))
    except Exception:  # noqa: BLE001
        return sm.nearest_checkpoint(SPAWN_XZ)


def _resolve_run_dir(run_dir, arm, runs_dir=None):
    """Return an absolute, existing, attempt-unique output directory.

    An explicit path is owned by the caller (Distributed Sari Bench creates one per attempt).  A
    local invocation gets an atomically-created directory, so even same-arm runs started in the
    same second cannot select the same fallback. `runs_dir` relocates just that fallback's parent
    (default agent/subtask_run_outputs/) while keeping the timestamped per-run name; it is
    ignored when `run_dir` pins an exact directory.
    """
    if run_dir:
        resolved = os.path.abspath(os.fspath(run_dir))
        os.makedirs(resolved, exist_ok=True)
        return resolved

    base = runs_dir or os.path.join(_AGENT_DIR, "subtask_run_outputs")
    os.makedirs(base, exist_ok=True)
    prefix = f"{datetime.now():%m%d_%H%M%S}_{arm}_"
    return tempfile.mkdtemp(prefix=prefix, dir=base)


@dataclass(frozen=True)
class OrchestrationConfig:
    """Stable inputs for one orchestration run."""

    task: str
    arm: str = "graph"
    caps: tuple = (0, 0.0)
    out: str | None = None
    run_dir: str | None = None
    resolver_backend: str = "endpoint"
    reset_start: bool = False
    restart_env: bool = False
    leg_retries: int = 1
    output_dir: str | None = None
    completion_guard: str = "deterministic"
    ocr_url: str | None = None
    runs_dir: str | None = None
    context_policy: str = "baseline"


@dataclass
class _RunState:
    """Mutable data shared by the orchestration phases."""

    config: OrchestrationConfig
    policy: object
    run_dir: str
    response_memory: dict
    resolved_ocr_url: str = ""
    client: object | None = None
    agent: object | None = None
    store_map: object | None = None
    legs: list = field(default_factory=list)
    leg_rows: list = field(default_factory=list)
    visited: set = field(default_factory=set)
    carried_names: dict | None = None
    cumulative_context: str = ""
    resolver_calls: int = 0
    llm_calls: int = 0
    success: bool = True
    started_at: float = 0.0
    error: BaseException | None = None
    ready_to_finalize: bool = False


def _new_run_state(config):
    """Create the run directory and persist the request before external setup."""
    policy = resolve_context_policy(config.context_policy)
    run_dir = _resolve_run_dir(
        config.run_dir, config.arm, runs_dir=config.runs_dir
    )
    os.environ["SARI_RUN_DIR"] = run_dir
    if config.ocr_url:
        os.environ["SARI_OCR_URL"] = config.ocr_url

    memory = new_response_memory(config.task)
    save_response_memory(run_dir, memory)
    return _RunState(
        config=config,
        policy=policy,
        run_dir=run_dir,
        response_memory=memory,
    )


def _initialize_runtime(state):
    """Check shared services and construct the model and embodied agent."""
    from vision.ocr_client import check_ocr_health, resolve_ocr_url

    config = state.config
    state.resolved_ocr_url = resolve_ocr_url(config.ocr_url)
    health = check_ocr_health(state.resolved_ocr_url)
    print(f"[ORCHESTRATOR] OCR ready: {health['model']} at {state.resolved_ocr_url}")

    # Install before any reasoner runs so planning tokens are included.
    token_meter.install(state.run_dir)
    state.client = _llm_client()
    init_logger(run_name="runtime", directory=state.run_dir)

    from sim.env import default_uri, wait_for_ready

    if not wait_for_ready():
        raise RuntimeError(
            f"Sandbox at {default_uri()} never reported ready; refusing to start the task against "
            "an environment that may still be mid-reset."
        )

    state.agent = EmbodiedAgent(
        vlm_config=VLM_CONFIG,
        associative_config=ASSOCIATIVE_CONFIG,
        mode="lean",
        nav_mode=config.arm,
        resolver_backend=config.resolver_backend,
        map_output_dir=config.output_dir,
        run_dir=state.run_dir,
        context_policy=state.policy,
    )
    state.started_at = time.time()
    state.ready_to_finalize = True
    token_meter.dump(state.run_dir)
    _print_run_header(state)


def _print_run_header(state):
    """Print the compact run configuration shown in interactive sessions."""
    config = state.config

    def cap(value, unit):
        return "unlimited" if not value else f"{value} {unit}"

    print(f"[ORCHESTRATOR] task: {config.task!r}")
    print(
        f"[ORCHESTRATOR] arm={config.arm}  context_policy={config.context_policy}  "
        f"completion_guard={config.completion_guard}  "
        f"caps={cap(config.caps[0], 'steps')} / {cap(config.caps[1], 'min')} per leg  "
        f"run dir: {state.run_dir}"
    )


def _plan_task(state):
    """Decompose the request and resolve all map targets once."""
    config = state.config
    subtasks = decompose_task(state.client, config.task)
    state.llm_calls = 1
    state.store_map = _load_store_map(config.output_dir)
    resolve_call = make_resolve_call(config.resolver_backend)
    state.legs, state.resolver_calls = plan_legs(state.store_map, resolve_call, subtasks)
    state.llm_calls += state.resolver_calls
    _save_plan(state)


def _save_plan(state):
    """Persist the latest plan for crash diagnosis."""
    set_planned_subtasks(state.response_memory, state.legs)
    save_response_memory(state.run_dir, state.response_memory)


def _prepare_environment(state):
    """Apply optional store and pose resets before ordering the plan."""
    config = state.config
    if config.restart_env:
        try:
            from sim.env import Reset as reset_environment

            reset_environment()
            # Unity reloads store objects asynchronously.
            time.sleep(1.5)
            print("[ORCHESTRATOR] hard env reset: store restored to initial state.")
        except Exception as error:  # noqa: BLE001 - resets are best effort
            print(f"[ORCHESTRATOR] restart_env skipped ({type(error).__name__}: {error})")

    if config.reset_start:
        try:
            from orchestrator.start_pose import return_to_start

            return_to_start(state.agent, output_dir=config.output_dir)
        except Exception as error:  # noqa: BLE001 - resets are best effort
            print(f"[ORCHESTRATOR] return_to_start skipped ({type(error).__name__}: {error})")


def _order_and_report_plan(state):
    """Order independent legs from the live pose and print the result."""
    state.legs = order_legs(
        state.store_map, state.legs, _current_nearest_cp(state.store_map)
    )
    _save_plan(state)

    print(
        f"[ORCHESTRATOR] {len(state.legs)} leg(s) "
        f"(resolver calls: {state.resolver_calls}):"
    )
    for index, leg in enumerate(state.legs, 1):
        infeasible = (
            "" if leg.get("feasible", True)
            else "  [INFEASIBLE: target resolved to no checkpoint]"
        )
        candidates = leg.get("candidates")
        print(
            f"  {index}. [{leg.get('type')}] {leg.get('text')}"
            + (f"  -> cps {candidates}" if candidates else "")
            + infeasible
        )

    infeasible = [
        index for index, leg in enumerate(state.legs, 1)
        if not leg.get("feasible", True)
    ]
    if infeasible:
        print(
            f"[ORCHESTRATOR] WARNING: leg(s) {infeasible} resolved to no checkpoint; "
            "running so the failure is measured, not assumed."
        )


def _retry_context(state, previous):
    """Add the prior rejection reason to a retry's context."""
    reason = (
        (previous.get("final_state") or {}).get("last_halt_refused")
        or previous["end_reason"]
    )
    context = state.cumulative_context + (
        "\n\n--- YOUR PREVIOUS ATTEMPT AT THIS EXACT SUBTASK FAILED "
        f"({previous['end_reason']}) ---\n"
        f"Why it was not accepted: {reason}\n"
        "Fix that specifically this time; everything you learned is still in memory."
    )
    return context, reason


def _record_leg_attempt(state, leg, leg_index, attempt, metrics, tokens_before):
    """Persist one attempt and add its compact metrics to the summary."""
    state.carried_names = (metrics.get("final_state") or {}).get("gripped_names")
    state.llm_calls += metrics["llm_calls"]
    leg_tokens = token_meter.delta(tokens_before)
    state.leg_rows.append(
        {
            **{
                key: value for key, value in metrics.items()
                if key not in ("final_state", "new_semantic_entries")
            },
            "attempt": attempt,
            "tokens_in": leg_tokens["tokens_in"],
            "tokens_out": leg_tokens["tokens_out"],
            "tokens_by_role": leg_tokens["by_role"],
        }
    )
    record_attempt(
        state.response_memory,
        leg_number=leg_index + 1,
        attempt_number=attempt,
        subtask=leg,
        metrics=metrics,
        episodic_reflection=getattr(state.agent.vlm_agent, "episodic_memory", ""),
    )
    save_response_memory(state.run_dir, state.response_memory)
    token_meter.dump()
    print(
        f"### leg {leg_index + 1} attempt {attempt} {metrics['end_reason']}: "
        f"success={metrics['success']} t_grip={metrics['t_grip']} "
        f"t_checkout={metrics['t_checkout']} steps={metrics['timesteps']} "
        f"halts_refused={metrics['halts_refused']} wall={metrics['wall_s']}s"
    )


def _run_leg_with_retries(state, leg, leg_index):
    """Run one leg until it succeeds or its retry budget is exhausted."""
    previous = None
    total_attempts = max(1, state.config.leg_retries + 1)
    future_legs = state.legs[leg_index + 1:]

    for attempt in range(1, total_attempts + 1):
        context = state.cumulative_context
        if previous is not None:
            context, reason = _retry_context(state, previous)
            print(
                f"[ORCHESTRATOR] retrying leg {leg_index + 1} "
                f"(attempt {attempt}/{total_attempts}): {reason}"
            )

        suffix = "" if attempt == 1 else f"_retry{attempt - 1}"
        tokens_before = token_meter.snapshot()
        metrics = run_leg(
            state.agent,
            leg,
            state.store_map,
            state.config.caps,
            log_path=os.path.join(state.run_dir, f"leg{leg_index:02d}{suffix}.jsonl"),
            context=context,
            future_legs=future_legs,
            visited=state.visited,
            leg_idx=leg_index + 1,
            completion_guard=state.config.completion_guard,
            carried_names=state.carried_names,
        )
        _record_leg_attempt(state, leg, leg_index, attempt, metrics, tokens_before)
        if metrics["success"]:
            return metrics, attempt
        previous = metrics

    return previous, total_attempts


def _carry_findings_forward(state, leg, leg_index, attempt, metrics):
    """Generate and persist the optional between-leg findings summary."""
    if state.policy.findings_enabled:
        print("[ORCHESTRATOR] Generating findings summary...")
    findings = _generate_findings_if_enabled(
        state.policy,
        state.client,
        completed_subtask=leg.get("text", ""),
        final_state=metrics["final_state"],
        new_semantic_entries=metrics["new_semantic_entries"],
    )
    if findings is None:
        return

    state.llm_calls += 1
    attach_findings(state.response_memory, leg_index + 1, attempt, findings)
    save_response_memory(state.run_dir, state.response_memory)
    print(f"[FINDINGS SUMMARY]\n{findings}\n")
    state.cumulative_context += f"\n\n--- LEG {leg_index + 1} FINDINGS ---\n{findings}"


def _execute_plan(state):
    """Run planned legs in order, aborting after an exhausted failure."""
    for leg_index, leg in enumerate(state.legs):
        print(f"\n[ORCHESTRATOR] ── Leg {leg_index + 1}/{len(state.legs)} ──")
        metrics, attempt = _run_leg_with_retries(state, leg, leg_index)
        if not metrics["success"]:
            state.success = False
            remaining = len(state.legs) - leg_index - 1
            print(
                f"[ORCHESTRATOR] leg {leg_index + 1} did not complete "
                f"({metrics['end_reason']}) — aborting the remaining {remaining} leg(s)."
            )
            return
        if leg_index + 1 < len(state.legs):
            _carry_findings_forward(state, leg, leg_index, attempt, metrics)


def _finalize_response(state):
    """Finalize the journal and synthesize the user-facing response."""
    finalize_response_memory(
        state.response_memory, success=state.success, planned_subtasks=state.legs
    )
    if state.error is not None and not state.response_memory["final"].get("failure_reason"):
        state.response_memory["final"]["failure_reason"] = (
            f"{type(state.error).__name__}: {state.error}"
        )
    save_response_memory(state.run_dir, state.response_memory)

    state.llm_calls += 1
    response, source = synthesize_response(
        state.response_memory,
        lambda system, user: _llm_call(
            state.client, system, user, token_meter.ROLE_RESPONDER
        ),
    )
    state.response_memory["response"] = response
    state.response_memory["response_source"] = source
    save_response_memory(state.run_dir, state.response_memory)
    write_response_artifact(state.run_dir, response)
    return response, source


def _build_summary(state, response, response_source):
    """Build the stable summary.json payload."""
    config = state.config
    token_totals = token_meter.totals()
    summary = {
        "task": config.task,
        "arm": config.arm,
        "context_policy": asdict(state.policy),
        "completion_guard": config.completion_guard,
        "ocr_url": state.resolved_ocr_url,
        "run_config": {
            "arm": config.arm,
            "context_policy": config.context_policy,
            "max_steps": config.caps[0],
            "max_minutes": config.caps[1],
            "resolver_backend": config.resolver_backend,
            "completion_guard": config.completion_guard,
            "leg_retries": config.leg_retries,
            "map_dir": str(Path(config.output_dir).resolve()) if config.output_dir else None,
            "reset_start": config.reset_start,
            "restart_env": config.restart_env,
            "ocr_url": state.resolved_ocr_url,
        },
        "success": state.success,
        "response": response,
        "response_source": response_source,
        "legs_planned": len(state.legs),
        "legs_completed": sum(1 for row in state.leg_rows if row.get("success")),
        "resolver_calls": state.resolver_calls,
        "llm_calls": state.llm_calls,
        "tokens_in": token_totals["tokens_in"],
        "tokens_out": token_totals["tokens_out"],
        "tokens": token_totals,
        "wall_s": round(time.time() - state.started_at, 1),
        "legs": state.leg_rows,
    }
    summary.update(planned_subtask_metrics(state.legs))
    if config.arm == "graph-advised":
        advised = getattr(state.agent, "_advised_stats", [])
        summary["advised"] = {
            "hops": len(advised),
            "agreed": sum(1 for item in advised if item["agreed"]),
            "invalid": sum(1 for item in advised if item["invalid"]),
            "stops": sum(1 for item in advised if item["stop_here"]),
        }
    return summary


def _write_summary(state, summary):
    """Write and report the completed run summary."""
    out_path = state.config.out or os.path.join(state.run_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    token_meter.dump()
    print("-" * 40)
    print(
        f"[ORCHESTRATOR] task success={state.success}  "
        f"legs {summary['legs_completed']}/{summary['legs_planned']}  "
        f"llm={state.llm_calls}  "
        f"tokens in/out={summary['tokens_in']}/{summary['tokens_out']}  "
        f"wall={summary['wall_s']}s  -> {out_path}"
    )
    print("-" * 40)
    print(f"[RESPONSE]\n{summary['response']}")


def _finalize_run(state):
    """Produce response and summary artifacts for a started runtime."""
    response, source = _finalize_response(state)
    summary = _build_summary(state, response, source)
    _write_summary(state, summary)
    return summary


def _close_run(state):
    """Release the agent without letting notification failures break the run."""
    if state.agent is not None:
        try:
            state.agent.close()
        except Exception:  # noqa: BLE001 - cleanup is best effort
            pass
    if state.ready_to_finalize:
        try:
            chime.beep()
        except Exception:  # noqa: BLE001 - a chime is not part of task success
            pass


def orchestrate(config: OrchestrationConfig):
    """Plan and execute a task, then persist its response and metrics."""
    state = _new_run_state(config)
    summary = None
    active_error = None

    try:
        _initialize_runtime(state)
        _plan_task(state)
        _prepare_environment(state)
        _order_and_report_plan(state)
        _execute_plan(state)
    except BaseException as error:
        state.success = False
        state.error = error
        active_error = error
        raise
    finally:
        try:
            if state.ready_to_finalize:
                summary = _finalize_run(state)
        except Exception as finalization_error:
            if active_error is None:
                raise
            print(
                f"[ORCHESTRATOR] finalization also failed "
                f"({type(finalization_error).__name__}: {finalization_error})",
                file=sys.stderr,
            )
        finally:
            _close_run(state)

    return summary
