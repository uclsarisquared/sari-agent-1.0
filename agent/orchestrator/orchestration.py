"""Whole-task orchestration, retries, runtime setup, and finalization."""

import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from sim import chime
from sim.env import TransformAgent, init_logger
from agent_core import token_meter
from agent_core.agent import EmbodiedAgent
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

_OVERHAUL_DIR = str(Path(__file__).resolve().parent.parent)

def _load_store_map(output_dir=None):
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

    base = runs_dir or os.path.join(_OVERHAUL_DIR, "subtask_run_outputs")
    os.makedirs(base, exist_ok=True)
    prefix = f"{datetime.now():%m%d_%H%M%S}_{arm}_"
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def orchestrate(task, arm="graph", caps=(0, 0.0), out=None, run_dir=None,
                resolver_backend="qwen", reset_start=False, restart_env=False, leg_retries=1,
                output_dir=None, completion_guard="deterministic", ocr_url=None, runs_dir=None,
                context_policy="baseline"):
    """Decompose `task` -> typed legs, resolve each leg on the map (plan time), order the legs, then
    run each with run_leg until the AGENT stops (predicate-granted) or a per-leg cap fires. Shared
    semantic/episodic memory + a between-leg findings summary carry context forward. A failed leg is
    RETRIED up to `leg_retries` times (default 1) with the failure reason fed into the retry's
    context (orchestrator-level self-correction, 2026-07-23 - a halt_forced leg used to abort the
    task outright); only when the retries are also exhausted does it ABORT the remaining legs (a
    failed pickup shouldn't burn a checkout leg). Writes a summary.json + per-leg JSONL to run_dir
    (eval_pickup layout; a retry attempt logs to leg<NN>_retry<K>.jsonl).

    arm: 'graph' (default - the measured-better navigator, right for long-horizon), 'vlm'
    (control), or 'graph-advised' (graph targets, per-hop advisor-VLM drive - see
    agent._advised_goto; adds one advisor call per graph hop, counted in llm_calls).
    caps: (max_steps, max_minutes) PER LEG; either set to 0 means NO LIMIT for that dimension
    (default (0, 0.0) = uncapped, so a leg ends only on a real terminal reason).

    Output location: `run_dir` is the EXACT directory this run's artifacts (per-leg JSONL +
    screenshots + summary.json) land in; when None, an auto-named `<MMDD_HHMMSS>_<arm>` dir is
    created under `runs_dir` (the base, default agent/subtask_run_outputs/). Pass `run_dir` to
    pin an exact folder, or `runs_dir` to just relocate the parent while keeping the timestamped
    per-run name.

    completion_guard: 'deterministic' (default, unchanged baseline) or 'vlm'. The latter adds visual
    grounding for targeted pickup, compare, and unknown legs. Inspect always uses its mandatory
    image-bound verifier; checkout and goto remain deterministic, as do the physical/structural
    prerequisites retained by the guarded leg types.

    reset_start (default FALSE): drive to the fixed spawn checkpoint ONCE before the first leg
    (return_to_start; pose-only, never between legs - it stows hands, which would drop a carry). This
    is EVAL-reproducibility machinery, not an agent capability: it makes a batteried run (6.4's
    eval_longhorizon) start every task from the identical pose so metrics compare. A plain interactive
    run leaves it OFF - the agent starts from wherever it is (returning to spawn every time is awkward
    and adds nothing the resolver + graph nav don't already handle).

    restart_env (default FALSE): hard-reset the STORE to its pristine initial state before the first
    leg via env.Reset() (Unity's ResetEnvironment) - items back on shelves, prior checkouts undone,
    agent teleported to spawn. Distinct from reset_start, which only MOVES the agent and leaves the
    shelf state a previous run displaced. Use it when a fresh task must not inherit the last run's
    grabbed/checked-out items (e.g. re-running 'pick up 2 Jin Ramen' after a run that already removed
    two). NOTE: eval_pickup's docstrings say 'never call Reset()' because ResetEnvironment used to
    DOUBLE every non-RetailItem object (price tags, cans); that warning PREDATES the C# fix -
    ResetEnvironment now calls ItemPoolingManager.ClearPool() + ShelfBuilder.DeleteAllPriceTags()
    before reloading (DataHandler.cs:617). Verify the duplication is gone on your build before relying
    on this in a batteried eval; it stays OFF by default."""
    policy = resolve_context_policy(context_policy)

    # Resolve the attempt context before ANY logger, model, or agent is constructed. Helpers deep in
    # perception/sim resolve SARI_RUN_DIR at call time, keeping their scratch output attempt-local
    # without adding configuration to ordinary single-run commands.
    run_dir = _resolve_run_dir(run_dir, arm, runs_dir=runs_dir)
    os.environ["SARI_RUN_DIR"] = run_dir
    if ocr_url:
        os.environ["SARI_OCR_URL"] = ocr_url
    response_memory = new_response_memory(task)
    # This first write happens before service/model/simulator setup. A later forced termination may
    # prevent a final answer, but it should still leave the original request available for diagnosis.
    save_response_memory(run_dir, response_memory)

    # OCR is a required shared service, even for tasks that may not reach checkout. Fail before the
    # first simulator command so a missing daemon cannot consume a sandbox lease or alter sim state.
    from vision.ocr_client import check_ocr_health, resolve_ocr_url
    resolved_ocr_url = resolve_ocr_url(ocr_url)
    health = check_ocr_health(resolved_ocr_url)
    print(f"[ORCHESTRATOR] OCR ready: {health['model']} at {resolved_ocr_url}")

    # Before ANY reasoner runs, so the decomposer's and resolver's tokens are counted too.
    token_meter.install(run_dir)
    client = _llm_client()
    init_logger(run_name="runtime", directory=run_dir)

    # Barrier before ANY sim traffic. Under Distributed Sari Bench this process is launched the
    # moment a sandbox is leased, which can be while that sandbox is still booting or still
    # resetting from the previous attempt - so wait rather than fail the run on a refused
    # connection or a reply from a half-built store. A local sim that is already up returns
    # immediately, so this costs a plain run nothing.
    from sim.env import default_uri, wait_for_ready
    if not wait_for_ready():
        raise RuntimeError(
            f"Sandbox at {default_uri()} never reported ready; refusing to start the task against "
            "an environment that may still be mid-reset.")

    agent = EmbodiedAgent(vlm_config=VLM_CONFIG, associative_config=ASSOCIATIVE_CONFIG,
                          mode='lean', nav_mode=arm, resolver_backend=resolver_backend,
                          map_output_dir=output_dir, run_dir=run_dir,
                          context_policy=policy)

    # From here the meter also writes run_dir/tokens.json as it goes: summary.json is only written at
    # exit, so an attempt the harness SIGKILLs would otherwise report no token cost at all.
    token_meter.dump(run_dir)
    t0 = time.time()
    print(f"[ORCHESTRATOR] task: {task!r}")
    _cap = lambda v, unit: "unlimited" if not v else f"{v} {unit}"
    print(f"[ORCHESTRATOR] arm={arm}  context_policy={context_policy}  "
          f"completion_guard={completion_guard}  "
          f"caps={_cap(caps[0], 'steps')} / {_cap(caps[1], 'min')} per leg  run dir: {run_dir}")

    # -- decompose (1 LLM) + resolve each leg on the map (N LLM, plan time) --
    subtasks = decompose_task(client, task)
    task_llm = 1
    sm = _load_store_map(output_dir)
    resolve_call = make_resolve_call(resolver_backend)
    legs, n_resolves = plan_legs(sm, resolve_call, subtasks)
    task_llm += n_resolves
    # The plan is already valuable diagnostic state. Persist it before any optional reset or other
    # simulator traffic; ordering below may update the sequence, at which point it is saved again.
    set_planned_subtasks(response_memory, legs)
    save_response_memory(run_dir, response_memory)

    # -- hard STORE reset (OPT-IN, default off): put the shelves back before the task starts, so a
    #    fresh run doesn't inherit items a previous run grabbed/checked out. Done FIRST (before the
    #    pose reset and before ordering) so return_to_start re-syncs the nav pose to the post-reset
    #    spawn and order_legs reads the true start. See the docstring re: the (now-fixed) duplication
    #    warning in eval_pickup.
    if restart_env:
        try:
            from sim.env import Reset as _reset_env
            _reset_env()
            time.sleep(1.5)   # let Unity destroy + LoadStore() re-instantiate before the first frame
            print("[ORCHESTRATOR] hard env reset (ResetEnvironment): store restored to initial state.")
        except Exception as e:  # noqa: BLE001 - a reset hiccup shouldn't abort the whole task
            print(f"[ORCHESTRATOR] restart_env skipped ({type(e).__name__}: {e})")

    # -- per-TASK reset (OPT-IN, default off): eval-reproducibility only; see the docstring. Done
    #    BEFORE ordering so legs order from the true post-reset start. Pose-only, never between legs.
    if reset_start:
        try:
            from evals.eval_pickup import return_to_start
            return_to_start(agent, output_dir=output_dir)
        except Exception as e:  # noqa: BLE001 - a reset hiccup shouldn't abort the whole task
            print(f"[ORCHESTRATOR] return_to_start skipped ({type(e).__name__}: {e})")

    # Order independent pickup->checkout pairs from where the agent ACTUALLY is (spawn if we reset,
    # else its current pose) - not an assumed spawn corner.
    legs = order_legs(sm, legs, _current_nearest_cp(sm))
    set_planned_subtasks(response_memory, legs)
    save_response_memory(run_dir, response_memory)

    print(f"[ORCHESTRATOR] {len(legs)} leg(s) (resolver calls: {n_resolves}):")
    for i, lg in enumerate(legs, 1):
        feas = "" if lg.get("feasible", True) else "  [INFEASIBLE: target resolved to no checkpoint]"
        cps = lg.get("candidates")
        print(f"  {i}. [{lg.get('type')}] {lg.get('text')}"
              + (f"  -> cps {cps}" if cps else "") + feas)
    infeasible = [i + 1 for i, lg in enumerate(legs) if not lg.get("feasible", True)]
    if infeasible:
        print(f"[ORCHESTRATOR] WARNING: leg(s) {infeasible} resolved to no checkpoint - the plan may "
              f"be doomed, but running so the failure is measured, not assumed.")

    cumulative_context = ""
    visited = set()                 # task-level visit trace (compare predicate), grown by every leg
    leg_rows = []
    task_success = True
    # Carries `gripped_names` across leg/attempt boundaries so a hand still gripping keeps its
    # recorded SKU (see the seeding comment in `_run_leg_impl`) - updated after every run_leg call.
    carried_names = None
    try:
        for i, leg in enumerate(legs):
            future = legs[i + 1:]
            print(f"\n[ORCHESTRATOR] ── Leg {i + 1}/{len(legs)} ──")
            attempt, m = 0, None
            while True:
                attempt += 1
                leg_context = cumulative_context
                if attempt > 1:
                    # Orchestrator-level self-correction (2026-07-23): re-run the leg with WHY the
                    # last attempt was not accepted in front of the fresh agent. Semantic/episodic
                    # memory already persists, so the retry keeps everything the failure learned.
                    fail_reason = ((m.get("final_state") or {}).get("last_halt_refused")
                                   or m["end_reason"])
                    leg_context = cumulative_context + (
                        f"\n\n--- YOUR PREVIOUS ATTEMPT AT THIS EXACT SUBTASK FAILED "
                        f"({m['end_reason']}) ---\n"
                        f"Why it was not accepted: {fail_reason}\n"
                        f"Fix that specifically this time; everything you learned is still in memory.")
                    print(f"[ORCHESTRATOR] retrying leg {i + 1} "
                          f"(attempt {attempt}/{1 + leg_retries}): {fail_reason}")
                suffix = "" if attempt == 1 else f"_retry{attempt - 1}"
                tokens_before = token_meter.snapshot()
                m = run_leg(agent, leg, sm, caps,
                            log_path=os.path.join(run_dir, f"leg{i:02d}{suffix}.jsonl"),
                            context=leg_context, future_legs=future,
                            visited=visited, leg_idx=i + 1,
                            completion_guard=completion_guard, carried_names=carried_names)
                carried_names = (m.get("final_state") or {}).get("gripped_names")
                task_llm += m["llm_calls"]
                # Per-leg token cost, so a leg that spun to its cap is visibly the expensive one.
                # A retried leg's rows are separate, exactly like its llm_calls. `tokens_by_role`
                # splits the same window by reasoner, so "this leg was expensive" can be followed by
                # "because it kept re-running perception" without re-instrumenting anything.
                leg_tokens = token_meter.delta(tokens_before)
                leg_rows.append({**{k: v for k, v in m.items()
                                    if k not in ("final_state", "new_semantic_entries")},
                                 "attempt": attempt,
                                 "tokens_in": leg_tokens["tokens_in"],
                                 "tokens_out": leg_tokens["tokens_out"],
                                 "tokens_by_role": leg_tokens["by_role"]})
                record_attempt(
                    response_memory,
                    leg_number=i + 1,
                    attempt_number=attempt,
                    subtask=leg,
                    metrics=m,
                    episodic_reflection=getattr(agent.vlm_agent, "episodic_memory", ""),
                )
                save_response_memory(run_dir, response_memory)
                token_meter.dump()
                print(f"### leg {i+1} attempt {attempt} {m['end_reason']}: success={m['success']} "
                      f"t_grip={m['t_grip']} t_checkout={m['t_checkout']} steps={m['timesteps']} "
                      f"halts_refused={m['halts_refused']} wall={m['wall_s']}s")
                if m["success"] or attempt > leg_retries:
                    break

            if not m["success"]:
                task_success = False
                print(f"[ORCHESTRATOR] leg {i+1} did not complete ({m['end_reason']}) — "
                      f"aborting the remaining {len(legs) - i - 1} leg(s).")
                break

            if i + 1 < len(legs):
                if policy.findings_enabled:
                    print("[ORCHESTRATOR] Generating findings summary...")
                findings = _generate_findings_if_enabled(
                    policy,
                    client,
                    completed_subtask=leg.get("text", ""),
                    final_state=m["final_state"],
                    new_semantic_entries=m["new_semantic_entries"],
                )
                if findings is not None:
                    task_llm += 1
                    attach_findings(response_memory, i + 1, attempt, findings)
                    save_response_memory(run_dir, response_memory)
                    print(f"[FINDINGS SUMMARY]\n{findings}\n")
                    cumulative_context += f"\n\n--- LEG {i + 1} FINDINGS ---\n{findings}"
    finally:
        active_error = sys.exc_info()[1]
        if active_error is not None:
            # An unexpected exception may still reach this finalization block. It must never leave a
            # successful task verdict behind or let the responder claim the unrun remainder worked.
            task_success = False
        # Final response synthesis is one logical LLM call for the whole original task, regardless
        # of how many subtasks or retry attempts ran. The responder sees only the compact journal;
        # its helper catches model failures and deterministically produces a non-empty answer.
        finalize_response_memory(response_memory, success=task_success, planned_subtasks=legs)
        if active_error is not None and not response_memory["final"].get("failure_reason"):
            response_memory["final"]["failure_reason"] = (
                f"{type(active_error).__name__}: {active_error}"
            )
        save_response_memory(run_dir, response_memory)
        task_llm += 1
        response, response_source = synthesize_response(
            response_memory,
            lambda system, user: _llm_call(
                client, system, user, token_meter.ROLE_RESPONDER
            ),
        )
        response_memory["response"] = response
        response_memory["response_source"] = response_source
        save_response_memory(run_dir, response_memory)
        write_response_artifact(run_dir, response)

        # Whole-run token cost: prompt (in) / completion (out) across EVERY reasoner, incl. the
        # decomposer, the resolver, per-step perception and the findings summaries - not just the
        # legs' own deltas, which miss the between-leg work. by_model would split actor from advisor
        # only if they ever stopped being the same model, which is why by_role exists instead: it is
        # what makes an ablation able to say which component the tokens it removed were going to.
        token_totals = token_meter.totals()
        summary = {"task": task, "arm": arm, "context_policy": asdict(policy),
                   "completion_guard": completion_guard,
                   "ocr_url": resolved_ocr_url,
                   "run_config": {
                       "arm": arm,
                       "context_policy": context_policy,
                       "max_steps": caps[0],
                       "max_minutes": caps[1],
                       "resolver_backend": resolver_backend,
                       "completion_guard": completion_guard,
                       "leg_retries": leg_retries,
                       "map_dir": str(Path(output_dir).resolve()) if output_dir else None,
                       "reset_start": reset_start,
                       "restart_env": restart_env,
                       "ocr_url": resolved_ocr_url,
                   },
                   "success": task_success,
                   "response": response, "response_source": response_source,
                   "legs_planned": len(legs),
                   "legs_completed": sum(1 for r in leg_rows if r.get("success")),
                   "resolver_calls": n_resolves, "llm_calls": task_llm,
                   "tokens_in": token_totals["tokens_in"],
                   "tokens_out": token_totals["tokens_out"],
                   "tokens": token_totals,
                   "wall_s": round(time.time() - t0, 1), "legs": leg_rows}
        summary.update(planned_subtask_metrics(legs))
        if arm == "graph-advised":
            # Whole-task advisor attribution (per-hop detail rides the agent's logger lines):
            # agree ~= hops means the graph arm with a per-hop tax; deviations/stops are the
            # rows the arm exists to surface (VLMAdvisedPlanner's read-together rule).
            st = getattr(agent, "_advised_stats", [])
            summary["advised"] = {"hops": len(st),
                                  "agreed": sum(1 for s in st if s["agreed"]),
                                  "invalid": sum(1 for s in st if s["invalid"]),
                                  "stops": sum(1 for s in st if s["stop_here"])}
        out_path = out or os.path.join(run_dir, "summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        token_meter.dump()
        print("-" * 40)
        print(f"[ORCHESTRATOR] task success={task_success}  "
              f"legs {summary['legs_completed']}/{summary['legs_planned']}  "
              f"llm={task_llm}  tokens in/out={token_totals['tokens_in']}/{token_totals['tokens_out']}  "
              f"wall={summary['wall_s']}s  -> {out_path}")
        print("-" * 40)
        print(f"[RESPONSE]\n{response}")
        try:
            if agent._graph_nav:
                agent._graph_nav[1].close()
        except Exception:  # noqa: BLE001
            pass
        chime.beep()
    return summary

