"""Phase 4.2 - the paired pick-up A/B: does purely improving navigation improve time/success?

Runs the SAME pick-up tasks through the SAME agent twice, once per navigation arm:

    python validation/evals/pickup_navigation.py --arm vlm
    python validation/evals/pickup_navigation.py --arm graph
    python validation/evals/pickup_navigation.py --arm graph-advised
    python validation/evals/pickup_navigation.py --arm vlm --tasks 0 2

Everything else - semantic learner, perception, manipulation, prompts, memory (the regenerated
substrate) - is byte-identical between arms (see mapping/plans/phase4.2_dual_stack_integration.md).
Between tasks the harness DRIVES the agent back to a fixed start checkpoint (never
env.Reset() - see return_to_start's docstring for the measured Unity duplication bug).

Metrics per task, split at the boundary that matters:
    t_manip  - wall-clock to FIRST manipulation-mode entry ("target acquired"; this is where
               arm differences SHOULD live)
    t_grip   - wall-clock to a True gripped state (divergence past t_manip is manipulation
               noise, identical machinery in both arms)
    success  - gripped AND the gripped/hovered object name overlaps the target
    timesteps, llm_calls (arm B adds the resolver's ONE call - counted, not hidden), errors

Caps: --max-steps timesteps or --max-minutes wall-clock per task, whichever first. A task that
hits a cap scores success=False with whatever partial metrics it earned - report honestly.
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "agent")  # agent/ — packages + mapping live under here
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from agent_core.llm import agent_vlm_config
from agent_core.runtime import EmbodiedAgent
from sim.env import _REQUEST_SCREENSHOT_, downscale_for_storage
from orchestrator.action_dispatch import dispatch_action, parse_actor_response
from orchestrator.leg_runner import _fresh_agent_state, write_step_output
from orchestrator.start_pose import return_to_start
from sim.hand_reset import reset_hands_in_front2

# Targets verified present in the live store and reachable on open shelving (fridge items are
# excluded on purpose: grabbing through glass doors is an unmeasured sim behaviour, and this
# A/B is about navigation, not door physics).
TASKS = [
    ("Get crackers in a wrapper, not a box."),
    ("Get green colored chips."),
    ("Get orange pringles."),
    ("Get an item that weighs more than 100 grams."),
    ("Get an imported drink."),
    ("Get crackers but not from Glico."),
    ("Find chips that costs less than 30 Pesos."),
    ("Find a product with stevia."),
    ("Find Bingo Corned Beef. If you see one in the bottom shelf, pick that one."),
    ("Find an item near the checkout."),
    ("Find a large Pik Nik."),
    ("Find a canned food and check its price.")
]

EXTRACT = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)


# name_matches re-homed to subtask_completion (Phase 6.3) so the pickup completion predicate and this
# eval share ONE implementation - import it, don't redefine. (Behaviour preserved; the imported version
# additionally lower-cases each keyword, which only makes matching more forgiving.)
from orchestrator.subtask_completion import name_matches


def run_one(agent, task, keywords, max_steps, max_minutes, log_path=None, output_dir=None):
    return_to_start(agent, output_dir=output_dir)
    time.sleep(1.0)
    state = _fresh_agent_state()
    t0 = time.time()
    m = {"t_manip": None, "t_grip": None, "success": False, "timesteps": 0,
         "llm_calls": 0, "errors": 0, "end_reason": None}
    log_fh = open(log_path, "a", encoding="utf-8") if log_path else None

    # Per-step debug screenshots live alongside the JSONL in a folder named after the task log:
    # pickup_runs/<run>/task<NN>/step<NN>.png (the observation the agent acted on), and the
    # centring tool's per-look frames in pickup_runs/<run>/task<NN>/step<NN>_center/.
    shots_dir = os.path.splitext(log_path)[0] if log_path else None
    if shots_dir:
        os.makedirs(shots_dir, exist_ok=True)

    def log(record):
        if log_fh:
            record["wall"] = round(time.time() - t0, 1)
            log_fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            log_fh.flush()   # crash-proof: a dead run still leaves its trail

    log({"event": "task_start", "task": task, "arm": agent.nav_mode,
         "ts": datetime.now().isoformat(timespec="seconds")})

    agent.vlm_agent.reset_history()
    agent.vlm_agent.episodic_memory = ""
    agent.set_semantic_memory()
    # reset per-task graph-nav state so the resolver runs fresh for this task
    agent._nav_task = None
    agent._advised_stats = []   # per-task advised-hop telemetry (graph-advised arm; empty elsewhere)

    for step in range(1, max_steps + 1):
        if (time.time() - t0) / 60 > max_minutes:
            m["end_reason"] = "time_cap"
            break
        m["timesteps"] = step
        img_bytes = _REQUEST_SCREENSHOT_()["image"]
        if shots_dir:
            # Cap the SAVED debug frame at 1080p; the VLM below gets NATIVE bytes. Storage-only, so the
            # frozen-baseline metrics are unaffected (the agent never reads this file back).
            with open(os.path.join(shots_dir, f"step{step:02d}.png"), "wb") as fh:
                fh.write(downscale_for_storage(img_bytes))
        imageb64 = base64.b64encode(img_bytes).decode("utf-8")
        request = {"task": task, "image": imageb64, "state": state}
        try:
            adv0 = agent._advised_llm_calls
            response = agent.execute_lean(request, step)
            m["llm_calls"] += 3  # semantic + VLM + episodic per execute_lean call
            m["llm_calls"] += agent._advised_llm_calls - adv0  # per-hop advisor calls, counted not hidden
            if agent.nav_mode in ("graph", "graph-advised") and step == 1:
                m["llm_calls"] += 1  # the resolver's one call, counted not hidden
        except Exception as e:
            m["errors"] += 1
            print(f"    [step {step}] execute_lean error: {type(e).__name__}: {e}")
            log({"event": "error", "step": step, "error": f"{type(e).__name__}: {e}"})
            if m["errors"] >= 3:
                m["end_reason"] = "errors"
                break
            continue

        if shots_dir:
            write_step_output(shots_dir, step, response)

        mode = response.get("agent_mode")
        if mode == "manipulation" and m["t_manip"] is None:
            m["t_manip"] = round(time.time() - t0, 1)
        if response.get("halt"):
            m["end_reason"] = "agent_stop"
            log({"event": "agent_stop", "step": step})
            break

        # Same tolerant parse as the orchestrator (parse_actor_response) so the A/B baseline and the
        # orchestrator recover the identical set of actor replies - a "Kellogg's" apostrophe must not
        # waste a step here while the orchestrator salvages it, or the comparison drifts.
        parsed = parse_actor_response(response["text"], EXTRACT)
        if parsed is None:
            m["errors"] += 1
            log({"event": "parse_error", "step": step, "raw": response["text"][:400]})
            continue
        notes = parsed.get("notes", {})
        acted = []
        blocked_reason = False
        center_msg = None
        last_reach = None
        for action, times in zip(parsed.get("actions", []), parsed.get("times", [])):
            action = action.strip()
            inline = None
            im = re.match(r'^(\w+)\([\'"]?(.*?)[\'"]?\)$', action)
            if im:
                action, inline = im.group(1), im.group(2)
            center_dir = None
            if shots_dir and action == "center_object_on_screen":
                center_dir = os.path.join(shots_dir, f"step{step:02d}_center")
                os.makedirs(center_dir, exist_ok=True)
            res = dispatch_action(action, int(times), notes, inline, mode=mode,
                                  debug_dir=center_dir) or {}
            if res.get("blocked"):
                blocked_reason = res.get("reason", True)
            if res.get("center_message"):
                center_msg = res["center_message"]
            if res.get("last_reach"):
                last_reach = res["last_reach"]
            acted.append([action, int(times)])

        state = _fresh_agent_state()
        state["mode"] = mode
        state["last_action_blocked"] = blocked_reason
        state["last_center"] = center_msg
        state["last_reach"] = last_reach   # Phase D: measured reach verdict (see AGENT_STATE_DOC p)
        log({"event": "step", "step": step, "mode": mode,
             "nav_note": (response.get("nav_note") or "")[:200] or None,
             "actions": acted, "blocked": blocked_reason or None,
             "center": center_msg, "reach": last_reach,
             "pos": state.get("translation"), "rot": state.get("rotation"),
             "hovered": [state.get("leftHoveredObject"), state.get("rightHoveredObject")],
             "gripped": [state.get("leftGrippedState"), state.get("rightGrippedState")],
             "reasoning": (parsed.get("reasoning") or "")[:500],
             "status": notes.get("status")})
        gripped = state.get("leftGrippedState") or state.get("rightGrippedState")
        if gripped:
            m["t_grip"] = round(time.time() - t0, 1)
            # SUCCESS = gripped the RIGHT thing. hovered fields can clear after a grip, so a
            # name miss here may be a false negative - gripped_names is recorded for manual
            # audit, but the headline number never counts "gripped something" as success.
            m["gripped_names"] = [state.get("leftHoveredObject"), state.get("rightHoveredObject")]
            m["name_matched"] = name_matches(state, keywords)
            m["success"] = m["name_matched"]
            m["end_reason"] = "gripped"
            break

    m["wall_s"] = round(time.time() - t0, 1)
    if m["end_reason"] is None:
        m["end_reason"] = "step_cap"
    if agent.nav_mode == "graph-advised":
        # The attribution numbers the arm exists for (read TOGETHER with success/t_manip, per
        # VLMAdvisedPlanner's rule): agree ~= hops means graph-with-a-tax; deviations and
        # stop_here rows are the ones worth reading individually in the JSONL.
        st = agent._advised_stats
        m["advised"] = {"hops": len(st),
                        "agreed": sum(1 for s in st if s["agreed"]),
                        "deviated": sum(1 for s in st if not s["agreed"] and not s["invalid"]
                                        and not s["stop_here"]),
                        "invalid": sum(1 for s in st if s["invalid"]),
                        "stops": sum(1 for s in st if s["stop_here"])}
        log({"event": "advised_hops", "hops": st})
    log({"event": "task_end", **{k: v for k, v in m.items()}})
    if log_fh:
        log_fh.close()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["vlm", "graph", "graph-advised"], required=True)
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="task indices to run")
    ap.add_argument("--task", default=None,
                    help="Run ONE ad-hoc task instead of the built-in TASKS list.")
    ap.add_argument("--match", default=None,
                    help="Comma-separated keywords that count as success for --task (substring "
                         "match on the gripped/hovered object name). If omitted, success stays "
                         "False but the picked-up item is still reported.")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--output-dir", default=None,
                    help="mapping output dir to load the map from (topology/annotations/grid). "
                         "Default: mapping/output (StoreMap's DEFAULT_OUTPUT_DIR).")
    args = ap.parse_args()

    if args.task:
        kw = tuple(k.strip().lower() for k in args.match.split(",")) if args.match else ()
        todo = [(args.task, kw)]
    else:
        todo = [TASKS[i] for i in args.tasks] if args.tasks else TASKS
    # Agent runtime = the OpenAI-compatible endpoint from config.env
    # (user directive 2026-07-19; OpenRouter retired on 402).
    agent = EmbodiedAgent(
        vlm_config=agent_vlm_config(temperature=0.5),
        associative_config=agent_vlm_config(temperature=0.3),
        mode='lean', nav_mode=args.arm, map_output_dir=args.output_dir)

    # Startup hand-centering DISABLED (user, 2026-07-21): Unity now owns the default hand pose.
    # Was needed here because without it the default pose filled the camera (giant ghost hands);
    # re-enable if that regresses.
    # reset_hands_in_front2(extra_elevation=-0.1, hand="left")

    print(f"== pick-up A/B, arm={args.arm}, {len(todo)} task(s), "
          f"caps: {args.max_steps} steps / {args.max_minutes} min ==")
    run_dir = os.path.join(
        os.path.dirname(_AGENT_DIR), "validation", "artifacts", "evals", "pickup_navigation",
        f"{datetime.now():%m%d_%H%M%S}_{args.arm}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"   run log dir: {run_dir}")
    rows = []
    for idx, (task, kw) in enumerate(todo):
        print(f"\n### {task}")
        m = run_one(agent, task, kw, args.max_steps, args.max_minutes,
                    log_path=os.path.join(run_dir, f"task{idx:02d}.jsonl"),
                    output_dir=args.output_dir)
        rows.append({"task": task, "arm": args.arm, **m})
        print(f"### {m['end_reason']}: success={m['success']} t_manip={m['t_manip']} "
              f"t_grip={m['t_grip']} steps={m['timesteps']} llm={m['llm_calls']} "
              f"wall={m['wall_s']}s")

    agent.close()

    out = args.out or os.path.join(run_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    n_ok = sum(1 for r in rows if r["success"])
    print(f"\n== arm={args.arm}: {n_ok}/{len(rows)} succeeded -> {out}")


if __name__ == "__main__":
    main()
