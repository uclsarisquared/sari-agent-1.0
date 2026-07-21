"""Phase 4.2 - the paired pick-up A/B: does purely improving navigation improve time/success?

Runs the SAME pick-up tasks through the SAME agent twice, once per navigation arm:

    python eval_pickup.py --arm vlm      # control: old VLM-driven navigation
    python eval_pickup.py --arm graph    # graph: resolver + goto_checkpoint dispatch
    python eval_pickup.py --arm vlm --tasks 0 2    # subset by index (paired reruns)

Everything else - semantic learner, perception, manipulation, prompts, memory (the regenerated
substrate) - is byte-identical between arms (see slamtest/plans/phase4.2_dual_stack_integration.md).
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
import ast
import base64
import json
import os
import re
import sys
import time
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from agent import EmbodiedAgent, ucl_qwen_config
from env import _REQUEST_SCREENSHOT_
from subtask_agents import dispatch_action, _fresh_agent_state
from hand_reset import reset_hands_in_front2

# Targets verified present in the live store and reachable on open shelving (fridge items are
# excluded on purpose: grabbing through glass doors is an unmeasured sim behaviour, and this
# A/B is about navigation, not door physics).
TASKS = [
    ("Find and pick up a can of Century Tuna.", ("century", "tuna")),
    ("Find and pick up a can of Piattos chips.", ("piattos",)),
    ("Find and pick up a pack of instant noodles.", ("noodle", "pancit", "canton", "ramen", "mi goreng", "lucky")),
    ("Find and pick up the Ritz crackers.", ("ritz",)),
    ("Find and pick up a bag of Clover Chips.", ("clover",)),
]

EXTRACT = re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)


def name_matches(state, keywords):
    fields = [state.get("leftHoveredObject") or "", state.get("rightHoveredObject") or ""]
    blob = " ".join(str(f) for f in fields).lower()
    return any(k in blob for k in keywords)


def return_to_start(agent):
    """Drive the agent back to a fixed start checkpoint between tasks - harness machinery,
    identical for both arms, so every task starts from the same place.

    This REPLACES env.Reset(): MEASURED 2026-07-19, Unity's ResetEnvironment destroys only
    'RetailItem'-tagged objects before LoadStore() re-instantiates the whole store, so each
    reset DOUBLES everything not carrying that tag (price tags visibly doubled; clipping can
    stacks). Sim-side C# fix pending (DataHandler.cs:606). Until then: never call Reset().
    Known cost: items displaced by earlier tasks stay displaced - each task targets a
    different product, so at most one prior grab per shelf; noted, not hidden."""
    from store_map import StoreMap, NavSession
    if not hasattr(return_to_start, "_nav"):
        sm = StoreMap()
        return_to_start._nav = (sm, NavSession(sm, stow_hands=False),
                                sm.nearest_checkpoint((-3.0, -5.0)))  # nearest to spawn
    sm, nav, start_cp = return_to_start._nav
    from explore import step_agent
    from capture_walk import face
    nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
    # Through the agent's tracker, not raw SetHandsActive - a raw call here would desync
    # agent._hands_active and the mode toggle would skip a needed transition.
    agent._set_hands(False)
    nav.goto(start_cp, face_shelf=False)
    # Face world yaw 0 at the start point so every task begins from the identical pose. goto with
    # face_shelf=False leaves the yaw wherever the drive ended (it only re-levels the pitch);
    # face() is an absolute rotate-in-place on the yaw axis, so the levelled pitch is untouched.
    nav.pos, nav.rot = face(nav.args, nav.pos, nav.rot, 0.0)


def run_one(agent, task, keywords, max_steps, max_minutes, log_path=None):
    return_to_start(agent)
    time.sleep(1.0)
    state = _fresh_agent_state()
    t0 = time.time()
    m = {"t_manip": None, "t_grip": None, "success": False, "timesteps": 0,
         "llm_calls": 0, "errors": 0, "end_reason": None}
    log_fh = open(log_path, "a", encoding="utf-8") if log_path else None

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

    for step in range(1, max_steps + 1):
        if (time.time() - t0) / 60 > max_minutes:
            m["end_reason"] = "time_cap"
            break
        m["timesteps"] = step
        imageb64 = base64.b64encode(_REQUEST_SCREENSHOT_()["image"]).decode("utf-8")
        request = {"task": task, "image": imageb64, "state": state}
        try:
            response = agent.execute_lean(request, step)
            m["llm_calls"] += 3  # semantic + VLM + episodic per execute_lean call
            if agent.nav_mode == "graph" and step == 1:
                m["llm_calls"] += 1  # the resolver's one call, counted not hidden
        except Exception as e:
            m["errors"] += 1
            print(f"    [step {step}] execute_lean error: {type(e).__name__}: {e}")
            log({"event": "error", "step": step, "error": f"{type(e).__name__}: {e}"})
            if m["errors"] >= 3:
                m["end_reason"] = "errors"
                break
            continue

        mode = response.get("agent_mode")
        if mode == "manipulation" and m["t_manip"] is None:
            m["t_manip"] = round(time.time() - t0, 1)
        if response.get("halt"):
            m["end_reason"] = "agent_stop"
            log({"event": "agent_stop", "step": step})
            break

        try:
            parsed = ast.literal_eval(EXTRACT.search(response["text"])[1])
        except Exception:
            m["errors"] += 1
            log({"event": "parse_error", "step": step, "raw": response["text"][:400]})
            continue
        notes = parsed.get("notes", {})
        acted = []
        blocked_reason = False
        center_msg = None
        for action, times in zip(parsed.get("actions", []), parsed.get("times", [])):
            action = action.strip()
            inline = None
            im = re.match(r'^(\w+)\([\'"]?(.*?)[\'"]?\)$', action)
            if im:
                action, inline = im.group(1), im.group(2)
            res = dispatch_action(action, int(times), notes, inline, mode=mode) or {}
            if res.get("blocked"):
                blocked_reason = res.get("reason", True)
            if res.get("center_message"):
                center_msg = res["center_message"]
            acted.append([action, int(times)])

        state = _fresh_agent_state()
        state["mode"] = mode
        state["last_action_blocked"] = blocked_reason
        state["last_center"] = center_msg
        log({"event": "step", "step": step, "mode": mode,
             "nav_note": (response.get("nav_note") or "")[:200] or None,
             "actions": acted, "blocked": blocked_reason or None,
             "center": center_msg,
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
    log({"event": "task_end", **{k: v for k, v in m.items()}})
    if log_fh:
        log_fh.close()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["vlm", "graph"], required=True)
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="task indices to run")
    ap.add_argument("--task", default=None,
                    help="Run ONE ad-hoc task instead of the built-in TASKS list.")
    ap.add_argument("--match", default=None,
                    help="Comma-separated keywords that count as success for --task (substring "
                         "match on the gripped/hovered object name). If omitted, success stays "
                         "False but the picked-up item is still reported.")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.task:
        kw = tuple(k.strip().lower() for k in args.match.split(",")) if args.match else ()
        todo = [(args.task, kw)]
    else:
        todo = [TASKS[i] for i in args.tasks] if args.tasks else TASKS
    # Agent runtime = UCL qwen (user directive 2026-07-19; OpenRouter retired on 402).
    agent = EmbodiedAgent(
        vlm_config=ucl_qwen_config(temperature=0.5),
        associative_config=ucl_qwen_config(temperature=0.3),
        mode='lean', nav_mode=args.arm)

    # Startup hand-centering DISABLED (user, 2026-07-21): Unity now owns the default hand pose.
    # Was needed here because without it the default pose filled the camera (giant ghost hands);
    # re-enable if that regresses.
    # reset_hands_in_front2(extra_elevation=-0.1, hand="left")

    print(f"== pick-up A/B, arm={args.arm}, {len(todo)} task(s), "
          f"caps: {args.max_steps} steps / {args.max_minutes} min ==")
    run_dir = os.path.join(_THIS_DIR, "slamtest", "output", "pickup_runs",
                           f"{datetime.now():%m%d_%H%M%S}_{args.arm}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"   run log dir: {run_dir}")
    rows = []
    for idx, (task, kw) in enumerate(todo):
        print(f"\n### {task}")
        m = run_one(agent, task, kw, args.max_steps, args.max_minutes,
                    log_path=os.path.join(run_dir, f"task{idx:02d}.jsonl"))
        rows.append({"task": task, "arm": args.arm, **m})
        print(f"### {m['end_reason']}: success={m['success']} t_manip={m['t_manip']} "
              f"t_grip={m['t_grip']} steps={m['timesteps']} llm={m['llm_calls']} "
              f"wall={m['wall_s']}s")

    if agent._graph_nav:
        agent._graph_nav[1].close()

    out = args.out or os.path.join(run_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    n_ok = sum(1 for r in rows if r["success"])
    print(f"\n== arm={args.arm}: {n_ok}/{len(rows)} succeeded -> {out}")


if __name__ == "__main__":
    main()
