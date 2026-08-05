"""carry_probe.py - Phase 6.1 GATE (step 5): scripted, no-LLM carry test. 🎮 needs the sim in Play mode.

The gate the phase6 plan sets before any agent-loop work starts:

    grab a verified-reachable item -> park the hand at REST -> drive a 3-4 checkpoint route via
    NavSession.goto -> after EVERY leg assert leftGrippedState is still True, the drive never froze
    (goto arrived), and save a screenshot for the occlusion/parenting eyeball.
    PASS = 3/3 routes with ZERO drops.

This exercises the whole 6.1 chain end to end with NO VLM in the loop:
  - manipulation.extend_arm_until_grabbed now sets GRAB, reaches, and restores REST itself;
  - the hand stays ACTIVE at REST through navigation (it is no longer disabled), so the gripped
    item rides along instead of being dropped at the first mode change.

Usage:
    python validation/probes/carry_probe.py                       # default route
    python validation/probes/carry_probe.py --grab-cp 32 --route 15,20,32,54 --routes 3
    python validation/probes/carry_probe.py --legs 4 --routes 3

At each route the probe drives to --grab-cp and tries to grab. If the auto-grab misses (the item
isn't under the hand), it drops into a tiny nudge loop (no VLM) so you can line the item up and retry;
once gripped, the scripted carry runs on its own. A carried item that survives every leg of every
route is the pass.
"""
import argparse
import json
import os
import sys
from datetime import datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "agent")                       # agent/
_MAPPING = os.path.join(_ROOT, "mapping")
for _p in (_ROOT, _MAPPING):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: F401,E402 - mapping category dirs onto sys.path (flat-import contract)


def _auto_route(sm, grab_cp, legs):
    """A drivable route of `legs` distinct checkpoints, spread out by graph distance from grab_cp and
    ending at the checkout counter (goto uses grid A* between any valid checkpoints, so the route only
    needs valid ids, not graph-adjacent ones)."""
    counter = sm.counter_checkpoint()
    others = [c for c in sm.by_id
              if c != grab_cp and c != counter and sm.by_id[c].get("world_xz")]
    others.sort(key=lambda c: (sm.hops(grab_cp, c) if sm.hops(grab_cp, c) is not None else 999))
    picks = []
    if others and legs > 1:
        step = max(1, len(others) // (legs - 1))
        picks = others[::step][:legs - 1]
    return picks + ([counter] if counter is not None else [])


def secure_grip(nav, uri):
    """Get a real item into the LEFT hand. Auto-attempts extend_arm_until_grabbed (which sets GRAB,
    reaches, restores REST). On a miss, a nudge loop (no VLM) lets the operator line the item up and
    retry. Returns True once gripped, False if the operator aborts."""
    from sim.env import (RequestLidarCenter, TransformHands, SetCrouch,
                     move_forward, move_backward, move_left, move_right,
                     pan_left, pan_right, tilt_up, tilt_down)
    from manip.manipulation import extend_arm_until_grabbed, plan_reach

    MOVES = {"w": move_forward, "s": move_backward, "a": move_left, "d": move_right,
             "j": pan_left, "l": pan_right, "i": tilt_up, "k": tilt_down}
    crouching = [False]

    def gripped_now():
        return bool(TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))["leftGrippedState"])

    while True:
        try:
            v = plan_reach(RequestLidarCenter())
            print(f"  plan_reach: {v['verdict']}"
                  + (f" (move {v['move_steps']})" if v.get("move_steps") else ""))
        except Exception as e:
            print(f"  (plan_reach unavailable: {type(e).__name__})")
        res = extend_arm_until_grabbed()
        if res.get("gripped") and gripped_now():
            print(f"  GRIPPED {res.get('hovered')!r} - hand parked at REST, ready to carry.")
            return True
        print(f"  no grip yet ({res.get('reason') or res})")
        try:
            line = input("  nudge [w/s/a/d/j/l/i/k N | c crouch | Enter=retry | q=abort] > ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if not line:
            continue
        cmd, *rest = line.split()
        cmd = cmd.lower()
        n = int(rest[0]) if (rest and rest[0].lstrip("-").isdigit()) else 1
        if cmd in ("q", "quit"):
            return False
        elif cmd == "c":
            crouching[0] = not crouching[0]
            SetCrouch(crouching[0], uri=uri)
            print(f"  crouch {'ON' if crouching[0] else 'OFF'}")
        elif cmd in MOVES:
            MOVES[cmd](n)


def run(args):
    from nav.store_map import StoreMap, NavSession
    from sim.env import TransformHands, ToggleLeftGrip, RequestScreenshot

    try:
        RequestScreenshot(save_image=False, uri=args.uri)
    except Exception as e:
        print(f"Could not reach the sim ({type(e).__name__}: {e}). Is it in Play mode on {args.uri}?")
        return 2

    sm = StoreMap()
    if args.grab_cp not in sm.by_id:
        print(f"grab-cp {args.grab_cp} is not a checkpoint in the graph.")
        return 2
    route = ([int(x) for x in args.route.split(",") if x.strip()]
             if args.route else _auto_route(sm, args.grab_cp, args.legs))
    route = [c for c in route if c in sm.by_id]
    if not route:
        print("empty/invalid route; pass --route a,b,c with valid checkpoint ids.")
        return 2

    run_dir = os.path.join(
        os.path.dirname(_ROOT), "validation", "artifacts", "probes", "carry",
        datetime.now().strftime("%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    jsonl = os.path.join(run_dir, "carry.jsonl")

    def log(**row):
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **row}) + "\n")

    # hands managed per-leg (NOT stowed) - stowing is the very drop-bug 6.1 removes.
    nav = NavSession(sm, stow_hands=False, uri=args.uri)
    print(f"grab-cp: {args.grab_cp}   route: {route}   routes: {args.routes}")
    print(f"run dir: {run_dir}\n")

    routes_passed = 0
    try:
        for r in range(1, args.routes + 1):
            print(f"--- route {r}/{args.routes} ---")
            if not nav.goto(args.grab_cp):
                print(f"  could not drive to grab-cp {args.grab_cp}; skipping route.")
                log(route=r, result="no_grab_cp")
                continue
            if not secure_grip(nav, args.uri):
                print("  aborted before a grip was secured.")
                log(route=r, result="no_grip")
                continue

            drops, frozen = 0, 0
            for i, cp in enumerate(route, 1):
                arrived = nav.goto(cp)
                hs = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
                gripped = bool(hs["leftGrippedState"])
                hand_t = hs["leftTranslation"]
                shot = os.path.join(run_dir, f"route{r}_leg{i}_cp{cp}.png")
                try:
                    nav.screenshot(shot)
                    shot_name = os.path.basename(shot)
                except Exception as e:
                    shot_name = f"(no shot: {type(e).__name__})"
                if not gripped:
                    drops += 1
                if not arrived:
                    frozen += 1
                where = nav.where()
                flag = "" if (gripped and arrived) else ("   <-- DROP" if not gripped
                                                         else "   <-- FROZE/blocked")
                print(f"  leg {i}: cp{cp} arrived={arrived} grip={gripped} "
                      f"nearest={where[3]} hand={tuple(round(v, 3) for v in hand_t)} "
                      f"[{shot_name}]{flag}")
                log(route=r, leg=i, cp=cp, arrived=arrived, gripped=gripped,
                    nearest_cp=where[3], hand_translation=list(hand_t), shot=shot_name)
                if not gripped:
                    break   # dropped mid-route - this route has failed, stop driving it

            ok = (drops == 0 and frozen == 0)
            routes_passed += int(ok)
            print(f"  route {r}: {'PASS' if ok else 'FAIL'} (drops={drops}, frozen_legs={frozen})")
            log(route=r, result="pass" if ok else "fail", drops=drops, frozen=frozen)

            # Release for the next route (a fresh grab each route). Only toggle if still holding.
            if bool(TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))["leftGrippedState"]):
                ToggleLeftGrip(uri=args.uri)
    finally:
        nav.close()

    verdict = "PASS" if routes_passed == args.routes else "FAIL"
    print(f"\n=== GATE 6.1: {routes_passed}/{args.routes} routes zero-drop -> {verdict} ===")
    print(f"log + screenshots: {run_dir}")
    return 0 if verdict == "PASS" else 1


def build_parser():
    p = argparse.ArgumentParser(description="Phase 6.1 gate: scripted carry probe (no LLM).")
    p.add_argument("--grab-cp", type=int, default=32,
                   help="checkpoint to grab an item at (a shelf node with a reachable item). Default 32.")
    p.add_argument("--route", type=str, default=None,
                   help="comma-separated checkpoint ids to carry through, e.g. 15,20,32,54. "
                        "Omit to auto-build a route ending at the checkout counter.")
    p.add_argument("--legs", type=int, default=4, help="auto-route length when --route is omitted.")
    p.add_argument("--routes", type=int, default=3, help="how many grab+carry routes to run (the gate is 3/3).")
    p.add_argument("--uri", type=str, default="ws://localhost:8080/commands")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
