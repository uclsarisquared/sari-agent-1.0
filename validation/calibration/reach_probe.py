"""reach_probe.py - interactive centre-LiDAR probe + reach-envelope calibration (Phase D).

Point the camera at ANY item - just put it at the CENTRE of the frame, since the LiDAR samples the
centre pixel - read the MEASURED distance/height, nudge into position while watching the numbers, and
log a grab attempt. This replaces the VLM-centred measure_reach_envelope.py: there is NO VLM here (so
a shelf of identical instances can't hop the target) and NO eyeballing distance (the LiDAR reports it,
and every read saves a screenshot with a crosshair so you can SEE exactly which item is under the
centre). Needs the sim in Play mode on ws://localhost:8080.

    python reach_probe.py

Commands at the `>` prompt (a trailing number is a repeat count, default 1):
  <Enter>      re-read the centre LiDAR + save a crosshair screenshot
  w N | s N    move forward | backward  N steps (0.1 m each), then re-read
  a N | d N    move left | right        N steps
  j N | l N    pan  left | right        N (2.5 deg each)
  i N | k N    tilt up   | down         N
  c            toggle crouch, then re-read
  h            toggle the hand prefabs  (grab needs them ON; reads don't care - hands are LiDAR-culled)
  g [label]    GRAB from here (raw tool, bypasses the plan_reach gate) and LOG the row
  x            release the left grip (drop a held item before the next grab)
  ?            reprint this help
  q            quit (stands the agent back up)

Each read prints, e.g.:
  distance=0.62m pitch=+22.1 cam_h=1.31 -> target_h=1.08 fwd_gap=0.57 | plan_reach: move(2)  [read_007.png]

CALIBRATION RECIPE - reach is a single SLANT-distance limit (the hand extends along the gaze), so you
sweep DISTANCE, not height (MEASURED 2026-07-22). Position with the Unity controller, put an item under
the crosshair, and log a grab with `g <label>` (then `x` to release) - you only need Enter/g/x here.
Grab a target from a RANGE of distances, from comfortably close out to where grabs start FAILING; a few
different heights/postures (crouch = LeftControl) confirm the limit is the same everywhere. Then run
`python validation/calibration/fit_envelope.py`, which reads the calibration artifact and prints:
  max_reach = the slant `distance` that separates grabs from misses, minus a small safety margin
The only columns that matter are `distance` and `grabbed`; fit_envelope flags any OVERLAP (a miss closer
than a grab), which would mean distance alone isn't deciding it. Paste max_reach into REACH_ENVELOPE;
re-run test_plan_reach.py. Where plan_reach's printed verdict disagrees with the actual grabbed result,
that's the guess the calibration is fixing.
"""
import csv
import os
import sys
from datetime import datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "agent")  # agent/ — runtime modules (env, manipulation) live there
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUT_DIR = os.path.join(
    os.path.dirname(_ROOT), "validation", "artifacts", "calibration", "reach")
CSV_PATH = os.path.join(OUT_DIR, "envelope.csv")
CSV_COLS = ["ts", "label", "posture", "distance", "pitch_deg", "camera_height", "target_height",
            "horizontal_gap", "plan_verdict", "plan_move_steps", "grabbed", "hovered"]

HELP = __doc__.split("Commands at", 1)[1]
HELP = "Commands at" + HELP.split("Each read prints", 1)[0]


def _fmt(v, nd=2):
    return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else "  -  "


def _crosshair(run_dir, idx):
    """Save the last screenshot with a green crosshair at the exact centre pixel the LiDAR sampled,
    so you can confirm WHICH instance is under the reading. Best-effort - never kills the loop."""
    try:
        from PIL import Image, ImageDraw
        src = os.path.join("screenshots", "ClientScreenshot.png")
        img = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(img)
        cx, cy = img.size[0] // 2, img.size[1] // 2
        d.line([(cx - 34, cy), (cx + 34, cy)], fill="lime", width=3)
        d.line([(cx, cy - 34), (cx, cy + 34)], fill="lime", width=3)
        out = os.path.join(run_dir, f"read_{idx:03d}.png")
        img.save(out)
        return os.path.basename(out)
    except Exception as e:
        return f"(no shot: {type(e).__name__})"


def main():
    from sim.env import (RequestLidarCenter, SetCrouch, SetHandsActive, RequestScreenshot,
                     ToggleLeftGrip, move_forward, move_backward, move_left, move_right,
                     pan_left, pan_right, tilt_up, tilt_down)
    from manip.manipulation import plan_reach, extend_arm_until_grabbed

    try:
        RequestScreenshot(save_image=True)
    except Exception as e:
        print(f"Could not reach the sim ({type(e).__name__}: {e}). Is it in Play mode on :8080?")
        sys.exit(1)

    run_dir = os.path.join(OUT_DIR, datetime.now().strftime("%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    MOVES = {"w": move_forward, "s": move_backward, "a": move_left, "d": move_right,
             "j": pan_left, "l": pan_right, "i": tilt_up, "k": tilt_down}

    state = {"idx": 0, "crouching": False}

    def read():
        state["idx"] += 1
        sample = RequestLidarCenter()
        plan = plan_reach(sample)
        RequestScreenshot(save_image=True)
        shot = _crosshair(run_dir, state["idx"])
        v = plan["verdict"]
        vtag = f"{v}({plan['move_steps']})" if v == "move" else v
        posture = "crouched" if state["crouching"] else "standing"
        print(f"  [{posture}] distance={_fmt(sample.get('distance'))}m "
              f"pitch={_fmt(sample.get('pitch_deg'), 1)} cam_h={_fmt(sample.get('camera_height'))} "
              f"-> target_h={_fmt(plan['target_height'])} fwd_gap={_fmt(plan['horizontal_gap'])} "
              f"| plan_reach: {vtag}  [{shot}]")
        if sample.get("error"):
            print(f"    (sim said: {sample['error']})")
        return sample, plan

    def grab(label):
        SetHandsActive(True)
        sample = RequestLidarCenter()          # measure at the exact grab pose
        plan = plan_reach(sample)
        res = extend_arm_until_grabbed()
        grabbed = bool(res.get("gripped"))
        posture = "crouched" if state["crouching"] else "standing"
        row = {"ts": datetime.now().isoformat(timespec="seconds"), "label": label,
               "posture": posture, "distance": sample.get("distance"),
               "pitch_deg": sample.get("pitch_deg"), "camera_height": sample.get("camera_height"),
               "target_height": plan["target_height"], "horizontal_gap": plan["horizontal_gap"],
               "plan_verdict": plan["verdict"], "plan_move_steps": plan["move_steps"],
               "grabbed": grabbed, "hovered": res.get("hovered")}
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if write_header:
                w.writeheader()
            w.writerow(row)
        flag = "GRABBED" if grabbed else "missed"
        agree = "" if (grabbed == (plan["verdict"] == "reachable")) else "  <-- plan_reach DISAGREES"
        print(f"  [GRAB/{posture}] {flag} hovered={res.get('hovered')!r} "
              f"target_h={_fmt(plan['target_height'])} fwd_gap={_fmt(plan['horizontal_gap'])} "
              f"(plan said {plan['verdict']}){agree}  -> logged to envelope.csv")

    print(HELP)
    print(f"logging to {CSV_PATH}\ncrosshair reads -> {run_dir}\n")
    read()
    try:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                read(); continue
            cmd, *rest = line.split()
            cmd = cmd.lower()
            count = int(rest[0]) if (rest and rest[0].lstrip("-").isdigit()) else 1
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("?", "help"):
                print(HELP)
            elif cmd in MOVES:
                MOVES[cmd](count); read()
            elif cmd == "c":
                state["crouching"] = not state["crouching"]
                SetCrouch(state["crouching"]); read()
            elif cmd == "h":
                state["hands"] = not state.get("hands", True)
                SetHandsActive(state["hands"])
                print(f"  hands {'ON' if state['hands'] else 'OFF'}")
            elif cmd == "g":
                grab(" ".join(rest))
            elif cmd == "x":
                ToggleLeftGrip(); print("  released left grip"); read()
            else:
                print(f"  ? unknown command {cmd!r} - type ? for help")
    finally:
        SetCrouch(False)   # stand back up; the crouch verdict is sticky
        print(f"\ndone. rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()
