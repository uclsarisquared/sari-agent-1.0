"""place_probe.py - interactive centre-LiDAR probe + place-envelope calibration (Phase 6.2). 🎮

The sibling of reach_probe.py, for RELEASING onto the checkout counter instead of grabbing off a
shelf. Grab an item to carry, drive it to the counter, centre the counter surface, then release from a
RANGE of distances and record whether the item landed ON the surface (judged by eye per trial - there
is no sim query for "is item X on surface Y", and honest eyes beat an unverified heuristic). Needs the
sim in Play mode on ws://localhost:8080.

    python place_probe.py

WHY a separate envelope (do NOT reuse the 0.85 m reach number): grabbing needs the HAND to reach the
item; placing needs the released item's FALL to intersect the surface. Different physics, different
number - and placing may care about release HEIGHT too (too high a drop can bounce/roll off), which is
why `c` (crouch) and the standing/crouched posture column are here. If height matters, a distinct
PLACE pose earns its existence over reusing GRAB (phase6 plan, "The place envelope - measured").

Commands at the `>` prompt (a trailing number is a repeat count, default 1):
  <Enter>        re-read the centre LiDAR + save a crosshair screenshot (surface distance/height/gap)
  counter        drive to the checkout counter via go_to_counter (the SAME deterministic tool the
                 agent uses - A*-drive + face cp54, carry-safe), then re-read. This is the realistic
                 arrival pose; sweep standoffs from HERE with w/s.
  goto N         drive to any checkpoint N (A*-drive + face) - e.g. to a shelf to grab, before `counter`
  center [y]     run center_to_counter (the FIXED-input centring variant), then re-read. Optional
                 y (0-1) overrides the VERTICAL aim for this call: the detector boxes the WHOLE
                 counter (front face included), so its bbox centre maps to a 3D point near the
                 top's FRONT EDGE - releases there roll off. y > 0.5 parks the bbox lower in
                 frame, so the centre ray (and the release) lands DEEPER on the surface. Dial y
                 until the crosshair sits mid-surface; the winning value gets PINNED into
                 perception.COUNTER_AIM_NORM (it is provisional for exactly this reason).
  scanner [y]    run center_to_scanner (the scan-pad sibling of `center`), then re-read. Optional
                 y (0-1) dials the vertical aim exactly like `center [y]`; the winning wording/aim
                 get PINNED into perception.SCANNER_TARGET_INFO / SCANNER_AIM_NORM. (probe M2)
  align [N]      lateral-align loop from cp54 (probe M4, the Option A prototype): centre the pad,
                 read the yaw error vs cp54's perpendicular, re-face the perpendicular, strafe
                 slant*sin(yaw_off) out, re-centre; at most N iterations (default 3). Prints the
                 residual per iteration - the number that decides Option A vs a spliced scan dock.
  adv [T]        approach until the centre-ray slant ~ T metres (default SCAN_REACH_M), re-centring
                 the SCANNER after each move - the "scan region within the hand's max reach" step;
                 the centre-LiDAR slant is the ONLY distance the gate uses (probe M3: does a sweep
                 from inside SCAN_REACH_M scan? no boundary-mapping ladder needed)
  sweep [label]  in-hand scan sweep (probe M1): requires HOLDING. GRAB pose, extend FULLY toward
                 the pad (the stall guard / Unity clamp stops the hand - there is NO stop-distance
                 calibration: the Easy trigger box covers the whole scan region, so a full
                 extension transits it), retract + REST. The grip is NEVER opened. Then it centres
                 the POS screen and region-OCRs its bbox for a NEW receipt line (scanned_measured),
                 AND asks YOU (scanned_verified) - both logged to scan_probe.csv, never promoted.
  screen         centre the POS screen + region-OCR its bbox, print every line (probe M5: is the
                 receipt OCR-able from here? does not disturb the sweep's receipt baseline)
  w N | s N      move forward | backward  N steps (0.1 m each), then re-read  (fine standoff sweep)
  a N | d N      move left | right        N steps
  j N | l N      pan  left | right        N (2.5 deg each)
  i N | k N      tilt up   | down         N
  c              toggle crouch, then re-read (tests whether a LOWER release lands better)
  g [label]      GRAB an item to carry (raw extend_arm_until_grabbed; sets GRAB, reaches, restores REST)
  p [label]      PLACE from here: extend the hand forward over the surface, release, then log the row
  pr [label]     PLACE at REST (release without extending) - A/B whether the forward reach matters
  gp <label> [T] AUTO macro (the whole 6.1+6.2 chain, no LLM): grab the item you're centred on
                 (or use the one you're already holding) -> go_to_counter -> center (2 retries) ->
                 auto-approach until the centre-ray slant ~ T metres -> center again -> drop -> ask
                 landed -> log. T defaults to TIGHTEN_TARGET_M; pass it to sample a chosen distance,
                 e.g. `gp 7 1.55` then `gp 8 1.35` to BRACKET the far boundary in two commands.
  x              release the left grip WITHOUT logging (abort a carry / reset)
  ?              reprint this help
  q              quit (stands the agent back up)

MANUAL RUN: `goto <shelf-cp>` -> `g <item>` -> `counter` -> `center` -> `p <item>`; then `s`/`c` to
sweep standoff and posture. AUTO RUN: centre on a shelf item, then `gp <item> [T]` does the rest -
repeat with different T to tighten the envelope fast.

SCAN RUN (6.2 restructure, probes M1-M5): carry an item in (`goto` + `g`, or be holding), `counter`,
then `scanner` (M2 - iterate the wording/aim until the pad locks), `align` (M4), `adv` (M3 - close
until the pad slant is inside SCAN_REACH_M), `sweep <label>` (M1 - full-extension in-hand sweep;
verifies via the POS-screen bbox OCR AND your eye). `screen` reads the receipt in its own bbox (M5).
There is NO drop-scan variant: the item never leaves the hand between shelf and tray. M7 (tray far
bound) stays on the old path: `gp <label> 1.35` / `1.45` / `1.55`.

A read prints, e.g.:
  [standing] dist=0.71m pitch=+34.0 cam_h=1.31 -> surface_h=0.91 fwd_gap=0.59 hit=True  [read_007.png]
surface_h is the counter-top world Y under the crosshair; fwd_gap is the horizontal distance to it.
hit=False means nothing solid at frame centre (looking PAST the counter / at the floor) - re-centre.

CALIBRATION RECIPE - carry an item in with `g`, `center` on the counter, then `p <label>` from a
range of standoffs (start at the cp54 dock ~0.20 m and back off with `s` until releases stop landing),
a couple at crouch to check height. Each `p` asks you by eye whether it LANDED, and logs
(slant_distance, pitch, camera_height, surface_height, landed). Then fit the boundary the same way
reach did (fit_envelope.py pattern) and wire it into plan_place. The columns that decide it are
`slant_distance` and `landed`; if crouched rows land where standing rows didn't at the same distance,
that is height mattering - split out a PLACE pose.
"""
import csv
import argparse
import math
import os
import sys
from datetime import datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "agent")  # agent/ — runtime modules (env, manipulation) live there
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUT_DIR = os.path.join(
    os.path.dirname(_ROOT), "validation", "artifacts", "calibration", "place")
CSV_PATH = os.path.join(OUT_DIR, "place_envelope.csv")
CSV_COLS = ["ts", "label", "posture", "release_mode", "slant_distance", "pitch_deg",
            "camera_height", "surface_height", "horizontal_gap", "hit", "landed", "shot"]

# Default `gp` approach target (centre-ray SLANT to the counter surface, metres). 1.45 sits mid the
# 1.29 m (last land) .. 1.64 m (dock miss) gap left by the first run - the single most informative
# spot to sample. Override per call (`gp <label> 1.55`) to bracket the boundary.
TIGHTEN_TARGET_M = 1.45

# ---- Scan probe (6.2 restructure: M1-M5) --------------------------------------------------------
SCAN_CSV_PATH = os.path.join(OUT_DIR, "scan_probe.csv")
SCAN_CSV_COLS = ["ts", "label", "slant_start", "pitch_deg", "camera_height", "steps_extended",
                 "scanned_measured", "scanned_verified", "still_holding", "new_lines", "shot"]

# The ONLY scan-distance number: get the pad's centre-LiDAR slant inside this before sweeping.
# Seeded from the measured grab REACH_ENVELOPE (0.85 m); probe M3 is a yes/no check that a
# full-extension sweep from inside it scans - NOT a boundary-mapping ladder. There is no
# stop-distance calibration inside the sweep itself: the user enlarged the Easy trigger box
# (2026-07-22) to cover the WHOLE scan region, so a full extension transits it, and the sweep
# stops on the hand's own stall/clamp, never on a measured distance.
SCAN_REACH_M = 0.85
SWEEP_MAX_STEPS = 25   # extension cap; Unity clamps the hand ~0.5 m out and the stall guard stops
                       # earlier, so this is a runaway bound, not a reach claim

HELP = "Commands at" + __doc__.split("Commands at", 1)[1].split("A read prints", 1)[0]


def _fmt(v, nd=2):
    return f"{v:+.{nd}f}" if isinstance(v, (int, float)) else "  -  "


def _geometry(sample):
    """Decompose one RequestLidarCenter sample into (surface_height, horizontal_gap) - the same trig
    plan_reach uses, but for the counter TOP: surface_h = cam_h - d*sin(pitch), fwd_gap = d*cos(pitch).
    Returns (None, None) when the sample lacks a pose (pre-Phase-D build) or missed."""
    d, pitch, cam_h = sample.get("distance"), sample.get("pitch_deg"), sample.get("camera_height")
    if d is None or pitch is None or cam_h is None or not sample.get("hit", False):
        return None, None
    theta = math.radians(float(pitch))
    return float(cam_h) - float(d) * math.sin(theta), float(d) * math.cos(theta)


def _crosshair(run_dir, idx, tag="read"):
    """Save the last screenshot with a green crosshair at the exact centre pixel the LiDAR sampled, so
    you can confirm WHAT is under the reading (and, on a place shot, whether the item landed there).
    Best-effort - never kills the loop."""
    try:
        from PIL import Image, ImageDraw
        src = os.path.join("screenshots", "ClientScreenshot.png")
        img = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(img)
        cx, cy = img.size[0] // 2, img.size[1] // 2
        d.line([(cx - 34, cy), (cx + 34, cy)], fill="lime", width=3)
        d.line([(cx, cy - 34), (cx, cy + 34)], fill="lime", width=3)
        out = os.path.join(run_dir, f"{tag}_{idx:03d}.png")
        img.save(out)
        return os.path.basename(out)
    except Exception as e:
        return f"(no shot: {type(e).__name__})"


def main():
    parser = argparse.ArgumentParser(description="Interactive placement and checkout OCR probe.")
    parser.add_argument("--ocr-url", default=None, help="OCR service base URL (else $SARI_OCR_URL/default).")
    args = parser.parse_args()

    from vision.ocr_client import check_ocr_health
    try:
        check_ocr_health(args.ocr_url)
    except Exception as error:
        print(f"OCR preflight failed before simulator work: {error}")
        return 1

    from collections import Counter
    from sim.env import (RequestLidarCenter, SetCrouch, SetHandsActive, RequestScreenshot,
                     TransformHands, ToggleLeftGrip, _XTNFWD_LEFT_, _PLLBCK_LEFT_,
                     move_forward, move_backward, move_left, move_right,
                     pan_left, pan_right, tilt_up, tilt_down, default_uri)
    from manip.manipulation import set_hand_pose, extend_arm_until_grabbed
    from vision.perception import (center_to_counter, COUNTER_AIM_NORM,
                            center_to_scanner, SCANNER_AIM_NORM,
                            center_to_screen, read_text_in_box)
    from nav.store_map import StoreMap, NavSession, go_to_counter
    from explore import step_agent
    from capture_walk import perpendicular_yaw, face     # mapping on sys.path via store_map
    from mapping import normalize_deg

    try:
        RequestScreenshot(save_image=True)
    except Exception as e:
        print(f"Could not reach the sim ({type(e).__name__}: {e}). Is it in Play mode on :8080?")
        sys.exit(1)

    # Carry-safe NavSession: stow_hands=False so a driven carry is NOT dropped (stowing is the very
    # 6.1 drop bug). We drive with the SAME machinery the agent uses (go_to_counter / nav.goto), then
    # fine-tune the standoff with the raw WASD primitives below.
    sm = StoreMap()
    nav = NavSession(sm, uri=default_uri(), stow_hands=False)

    run_dir = os.path.join(OUT_DIR, datetime.now().strftime("%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    MOVES = {"w": move_forward, "s": move_backward, "a": move_left, "d": move_right,
             "j": pan_left, "l": pan_right, "i": tilt_up, "k": tilt_down}

    # "receipt" is the running multiset of POS-screen lines last seen (region-OCR'd inside the
    # screen's own bbox). A sweep's scanned_measured is the DIFF against it: the receipt accumulates
    # across scans in a session, so a plain "is there a digit line" test would false-positive on the
    # SECOND item (the first is still on screen). Seeded empty; updated after every screen read.
    state = {"idx": 0, "crouching": False, "hands": True, "receipt": Counter()}

    def _hand_state():
        return TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    def _holding():
        return bool(_hand_state()["leftGrippedState"])

    def read(save=True):
        state["idx"] += 1
        sample = RequestLidarCenter()
        surface_h, fwd_gap = _geometry(sample)
        shot = ""
        if save:
            RequestScreenshot(save_image=True)
            shot = _crosshair(run_dir, state["idx"])
        posture = "crouched" if state["crouching"] else "standing"
        held = "  HOLDING" if _holding() else ""
        print(f"  [{posture}] dist={_fmt(sample.get('distance'))}m "
              f"pitch={_fmt(sample.get('pitch_deg'), 1)} cam_h={_fmt(sample.get('camera_height'))} "
              f"-> surface_h={_fmt(surface_h)} fwd_gap={_fmt(fwd_gap)} "
              f"hit={sample.get('hit')}{held}" + (f"  [{shot}]" if shot else ""))
        if sample.get("error"):
            print(f"    (sim said: {sample['error']})")
        if not sample.get("hit", True):
            print("    (no surface at frame centre - looking past the counter / at the floor; re-centre)")
        return sample

    def _release(mode):
        """Drop the held item. 'extend' mirrors the eventual place_held_item: move the held item to the
        forward-ready GRAB pose, extend to the reach clamp so it sits OVER the counter, ToggleGrip to
        release, retract, park REST. 'rest' just opens the hand from the carry pose (the item falls at
        the agent's front). LOCAL measurement copy (step-0 pattern) - production place_held_item is the
        next checklist item; the probe measures the envelope it will be gated on."""
        if mode == "extend":
            set_hand_pose("grab")
            prev = _hand_state()["leftTranslation"]
            moved = 0
            for _ in range(25):                    # extend to the 0.5 clamp; no item to hover here
                cur = _XTNFWD_LEFT_()["leftTranslation"]
                if sum((cur[k] - prev[k]) ** 2 for k in range(3)) ** 0.5 <= 1e-4:
                    break
                prev = cur
                moved += 1
            released = not ToggleLeftGrip()["gripped"]
            for _ in range(moved):                 # retract exactly as far as we extended
                _PLLBCK_LEFT_()
            set_hand_pose("rest")
        else:                                      # 'rest': open the hand where it carries
            released = not ToggleLeftGrip()["gripped"]
        return released

    def place(label, mode):
        if not _holding():
            print("  not holding anything - grab an item first (g) before you can place.")
            return
        sample = RequestLidarCenter()              # measure the surface at the release pose FIRST
        surface_h, fwd_gap = _geometry(sample)
        released = _release(mode)
        state["idx"] += 1
        RequestScreenshot(save_image=True)
        shot = _crosshair(run_dir, state["idx"], tag="place")
        posture = "crouched" if state["crouching"] else "standing"
        print(f"  [PLACE/{mode}/{posture}] released={released} "
              f"dist={_fmt(sample.get('distance'))}m surface_h={_fmt(surface_h)} "
              f"fwd_gap={_fmt(fwd_gap)} hit={sample.get('hit')}  [{shot}]")
        if not released:
            print("    (grip did NOT open - nothing was placed; not logging a row)")
            return
        try:
            verdict = input("    landed ON the counter? [y]es / [n]o / [s]kip (don't log) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            verdict = "s"
        if verdict not in ("y", "yes", "n", "no"):
            print("    skipped - no row logged.")
            return
        landed = verdict in ("y", "yes")
        row = {"ts": datetime.now().isoformat(timespec="seconds"), "label": label or "",
               "posture": posture, "release_mode": mode, "slant_distance": sample.get("distance"),
               "pitch_deg": sample.get("pitch_deg"), "camera_height": sample.get("camera_height"),
               "surface_height": surface_h, "horizontal_gap": fwd_gap, "hit": sample.get("hit"),
               "landed": landed, "shot": shot}
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"    logged: landed={landed} at slant={_fmt(sample.get('distance'))}m -> {os.path.basename(CSV_PATH)}")

    def grab(label):
        SetHandsActive(True); state["hands"] = True
        res = extend_arm_until_grabbed()
        grabbed = bool(res.get("gripped"))
        tail = f" | {res['reason']}" if res.get("reason") else ""
        outcome = "GRABBED" if grabbed else ("REFUSED" if res.get("blocked") else "missed")
        print(f"  [GRAB] {outcome} hovered={res.get('hovered')!r}{tail}")
        if grabbed:
            print("  carrying at REST - drive to the counter (w/s/a/d), `center`, then `p <label>`.")

    def _resync():
        # Raw WASD moves the agent behind NavSession's back - re-read pose before an A* drive.
        nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)

    def drive(cp):
        if cp not in sm.by_id:
            print(f"  cp{cp} is not a checkpoint in the graph."); return
        set_hand_pose("rest")        # 6.1 carry-safe before any drive
        _resync()
        ok = nav.goto(cp)
        print(f"  goto cp{cp}: arrived={ok}")
        read()

    def drive_counter():
        set_hand_pose("rest")        # 6.1 carry-safe (go_to_counter won't touch the hands - caller's job)
        res = go_to_counter(nav)     # resyncs pose + A*-drive + face the cp54 landmark
        print(f"  go_to_counter: arrived={res.get('arrived')} cp={res.get('checkpoint')} "
              f"at ({res.get('x', 0):.2f},{res.get('z', 0):.2f})"
              + (f" - {res['reason']}" if res.get("reason") else ""))
        read()

    def center(aim_y=None):
        # Hands stay ACTIVE at REST while carrying - stowing them here would drop the item and undo
        # Phase 6.1. REST is out of frame and LiDAR-culled, so the centring read is clean.
        # aim_y dials the vertical aim live (see help): the counter bbox includes the vertical front
        # face, so at the default aim the centre ray lands near the top's FRONT EDGE (measured live
        # 2026-07-22, user screenshot) - a higher aim_y pushes the sample/release point deeper onto
        # the surface. The dialed-in winner gets pinned into perception.COUNTER_AIM_NORM.
        aim = (COUNTER_AIM_NORM[0], aim_y) if aim_y is not None else COUNTER_AIM_NORM
        res = center_to_counter(aim_norm=aim, debug_dir=run_dir)
        print(f"  center_to_counter: {res.get('outcome')} "
              f"(detected={res.get('detected')}, residual={res.get('residual_px')}, "
              f"aim={aim}) - {res.get('center_message')}")
        if res.get("outcome") == "not_detected":
            print("    counter not detected: pan/tilt it into view, or this is the front-item-bias / "
                  "glass finding - fall back to facing the cp54 perpendicular yaw.")
        read()
        return res

    def center_scanner(aim_y=None):
        # M2: the scan-pad sibling of center(). Same hands-stay-ACTIVE rule (the caller is carrying
        # by construction); same aim dial; the winning wording/aim get pinned into perception.
        aim = (SCANNER_AIM_NORM[0], aim_y) if aim_y is not None else SCANNER_AIM_NORM
        res = center_to_scanner(aim_norm=aim, debug_dir=run_dir)
        print(f"  center_to_scanner: {res.get('outcome')} "
              f"(detected={res.get('detected')}, residual={res.get('residual_px')}, "
              f"aim={aim}) - {res.get('center_message')}")
        if res.get("outcome") == "not_detected":
            print("    pad not detected: pan/tilt it into view, or iterate the "
                  "perception.SCANNER_TARGET_INFO wording - that is the M2 finding.")
        read()
        return res

    def _new_lines(before, after):
        """Multiset diff: lines in `after` not accounted for by `before`. A second `1x <item>`
        receipt line is exactly the signal a plain set-diff would swallow, so count multiplicities."""
        pool = Counter(before)
        out = []
        for l in after:
            if pool[l] > 0:
                pool[l] -= 1
            else:
                out.append(l)
        return out

    def read_screen():
        """Centre the POS screen and OCR ONLY its bbox (region-cropped - the receipt is a small
        bright rectangle the full-frame read would drown in shelf/label text). Returns (lines, box,
        detected). Leaves state["receipt"] UNCHANGED - the caller decides when the diff is 'consumed'
        (sweep updates it; the standalone `screen` command just prints)."""
        res = center_to_screen(debug_dir=run_dir)
        box = res.get("box")
        print(f"  center_to_screen: {res.get('outcome')} "
              f"(detected={res.get('detected')}, residual={res.get('residual_px')}) "
              f"- {res.get('center_message')}")
        if not box:
            print("    screen not detected: pan/tilt it into view, or iterate "
                  "perception.SCREEN_TARGET_INFO (M5). No OCR possible without a box.")
            return [], None, False
        lines = read_text_in_box(box)
        return lines, box, True

    def sweep(label):
        """M1: the in-hand scan sweep. Full extension toward the centred pad - the stall guard /
        Unity clamp stops the hand, NOT a calibrated distance (the Easy trigger box covers the
        whole scan region, so the precondition is only that the pad slant is inside SCAN_REACH_M -
        that is `adv`'s job). Retracts and restores REST; the grip is never opened.

        Two verification channels, both logged (never promote one to the other):
          scanned_measured  - after the sweep, centre the POS screen and region-OCR its bbox; a NEW
                              receipt line vs state["receipt"] (the running baseline) => measured scan.
          scanned_verified  - the user's own beep/eye call (the ground truth while no checkout
                              websocket query exists).
        still_holding is the other half of M1 (the sweep must not drop the item)."""
        if not _holding():
            print("  not holding anything - the sweep scans the HELD item (grab one with g first).")
            return
        s0 = RequestLidarCenter()
        _surface_h, fwd_gap = _geometry(s0)
        slant0 = s0.get("distance")
        print(f"  [SWEEP] start slant={_fmt(slant0)}m fwd_gap={_fmt(fwd_gap)} hit={s0.get('hit')}")
        if not s0.get("hit"):
            print("    (no LiDAR hit at frame centre - `scanner` first so the slant column means the pad)")
        elif slant0 is not None and float(slant0) > SCAN_REACH_M:
            print(f"    (pad slant {float(slant0):.2f} m is OUTSIDE SCAN_REACH_M={SCAN_REACH_M} - "
                  f"`adv` closer first; sweeping anyway logs an honest miss row)")

        set_hand_pose("grab")
        moved = 0
        try:
            prev = _hand_state()["leftTranslation"]
            for _ in range(SWEEP_MAX_STEPS):           # full extension; stall guard stops earlier
                cur = _XTNFWD_LEFT_()["leftTranslation"]
                if sum((cur[k] - prev[k]) ** 2 for k in range(3)) ** 0.5 <= 1e-4:
                    break
                prev = cur
                moved += 1
        finally:
            for _ in range(moved):
                _PLLBCK_LEFT_()
            set_hand_pose("rest")

        still = _holding()
        state["idx"] += 1
        RequestScreenshot(save_image=True)
        shot = _crosshair(run_dir, state["idx"], tag="sweep")
        print(f"  [SWEEP] extended {moved} step(s): still_holding={still}  [{shot}]")
        if not still:
            print("    WARNING: the grip OPENED during the sweep - the M1 failure mode; log it honestly.")

        # Measured channel: centre the screen, region-OCR its bbox, diff vs the running receipt.
        print("  [SWEEP] reading the POS screen for the receipt delta...")
        lines, _box, detected = read_screen()
        new = _new_lines(state["receipt"], lines) if detected else []
        scanned_measured = any(any(c.isdigit() for c in l) for l in new)
        if detected:
            state["receipt"] = Counter(lines)          # consume: this screen is the new baseline
            print(f"  [SWEEP] screen: {len(lines)} line(s), {len(new)} new -> scanned_measured={scanned_measured}")
            for l in new:
                print(f"    + {l}")
        else:
            print("  [SWEEP] screen not read - scanned_measured unavailable (rely on your verdict).")

        try:
            verdict = input("    did it SCAN (beep / new receipt line, by eye)? "
                            "[y]es / [n]o / [s]kip (don't log) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            verdict = "s"
        if verdict not in ("y", "yes", "n", "no"):
            print("    skipped - no row logged.")
            return
        scanned_verified = verdict in ("y", "yes")
        if scanned_measured != scanned_verified:
            print(f"    NOTE: measured({scanned_measured}) != verified({scanned_verified}) - "
                  "an honest M5 discrepancy (OCR miss or occlusion); both are logged.")
        row = {"ts": datetime.now().isoformat(timespec="seconds"), "label": label or "",
               "slant_start": slant0, "pitch_deg": s0.get("pitch_deg"),
               "camera_height": s0.get("camera_height"), "steps_extended": moved,
               "scanned_measured": scanned_measured, "scanned_verified": scanned_verified,
               "still_holding": still, "new_lines": " | ".join(new), "shot": shot}
        write_header = not os.path.exists(SCAN_CSV_PATH)
        with open(SCAN_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SCAN_CSV_COLS)
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"    logged: scanned_measured={scanned_measured} scanned_verified={scanned_verified} "
              f"still_holding={still} at slant={_fmt(slant0)}m -> {os.path.basename(SCAN_CSV_PATH)}")

    def screen_read():
        """M5 (standalone): centre the POS screen and region-OCR its bbox, print every line. Does
        NOT consume the receipt baseline - use it to check readability without disturbing a sweep's
        diff. `screen` at the scan pose answers 'is the receipt OCR-able from here?'."""
        lines, box, detected = read_screen()
        state["idx"] += 1
        shot = _crosshair(run_dir, state["idx"], tag="screen")
        print(f"  [SCREEN] detected={detected} {len(lines)} line(s)  [{shot}]")
        for l in lines:
            print(f"    | {l}")

    def align(max_iter=3):
        """M4: the Option A lateral-alignment prototype. Centre the pad, read the yaw error vs
        cp54's perpendicular, re-face the perpendicular, strafe slant*sin(yaw_off) out, re-centre.
        Converged = yaw error within one pan step (2.5 deg) or a strafe that rounds to zero steps.
        The per-iteration residual it prints is the number that decides Option A vs a spliced
        scan-dock node (Option B)."""
        cp_id = sm.counter_checkpoint()
        if cp_id is None:
            print("  align: no landmark checkpoint in the graph.")
            return
        perp = perpendicular_yaw(sm.by_id[cp_id])
        for it in range(1, max_iter + 1):
            res = center_scanner()
            if res.get("outcome") == "not_detected":
                print(f"  align {it}/{max_iter}: pad not detected - aborting (fix M2 first).")
                return
            _resync()
            yaw_off = normalize_deg(nav.rot[1] - perp)
            s = RequestLidarCenter()
            if not s.get("hit"):
                print(f"  align {it}/{max_iter}: no LiDAR hit at centre - aborting.")
                return
            lateral = float(s["distance"]) * math.sin(math.radians(yaw_off))
            steps = round(abs(lateral) / 0.10)
            print(f"  align {it}/{max_iter}: yaw_off={yaw_off:+.1f} deg "
                  f"slant={float(s['distance']):.2f} m -> lateral={lateral:+.2f} m ({steps} step(s))")
            if abs(yaw_off) <= 2.5 or steps == 0:
                print(f"  align: CONVERGED in {it} iteration(s) (residual yaw {yaw_off:+.1f} deg).")
                return
            nav.pos, nav.rot = face(nav.args, nav.pos, nav.rot, perp)
            (move_right if lateral > 0 else move_left)(steps)
        print(f"  align: NOT converged after {max_iter} iterations - an honest M4 data point "
              f"(the Option B trigger).")

    def approach(target_slant, max_iter=6, recenter=None):
        """Move (forward/back) until the centre-ray slant to the centred surface ~ target_slant,
        re-centring after each move (moving de-centres the surface point) via `recenter` (default:
        the counter; `adv` passes the scanner variant). Reuses plan_reach's geometry inline: moving
        forward shrinks the horizontal gap, so the gap needed for a target slant is
        sqrt(target^2 - vertical^2). Iterates because Unity clamps a big move and the re-centre nudges
        the sample. The scripted stand-in for 6.2's place-approach loop (and the number that decides
        whether go_to_counter needs an approach-past-cp54 step)."""
        recenter = recenter or center
        for _ in range(max_iter):
            s = RequestLidarCenter()
            if not s.get("hit"):
                print("  approach: no surface at centre - re-centring."); recenter(); continue
            slant = float(s["distance"]); theta = math.radians(float(s["pitch_deg"]))
            vertical = slant * math.sin(theta); gap = slant * math.cos(theta)
            if target_slant <= vertical:
                print(f"  approach: target {target_slant:.2f} m is below the {vertical:.2f} m vertical "
                      f"drop - unreachable by moving; dropping from here."); return s
            needed = math.sqrt(target_slant ** 2 - vertical ** 2)
            steps = round((gap - needed) / 0.10)
            if steps == 0:
                print(f"  approach: at target (slant {slant:.2f} m ~ {target_slant:.2f} m)."); return s
            (move_forward if steps > 0 else move_backward)(min(abs(steps), 10))
            print(f"  approach: slant {slant:.2f} -> {target_slant:.2f} m "
                  f"({'fwd' if steps > 0 else 'back'} {abs(steps)} step(s))")
            recenter()
        print(f"  approach: hit the {max_iter}-move cap; dropping from wherever we are.")
        return RequestLidarCenter()

    def grab_and_place(label, target_slant):
        """gp macro: grab the centred item (or use what's already held) -> carry to the counter ->
        centre (2 retries) -> auto-approach to target_slant -> centre again -> drop -> ask + log. The
        whole 6.1+6.2 chain scripted, no LLM. Each call logs one place row via place()."""
        if not _holding():
            SetHandsActive(True); state["hands"] = True
            res = extend_arm_until_grabbed()
            if not (res.get("gripped") and _holding()):
                print(f"  gp: grab FAILED ({res.get('reason') or res}) - reposition on the item and retry.")
                return
            print(f"  gp: grabbed {res.get('hovered')!r}")
        else:
            print("  gp: already holding - skipping grab.")

        set_hand_pose("rest")                       # 6.1 carry-safe before the drive
        res = go_to_counter(nav)
        if not res.get("arrived"):
            print(f"  gp: could not reach the counter ({res.get('reason', 'path blocked')}) - aborting.")
            return
        print(f"  gp: at counter cp{res.get('checkpoint')}; approaching to slant ~{target_slant:.2f} m")

        for attempt in range(3):                    # centre with up to 2 retries
            if center().get("outcome") == "success":
                break
            print(f"  gp: centre retry {attempt + 1}/2")
        else:
            print("  gp: counter not centred after retries - proceeding best-effort.")

        approach(target_slant)
        center()                                    # centre again after the approach moves
        place(label, mode="extend")                 # drop + screenshot + ask landed + log

    def _parse_gp(rest):
        """`gp <label> [T]` - only treat a trailing token as the target slant T when there are >=2
        tokens, so a lone `gp 7` reads 7 as the LABEL (not a 7 m target)."""
        toks = list(rest)
        target = TIGHTEN_TARGET_M
        if len(toks) >= 2:
            try:
                target = float(toks[-1]); toks = toks[:-1]
            except ValueError:
                pass
        return (" ".join(toks) or "gp"), target

    print(HELP)
    print(f"logging to {CSV_PATH}\ncrosshair reads + place shots -> {run_dir}\n")
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
            elif cmd == "counter":
                drive_counter()
            elif cmd == "goto":
                if rest and rest[0].lstrip("-").isdigit():
                    drive(int(rest[0]))
                else:
                    print("  goto needs a checkpoint id, e.g. `goto 15`.")
            elif cmd == "center":
                try:
                    center(float(rest[0]) if rest else None)
                except ValueError:
                    print("  center takes an optional vertical aim 0-1, e.g. `center 0.62`.")
            elif cmd == "scanner":
                try:
                    center_scanner(float(rest[0]) if rest else None)
                except ValueError:
                    print("  scanner takes an optional vertical aim 0-1, e.g. `scanner 0.62`.")
            elif cmd == "align":
                align(count if rest else 3)
            elif cmd == "adv":
                try:
                    approach(float(rest[0]) if rest else SCAN_REACH_M, recenter=center_scanner)
                except ValueError:
                    print("  adv takes an optional target slant in metres, e.g. `adv 0.85`.")
            elif cmd == "sweep":
                sweep(" ".join(rest))
            elif cmd == "screen":
                screen_read()
            elif cmd in MOVES:
                MOVES[cmd](count); read()
            elif cmd == "c":
                state["crouching"] = not state["crouching"]
                SetCrouch(state["crouching"]); read()
            elif cmd == "h":
                state["hands"] = not state["hands"]
                SetHandsActive(state["hands"])
                print(f"  hands {'ON' if state['hands'] else 'OFF'} "
                      + ("(WARNING: hands off drops a carried item)" if not state["hands"] else ""))
            elif cmd == "g":
                grab(" ".join(rest))
            elif cmd == "p":
                place(" ".join(rest), mode="extend")
            elif cmd == "pr":
                place(" ".join(rest), mode="rest")
            elif cmd == "gp":
                lbl, tgt = _parse_gp(rest)
                grab_and_place(lbl, tgt)
            elif cmd == "x":
                ToggleLeftGrip(); print("  released left grip (no row logged)"); read()
            else:
                print(f"  ? unknown command {cmd!r} - type ? for help")
    finally:
        SetCrouch(False)   # stand back up; the crouch verdict is sticky
        nav.close()        # no-op restore for stow_hands=False, but keep the session tidy
        print(f"\ndone. rows -> {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
