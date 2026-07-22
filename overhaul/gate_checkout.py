"""gate_checkout.py - Phase 6.2 GATE: N scripted checkout runs via checkout_held_item, honest
four-number scoring, 4/5 pass. Doubles as the tray-envelope re-measure (logs each drop's slant +
whether it landed in the tray by eye). 🎮  Needs the sim in Play mode on ws://localhost:8080.

    python gate_checkout.py                 # 5 runs; you align on an item each run, it grabs in place
    python gate_checkout.py --runs 5 --pass 4
    python gate_checkout.py --shelf 15      # auto goto cp15 + grab each run (falls to manual on a miss)

Each run:
  1. get ONE item in hand (you align the agent on it + Enter, or --shelf drives+grabs)
  2. checkout_held_item(nav): drive to counter -> align pad -> scan -> bag  (deterministic, NO LLM)
  3. you confirm by eye: did it BEEP (scanned_verified) and is it IN the tray (placed_verified)
  4. the row is logged (measured + verified) to slamtest/output/placetests/gate_checkout.csv

PASS = at least --pass runs (default 4 of 5) with scanned_verified AND placed_verified.

HONEST SCORING: the MEASURED numbers (scanned = OCR receipt delta; placed = released under a
'placeable' verdict) come from the tool; the VERIFIED numbers (beep + in-tray) are YOUR eyes. They are
logged SEPARATELY and never promoted. A measured PASS with a verified FAIL is the discrepancy the gate
exists to surface (e.g. a drop that reads 'placeable' but rolls off the tray lip).

TRAY-ENVELOPE RE-MEASURE (finding 8): the `place_slant` + `placed_verified` columns ARE the re-measure
- they record, for the tray target with the clip-standoff, at what slant a drop actually lands. If
every verified-in-tray row sits at place_max=1.28 m or below with no misses, the envelope holds; a
verified miss at a slant <= 1.28 says re-fit.
"""
import argparse
import csv
import os
import sys
from datetime import datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

CSV_PATH = os.path.join(_THIS, "slamtest", "output", "placetests", "gate_checkout.csv")
CSV_COLS = ["ts", "run", "grabbed_item", "aligned", "align_slant", "align_lateral",
            "scanned_measured", "still_holding", "placed_measured", "place_verdict",
            "place_slant", "release_z", "scanned_verified", "placed_verified",
            "success_measured", "success_verified", "reason"]


def _ask_yn(prompt):
    """y/n from the user; None on skip/EOF (counts as NOT-success but logged distinct from a firm no)."""
    try:
        a = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if a in ("y", "yes"):
        return True
    if a in ("n", "no"):
        return False
    return None


def _dig(d, *keys):
    """Safe nested get: _dig(result, 'steps', 'place', 'distance') -> value or None."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def main():
    ap = argparse.ArgumentParser(description="Phase 6.2 Gate: N checkout runs, 4/5 pass.")
    ap.add_argument("--runs", type=int, default=5, help="number of checkout runs (default 5)")
    ap.add_argument("--pass", dest="pass_n", type=int, default=4,
                    help="runs that must be verified-success to PASS (default 4)")
    ap.add_argument("--shelf", type=int, default=None,
                    help="checkpoint to auto goto+grab from each run (falls to manual on a miss)")
    ap.add_argument("--hand", default="left", choices=("left", "right"))
    args = ap.parse_args()

    from env import RequestScreenshot, SetHandsActive, TransformHands
    from manipulation import set_hand_pose, extend_arm_until_grabbed
    from store_map import StoreMap, NavSession, checkout_held_item
    from explore import step_agent

    try:
        RequestScreenshot(save_image=True)
    except Exception as e:
        print(f"Could not reach the sim ({type(e).__name__}: {e}). Is it in Play mode on :8080?")
        return 1

    sm = StoreMap()
    nav = NavSession(sm, stow_hands=False)
    SetHandsActive(True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    grip_key = "leftGrippedState" if args.hand == "left" else "rightGrippedState"

    def _holding():
        return bool(TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)).get(grip_key))

    def ensure_holding(run):
        """Get one item in hand for this run. Returns the item name, 'already-holding', or None (abort
        the gate). A failed GRAB never counts against the checkout tally - it just re-prompts."""
        if _holding():
            return "already-holding"
        if args.shelf is not None:                       # auto path: drive to the shelf and grab
            if args.shelf not in sm.by_id:
                print(f"  cp{args.shelf} not in the graph - using manual grab.")
            else:
                set_hand_pose("rest", hand=args.hand)
                nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
                if nav.goto(args.shelf):
                    res = extend_arm_until_grabbed(hand=args.hand)
                    if res.get("gripped") and _holding():
                        print(f"  grabbed {res.get('hovered')!r} at cp{args.shelf}")
                        return res.get("hovered")
                    print(f"  auto-grab at cp{args.shelf} missed ({res.get('reason') or ''}) - manual.")
                else:
                    print(f"  could not drive to cp{args.shelf} - manual.")
        while True:                                      # manual path: user aligns, we grab in place
            try:
                cmd = input(f"  Run {run}: align the agent on an item, then Enter to grab "
                            f"(q = abort gate) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if cmd == "q":
                return None
            res = extend_arm_until_grabbed(hand=args.hand)
            if res.get("gripped") and _holding():
                print(f"  grabbed {res.get('hovered')!r}")
                return res.get("hovered")
            print(f"  grab missed ({res.get('reason') or ''}) - re-align and Enter to retry (q = abort).")

    rows = []
    try:
        for run in range(1, args.runs + 1):
            print(f"\n{'=' * 70}\n  GATE RUN {run}/{args.runs}\n{'=' * 70}")
            item = ensure_holding(run)
            if item is None:
                print("  aborted - ending the gate early."); break

            result = checkout_held_item(nav, hand=args.hand)
            scanned_m = bool(result.get("scanned"))
            placed_m = bool(result.get("placed"))
            aligned = bool(result.get("aligned"))
            still = bool(_dig(result, "steps", "scan", "still_holding"))
            print(f"  MEASURED: aligned={aligned} scanned={scanned_m} placed={placed_m} "
                  f"still_holding={still} - {result.get('reason')}")

            print("  --- confirm by eye ---")
            scanned_v = _ask_yn("  did it BEEP / show a new receipt line? [y/n] > ")
            placed_v = _ask_yn("  is the item IN the tray? [y/n] > ")

            row = {
                "ts": datetime.now().isoformat(timespec="seconds"), "run": run,
                "grabbed_item": item, "aligned": aligned,
                "align_slant": _dig(result, "steps", "align", "slant"),
                "align_lateral": _dig(result, "steps", "align", "lateral"),
                "scanned_measured": scanned_m, "still_holding": still, "placed_measured": placed_m,
                "place_verdict": _dig(result, "steps", "place", "verdict"),
                "place_slant": _dig(result, "steps", "place", "distance"),
                "release_z": _dig(result, "steps", "place", "release_z"),
                "scanned_verified": scanned_v, "placed_verified": placed_v,
                "success_measured": scanned_m and placed_m,
                "success_verified": (scanned_v is True) and (placed_v is True),
                "reason": result.get("reason"),
            }
            rows.append(row)
            write_header = not os.path.exists(CSV_PATH)
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLS)
                if write_header:
                    w.writeheader()
                w.writerow(row)
            print(f"  logged run {run}: measured_success={row['success_measured']} "
                  f"verified_success={row['success_verified']} -> {os.path.basename(CSV_PATH)}")
    finally:
        nav.close()

    # ---- Tally ----------------------------------------------------------------------------------
    n = len(rows)
    print(f"\n{'=' * 70}\n  GATE 6.2 RESULT ({n} run(s))\n{'=' * 70}")
    if not n:
        print("  no completed runs."); return 1
    meas = sum(1 for r in rows if r["success_measured"])
    veri = sum(1 for r in rows if r["success_verified"])
    for r in rows:
        print(f"  run {r['run']}: measured={'PASS' if r['success_measured'] else 'fail'} "
              f"verified={'PASS' if r['success_verified'] else 'fail'} "
              f"(scan m/v={r['scanned_measured']}/{r['scanned_verified']}, "
              f"place m/v={r['placed_measured']}/{r['placed_verified']}, "
              f"place_slant={r['place_slant']})")
    passed = veri >= args.pass_n
    print(f"\n  measured success: {meas}/{n}   verified success: {veri}/{n}")
    print(f"  GATE 6.2 {'PASS' if passed else 'FAIL'} "
          f"(need {args.pass_n} verified of {n}; got {veri})")
    disc = [r["run"] for r in rows if r["success_measured"] != r["success_verified"]]
    if disc:
        print(f"  measured/verified DISCREPANCY on run(s) {disc} - inspect (aim confound / OCR miss).")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
