"""Checkout acceptance check: N scripted runs via checkout_held_item with honest
four-number scoring, 4/5 pass. Doubles as the tray-envelope re-measure (logs each drop's slant +
whether it landed in the tray by eye). 🎮  Needs the sim in Play mode on ws://localhost:8080.

    python validation/acceptance/checkout.py                 # default 5 runs, 4 required
    python validation/acceptance/checkout.py --runs 5 --pass 4
    python validation/acceptance/checkout.py --shelf 15
    python validation/acceptance/checkout.py --two-hands

Each run:
  1. get item(s) in hand (you align the agent on it + Enter, or --shelf drives+grabs). --two-hands
     fills BOTH hands (grab left, then right).
  2. the deterministic checkout macro, NO LLM: single -> checkout_held_item(nav) (drive -> align -> scan
     -> bag); --two-hands -> checkout_held_items(nav), the FUSED pass (drive+align once, sweep-scan
     BOTH before bagging EITHER, then bag both).
  3. you confirm by eye, PER ITEM: did it BEEP (scanned_verified) and is it IN the tray (placed_verified)
  4. a row PER ITEM is logged under validation/artifacts/acceptance/checkout/
     (single) or checkout_two_hands.csv (--two-hands, with a `hand` column).

PASS = at least --pass RUNS (default 4 of 5) verified-success. A --two-hands run passes only when BOTH
its items are scanned_verified AND placed_verified (the run is the unit; each item is a logged row).

HONEST SCORING: the MEASURED numbers (scanned = OCR receipt delta; placed = released under a
'placeable' verdict) come from the tool; the VERIFIED numbers (beep + in-tray) are YOUR eyes. They are
logged SEPARATELY and never promoted. A measured PASS with a verified FAIL is the discrepancy this check
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
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "agent")  # agent/ — runtime modules (env, store_map) live there
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ARTIFACT_DIR = os.path.join(
    os.path.dirname(_ROOT), "validation", "artifacts", "acceptance", "checkout")
CSV_PATH = os.path.join(_ARTIFACT_DIR, "checkout.csv")
CSV_COLS = ["ts", "run", "grabbed_item", "aligned", "align_slant", "align_lateral",
            "scanned_measured", "still_holding", "placed_measured", "place_verdict",
            "place_slant", "release_z", "scanned_verified", "placed_verified",
            "success_measured", "success_verified", "reason"]
# --two-hands logs a row PER ITEM to its OWN file (an added `hand` column would misalign the existing
# single-hand CSV, whose header is only written once for a fresh file).
CSV_PATH_2H = os.path.join(_ARTIFACT_DIR, "checkout_two_hands.csv")
CSV_COLS_2H = ["ts", "run", "hand"] + CSV_COLS[2:]


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


def _item_signals(result, side, two):
    """Per-ITEM measured signals from a checkout result. The single macro is flat (scanned/placed +
    steps.scan/steps.place); the fused macro nests per hand (per_hand[side] + steps.scans[side] /
    steps.places[side]). Returns (scanned, placed, still_holding, place_verdict, place_slant, release_z)."""
    if two:
        return (bool(_dig(result, "per_hand", side, "scanned")),
                bool(_dig(result, "per_hand", side, "placed")),
                bool(_dig(result, "steps", "scans", side, "still_holding")),
                _dig(result, "steps", "places", side, "verdict"),
                _dig(result, "steps", "places", side, "distance"),
                _dig(result, "steps", "places", side, "release_z"))
    return (bool(result.get("scanned")), bool(result.get("placed")),
            bool(_dig(result, "steps", "scan", "still_holding")),
            _dig(result, "steps", "place", "verdict"),
            _dig(result, "steps", "place", "distance"),
            _dig(result, "steps", "place", "release_z"))


def main():
    ap = argparse.ArgumentParser(description="Phase 6.2 acceptance: N checkout runs, 4/5 pass.")
    ap.add_argument("--runs", type=int, default=5, help="number of checkout runs (default 5)")
    ap.add_argument("--pass", dest="pass_n", type=int, default=4,
                    help="runs that must be verified-success to PASS (default 4)")
    ap.add_argument("--shelf", type=int, default=None,
                    help="checkpoint to auto goto+grab from each run (falls to manual on a miss)")
    ap.add_argument("--hand", default="left", choices=("left", "right"),
                    help="which hand to grab+checkout in SINGLE-item mode (ignored under --two-hands)")
    ap.add_argument("--two-hands", action="store_true",
                    help="grab an item in EACH hand and check both out in one fused pass "
                         "(checkout_held_items); logs a row per item to checkout_two_hands.csv")
    ap.add_argument("--ocr-url", default=None, help="OCR service base URL (else $SARI_OCR_URL/default).")
    args = ap.parse_args()

    from vision.ocr_client import check_ocr_health
    try:
        check_ocr_health(args.ocr_url)
    except Exception as error:
        print(f"OCR preflight failed before simulator work: {error}")
        return 1

    from sim.env import RequestScreenshot, SetHandsActive, TransformHands, default_uri
    from manip.manipulation import set_hand_pose, extend_arm_until_grabbed
    from nav.store_map import StoreMap, NavSession, checkout_held_item, checkout_held_items
    from explore import step_agent

    try:
        RequestScreenshot(save_image=True)
    except Exception as e:
        print(f"Could not reach the sim ({type(e).__name__}: {e}). Is it in Play mode on :8080?")
        return 1

    sm = StoreMap()
    nav = NavSession(sm, uri=default_uri(), stow_hands=False)
    SetHandsActive(True)
    two = args.two_hands
    csv_path = CSV_PATH_2H if two else CSV_PATH
    csv_cols = CSV_COLS_2H if two else CSV_COLS
    sides = ("left", "right") if two else (args.hand,)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def _holding(side):
        key = "leftGrippedState" if side == "left" else "rightGrippedState"
        return bool(TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)).get(key))

    def grab_into(side, run):
        """Get ONE item into the `side` hand. Returns the item name, 'already-holding', or None (abort
        the acceptance check). A failed GRAB never counts against the checkout tally - it just re-prompts."""
        if _holding(side):
            return "already-holding"
        if args.shelf is not None:                       # auto path: drive to the shelf and grab
            if args.shelf not in sm.by_id:
                print(f"  cp{args.shelf} not in the graph - using manual grab.")
            else:
                set_hand_pose("rest", hand=side)
                nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
                if nav.goto(args.shelf):
                    res = extend_arm_until_grabbed(hand=side)
                    if res.get("gripped") and _holding(side):
                        print(f"  grabbed {res.get('hovered')!r} into the {side} hand at cp{args.shelf}")
                        return res.get("hovered")
                    print(f"  auto-grab at cp{args.shelf} missed ({res.get('reason') or ''}) - manual.")
                else:
                    print(f"  could not drive to cp{args.shelf} - manual.")
        while True:                                      # manual path: user aligns, we grab in place
            try:
                cmd = input(f"  Run {run}: align the agent on an item for the {side} hand, then Enter "
                            f"to grab (q = abort check) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if cmd == "q":
                return None
            res = extend_arm_until_grabbed(hand=side)
            if res.get("gripped") and _holding(side):
                print(f"  grabbed {res.get('hovered')!r} into the {side} hand")
                return res.get("hovered")
            print(f"  grab missed ({res.get('reason') or ''}) - re-align and Enter to retry (q = abort).")

    def ensure_holding(run):
        """Fill the hand(s) this check needs. Returns {side: item_name} or None to abort. Two-hand mode
        fills BOTH hands (left then right); single mode fills args.hand only."""
        names = {}
        for side in sides:
            got = grab_into(side, run)
            if got is None:
                return None
            names[side] = got
        return names

    run_summaries = []    # (run, measured_ok, verified_ok) per RUN - AND over its item(s)
    try:
        for run in range(1, args.runs + 1):
            print(f"\n{'=' * 70}\n  ACCEPTANCE RUN {run}/{args.runs}{'  (two-hand)' if two else ''}"
                  f"\n{'=' * 70}")
            names = ensure_holding(run)
            if names is None:
                print("  aborted - ending the acceptance check early."); break

            result = checkout_held_items(nav) if two else checkout_held_item(nav, hand=args.hand)
            aligned = bool(result.get("aligned"))
            print(f"  MEASURED: aligned={aligned} scanned={result.get('scanned')} "
                  f"placed={result.get('placed')} - {result.get('reason')}")

            run_meas_ok, run_veri_ok = True, True
            for side in sides:
                scanned_m, placed_m, still, verdict, slant, rz = _item_signals(result, side, two)
                item = names.get(side)
                label = f"the {side} item " if two else "it "
                print(f"  --- confirm {label}({item!r}) by eye ---"
                      f"  [measured scanned={scanned_m} placed={placed_m} still_holding={still}]")
                scanned_v = _ask_yn(f"  did {label}BEEP / add a receipt line? [y/n] > ")
                placed_v = _ask_yn(f"  is {label}IN the tray? [y/n] > ")
                item_meas = scanned_m and placed_m
                item_veri = (scanned_v is True) and (placed_v is True)
                run_meas_ok &= item_meas
                run_veri_ok &= item_veri

                row = {
                    "ts": datetime.now().isoformat(timespec="seconds"), "run": run,
                    "grabbed_item": item, "aligned": aligned,
                    "align_slant": _dig(result, "steps", "align", "slant"),
                    "align_lateral": _dig(result, "steps", "align", "lateral"),
                    "scanned_measured": scanned_m, "still_holding": still, "placed_measured": placed_m,
                    "place_verdict": verdict, "place_slant": slant, "release_z": rz,
                    "scanned_verified": scanned_v, "placed_verified": placed_v,
                    "success_measured": item_meas, "success_verified": item_veri,
                    "reason": result.get("reason"),
                }
                if two:
                    row["hand"] = side
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=csv_cols)
                    if write_header:
                        w.writeheader()
                    w.writerow(row)
                print(f"    logged {side if two else 'item'}: measured_success={item_meas} "
                      f"verified_success={item_veri} (place_slant={slant}) -> {os.path.basename(csv_path)}")

            run_summaries.append((run, run_meas_ok, run_veri_ok))
            print(f"  run {run}: measured_ok={run_meas_ok} verified_ok={run_veri_ok}")
    finally:
        nav.close()

    # ---- Tally (the RUN is the pass/fail unit; a two-hand run needs BOTH items verified) ---------
    n = len(run_summaries)
    print(f"\n{'=' * 70}\n  ACCEPTANCE 6.2 RESULT ({n} run(s){', two-hand' if two else ''})\n{'=' * 70}")
    if not n:
        print("  no completed runs."); return 1
    meas = sum(1 for (_, m_, _) in run_summaries if m_)
    veri = sum(1 for (_, _, v_) in run_summaries if v_)
    for (run, m_, v_) in run_summaries:
        print(f"  run {run}: measured={'PASS' if m_ else 'fail'} verified={'PASS' if v_ else 'fail'}")
    passed = veri >= args.pass_n
    print(f"\n  measured success: {meas}/{n}   verified success: {veri}/{n}")
    print(f"  ACCEPTANCE 6.2 {'PASS' if passed else 'FAIL'} "
          f"(need {args.pass_n} verified of {n}; got {veri})")
    print(f"  per-item detail (measured vs verified, place_slant) is in {os.path.basename(csv_path)}.")
    disc = [run for (run, m_, v_) in run_summaries if m_ != v_]
    if disc:
        print(f"  measured/verified DISCREPANCY on run(s) {disc} - inspect (aim confound / OCR miss).")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
