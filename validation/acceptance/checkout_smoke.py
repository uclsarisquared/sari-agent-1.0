"""One-item live smoke of the Phase 6.2 checkout chain (align -> scan -> bag). 🎮

Drives the FOUR built primitives in sequence on a single carried item, printing each one's structured
result, so the chain can be shaken out BEFORE it is wrapped in the deterministic checkout_held_item()
macro and before the measured acceptance check. This is the supervised dry-run the phase6.2 plan calls for: it is
where align_to_scanner's live margins and place_held_item's release show themselves.

    python validation/acceptance/checkout_smoke.py --grabnow
    python validation/acceptance/checkout_smoke.py --shelf 15
    python validation/acceptance/checkout_smoke.py
    python validation/acceptance/checkout_smoke.py --grabnow --auto
    python validation/acceptance/checkout_smoke.py --two-hands

Needs the sim in Play mode on ws://localhost:8080.

The chain (no LLM anywhere - all deterministic):
  1. (optional) goto <shelf> + extend_arm_until_grabbed   -> get one item in hand
  2. go_to_counter(nav)                                    -> drive to cp54, face it
  3. align_to_scanner(nav)                                 -> square on the scan pad, slant <= reach
  4. read the POS-screen baseline receipt, THEN re-acquire the pad -> the baseline read only happens
        AFTER alignment, because at the arrival pose (~1.64 m out, oblique) the screen is too small to
        OCR; aligned we are close and square, so it is legible. Reading it yaws the body to the SCREEN,
        so we center_to_scanner again to face the PAD before the sweep (a camera/yaw move only - the
        body does not translate, so the pad slant align achieved is preserved).
  5. scan_held_item(baseline=...)                          -> sweep; measured via the receipt delta
  6. place_held_item()                                     -> depth-gated release into the tray

HONEST SCORING: the script reports the MEASURED signals only (aligned, scanned_measured, still_holding,
placed==released-under-a-placeable-verdict). The BEEP and whether the item is physically IN the tray
are the user's eye - the script asks for neither and fakes neither. A green measured chain with a
"didn't actually land / didn't beep" from your eyes is exactly the discrepancy this smoke exists to
surface (e.g. the aim confound), so watch the sim.
"""
import argparse
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "agent")  # agent/ — runtime modules live there
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _holding(hand="left"):
    from sim.env import TransformHands
    key = "leftGrippedState" if hand == "left" else "rightGrippedState"
    return bool(TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)).get(key))


def _banner(n, title):
    print(f"\n{'=' * 70}\n  PHASE {n}: {title}\n{'=' * 70}")


def _confirm(auto, prompt="  [Enter] to proceed, 'n' to abort > "):
    if auto:
        return True
    try:
        return input(prompt).strip().lower() not in ("n", "no", "q", "quit", "abort")
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    ap = argparse.ArgumentParser(description="One-item live smoke of the 6.2 checkout chain.")
    ap.add_argument("--shelf", type=int, default=None,
                    help="checkpoint to drive to and grab from first; omit to use the held item")
    ap.add_argument("--grabnow", action="store_true",
                    help="grab from the CURRENT pose (you aligned the agent on the item yourself) - "
                         "no drive; takes precedence over --shelf")
    ap.add_argument("--hand", default="left", choices=("left", "right"),
                    help="which hand for the single-item chain (ignored under --two-hands)")
    ap.add_argument("--auto", action="store_true", help="don't pause between phases")
    ap.add_argument("--two-hands", action="store_true",
                    help="drive the FUSED both-hands chain: fill both hands, sweep-scan LEFT then RIGHT "
                         "(re-facing the pad between), then bag both - the primitive-level mirror of "
                         "store_map.checkout_held_items")
    ap.add_argument("--ocr-url", default=None, help="OCR service base URL (else $SARI_OCR_URL/default).")
    args = ap.parse_args()

    from vision.ocr_client import check_ocr_health
    try:
        check_ocr_health(args.ocr_url)
    except Exception as error:
        print(f"OCR preflight failed before simulator work: {error}")
        return 1

    from sim.env import RequestScreenshot, SetHandsActive, default_uri
    from manip.manipulation import set_hand_pose, extend_arm_until_grabbed
    from vision.perception import (scan_held_item, place_held_item,
                            center_to_screen, center_to_scanner, read_text_in_box)
    from nav.store_map import StoreMap, NavSession, go_to_counter, align_to_scanner
    from explore import step_agent

    try:
        RequestScreenshot(save_image=True)
    except Exception as e:
        print(f"Could not reach the sim ({type(e).__name__}: {e}). Is it in Play mode on :8080?")
        return 1

    sm = StoreMap()
    nav = NavSession(sm, uri=default_uri(),
                     stow_hands=False)          # carry-safe: never stow a held item
    SetHandsActive(True)
    results = {}

    def _grab_side(side):
        res = extend_arm_until_grabbed(hand=side)
        print(f"  {side} grab: gripped={res.get('gripped')} hovered={res.get('hovered')!r}"
              + (f" | {res['reason']}" if res.get("reason") else ""))
        return bool(res.get("gripped") and _holding(side))

    def _fill_both_hands():
        """Two-hand Phase 1: get an item into EACH hand. --shelf drives to the shelf once first; then
        each EMPTY hand is filled by a manual align+grab (you reposition the agent on each item, since
        two items rarely sit under one pose). Returns True once both hands hold, False on a failed grab
        or abort."""
        if args.shelf is not None:
            if args.shelf not in sm.by_id:
                print(f"  cp{args.shelf} is not in the graph - pick a shelf checkpoint."); return False
            set_hand_pose("rest", hand="left")
            nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
            if not nav.goto(args.shelf):
                print(f"  could not drive to cp{args.shelf} (path blocked)."); return False
        for side in ("left", "right"):
            if _holding(side):
                print(f"  {side} hand already holding - keeping it."); continue
            if not args.auto and not _confirm(False, f"  align the agent on an item for the {side} "
                                              f"hand, then [Enter] to grab ('n' aborts) > "):
                print("  aborted."); return False
            if not _grab_side(side):
                print(f"  {side} grab FAILED - re-align on the item (centred + in reach) and rerun.")
                return False
        return True

    try:
        # ---- Phase 1: get item(s) in hand -------------------------------------------------------
        if args.two_hands:
            _banner(1, "get an item in EACH hand (left, then right)")
            if not _fill_both_hands():
                return 1
        elif args.grabnow:
            _banner(1, "grab from the current pose (you aligned it)")
            res = extend_arm_until_grabbed(hand=args.hand)
            results["grab"] = res
            print(f"  grab: gripped={res.get('gripped')} hovered={res.get('hovered')!r}"
                  + (f" | {res['reason']}" if res.get("reason") else ""))
            if not (res.get("gripped") and _holding(args.hand)):
                print("  grab FAILED - re-align the agent on the item (get it centred + in reach) "
                      "and rerun.")
                return 1
        elif args.shelf is not None:
            _banner(1, f"grab an item at cp{args.shelf}")
            if args.shelf not in sm.by_id:
                print(f"  cp{args.shelf} is not in the graph - pick a shelf checkpoint."); return 1
            set_hand_pose("rest", hand=args.hand)
            nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
            if not nav.goto(args.shelf):
                print(f"  could not drive to cp{args.shelf} (path blocked)."); return 1
            res = extend_arm_until_grabbed(hand=args.hand)
            results["grab"] = res
            print(f"  grab: gripped={res.get('gripped')} hovered={res.get('hovered')!r}"
                  + (f" | {res['reason']}" if res.get("reason") else ""))
            if not (res.get("gripped") and _holding(args.hand)):
                print("  grab FAILED - reposition on the item and retry, or grab manually first.")
                return 1
        else:
            _banner(1, "use the item already in hand")
            if not _holding(args.hand):
                print(f"  the {args.hand} hand is empty - pass --shelf N to grab first, or grab "
                      f"manually before running."); return 1
            print("  holding confirmed.")

        # ---- Phase 2: drive to the checkout -----------------------------------------------------
        _banner(2, "go_to_counter")
        set_hand_pose("rest", hand=args.hand)
        res = go_to_counter(nav)
        results["counter"] = res
        print(f"  arrived={res.get('arrived')} cp={res.get('checkpoint')} "
              f"at ({res.get('x', 0):.2f},{res.get('z', 0):.2f})"
              + (f" - {res['reason']}" if res.get("reason") else ""))
        if not res.get("arrived"):
            print("  could not reach the counter - aborting."); return 1

        # ---- Phase 3: align to the scan pad -----------------------------------------------------
        _banner(3, "align_to_scanner")
        res = align_to_scanner(nav)
        results["align"] = res
        print(f"  aligned={res.get('aligned')} slant={res.get('slant')} "
              f"residual_yaw={res.get('residual_yaw'):+.1f} deg - {res.get('reason')}")
        if not res.get("aligned"):
            print("  NOT aligned - the scan will likely miss. Proceeding so you can see the sweep, "
                  "but this is the align-margin finding to report.")

        # ---- Phase 4: baseline receipt from the ALIGNED pose, then re-acquire the pad ------------
        # Read the screen ONLY now: at arrival it was too far/oblique to OCR; aligned we are close and
        # square. This yaws the body to the screen, so we center_to_scanner again (camera/yaw only, no
        # translation) to face the PAD before the sweep, preserving the slant align set.
        _banner(4, "read the POS-screen baseline receipt, then re-acquire the pad")
        sres = center_to_screen()
        baseline = read_text_in_box(sres.get("box")) if sres.get("box") else []
        results["baseline"] = baseline
        print(f"  screen detected={bool(sres.get('box'))}; baseline {len(baseline)} line(s):")
        for l in baseline:
            print(f"    | {l}")
        if not sres.get("box"):
            print("  (screen still not locked even when aligned - scan's measured delta will be "
                  "unreliable; watch the beep. This is an M5/aim finding to report.)")
        rescan = center_to_scanner()      # yaw back to the pad so the sweep goes the right way
        print(f"  re-acquired pad: {rescan.get('outcome')} (residual {rescan.get('residual_px')})")
        if rescan.get("outcome") == "not_detected":
            print("  pad not re-locked after looking at the screen - the sweep may aim wrong; "
                  "reporting and proceeding so you can see it.")
        if not _confirm(args.auto):
            print("  aborted before the sweep."); return 0

        if args.two_hands:
            # ---- Phase 5 (two-hand): sweep BOTH before bagging EITHER ---------------------------
            # The fused order (checkout_held_items): scan LEFT -> read screen -> scan RIGHT -> read
            # screen, threading the accumulating receipt, so a missed sweep is caught while both items
            # are still in hand - versus scan-drop-scan-drop. Re-face the pad between sweeps (each scan
            # ends by yawing to the SCREEN to read the receipt).
            _banner(5, "scan_held_item x2 (sweep LEFT, then RIGHT; items stay in hand)")
            base = baseline
            scans = {}
            for i, side in enumerate(("left", "right")):
                if not _holding(side):
                    print(f"  {side} hand empty - skipping its sweep."); continue
                r = scan_held_item(hand=side, baseline=base)
                scans[side] = r
                print(f"  [{side}] scanned_measured={r.get('scanned')} "
                      f"still_holding={r.get('still_holding')} screen_detected={r.get('screen_detected')}")
                for l in r.get("new_lines", []):
                    print(f"    + new receipt line: {l}")
                print(f"    -> {r.get('reason')}")
                base = r.get("receipt") or base                 # thread the accumulating receipt
                if i == 0:                                       # re-face the pad for the second sweep
                    rc = center_to_scanner()
                    print(f"  re-acquired pad for the next sweep: {rc.get('outcome')} "
                          f"(residual {rc.get('residual_px')})")
            results["scans"] = scans
            if not any(_holding(s) for s in ("left", "right")):
                print("  both grips opened during the sweeps (should not happen) - stopping."); return 1
            if not _confirm(args.auto, "  did BOTH scan by your eye/ear? [Enter] to bag both, 'n' aborts > "):
                print("  aborted before bagging (items still in hand)."); return 0

            # ---- Phase 6 (two-hand): bag BOTH --------------------------------------------------
            _banner(6, "place_held_item x2 (bag LEFT, then RIGHT)")
            places = {}
            for side in ("left", "right"):
                if not _holding(side):
                    continue
                r = place_held_item(hand=side)
                places[side] = r
                print(f"  [{side}] placed={r.get('placed')} released={r.get('released')} "
                      f"verdict={r.get('verdict')} slant={r.get('distance')} - {r.get('reason')}")
            results["places"] = places

            # ---- Summary (two-hand) ------------------------------------------------------------
            _banner("SUMMARY", "MEASURED two-hand chain (your eyes own beep + in-tray, PER item)")
            a = results.get("align", {})
            print(f"  aligned         : {a.get('aligned')}  (slant {a.get('slant')}, "
                  f"residual_yaw {a.get('residual_yaw')})")
            chain_ok = bool(a.get("aligned"))
            for side in ("left", "right"):
                sc = results.get("scans", {}).get(side, {})
                pl = results.get("places", {}).get(side, {})
                print(f"  {side:5}: scanned={sc.get('scanned')} still_holding={sc.get('still_holding')} "
                      f"placed={pl.get('placed')} (verdict {pl.get('verdict')}, slant {pl.get('distance')})")
                chain_ok = chain_ok and bool(sc.get("scanned") and pl.get("placed"))
            print(f"\n  MEASURED chain {'PASS' if chain_ok else 'INCOMPLETE'} - now confirm by eye, PER "
                  f"item: did EACH beep, and is EACH in the tray? A measured PASS with an eye FAIL is the "
                  f"discrepancy to chase (e.g. the aim confound).")
            return 0

        # ---- Phase 5: scan the held item --------------------------------------------------------
        _banner(5, "scan_held_item (sweep; item stays in hand)")
        res = scan_held_item(hand=args.hand, baseline=baseline)
        results["scan"] = res
        print(f"  scanned_measured={res.get('scanned')} still_holding={res.get('still_holding')} "
              f"screen_detected={res.get('screen_detected')}")
        for l in res.get("new_lines", []):
            print(f"    + new receipt line: {l}")
        print(f"  -> {res.get('reason')}")
        if not res.get("still_holding"):
            print("  the grip OPENED during the sweep (should not happen) - stopping before place.")
            return 1
        if not _confirm(args.auto, "  did it scan by your eye/ear too? [Enter] to bag, 'n' to abort > "):
            print("  aborted before the bag step (item still in hand)."); return 0

        # ---- Phase 6: bag it --------------------------------------------------------------------
        _banner(6, "place_held_item (the one deliberate release, into the tray)")
        res = place_held_item(hand=args.hand)
        results["place"] = res
        print(f"  placed={res.get('placed')} released={res.get('released')} "
              f"verdict={res.get('verdict')} slant={res.get('distance')} - {res.get('reason')}")

        # ---- Summary ----------------------------------------------------------------------------
        _banner("SUMMARY", "MEASURED chain result (your eyes own beep + in-tray)")
        a, sc, pl = results.get("align", {}), results.get("scan", {}), results.get("place", {})
        print(f"  aligned         : {a.get('aligned')}  (slant {a.get('slant')}, "
              f"residual_yaw {a.get('residual_yaw')})")
        print(f"  scanned_measured: {sc.get('scanned')}  (still_holding {sc.get('still_holding')})")
        print(f"  placed_measured : {pl.get('placed')}  (released {pl.get('released')}, "
              f"verdict {pl.get('verdict')})")
        chain_ok = bool(a.get("aligned") and sc.get("scanned") and sc.get("still_holding")
                        and pl.get("placed"))
        print(f"\n  MEASURED chain {'PASS' if chain_ok else 'INCOMPLETE'} - "
              f"now confirm by eye: did it BEEP, and is the item IN the tray? Report both; a measured "
              f"PASS with an eye FAIL is the discrepancy to chase (e.g. the aim confound).")
        return 0
    finally:
        nav.close()


if __name__ == "__main__":
    raise SystemExit(main())
