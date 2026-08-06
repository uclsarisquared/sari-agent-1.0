"""Phase 4 step 5 - the LIVE stratified locate trials. Needs the sim in Play mode.

Runs locate tasks END TO END (resolve -> drive -> verify -> verdict) through ONE NavSession -
the agent drives on from wherever the previous task left it, which is both faster and the
realistic condition - and scores each outcome against the best ground truth available:

  * FRIDGE-REGION products (shelves 6/7/8): `Store 2 v2.json` placements are slot-level truth,
    so found/not-found is scored hard.
  * EVERYWHERE ELSE: no placement truth exists (shelfItems covers the fridges only), so a
    "found" is scored as END-TO-END CONSISTENCY - resolver, driver and verifier agreeing at an
    index-expected checkpoint. That is a weaker claim and the report says so; do not quote the
    non-fridge rows as placement-verified.
  * CONTROLS: known-absent products must end not_found; no-shelf categories must end
    unresolvable or not_found (both are honest negatives - qwen tends to refuse at resolution,
    claude tends to route-then-fail-verification; either endpoint is correct behaviour, and
    which one fires is recorded).

Success criterion (per the phase doc's Evaluation contract): the run ends with the target in
view from a checkpoint + an honest verdict. No manipulation, no centring, no delivery.

    python mapping/scoring/eval_locate.py                     # zoom ladder ON (default)
    python mapping/scoring/eval_locate.py --no-zoom           # ablation arm
    python mapping/scoring/eval_locate.py --only slang --backend claude-cli
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/scoring
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from nav.store_map import StoreMap, NavSession  # noqa: E402
from nav import locate_task  # noqa: E402

# Fridge truth: which checkpoints can a placed SKU be verified from (shelf -> facing cps,
# measured in score_index.py). Used for the hard-scored rows.
FRIDGE_FACING = {6: {15, 16, 19}, 7: {19, 20, 44}, 8: {20, 31, 43, 44}}


def build_trials(sm):
    """(task, stratum, expectation) triples. Expectation kinds:
    ('found_at', {cps})   - must end found, at one of these checkpoints [hard truth or index]
    ('not_found', None)   - stocked-absent control: must end not_found
    ('negative', None)    - no-shelf control: unresolvable OR not_found both pass
    """
    cat = lambda c: set(sm.category_checkpoints(c))
    T = [
        # -- precise_name --
        ("Find the Century Tuna.",              "precise_name", ("found_at", {50, 51})),
        ("Find the Coca-Cola Light 500ml.",     "precise_name", ("found_at", FRIDGE_FACING[7])),   # shelf7 row1 slot3, PLACEMENT TRUTH
        ("Find the Lucky Me! Pancit Canton.",   "precise_name", ("found_at", {18, 32, 45, 52})),
        ("Find the Leslie's Clover Chips.",     "precise_name", ("found_at", {18, 21, 26, 27, 42, 45, 52})),
        ("Find the Pik-Nik.",                   "precise_name", ("found_at", {18, 26})),           # index 'Nik-Nik'; reconciler-dependent
        # -- general_name --
        ("Find corned beef.",                   "general_name", ("found_at", {50, 51})),
        ("Find sardines.",                      "general_name", ("found_at", {50, 51})),
        ("Find bottled water.",                 "general_name", ("found_at", FRIDGE_FACING[6] | {20})),  # shelf6 waters, PLACEMENT TRUTH
        ("Find instant noodles.",               "general_name", ("found_at", {18, 32, 45, 52})),
        ("Find butter cookies.",                "general_name", ("found_at", {33, 34, 47, 48})),
        # -- shelf_with --
        ("Find a shelf with chips.",            "shelf_with",   ("found_at", cat("Chips"))),
        ("Find a shelf with canned goods.",     "shelf_with",   ("found_at", cat("Can"))),
        ("Find a shelf with soda.",             "shelf_with",   ("found_at", cat("Soda"))),        # shelf7/8 sodas, PLACEMENT TRUTH
        ("Find a shelf with biscuits.",         "shelf_with",   ("found_at", cat("Biscuit"))),
        ("Find a shelf with instant noodles.",  "shelf_with",   ("found_at", cat("Noodles"))),
        # -- slang --
        ("Find the coke.",                      "slang",        ("found_at", FRIDGE_FACING[7])),   # colas live in shelf7, PLACEMENT TRUTH
        ("Find some softdrinks.",               "slang",        ("found_at", cat("Soda"))),
        ("Find the de lata.",                   "slang",        ("found_at", cat("Can"))),
        ("Where are the chichirya?",            "slang",        ("found_at", cat("Chips"))),
        ("Find the mineral water.",             "slang",        ("found_at", FRIDGE_FACING[6] | {20})),
        # -- controls --
        ("Locate the Coke Zero in the store.",  "control",      ("not_found", None)),              # catalog product, NOT stocked
        ("Locate the orange juice.",            "control",      ("negative", None)),               # Juice: no shelf node
        ("Find the soju.",                      "control",      ("negative", None)),               # Liquor: no shelf node
    ]
    return T


def score(expect, report):
    kind, arg = expect
    outcome = report.get("outcome")
    if kind == "found_at":
        if outcome != "found":
            return 0.0, f"expected found, got {outcome}"
        cp = report["found"]["checkpoint"]
        return (1.0, f"found at cp{cp} (in expected set)") if cp in arg else \
               (0.0, f"found at cp{cp}, NOT in expected {sorted(arg)}")
    if kind == "not_found":
        return (1.0, "honest not_found") if outcome == "not_found" else \
               (0.0, f"expected not_found, got {outcome}")
    if kind == "negative":
        return (1.0, f"honest negative ({outcome})") if outcome in ("not_found", "unresolvable") \
            else (0.0, f"expected honest negative, got {outcome}")
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["claude-cli", "qwen"], default="claude-cli")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--no-zoom", action="store_true", help="ablation: disable the zoom ladder")
    ap.add_argument("--only", default=None, help="run one stratum only")
    ap.add_argument("--max-visits", type=int, default=4)
    ap.add_argument("--uri", default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir or os.path.join(
        _MAPPING_DIR, "output", "locate_trials",
        f"{datetime.now():%m%d_%H%M%S}_{args.backend}{'_nozoom' if args.no_zoom else ''}")
    os.makedirs(run_dir, exist_ok=True)
    call = locate_task.make_backend(args)
    sm = StoreMap()
    trials = [t for t in build_trials(sm) if not args.only or t[1] == args.only]
    print(f"== live locate trials: {len(trials)} tasks, backend={args.backend}, "
          f"zoom={'OFF' if args.no_zoom else 'on'}, reconciled_index={sm.reconciled} ==")

    nav = NavSession(sm, uri=args.uri)
    rows, by_stratum = [], {}
    t0 = time.time()
    try:
        for i, (task, stratum, expect) in enumerate(trials):
            print(f"\n### [{i+1}/{len(trials)}] {stratum}: {task}")
            t1 = time.time()
            report = {"task": task, "visits": [], "outcome": None}
            try:
                resolution, _ = locate_task.resolve(call, sm, task)
                report["resolution"] = resolution
                if resolution["tier"] == "unresolvable" or not resolution["candidates"]:
                    report["outcome"] = "unresolvable"
                    print("[locate] UNRESOLVABLE (resolver refused)")
                else:
                    locate_task.drive_and_verify(
                        call, sm, nav, task, resolution, report, run_dir,
                        max_visits=args.max_visits, zoom=not args.no_zoom,
                        tag=f"t{i:02d}_")
            except Exception as e:
                report["outcome"] = f"error"
                report["error"] = f"{type(e).__name__}: {e}"
                print(f"[locate] ERROR {report['error'][:140]}")
            dt = time.time() - t1
            s, detail = score(expect, report)
            rows.append({"task": task, "stratum": stratum, "score": s, "detail": detail,
                         "outcome": report.get("outcome"), "seconds": round(dt, 1),
                         "visits": len(report.get("visits", [])), "report": report})
            by_stratum.setdefault(stratum, []).append(s)
            print(f"### score {s:.0f} - {detail}  ({dt:.0f}s)")
    finally:
        nav.close()

    print(f"\n== TRIAL SUMMARY ({(time.time()-t0)/60:.1f} min) ==")
    for st, sc in by_stratum.items():
        print(f"   {st:<13} {sum(sc)}/{len(sc)}")
    total = sum(r["score"] for r in rows)
    print(f"   {'OVERALL':<13} {total:.0f}/{len(rows)}  ({100*total/len(rows):.0f}%)")

    out = os.path.join(run_dir, "trials_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "zoom": not args.no_zoom, "rows": rows,
                   "by_stratum": {s: sum(v) / len(v) for s, v in by_stratum.items()},
                   "overall": total / len(rows)}, f, indent=2, ensure_ascii=False)
    print(f"   -> {out}")


if __name__ == "__main__":
    main()
