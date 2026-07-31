#!/usr/bin/env python3
"""Audit the early-stop path: which attempts were destroyed, and on what evidence.

When the auto predicate in subtask_completion.py grants success, the operator can
stop the prompt's remaining tries as "already_successful". If that grant is later
overturned on review, those tries are gone: the prompt loses its remaining
chances to be solved, and the arm carries a prompt it may well have solved.

This walks each battery's `human_verified_winners` against the verdict actually
recorded for the winning try.

Usage:  python3 analysis/context-ablation/censoring.py [--bench-runs DIR]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from collect import BATTERIES

ARMS = ["baseline", "a1", "a2c", "a3", "a4", "a5", "a6-2", "a6-4"]


def verdict_of(base: pathlib.Path, key: str) -> tuple[str, bool]:
    """(human verdict, auto success) for an attempt key like 'hard_03/try01'."""
    path = base / key / "attempt.json"
    if not path.exists():
        return ("", False)
    manifest = json.loads(path.read_text())
    return (manifest.get("verified_verdict") or "", bool(manifest.get("success")))


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-runs", type=pathlib.Path,
                        default=here.parent.parent / "bench_runs")
    args = parser.parse_args()

    rows = []
    detail = []
    for arm in ARMS:
        base = args.bench_runs / BATTERIES[arm]
        battery = json.loads((base / "battery.json").read_text())
        winners = battery.get("human_verified_winners") or {}
        killed = sum(1 for p in base.glob("*/try*/attempt.json")
                     if json.loads(p.read_text()).get("stop_reason") == "already_successful")
        wasted = 0
        for prompt_id, record in winners.items():
            key = record.get("winning_attempt_key") or ""
            if not key:
                continue
            verdict, auto = verdict_of(base, key)
            siblings = sum(1 for p in (base / prompt_id).glob("try*/attempt.json")
                           if json.loads(p.read_text()).get("stop_reason") == "already_successful")
            if verdict == "fail" and siblings:
                wasted += siblings
                detail.append([arm, prompt_id, key, f"auto={auto}", f"review={verdict}",
                               f"{siblings} sibling(s) destroyed"])
        rows.append([arm, len(winners), killed, wasted])

    headers = ["arm", "prompts stopped early", "attempts destroyed",
               "destroyed on a grant review overturned"]
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    print("\nCases where a prompt's remaining tries were destroyed for a grant that did")
    print("not survive review:")
    if not detail:
        print("  (none)")
    for row in detail:
        print("  " + "  ".join(str(c) for c in row))
    total = sum(r[3] for r in rows)
    print(f"\n{total} attempts across the battery were spent this way. Each one is a")
    print("chance to solve that prompt that the harness threw away on a false positive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
