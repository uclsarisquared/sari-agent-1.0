#!/usr/bin/env python3
"""Count runtime-log symptoms per arm for the context-window ablation.

Two symptoms motivated arms in this batch and are only visible in runtime.log:
  * "[learner] unparseable reply" -- the ROUTE-TRACING crash A2c was meant to
    remove, where the learner emits a reply ast.literal_eval cannot read.
  * "[graph-nav] all candidates visited" -- the navigator exhausting its
    candidate list and restarting, the signature of a lost agent.
Neither reaches attempts.jsonl, so they are counted here and normalised by the
number of steps each arm actually ran.

Usage:  python3 analysis/context-ablation/logscan.py [--bench-runs DIR]
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import re

from collect import BATTERIES

ARMS = ["baseline", "a1", "a2c", "a3", "a4", "a5", "a6-2", "a6-4"]
PATTERNS = {
    "learner_unparseable": re.compile(r"\[learner\] unparseable reply"),
    "episodic_unparseable": re.compile(r"\[episodic\] unparseable reply"),
    "nav_candidates_exhausted": re.compile(r"\[graph-nav\] all candidates visited"),
    "api_retry": re.compile(r"\[api-retry\]"),
}


def steps_by_arm(csv_path: pathlib.Path) -> dict[str, float]:
    totals: dict[str, float] = collections.defaultdict(float)
    if not csv_path.exists():
        return totals
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            totals[row["arm"]] += float(row["steps"] or 0)
    return totals


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-runs", type=pathlib.Path,
                        default=here.parent.parent / "bench_runs")
    parser.add_argument("--attempts", type=pathlib.Path, default=here / "attempts.csv")
    args = parser.parse_args()

    steps = steps_by_arm(args.attempts)
    counts = {arm: collections.Counter() for arm in ARMS}
    for arm in ARMS:
        base = args.bench_runs / BATTERIES[arm]
        for log in base.glob("*/try*/runtime.log"):
            text = log.read_text(errors="replace")
            for name, pattern in PATTERNS.items():
                counts[arm][name] += len(pattern.findall(text))

    headers = ["arm", "steps"] + list(PATTERNS) + ["unparseable /100 steps"]
    rows = []
    for arm in ARMS:
        bad = counts[arm]["learner_unparseable"] + counts[arm]["episodic_unparseable"]
        n = steps.get(arm, 0)
        rows.append([arm, int(n)] + [counts[arm][name] for name in PATTERNS] +
                    [f"{100 * bad / n:.2f}" if n else "-"])
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print("\nsteps is every step the arm ran, including attempts excluded from the")
    print("success tables, because a crash counts wherever it happened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
