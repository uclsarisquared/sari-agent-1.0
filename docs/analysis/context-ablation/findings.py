#!/usr/bin/env python3
"""Measure the findings summaries the A3/A4 arms target.

Findings summaries are only echoed into agent.log, under a "[FINDINGS SUMMARY]"
header, so their sizes are not in any structured artefact. This extracts them and
checks two things A4's design raises:

  * how long they actually are, per arm;
  * how often A4's 600-char cap fires at all, given that A4 also swaps in a much
    shorter system prompt (subtask_agents.py:239-245) -- if the shorter prompt
    already lands under the cap, A4 is measuring a prompt rewrite, not a cap.

Usage:  python3 analysis/context-ablation/findings.py [--bench-runs DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics

from collect import BATTERIES

ARMS = ["baseline", "a1", "a2c", "a3", "a4", "a5", "a6-2", "a6-4"]
HEADER = re.compile(r"^\[FINDINGS SUMMARY\]\s*$", re.M)
# The block runs to the next bracketed log header or a leg banner.
STOP = re.compile(r"^(\[[A-Z][A-Z ]+\]|--- LEG )", re.M)
A4_CAP = 600


def blocks(text: str) -> list[str]:
    out = []
    for match in HEADER.finditer(text):
        rest = text[match.end():].lstrip("\n")
        stop = STOP.search(rest)
        out.append(rest[:stop.start()].strip() if stop else rest.strip())
    return out


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-runs", type=pathlib.Path,
                        default=here.parent.parent / "bench_runs")
    args = parser.parse_args()

    rows = []
    for arm in ARMS:
        lengths = []
        for log in (args.bench_runs / BATTERIES[arm]).glob("*/try*/agent.log"):
            lengths.extend(len(b) for b in blocks(log.read_text(errors="replace")) if b)
        if not lengths:
            rows.append([arm, 0, "-", "-", "-", "-", "-"])
            continue
        # A hard slice at the cap leaves the text exactly cap chars long.
        at_cap = sum(1 for n in lengths if n == A4_CAP)
        rows.append([arm, len(lengths), f"{min(lengths):,}",
                     f"{statistics.median(lengths):,.0f}",
                     f"{statistics.fmean(lengths):,.0f}", f"{max(lengths):,}",
                     f"{at_cap}/{len(lengths)}"])
    headers = ["arm", "summaries", "min", "median", "mean", "max", "exactly 600 chars"]
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print("\nA3 should show zero summaries: findings generation is skipped entirely.")
    print("For A4, 'exactly 600 chars' counts summaries the cap actually truncated;")
    print("everything below that length was produced short by the rewritten prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
