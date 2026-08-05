#!/usr/bin/env python3
"""Build a per-leg table for the context-window ablation.

The leg is the right unit for any question about context growth: the actor's
message history is rebuilt at each leg boundary, so an attempt-level fit that
pools legs measures a mixture of leg lengths rather than growth within one.
Per-leg role tokens live in each attempt's summary.json.

Usage:  python3 analysis/context-ablation/collect_legs.py [--bench-runs DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

from collect import BATTERIES, ROLES


def collect(bench_runs: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for arm, battery in BATTERIES.items():
        base = bench_runs / battery
        if not base.exists():
            continue
        for summary_path in sorted(base.glob("*/try*/summary.json")):
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                print(f"warning: unreadable {summary_path}", file=sys.stderr)
                continue
            run_dir = summary_path.parent
            for index, leg in enumerate(summary.get("legs") or []):
                by_role = leg.get("tokens_by_role") or {}
                row = {
                    "arm": arm,
                    "battery": battery,
                    "prompt_id": run_dir.parent.name,
                    "try": run_dir.name,
                    "leg_index": index,
                    "leg_type": leg.get("type", ""),
                    "leg_attempt": leg.get("attempt", 1),
                    "success": int(bool(leg.get("success"))),
                    "end_reason": leg.get("end_reason", ""),
                    "timesteps": leg.get("timesteps", 0),
                    "llm_calls": leg.get("llm_calls", 0),
                    "tokens_in": leg.get("tokens_in", 0),
                    "tokens_out": leg.get("tokens_out", 0),
                    "wall_s": leg.get("wall_s", 0.0),
                    "halts_refused": leg.get("halts_refused", 0),
                    "halt_forced": int(bool(leg.get("halt_forced"))),
                    "errors": leg.get("errors", 0),
                }
                for role in ROLES:
                    stats = by_role.get(role) or {}
                    row[f"{role}_in"] = stats.get("tokens_in", 0)
                    row[f"{role}_out"] = stats.get("tokens_out", 0)
                    row[f"{role}_calls"] = stats.get("calls", 0)
                rows.append(row)
    return rows


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-runs", type=pathlib.Path,
                        default=here.parent.parent / "bench_runs")
    parser.add_argument("--out", type=pathlib.Path, default=here / "legs.csv")
    args = parser.parse_args()
    rows = collect(args.bench_runs)
    if not rows:
        print("no legs found", file=sys.stderr)
        return 1
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} legs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
