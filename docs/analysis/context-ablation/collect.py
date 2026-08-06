#!/usr/bin/env python3
"""Build a per-attempt table for the context-window ablation batteries.

Reads bench_runs/ directly; no dependencies outside the standard library.
Writes attempts.csv (one row per attempt) next to this script so every number in
the report can be re-derived and diffed.

Usage:  python3 analysis/context-ablation/collect.py [--bench-runs DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

# Batteries that make up the ablation. The baseline battery predates the naming
# convention, so it is listed by id rather than discovered by glob.
BATTERIES = {
    "baseline": "20260730_040404",
    "a1": "20260730_143507_context-ablation-a1",
    "a2c": "20260730_123306_context-ablation-a2c",
    "a3": "20260730_172208_context-ablation-a3",
    "a4": "20260730_192703_context-ablation-a4",
    "a5": "20260730_051207_context-ablation-a5",
    "a6-2": "20260730_080607_context-ablation-a6-2",
    "a6-4": "20260730_102443_context-ablation-a6-4",
    # Different prompt set (hard_prompts.json, 18 prompts) and never finished:
    # kept in the table, excluded from every arm-to-arm comparison.
    "hard-baseline": "20260730_205702_context-ablation-hard-baseline",
}

ROLES = ("actor", "semantic", "episodic", "perception", "guard",
         "findings", "resolver", "decomposer", "responder")

# An attempt only exercises the policy if the agent actually ran to its own
# stopping point. Operator kills and harness timeouts stop the clock from
# outside, so their token and wall figures describe the harness, not the arm.
CLEAN_OUTCOMES = {"completed"}


def leg_files(run_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(run_dir.glob("leg*.jsonl"))


def count_steps(run_dir: pathlib.Path) -> tuple[int, int, int]:
    """Return (steps, legs_run, halt_requests) read from the leg event logs."""
    steps = halts = 0
    legs = leg_files(run_dir)
    for path in legs:
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "step":
                steps += 1
            elif row.get("event") == "halt_request":
                halts += 1
    return steps, len(legs), halts


def semantic_chars(run_dir: pathlib.Path) -> int | None:
    path = run_dir / "semantic_memory.txt"
    return len(path.read_bytes()) if path.exists() else None


def manifest(run_dir: pathlib.Path) -> dict:
    """The attempt manifest carries the human review; attempts.jsonl does not."""
    path = run_dir / "attempt.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def collect(bench_runs: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for arm, battery in BATTERIES.items():
        base = bench_runs / battery
        attempts_path = base / "attempts.jsonl"
        if not attempts_path.exists():
            print(f"warning: {battery} has no attempts.jsonl; skipped", file=sys.stderr)
            continue
        for line in attempts_path.read_text().splitlines():
            if not line.strip():
                continue
            att = json.loads(line)
            run_dir = pathlib.Path(att.get("run_dir") or "")
            if not run_dir.is_absolute() or not run_dir.exists():
                run_dir = base / att["prompt_id"] / f"try{att['attempt']:02d}"
            steps, legs_run, halts = count_steps(run_dir) if run_dir.exists() else (0, 0, 0)
            by_role = att.get("tokens_by_role") or {}
            man = manifest(run_dir)
            verdict = man.get("verified_verdict") or ""
            # Scoring is deliberately flat: every prompt gets a denominator of 3,
            # and an attempt counts only if a human passed it. Attempts halted
            # because a sibling had already won score as non-successes like any
            # other, so no arm's denominator shrinks. `excluded` is still recorded
            # so the effect of that choice stays visible, but nothing scores on it.
            excluded = verdict in ("already_successful", "invalid") or \
                att.get("end_reason") == "already_successful"
            if verdict in ("pass", "fail"):
                final = int(verdict == "pass")
            else:
                final = int(bool(att.get("success")))
            row = {
                "arm": arm,
                "battery": battery,
                "prompt_id": att["prompt_id"],
                "family": att.get("family", ""),
                "attempt": att["attempt"],
                "success": int(bool(att.get("success"))),
                "verified_verdict": verdict,
                "verified": int(bool(verdict)),
                "success_final": final,
                "excluded": int(excluded),
                "false_pass": int(bool(att.get("success")) and verdict == "fail"),
                "false_fail": int(not att.get("success") and verdict == "pass"),
                "outcome": att.get("outcome", ""),
                "end_reason": att.get("end_reason", ""),
                "clean": int(att.get("outcome") in CLEAN_OUTCOMES),
                "wall_seconds": att.get("wall_seconds", 0.0),
                "tokens_in": att.get("tokens_in", 0),
                "tokens_out": att.get("tokens_out", 0),
                "llm_calls": att.get("llm_calls", 0),
                "legs_planned": (att.get("legs") or {}).get("planned", 0),
                "legs_completed": (att.get("legs") or {}).get("completed", 0),
                "leg_files": legs_run,
                "steps": steps,
                "halt_requests": halts,
                "semantic_chars": semantic_chars(run_dir) if run_dir.exists() else None,
                "requeues": att.get("requeues", 0),
            }
            # leg_files counts retry legs separately, so it exceeds legs_planned
            # exactly when a leg was retried.
            row["leg_retries"] = max(0, legs_run - row["legs_planned"])
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
    parser.add_argument("--out", type=pathlib.Path, default=here / "attempts.csv")
    args = parser.parse_args()

    rows = collect(args.bench_runs)
    if not rows:
        print("no attempts found", file=sys.stderr)
        return 1
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} attempts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
