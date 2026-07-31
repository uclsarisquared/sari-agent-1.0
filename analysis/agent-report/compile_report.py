#!/usr/bin/env python3
"""Compile a cross-difficulty performance report for three Sari Agent batteries.

Every number in the report comes out of this script: it reads each attempt's on-disk manifest
(`attempt.json`), its token meter (`tokens.json`) and the battery spine (`attempts.jsonl`), and
reuses `sari_bench.watch.scan` for the verdict semantics so "human grading" here means exactly what
the dashboard and `sari_bench report` mean by it.

Usage:  python3 analysis/agent-report/compile_report.py [--csv OUT.csv] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sari_bench.watch import scan  # noqa: E402

BENCH_ROOT = REPO / "bench_runs"
BATTERIES = [
    ("easy", "20260727_020820-easy"),
    ("medium", "20260728_143857-medium"),
    ("hard", "20260730_205702-hard"),
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_attempts(difficulty: str, battery_dir: Path) -> list[dict[str, Any]]:
    """One row per attempt directory, manifest-first with the spine row as fallback."""
    spine: dict[str, dict[str, Any]] = {}
    for line in (battery_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            spine[f"{row['prompt_id']}/try{int(row['attempt']):02d}"] = row

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(battery_dir.glob("*/try*")):
        if not run_dir.is_dir():
            continue
        key = f"{run_dir.parent.name}/{run_dir.name}"
        manifest = read_json(run_dir / "attempt.json")
        recorded = spine.get(key, {})
        summary = read_json(run_dir / "summary.json")
        meter = read_json(run_dir / "tokens.json")

        outcome = recorded.get("outcome") or manifest.get("outcome") or manifest.get("state") or "unfinished"
        end_reason = recorded.get("end_reason") or manifest.get("end_reason", "")
        # The harness predicate: what the agent itself claimed. Kept beside the human verdict so the
        # two can be compared rather than conflated.
        auto_success = bool(
            recorded.get("success") or summary.get("success") or manifest.get("success")
        ) if outcome == "completed" else False

        verdict = scan.effective_verdict(manifest)
        graded = bool(verdict) and verdict not in scan.EXCLUDED_VERDICTS

        # tokens.json is the meter the agent writes as it runs and is the most complete source;
        # the spine row and battery summary are the fallbacks for attempts that never wrote one.
        tokens_in = int(meter.get("tokens_in") or recorded.get("tokens_in")
                        or manifest.get("tokens_in") or summary.get("tokens_in") or 0)
        tokens_out = int(meter.get("tokens_out") or recorded.get("tokens_out")
                         or manifest.get("tokens_out") or summary.get("tokens_out") or 0)
        legs = recorded.get("legs") or {}
        # `summary.wall_seconds` only covers the battery's last session, so a resumed battery
        # under-reports its own span. Attempt timestamps are the honest clock.
        started_at = str(manifest.get("started_at") or "")
        ended_at = str(manifest.get("ended_at") or manifest.get("finalized_at") or "")

        rows.append({
            "difficulty": difficulty,
            "battery": battery_dir.name,
            "key": key,
            "prompt_id": run_dir.parent.name,
            "attempt": int(recorded.get("attempt") or manifest.get("attempt") or 0),
            "family": recorded.get("family") or manifest.get("family", ""),
            "prompt": recorded.get("prompt") or manifest.get("prompt", ""),
            "state": manifest.get("state", ""),
            "outcome": outcome,
            "end_reason": end_reason,
            "auto_success": auto_success,
            "verdict": verdict,
            "graded": graded,
            "human_pass": verdict == "pass",
            "wall_seconds": float(recorded.get("wall_seconds") or manifest.get("wall_seconds") or 0.0),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "llm_calls": int(meter.get("calls") or recorded.get("llm_calls") or summary.get("llm_calls") or 0),
            "started_at": started_at,
            "ended_at": ended_at,
            "legs_planned": legs.get("planned") or summary.get("legs_planned") or 0,
            "legs_completed": legs.get("completed") or summary.get("legs_completed") or 0,
        })
    return rows


def battery_span(rows: list[dict[str, Any]]) -> float:
    """Earliest attempt start to latest attempt end, in seconds.

    Batteries can be paused and resumed, so this measures elapsed calendar time over the whole
    battery - idle gaps included - and is deliberately not the same thing as summed agent runtime.
    """
    starts = [datetime.fromisoformat(r["started_at"]) for r in rows if r["started_at"]]
    ends = [datetime.fromisoformat(r["ended_at"]) for r in rows if r["ended_at"]]
    if not starts or not ends:
        return 0.0
    return (max(ends) - min(starts)).total_seconds()


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(difficulty: str, rows: list[dict[str, Any]], battery_summary: dict[str, Any]) -> dict[str, Any]:
    graded = [r for r in rows if r["graded"]]
    passes = [r for r in graded if r["human_pass"]]
    fails = [r for r in graded if not r["human_pass"]]
    excluded = [r for r in rows if r["verdict"] in scan.EXCLUDED_VERDICTS]
    ungraded = [r for r in rows if not r["verdict"]]
    timed = [r for r in rows if r["wall_seconds"] > 0]

    # Prompt-level: a prompt counts as solved when any attempt of it was graded a human pass.
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    solved_prompts = sorted(pid for pid, rs in by_prompt.items() if any(r["human_pass"] for r in rs))
    # A prompt is "attempted under grading" once at least one of its tries carries a real verdict.
    reviewed_prompts = sorted(pid for pid, rs in by_prompt.items() if any(r["graded"] for r in rs))

    # Agreement between the harness's own success predicate and the human verdict, over graded rows.
    agree = sum(1 for r in graded if r["auto_success"] == r["human_pass"])
    false_pass = sum(1 for r in graded if r["auto_success"] and not r["human_pass"])
    false_fail = sum(1 for r in graded if not r["auto_success"] and r["human_pass"])

    return {
        "difficulty": difficulty,
        "battery": rows[0]["battery"],
        "prompts": len(by_prompt),
        "attempts": len(rows),
        # --- human grading ---
        "graded_attempts": len(graded),
        "graded_pass": len(passes),
        "graded_fail": len(fails),
        "excluded_attempts": len(excluded),
        "ungraded_attempts": len(ungraded),
        "grading_coverage": len(graded) / len(rows) if rows else None,
        "human_success_rate": len(passes) / len(graded) if graded else None,
        "prompts_solved": len(solved_prompts),
        "prompts_reviewed": len(reviewed_prompts),
        "prompt_solve_rate_reviewed": (
            len(solved_prompts) / len(reviewed_prompts) if reviewed_prompts else None),
        "prompt_solve_rate_all": len(solved_prompts) / len(by_prompt) if by_prompt else None,
        "solved_prompt_ids": solved_prompts,
        # --- time ---
        "avg_success_seconds": mean([r["wall_seconds"] for r in passes if r["wall_seconds"] > 0]),
        "median_success_seconds": median([r["wall_seconds"] for r in passes if r["wall_seconds"] > 0]),
        "avg_graded_fail_seconds": mean([r["wall_seconds"] for r in fails if r["wall_seconds"] > 0]),
        "avg_attempt_seconds_all": mean([r["wall_seconds"] for r in timed]),
        "total_attempt_seconds": sum(r["wall_seconds"] for r in rows),
        "battery_span_seconds": battery_span(rows),
        "battery_last_session_seconds": float(battery_summary.get("wall_seconds") or 0.0),
        "parallelism_factor": (
            sum(r["wall_seconds"] for r in rows) / battery_span(rows)
            if battery_span(rows) else None),
        # --- tokens ---
        "tokens_total": sum(r["tokens_total"] for r in rows),
        "tokens_in_total": sum(r["tokens_in"] for r in rows),
        "tokens_out_total": sum(r["tokens_out"] for r in rows),
        "avg_tokens_per_attempt": mean([float(r["tokens_total"]) for r in rows]),
        "avg_tokens_per_pass": mean([float(r["tokens_total"]) for r in passes]),
        "avg_tokens_per_graded_fail": mean([float(r["tokens_total"]) for r in fails]),
        # Cost of one solved prompt: every token the battery burned, over prompts actually solved.
        "tokens_per_solved_prompt": (
            sum(r["tokens_total"] for r in rows) / len(solved_prompts) if solved_prompts else None),
        "llm_calls_total": sum(r["llm_calls"] for r in rows),
        "avg_llm_calls_per_pass": mean([float(r["llm_calls"]) for r in passes]),
        # --- harness vs human ---
        "auto_vs_human_agreement": agree / len(graded) if graded else None,
        "auto_false_pass": false_pass,
        "auto_false_fail": false_fail,
        "outcomes": dict(Counter(r["outcome"] for r in rows).most_common()),
        "end_reasons": dict(Counter(r["end_reason"] for r in rows if r["end_reason"]).most_common()),
    }


def fmt_secs(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 60:.1f} min"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def fmt_tok(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.0f}k"


def render(stats: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    out: list[str] = []
    w = out.append
    w("=" * 78)
    w("SARI AGENT PERFORMANCE REPORT")
    w("=" * 78)
    for s in stats:
        w(f"  {s['difficulty']:<7} {s['battery']}  "
          f"{s['prompts']} prompts x {s['tries']} tries = {s['attempts']} attempts  "
          f"(arm={s['arm']}, completion_guard={s['completion_guard']})")
    w("")

    w("PER-DIFFICULTY HEADLINE")
    w("-" * 78)
    header = f"{'metric':<34}" + "".join(f"{s['difficulty']:>14}" for s in stats)
    w(header)
    def line(label: str, fn) -> None:
        w(f"{label:<34}" + "".join(f"{fn(s):>14}" for s in stats))

    line("avg time to a graded success", lambda s: fmt_secs(s["avg_success_seconds"]))
    line("median time to graded success", lambda s: fmt_secs(s["median_success_seconds"]))
    line("human grading success rate", lambda s: fmt_pct(s["human_success_rate"]))
    line("  (graded pass / graded)", lambda s: f"{s['graded_pass']}/{s['graded_attempts']}")
    line("avg tokens per graded success", lambda s: fmt_tok(s["avg_tokens_per_pass"]))
    line("total tokens used", lambda s: fmt_tok(s["tokens_total"]))
    line("tokens per solved prompt", lambda s: fmt_tok(s["tokens_per_solved_prompt"]))
    line("avg runtime per attempt (all)", lambda s: fmt_secs(s["avg_attempt_seconds_all"]))
    w("")

    w("COVERAGE AND PROMPT-LEVEL SOLVE")
    w("-" * 78)
    w(header)
    line("prompts solved (any try passed)", lambda s: f"{s['prompts_solved']}/{s['prompts']}")
    line("prompt solve rate (all prompts)", lambda s: fmt_pct(s["prompt_solve_rate_all"]))
    line("prompt solve rate (reviewed)", lambda s: fmt_pct(s["prompt_solve_rate_reviewed"]))
    line("grading coverage of attempts", lambda s: fmt_pct(s["grading_coverage"]))
    line("ungraded attempts", lambda s: str(s["ungraded_attempts"]))
    line("excluded (invalid/already-won)", lambda s: str(s["excluded_attempts"]))
    w("")

    w("COST AND TIME DETAIL")
    w("-" * 78)
    w(header)
    line("avg tokens per attempt", lambda s: fmt_tok(s["avg_tokens_per_attempt"]))
    line("avg tokens per graded failure", lambda s: fmt_tok(s["avg_tokens_per_graded_fail"]))
    line("tokens in / out", lambda s: f"{fmt_tok(s['tokens_in_total'])}/{fmt_tok(s['tokens_out_total'])}")
    line("total LLM calls", lambda s: f"{s['llm_calls_total']:,}")
    line("avg LLM calls per success", lambda s: "n/a" if s["avg_llm_calls_per_pass"] is None
         else f"{s['avg_llm_calls_per_pass']:.0f}")
    line("avg graded-failure runtime", lambda s: fmt_secs(s["avg_graded_fail_seconds"]))
    line("summed agent runtime", lambda s: f"{s['total_attempt_seconds'] / 3600:.1f} h")
    line("battery elapsed span", lambda s: f"{s['battery_span_seconds'] / 3600:.1f} h")
    line("parallelism factor", lambda s: "n/a" if s["parallelism_factor"] is None
         else f"{s['parallelism_factor']:.1f}x")
    w("")

    w("HARNESS SELF-REPORT vs HUMAN VERDICT (graded attempts only)")
    w("-" * 78)
    w(header)
    line("agreement", lambda s: fmt_pct(s["auto_vs_human_agreement"]))
    line("agent claimed win, human said no", lambda s: str(s["auto_false_pass"]))
    line("agent claimed loss, human said yes", lambda s: str(s["auto_false_fail"]))
    w("")

    timed = [r for r in rows if r["wall_seconds"] > 0]
    graded = [r for r in rows if r["graded"]]
    passes = [r for r in graded if r["human_pass"]]
    w("ACROSS ALL THREE BATTERIES")
    w("-" * 78)
    w(f"  attempts run                       {len(rows)}")
    w(f"  avg runtime per attempt (incl. failures)  "
      f"{fmt_secs(mean([r['wall_seconds'] for r in timed]))}")
    w(f"  median runtime per attempt         {fmt_secs(median([r['wall_seconds'] for r in timed]))}")
    w(f"  summed agent runtime               {sum(r['wall_seconds'] for r in rows) / 3600:.1f} h")
    w(f"  combined battery elapsed span      "
      f"{sum(s['battery_span_seconds'] for s in stats) / 3600:.1f} h")
    w(f"  tokens burned                      {sum(r['tokens_total'] for r in rows):,}")
    w(f"  LLM calls                          {sum(r['llm_calls'] for r in rows):,}")
    w(f"  graded attempts                    {len(graded)} ({len(passes)} pass, "
      f"{len(graded) - len(passes)} fail)")
    w(f"  overall human grading success rate {fmt_pct(len(passes) / len(graded) if graded else None)}")
    w("")

    w("BY TASK FAMILY (graded attempts, all difficulties)")
    w("-" * 78)
    fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in graded:
        fam[row["family"] or "(none)"].append(row)
    w(f"{'family':<16}{'graded':>8}{'pass':>7}{'rate':>9}{'avg pass time':>16}{'avg tokens':>13}")
    for name, frows in sorted(fam.items(), key=lambda kv: -len(kv[1])):
        fpass = [r for r in frows if r["human_pass"]]
        w(f"{name:<16}{len(frows):>8}{len(fpass):>7}{fmt_pct(len(fpass) / len(frows)):>9}"
          f"{fmt_secs(mean([r['wall_seconds'] for r in fpass if r['wall_seconds'] > 0])):>16}"
          f"{fmt_tok(mean([float(r['tokens_total']) for r in fpass])):>13}")
    w("")

    w("WHICH TRY WON (graded passes, by attempt index)")
    w("-" * 78)
    idx = Counter(r["attempt"] for r in passes)
    for i in sorted(idx):
        w(f"  try{i:02d}  {idx[i]:>3} pass(es)")
    w("")

    w("SLOWEST AND CHEAPEST GRADED PASSES")
    w("-" * 78)
    ranked = sorted([r for r in passes if r["wall_seconds"] > 0], key=lambda r: -r["wall_seconds"])
    for row in ranked[:3]:
        w(f"  slowest  {row['key']:<18} {fmt_secs(row['wall_seconds']):>9}  "
          f"{fmt_tok(row['tokens_total']):>7}  {row['prompt'][:44]}")
    for row in sorted(passes, key=lambda r: r["tokens_total"])[:3]:
        w(f"  cheapest {row['key']:<18} {fmt_secs(row['wall_seconds']):>9}  "
          f"{fmt_tok(row['tokens_total']):>7}  {row['prompt'][:44]}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("attempts.csv"))
    parser.add_argument("--json", type=Path, default=Path(__file__).with_name("stats.json"))
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for difficulty, name in BATTERIES:
        battery_dir = BENCH_ROOT / name
        if not battery_dir.is_dir():
            raise SystemExit(f"missing battery: {battery_dir}")
        rows = load_attempts(difficulty, battery_dir)
        all_rows.extend(rows)
        config = read_json(battery_dir / "battery.json")
        summary = summarize(difficulty, rows, read_json(battery_dir / "summary.json"))
        summary["tries"] = config.get("tries")
        summary["completion_guard"] = config.get("completion_guard") or "none"
        summary["arm"] = config.get("arm", "")
        stats.append(summary)

    report = render(stats, all_rows)
    print(report)

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    args.json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n[wrote] {args.csv}\n[wrote] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
