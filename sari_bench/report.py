"""Flattens a battery into CSVs: ``python -m sari_bench report``.

Two files, not one. Attempts and legs are different grains - an attempt has a variable number of
legs, so folding them into one row either truncates or explodes, and neither pivots. They join on
``run_dir``.

Both inputs are written incrementally (``attempts.jsonl`` as attempts finish, each attempt's own
``summary.json`` at its exit), so this is runnable mid-battery and a battery interrupted after six
hours still reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sari_bench.watch import scan

ATTEMPT_COLUMNS = [
    "battery_id", "prompt_id", "attempt", "family", "prompt", "looking_for",
    "outcome", "success", "end_reason", "exit_code", "wall_seconds", "wall_minutes",
    "legs_planned", "legs_completed", "requeues", "sandbox_id", "commands_uri",
    "arm", "killed_by", "collapse_score", "collapse_signals", "run_dir", "error",
]

LEG_COLUMNS = [
    "battery_id", "prompt_id", "attempt", "leg_index", "type", "text", "success",
    "end_reason", "timesteps", "llm_calls", "errors", "halts_refused", "halt_forced",
    "corrective_release", "t_manip", "t_grip", "t_checkout", "wall_s", "run_dir",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def collect(battery: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Builds the attempt rows and leg rows for one battery dir.

    ``attempts.jsonl`` is the spine, but it only holds FINISHED attempts. Run dirs whose manifest
    has not been closed out are folded in too, so a report taken mid-battery (or after a runner
    crash) still accounts for every attempt that started.
    """
    battery_id = battery.name
    by_run_dir: dict[str, dict[str, Any]] = {}

    for row in _read_jsonl(battery / "attempts.jsonl"):
        if row.get("run_dir"):
            by_run_dir[str(Path(row["run_dir"]))] = row

    attempt_rows: list[dict[str, Any]] = []
    leg_rows: list[dict[str, Any]] = []

    for run_dir in scan.run_dirs_of(battery):
        manifest = scan._read_json(run_dir / scan.ATTEMPT_MANIFEST)
        recorded = by_run_dir.get(str(run_dir), {})
        summary = scan._read_json(run_dir / "summary.json")

        prompt_id = recorded.get("prompt_id") or manifest.get("prompt_id") or run_dir.parent.name
        attempt = recorded.get("attempt") or manifest.get("attempt") or 0
        outcome = recorded.get("outcome") or manifest.get("outcome") or manifest.get("state") or "unfinished"
        wall = recorded.get("wall_seconds") or manifest.get("wall_seconds") or 0.0

        # A live/orphaned attempt gets its collapse score recorded, which is what makes
        # "how many attempts were in a death loop when I killed them" answerable after the fact.
        view = scan.scan_attempt(run_dir, battery, now=0.0)
        legs = recorded.get("legs") or {}

        attempt_rows.append({
            "battery_id": battery_id,
            "prompt_id": prompt_id,
            "attempt": attempt,
            "family": recorded.get("family") or manifest.get("family", ""),
            "prompt": recorded.get("prompt") or manifest.get("prompt", ""),
            "looking_for": manifest.get("looking_for", ""),
            "outcome": outcome,
            "success": bool(recorded.get("success") or summary.get("success")),
            "end_reason": recorded.get("end_reason") or manifest.get("end_reason", ""),
            "exit_code": recorded.get("exit_code", manifest.get("exit_code")),
            "wall_seconds": wall,
            "wall_minutes": round(float(wall) / 60.0, 2) if wall else 0.0,
            "legs_planned": legs.get("planned", summary.get("legs_planned")),
            "legs_completed": legs.get("completed", summary.get("legs_completed")),
            "requeues": recorded.get("requeues", 0),
            "sandbox_id": recorded.get("sandbox_id") or manifest.get("sandbox_id", ""),
            "commands_uri": recorded.get("commands_uri") or manifest.get("commands_uri", ""),
            "arm": manifest.get("arm") or summary.get("arm", ""),
            "killed_by": manifest.get("killed_by", ""),
            "collapse_score": (view.health or {}).get("score", 0.0),
            "collapse_signals": (view.health or {}).get("summary", ""),
            "run_dir": str(run_dir),
            "error": recorded.get("error", ""),
        })

        for index, leg in enumerate(summary.get("legs") or []):
            if not isinstance(leg, dict):
                continue
            leg_rows.append({
                "battery_id": battery_id,
                "prompt_id": prompt_id,
                "attempt": attempt,
                "leg_index": index,
                "type": leg.get("type", ""),
                "text": leg.get("text", ""),
                "success": bool(leg.get("success")),
                "end_reason": leg.get("end_reason", ""),
                "timesteps": leg.get("timesteps"),
                "llm_calls": leg.get("llm_calls"),
                "errors": leg.get("errors"),
                "halts_refused": leg.get("halts_refused"),
                "halt_forced": leg.get("halt_forced"),
                "corrective_release": leg.get("corrective_release"),
                "t_manip": leg.get("t_manip"),
                "t_grip": leg.get("t_grip"),
                "t_checkout": leg.get("t_checkout"),
                "wall_s": leg.get("wall_s"),
                "run_dir": str(run_dir),
            })

    attempt_rows.sort(key=lambda r: (str(r["prompt_id"]), r["attempt"]))
    return attempt_rows, leg_rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sari_bench report",
                                     description="Flatten a battery into attempts/legs CSVs.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Battery dir. Defaults to the newest under --bench-root.")
    parser.add_argument("--bench-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "bench_runs")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where the CSVs land. Defaults to the battery dir itself.")
    args = parser.parse_args(argv)

    battery = args.run_dir
    if battery is None:
        found = scan.find_batteries(args.bench_root.resolve())
        if not found:
            print(f"[sari-bench report] no battery dirs under {args.bench_root}")
            return 1
        battery = found[0]
        print(f"[sari-bench report] newest battery: {battery}")

    attempts, legs = collect(battery.resolve())
    out_dir = (args.out_dir or battery).resolve()
    _write_csv(out_dir / "attempts.csv", ATTEMPT_COLUMNS, attempts)
    _write_csv(out_dir / "legs.csv", LEG_COLUMNS, legs)
    successes = sum(1 for row in attempts if row["success"])
    print(f"[sari-bench report] {len(attempts)} attempt(s) ({successes} successful), "
          f"{len(legs)} leg(s) -> {out_dir}/attempts.csv, {out_dir}/legs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
