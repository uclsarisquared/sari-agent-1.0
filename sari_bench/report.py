"""Flattens a battery into CSVs: ``python -m sari_bench report``.

Three files, not one. Attempts, legs and per-role token spend are different grains - an attempt has
a variable number of legs and a variable set of reasoners, so folding them into one row either
truncates or explodes, and neither pivots. They all join on ``run_dir``.

``roles.csv`` is long-format on purpose (one row per attempt PER ROLE, not one column per role):
that is what makes "tokens by arm by role" a two-click pivot, and it means a reasoner added or
removed agent-side changes the row count rather than the header - so an old CSV and a new one still
stack.

Both inputs are written incrementally (the canonical ``attempts.jsonl`` index as attempts finish,
each attempt's own ``summary.json`` at its exit), so this is runnable mid-battery and a battery
interrupted after six hours still reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sari_bench.storage import canonical_attempt_rows
from sari_bench.watch import scan

ATTEMPT_COLUMNS = [
    "battery_id", "prompt_id", "attempt", "family", "prompt", "looking_for",
    "outcome", "success",
    # The human's answer, kept beside the predicate's rather than folded into it. `success` is what
    # overhaul/orchestrator/subtask_completion.py decided - and several of its predicates grant on
    # state they cannot ground. `verified_success` is a reviewer's call after watching the replay.
    # `verdict_agrees` is the column this whole path exists to produce; `success_final` is the one to
    # group by, preferring the human where there is one.
    # `verified_verdict` carries the answers the boolean cannot: "invalid", a run the reviewer threw
    # out, and "already_successful", a try halted because a sibling had already won the prompt. Those
    # rows leave `verified_success`, `verdict_agrees` and `success_final` blank - excluded from every
    # total rather than counted as failures.
    "verifiable", "verified", "verified_verdict", "verified_success", "verdict_agrees",
    "success_final",
    # True on exactly one row per solved prompt: its lowest-numbered human-verified pass, the same
    # try the dashboard's "success time" column shows. Averaging `wall_minutes` over these rows is
    # the dashboard's "avg success time" - without it the CSV can only average every passing try,
    # which double-counts a prompt that was dispatched more than once and won twice.
    "first_pass",
    "verified_by", "verified_at", "verified_note",
    "end_reason", "exit_code", "wall_seconds", "wall_minutes",
    "tokens_in", "tokens_out", "tokens_total", "llm_calls",
    "legs_planned", "legs_completed", "requeues", "sandbox_id", "commands_uri",
    "arm", "killed_by", "stop_reason", "stop_requested_at", "stop_requested_by",
    "winning_attempt_key", "collapse_score", "collapse_signals", "run_dir", "error",
]

LEG_COLUMNS = [
    "battery_id", "prompt_id", "attempt", "leg_index", "type", "text", "success",
    "end_reason", "timesteps", "llm_calls", "tokens_in", "tokens_out", "errors",
    "halts_refused", "halt_forced",
    "corrective_release", "t_manip", "t_grip", "t_checkout", "wall_s", "run_dir",
]

# One row per (attempt, role). `share_in`/`share_out` are the role's fraction of THIS attempt's
# metered spend, which is the column an ablation actually reads: absolute tokens move with task
# length, the share does not. They are computed against the summed role rows rather than the
# attempt's `tokens_in`, so they total to 1.0 even where the two disagree (see `_role_rows`).
ROLE_COLUMNS = [
    "battery_id", "prompt_id", "attempt", "arm", "family", "outcome", "success_final",
    "role", "tokens_in", "tokens_out", "tokens_total", "calls", "share_in", "share_out",
    "run_dir",
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


def _tokens_of(run_dir: Path, recorded: dict[str, Any], summary: dict[str, Any]) -> tuple[int, int]:
    """The attempt's (tokens_in, tokens_out), from the most authoritative record available.

    attempts.jsonl for a finished attempt, then the agent's summary.json, then the tokens.json the
    agent rewrites as it goes - which is what makes a still-running or SIGKILLed attempt account for
    its tokens at all.
    """
    for source in (recorded, summary, scan._read_json(run_dir / "tokens.json")):
        if isinstance(source, dict) and ("tokens_in" in source or "tokens_out" in source):
            return int(source.get("tokens_in") or 0), int(source.get("tokens_out") or 0)
    return 0, 0


def _roles_of(run_dir: Path, recorded: dict[str, Any], summary: dict[str, Any]
              ) -> dict[str, dict[str, int]]:
    """The attempt's per-role token spend, from the same authority chain as ``_tokens_of``.

    The key differs per source only because each writer names it for its own context: the runner
    records ``tokens_by_role`` on the attempt row, while the agent's summary.json and tokens.json
    both carry the meter's own ``by_role`` block. Empty when nothing recorded roles - an attempt from
    before per-role accounting contributes no rows to roles.csv rather than a row of zeroes.
    """
    tokens = summary.get("tokens") if isinstance(summary.get("tokens"), dict) else {}
    for raw in (recorded.get("tokens_by_role"), tokens.get("by_role"), summary.get("by_role"),
                scan._read_json(run_dir / "tokens.json").get("by_role")):
        rows = scan.normalize_by_role(raw)
        if rows:
            return rows
    return {}


def role_rows(attempt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expands ``collect``'s attempt rows into the long-format (attempt x role) grain.

    Reads the per-attempt block ``collect`` stashes under the private ``_tokens_by_role`` key rather
    than re-walking the run dirs: one traversal, one authority chain, no way for the two CSVs to
    disagree about what an attempt spent. Every writer here passes ``extrasaction="ignore"``, so the
    private key never reaches attempts.csv.
    """
    rows: list[dict[str, Any]] = []
    for attempt in attempt_rows:
        by_role = attempt.get("_tokens_by_role") or {}
        # Denominators from the role rows themselves, not from the attempt's `tokens_in`: an attempt
        # whose agent died mid-run can have a totals line newer than its role lines, and a share
        # column that does not sum to 1.0 would read as a measurement error rather than as skew.
        total_in = sum(row["tokens_in"] for row in by_role.values())
        total_out = sum(row["tokens_out"] for row in by_role.values())
        for name in scan.sorted_roles(by_role):
            row = by_role[name]
            rows.append({
                "battery_id": attempt["battery_id"],
                "prompt_id": attempt["prompt_id"],
                "attempt": attempt["attempt"],
                "arm": attempt["arm"],
                "family": attempt["family"],
                "outcome": attempt["outcome"],
                "success_final": attempt["success_final"],
                "role": name,
                "tokens_in": row["tokens_in"],
                "tokens_out": row["tokens_out"],
                "tokens_total": row["tokens_in"] + row["tokens_out"],
                "calls": row["calls"],
                "share_in": round(row["tokens_in"] / total_in, 4) if total_in else "",
                "share_out": round(row["tokens_out"] / total_out, 4) if total_out else "",
                "run_dir": attempt["run_dir"],
            })
    return rows


def collect(battery: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Builds the attempt rows and leg rows for one battery dir.

    ``attempts.jsonl`` is the spine, but it only holds FINISHED attempts. Run dirs whose manifest
    has not been closed out are folded in too, so a report taken mid-battery (or after a runner
    crash) still accounts for every attempt that started.
    """
    battery_id = battery.name
    by_run_dir: dict[str, dict[str, Any]] = {}

    for row in canonical_attempt_rows(battery):
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
        tokens_in, tokens_out = _tokens_of(run_dir, recorded, summary)

        # Same fallback chain as every other field in this row - and, now that `verdict_agrees`
        # compares against it, the same answer the dashboard shows. The manifest belongs in the chain
        # because the runner patches `success` into it when it publishes the attempts.jsonl row, so
        # an attempt whose row never made it to the spine still reports the verdict it earned.
        success = (
            bool(recorded.get("success") or summary.get("success") or manifest.get("success"))
            if outcome == "completed" else False
        )
        end_reason = recorded.get("end_reason") or manifest.get("end_reason", "")
        # Blank, never False, when nobody has looked: an unreviewed attempt must not read as "a human
        # said it failed" in a pivot table.
        verdict = scan.effective_verdict(manifest)
        verified = bool(verdict)
        # Blank for the excluded verdicts as well as for unreviewed, for the same reason: none of
        # them is a human saying the attempt failed, and a pivot table must not read one as that.
        verified_success = (
            "" if verdict == "" or verdict in scan.EXCLUDED_VERDICTS else verdict == "pass"
        )

        attempt_rows.append({
            "battery_id": battery_id,
            "prompt_id": prompt_id,
            "attempt": attempt,
            "family": recorded.get("family") or manifest.get("family", ""),
            "prompt": recorded.get("prompt") or manifest.get("prompt", ""),
            "looking_for": manifest.get("looking_for", ""),
            "outcome": outcome,
            "success": success,
            "verifiable": scan.is_verifiable(str(manifest.get("state") or ""), str(end_reason)),
            "verified": verified,
            "verified_verdict": verdict,
            "verified_success": verified_success,
            "verdict_agrees": (verified_success == success) if verified_success != "" else "",
            # An excluded attempt has no final answer at all - it is not the predicate's `success`
            # either, since the whole point of the verdict was that the run proved nothing.
            "success_final": "" if verdict in scan.EXCLUDED_VERDICTS else (
                verified_success if verified else success),
            "verified_by": manifest.get("verified_by", ""),
            "verified_at": manifest.get("verified_at", ""),
            "verified_note": manifest.get("verified_note", ""),
            "end_reason": end_reason,
            "exit_code": recorded.get("exit_code", manifest.get("exit_code")),
            "wall_seconds": wall,
            "wall_minutes": round(float(wall) / 60.0, 2) if wall else 0.0,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "llm_calls": recorded.get("llm_calls") or summary.get("llm_calls") or 0,
            "legs_planned": legs.get("planned", summary.get("legs_planned")),
            "legs_completed": legs.get("completed", summary.get("legs_completed")),
            "requeues": recorded.get("requeues", 0),
            "sandbox_id": recorded.get("sandbox_id") or manifest.get("sandbox_id", ""),
            "commands_uri": recorded.get("commands_uri") or manifest.get("commands_uri", ""),
            "arm": manifest.get("arm") or summary.get("arm", ""),
            "killed_by": manifest.get("killed_by", ""),
            "stop_reason": manifest.get("stop_reason", ""),
            "stop_requested_at": manifest.get("stop_requested_at", ""),
            "stop_requested_by": manifest.get("stop_requested_by", ""),
            "winning_attempt_key": (
                recorded.get("winning_attempt_key")
                or manifest.get("winning_attempt_key", "")
            ),
            "collapse_score": (view.health or {}).get("score", 0.0),
            "collapse_signals": (view.health or {}).get("summary", ""),
            "run_dir": str(run_dir),
            "error": recorded.get("error", ""),
            # Private, and dropped by every DictWriter here (extrasaction="ignore"). It rides the
            # attempt row so `role_rows` can expand the (attempt x role) grain without a second
            # traversal of the battery - see role_rows.
            "_tokens_by_role": _roles_of(run_dir, recorded, summary),
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
                "tokens_in": leg.get("tokens_in"),
                "tokens_out": leg.get("tokens_out"),
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
    # After the sort, so "first" means the lowest try number and not whatever order the run dirs came
    # back in. Only a human `pass` marks one: the predicate's `success` is exactly what the review
    # flow exists to second-guess, and an unreviewed halt has not won its prompt yet.
    seen: set[str] = set()
    for row in attempt_rows:
        first = row["verified_verdict"] == "pass" and str(row["prompt_id"]) not in seen
        if first:
            seen.add(str(row["prompt_id"]))
        row["first_pass"] = first
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
    roles = role_rows(attempts)
    out_dir = (args.out_dir or battery).resolve()
    _write_csv(out_dir / "attempts.csv", ATTEMPT_COLUMNS, attempts)
    _write_csv(out_dir / "legs.csv", LEG_COLUMNS, legs)
    _write_csv(out_dir / "roles.csv", ROLE_COLUMNS, roles)
    successes = sum(1 for row in attempts if row["success"])
    print(f"[sari-bench report] {len(attempts)} attempt(s) ({successes} successful), "
          f"{len(legs)} leg(s), {len(roles)} role row(s) -> {out_dir}/attempts.csv, "
          f"{out_dir}/legs.csv, {out_dir}/roles.csv")

    # Which reasoner the battery's tokens actually went to. Printed rather than left to the CSV
    # because it is the line an ablation is run to read, and because a battery whose attempts
    # recorded no roles at all should say so out loud instead of writing an empty file quietly.
    metered = sum(1 for row in attempts if row["_tokens_by_role"])
    if metered:
        per_role: dict[str, int] = {}
        for row in roles:
            per_role[row["role"]] = per_role.get(row["role"], 0) + row["tokens_total"]
        grand = sum(per_role.values()) or 1
        ranked = sorted(per_role.items(), key=lambda item: -item[1])
        share = ", ".join(f"{name} {value * 100 // grand}%" for name, value in ranked)
        print(f"[sari-bench report] tokens by role over {metered}/{len(attempts)} attempt(s) "
              f"with role accounting: {share}")
    elif attempts:
        print("[sari-bench report] no attempt recorded per-role tokens "
              "(agent predates role accounting); roles.csv is empty")

    # The predicate-vs-human line. Disagreements are the number the review flow exists to surface, so
    # it gets its own line rather than a column nobody opens the CSV to find.
    reviewed = [row for row in attempts if row["verified"]]
    if reviewed or any(row["verifiable"] for row in attempts):
        # Excluded runs are held out of the agree/disagree split entirely: nobody ruled on the
        # predicate. An invalid one is evidence about the harness; an already_successful one is
        # bookkeeping about a prompt some other try had already won.
        judged = [row for row in reviewed
                  if row["verified_verdict"] not in scan.EXCLUDED_VERDICTS]
        invalid = sum(1 for row in reviewed if row["verified_verdict"] == "invalid")
        halted = sum(1 for row in reviewed
                     if row["verified_verdict"] == scan.ALREADY_SUCCESSFUL)
        agree = sum(1 for row in judged if row["verdict_agrees"])
        waiting = sum(1 for row in attempts if row["verifiable"] and not row["verified"])
        print(f"[sari-bench report] {len(judged)} human-verified "
              f"({agree} agree, {len(judged) - agree} disagree with the predicate), "
              f"{invalid} marked invalid, {halted} halted as already successful, "
              f"{waiting} halt(s) awaiting review")

    # The dashboard's "avg success time", printed from the same rows the CSV carries so the two
    # cannot drift. A solved prompt whose wall clock was never recorded is left out of the mean
    # rather than averaged in as an instant win.
    won = [float(row["wall_seconds"]) for row in attempts
           if row["first_pass"] and row["wall_seconds"]]
    if won:
        mean = sum(won) / len(won)
        print(f"[sari-bench report] avg success time {int(mean // 60)}m{int(mean % 60):02d}s "
              f"over {len(won)} solved prompt(s) (first human-verified pass of each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
