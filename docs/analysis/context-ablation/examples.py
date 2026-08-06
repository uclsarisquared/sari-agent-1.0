#!/usr/bin/env python3
"""Extract real before/after evidence for each ablation arm from bench_runs/.

Every excerpt printed here is quoted verbatim out of a run artefact, so the
examples in ABLATION_REPORT.md can be regenerated and checked rather than
trusted. For A2c the "after" is produced by replaying the arm's own rule over a
baseline store, which is the only way to see the same entries with and without
the seam.

Usage:  python3 analysis/context-ablation/examples.py [--arm a1] [--bench-runs DIR]
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import statistics

from collect import BATTERIES
from findings import blocks

ENTRY = re.compile(r"^@ leg (\d+) step (\d+): (.*)$", re.M)
# A2c's shipped configuration, from agent_core/context_policy.py.
DEDUPE = 0.80
WINDOW = 8
KEEP_LAST = 12


def store_entries(path: pathlib.Path) -> list[tuple[str, str]]:
    """(marker, text) for each appended entry; the base map dump is excluded."""
    return [(f"leg {leg} step {step}", text)
            for leg, step, text in ENTRY.findall(path.read_text(errors="replace"))]


def rule(text: str, survivors: list[str]) -> tuple[bool, float, str]:
    """A2c's accept test: reject if too similar to any of the last WINDOW survivors."""
    best, against = 0.0, ""
    for existing in survivors[-WINDOW:]:
        ratio = difflib.SequenceMatcher(None, text, existing).ratio()
        if ratio > best:
            best, against = ratio, existing
    return (best <= DEDUPE, best, against)


def head(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_a1(runs: pathlib.Path) -> None:
    banner("A1 — SemanticLog.render() returns base only")
    print("Seam: agent_core/context.py:42-48. keep_last=0 renders zero appended entries.\n")
    before = runs / BATTERIES["baseline"] / "medium_01/try01/semantic_memory.txt"
    after = runs / BATTERIES["a1"] / "medium_01/try01/semantic_memory.txt"
    entries = store_entries(before)
    print(f"BEFORE  {before.relative_to(runs)}")
    print(f"        {before.stat().st_size:,} bytes, {len(entries)} appended entries")
    print("        last two entries the learner re-read every step:\n")
    for marker, text in entries[-2:]:
        print(f"          @ {marker}: {head(text)}")
    print(f"\nAFTER   {after.relative_to(runs)}")
    print(f"        {after.stat().st_size:,} bytes, "
          f"{len(store_entries(after))} appended entries")
    print("        the store ends at the last line of the base map dump:\n")
    print(f"          {head(after.read_text().strip().splitlines()[-1])}")


def show_a2c(runs: pathlib.Path) -> None:
    banner("A2c — drop an entry too similar to one of the last 8; render the last 12")
    print("Seam: agent_core/context.py:30-48 (append) and render().")
    print(f"Replaying the rule (ratio > {DEDUPE}, window {WINDOW}) over baseline stores,")
    print("so the same entries are seen with and without the seam.\n")

    kept = dropped = 0
    worked: list[tuple[str, str, str, float]] = []
    for path in sorted((runs / BATTERIES["baseline"]).glob("*/try*/semantic_memory.txt")):
        survivors: list[str] = []
        for marker, text in store_entries(path):
            accept, ratio, against = rule(text, survivors)
            if accept:
                survivors.append(text)
                kept += 1
            else:
                dropped += 1
                worked.append((str(path.parent.relative_to(runs)), marker, text, ratio))
                worked[-1] = (*worked[-1], against)
    total = kept + dropped
    print(f"Over {total} baseline entries: {kept} kept, {dropped} dropped "
          f"({dropped / total:.0%} near-duplicate).\n")
    print("Three real drops, each shown against the entry that suppressed it:\n")
    for source, marker, text, ratio, against in worked[:3]:
        print(f"  {source}  @ {marker}   similarity {ratio:.2f}")
        print(f"    DROPPED : {head(text)}")
        print(f"    AGAINST : {head(against)}\n")


def show_a3_a4(runs: pathlib.Path) -> None:
    banner("A3 — skip findings entirely / A4 — 'cap' findings at 600 chars")
    print("Seam: orchestrator/subtask_agents.py:221-250 and _generate_findings_if_enabled.\n")
    for arm, label in (("baseline", "BEFORE (baseline)"), ("a4", "AFTER (a4)"),
                       ("a3", "AFTER (a3)")):
        found = []
        for log in sorted((runs / BATTERIES[arm]).glob("*/try*/agent.log")):
            for block in blocks(log.read_text(errors="replace")):
                if block:
                    found.append((log, block))
        print(f"{label}: {len(found)} findings summaries emitted")
        if not found:
            print("    (none — generate_findings_summary is never called)\n")
            continue
        log, block = found[0]
        print(f"    {log.parent.relative_to(runs)}  —  {len(block):,} chars")
        print(f"    {head(block, 400)}\n")


def show_a5(runs: pathlib.Path) -> None:
    banner("A5 — drop '## EXISTING EPISODIC MEMORY:' from the actor's user message")
    print("Seam: agent_core/agent.py:1193-1196. The blob is rebuilt every step and")
    print("appended to the actor's history, which is never trimmed, so step k carries")
    print("k copies of something superseded k-1 times.\n")
    sizes = []
    for arm in ("baseline", "a5"):
        arm_sizes = [p.stat().st_size for p in
                     (runs / BATTERIES[arm]).glob("*/try*/episodic_memory.txt")]
        sizes.append((arm, arm_sizes))
        print(f"  {arm:9s} episodic_memory.txt: n={len(arm_sizes)} "
              f"median {statistics.median(arm_sizes):,.0f} bytes, "
              f"max {max(arm_sizes):,} bytes")
    path = runs / BATTERIES["baseline"] / "medium_01/try01/episodic_memory.txt"
    text = path.read_text(errors="replace")
    print(f"\nBEFORE  the actor received this line every step, {len(text):,} chars of it:")
    print(f"        {path.relative_to(runs)}\n")
    print(f"          ## EXISTING EPISODIC MEMORY: {head(text, 300)}")
    print("\nAFTER   the line is absent; the episodic learner still writes the same file")
    print("        and still reads get_history_text(n=8). Only the actor's copy is gone.")


def show_a6(runs: pathlib.Path) -> None:
    banner("A6 — strip image parts from all but the newest N user messages")
    print("Seam: agent_core/agent.py:365-395 (_outbound_history). self.history keeps")
    print("every frame; only the outgoing message list is filtered.\n")
    print("One 1280x720 frame is ~1,125 tokens by encoder geometry. What a leg sends:\n")
    rows = [("leg length (actor calls)", "baseline", "a6-4", "a6-2")]
    for n in (5, 10, 20, 40):
        rows.append((str(n), f"{n} frames", f"{min(n, 4)} frames", f"{min(n, 2)} frames"))
    for row in rows:
        print("    " + "".join(str(c).ljust(26) for c in row))
    print("\nLongest legs actually observed, by frames retained in the run directory:")
    for arm in ("baseline", "a6-2", "a6-4"):
        best = (0, None)
        for leg_dir in (runs / BATTERIES[arm]).glob("*/try*/leg*/"):
            count = len(list(leg_dir.glob("step*.png")))
            if count > best[0]:
                best = (count, leg_dir)
        print(f"  {arm:9s} {best[0]:3d} steps  {best[1].relative_to(runs) if best[1] else '-'}")
    print("\nAt those lengths the baseline actor is re-sending every frame it has ever")
    print("seen on every call; a6-2 is sending two.")


SHOWS = {"a1": show_a1, "a2c": show_a2c, "a3": show_a3_a4, "a4": show_a3_a4,
         "a5": show_a5, "a6": show_a6}


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-runs", type=pathlib.Path,
                        default=here.parent.parent / "bench_runs")
    parser.add_argument("--arm", choices=sorted(set(SHOWS)), help="show only one arm")
    args = parser.parse_args()
    order = [args.arm] if args.arm else ["a1", "a2c", "a3", "a5", "a6"]
    for arm in order:
        SHOWS[arm](args.bench_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
