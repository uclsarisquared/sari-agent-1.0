#!/usr/bin/env python3
"""Mine agent.log files for quotable model output.

The orchestrator logs each model reply right after a `<SPEAKER> RESPONSE:` marker, in one of two
shapes: a ```json fence, or a Python dict repr on a single line. It also logs the episodic memory
the agent rewrites for itself each timestep ("DENSE SUMMARY / WHAT WORKED / WHAT TO AVOID"). All
three are pulled out here so "interesting moment" is a repeatable search rather than a hunch.

Usage:
  python3 analysis/agent-report/mine_logs.py --theme recovery --limit 6
  python3 analysis/agent-report/mine_logs.py --memory --glob 'hard_09/try*'
  python3 analysis/agent-report/mine_logs.py --list-themes
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO / "bench_runs"
BATTERIES = ["20260727_020820-easy", "20260728_143857-medium", "20260730_205702-hard"]

SPEAKER = re.compile(r"^([A-Za-z][A-Za-z _]*?) RESPONSE:[ \t]*", re.MULTILINE)
MEMORY = re.compile(
    r"Episodic memory updated: @ timestep (\d+):\n(## DENSE SUMMARY:.*?)(?=\n\n|\n\[|\n\d{4}-)",
    re.DOTALL,
)
DECODER = json.JSONDecoder()

THEMES = {
    "budget": ("php", "budget", "afford", "cheaper", "total", "price", "cost"),
    "planning": ("plan", "first", "then", "before", "remaining", "checklist", "next leg"),
    "recovery": ("stuck", "again", "did not", "failed", "retry", "instead", "wrong",
                 "mistake", "avoid", "no longer", "stalled"),
    "counting": ("count", "how many", "number of", "rows", "total of"),
    "self-doubt": ("not sure", "unclear", "cannot see", "ambiguous", "might be",
                   "hard to tell", "unable to", "no longer visible"),
    "grip": ("grab", "grasp", "hand", "reach", "drop", "hold", "gripper"),
}


def parse_payload(text: str, start: int) -> dict[str, Any] | None:
    """One logged reply, whether it was fenced JSON or a Python dict repr."""
    head = text[start : start + 8]
    if head.lstrip().startswith("```"):
        brace = text.find("{", start)
        if brace < 0:
            return None
        try:
            payload, _ = DECODER.raw_decode(text, brace)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None
    # Dict reprs are logged on a single line, so the line is the whole payload.
    line = text[start : text.find("\n", start) if text.find("\n", start) > 0 else len(text)]
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        payload = ast.literal_eval(line)
    except (ValueError, SyntaxError):
        return None
    return payload if isinstance(payload, dict) else None


def replies(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in SPEAKER.finditer(text):
        payload = parse_payload(text, match.end())
        if payload:
            yield match.group(1).strip(), payload


def memories(path: Path) -> Iterator[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in MEMORY.finditer(text):
        yield int(match.group(1)), match.group(2).strip()


def attempt_logs(glob: str) -> Iterator[tuple[str, str, Path]]:
    for battery in BATTERIES:
        for log in sorted((BENCH_ROOT / battery).glob(f"{glob}/agent.log")):
            yield battery, f"{log.parent.parent.name}/{log.parent.name}", log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", default="planning", choices=sorted(THEMES))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--min-score", type=int, default=2)
    parser.add_argument("--glob", default="*/try*", help="attempt filter, e.g. 'hard_09/try*'")
    parser.add_argument("--memory", action="store_true", help="dump episodic memory instead")
    parser.add_argument("--list-themes", action="store_true")
    args = parser.parse_args()

    if args.list_themes:
        for name, words in THEMES.items():
            print(f"{name:<12} {', '.join(words)}")
        return 0

    if args.memory:
        shown = 0
        for battery, key, log in attempt_logs(args.glob):
            for timestep, body in memories(log):
                print("=" * 78)
                print(f"{battery}  {key}  timestep {timestep}")
                print("-" * 78)
                print(body)
                shown += 1
                if shown >= args.limit:
                    return 0
        return 0

    keywords = THEMES[args.theme]
    hits = []
    # A logged prompt replays earlier turns, so the same reply can appear many times over.
    seen: set[str] = set()
    for battery, key, log in attempt_logs(args.glob):
        for speaker, payload in replies(log):
            text = json.dumps(payload, ensure_ascii=False)
            if text in seen:
                continue
            seen.add(text)
            points = sum(1 for word in keywords if word in text.lower())
            if points >= args.min_score and len(text) >= args.min_chars:
                hits.append((points, len(text), battery, key, speaker, payload))

    hits.sort(key=lambda hit: (-hit[0], -hit[1]))
    for points, _, battery, key, speaker, payload in hits[: args.limit]:
        print("=" * 78)
        print(f"{battery}  {key}  [{speaker}]  theme={args.theme} score={points}")
        print("-" * 78)
        print(json.dumps(payload, indent=2, ensure_ascii=False)[:2400])
    print(f"\n{len(hits)} matching reply/replies; showing {min(args.limit, len(hits))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
