"""Probe the sandbox's RequestLidarCenter wire response and validate its schema.

This uses the same WebSocket command endpoint and JSON decoding contract as the grab
dispatcher. It does not move the agent or its hands.

Examples (run from the repository root):

    uv run python validation/probes/lidar_json_probe.py
    uv run python validation/probes/lidar_json_probe.py --uri ws://localhost:8080/commands
    uv run python validation/probes/lidar_json_probe.py --count 10 --delay 0.2

Exit status is zero only when every response is a JSON object containing the numeric
distance/pitch/height/range fields and a boolean hit field expected by the current agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from typing import Any


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from manip.manipulation import plan_reach  # noqa: E402
from sim.env import SendCommand, default_uri  # noqa: E402


EXPECTED_FIELDS = (
    "distance",
    "hit",
    "pitch_deg",
    "camera_height",
    "min_range",
    "max_range",
)
NUMERIC_FIELDS = (
    "distance",
    "pitch_deg",
    "camera_height",
    "min_range",
    "max_range",
)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_sample(sample: Any) -> list[str]:
    """Return schema/type errors that would make the response unsafe for reach planning."""
    if not isinstance(sample, dict):
        return [f"top-level JSON must be an object, got {type(sample).__name__}"]

    errors = []
    missing = [field for field in EXPECTED_FIELDS if field not in sample]
    if missing:
        errors.append(f"missing expected field(s): {', '.join(missing)}")

    if "hit" in sample and not isinstance(sample["hit"], bool):
        errors.append(f"hit must be boolean, got {sample['hit']!r}")

    for field in NUMERIC_FIELDS:
        if field in sample and not _is_finite_number(sample[field]):
            errors.append(f"{field} must be a finite number, got {sample[field]!r}")

    if all(_is_finite_number(sample.get(field))
           for field in ("distance", "min_range", "max_range")):
        if sample["min_range"] > sample["max_range"]:
            errors.append("min_range must not exceed max_range")
        if not sample["min_range"] <= sample["distance"] <= sample["max_range"]:
            errors.append(
                f"distance {sample['distance']} is outside "
                f"[min_range={sample['min_range']}, max_range={sample['max_range']}]"
            )

    return errors


def request_raw(uri: str) -> Any:
    """Send exactly one RequestLidarCenter command and return the wire reply."""
    return asyncio.get_event_loop().run_until_complete(
        SendCommand({"command": "RequestLidarCenter"}, uri)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the sandbox RequestLidarCenter JSON response used by grab planning."
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="WebSocket command URI. Defaults to $SARI_WS_URI, then ws://localhost:8080/commands.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of independent requests to make (default: 1).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds between requests (default: 0).",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.delay < 0:
        parser.error("--delay must be non-negative")

    uri = args.uri or default_uri()
    print(f"RequestLidarCenter endpoint: {uri}")
    print(f"SARI_WS_URI environment: {os.environ.get('SARI_WS_URI')!r}")

    failed = 0
    for index in range(1, args.count + 1):
        print(f"\n--- sample {index}/{args.count} ---")
        try:
            raw = request_raw(uri)
        except Exception as exc:  # transport/runtime errors are part of what this probe diagnoses
            failed += 1
            print(f"FAIL transport: {type(exc).__name__}: {exc}")
            if index < args.count and args.delay:
                time.sleep(args.delay)
            continue

        print(f"raw type: {type(raw).__name__}")
        print(f"raw reply: {raw!r}")

        try:
            sample = json.loads(raw)
        except (TypeError, ValueError) as exc:
            failed += 1
            print(f"FAIL JSON decoding: {type(exc).__name__}: {exc}")
            if index < args.count and args.delay:
                time.sleep(args.delay)
            continue

        print(f"decoded type: {type(sample).__name__}")
        try:
            print(json.dumps(sample, indent=2, sort_keys=True))
        except (TypeError, ValueError):
            print(repr(sample))

        errors = validate_sample(sample)
        if errors:
            failed += 1
            print("FAIL schema:")
            for error in errors:
                print(f"  - {error}")
        else:
            plan = plan_reach(sample)
            print("PASS schema")
            print(
                "reach plan: "
                f"verdict={plan['verdict']!r} move_steps={plan['move_steps']} "
                f"reason={plan['reason']}"
            )

        if index < args.count and args.delay:
            time.sleep(args.delay)

    passed = args.count - failed
    print(f"\nResult: {passed}/{args.count} response(s) passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
