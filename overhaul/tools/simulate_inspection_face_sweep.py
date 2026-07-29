#!/usr/bin/env python3
"""Run an automatic offline inspection-face sweep using the real tool functions."""

import argparse
import json

from inspection_face_simulator import call_tool, install


def main():
    parser = argparse.ArgumentParser(
        description="Simulate present_* and rotate_*_to_next_inspection_face agent calls.")
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    parser.add_argument("--z", type=float, default=37.0,
                        help="pre-existing Z/roll to preserve (default: 37)")
    parser.add_argument("--steps", type=int, default=7,
                        help="number of next-face calls (default: complete seven-turn sweep)")
    parser.add_argument("--restart", action="store_true",
                        help="call present_* again after the sweep to demonstrate X/Y reset")
    args = parser.parse_args()

    initial = (0, 0, args.z)
    hands = install(left_rotation=initial, right_rotation=initial)
    present = f"present_{args.hand}_item_for_inspection"
    rotate = f"rotate_{args.hand}_to_next_inspection_face"

    calls = [present] + [rotate] * max(0, args.steps)
    if args.restart:
        calls.append(present)

    for index, name in enumerate(calls, 1):
        result = call_tool(name)
        hands.assert_no_z_commands()
        state = hands.state()
        print(f"\n{index}. {name}")
        print(json.dumps({"result": result, "state": state}, indent=2, default=str))

    print("\nPASS: every emitted rotation delta kept Z at 0; the hand's existing Z was preserved.")


if __name__ == "__main__":
    main()
