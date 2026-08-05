#!/usr/bin/env python3
"""Interactive offline REPL that accepts inspection tool names like an agent."""

import argparse
import json

from inspection_face_simulator import TOOL_CALLS, call_tool, install


def main():
    parser = argparse.ArgumentParser(
        description="Interactively call the held-item inspection presentation/rotation tools.")
    parser.add_argument("--left-z", type=float, default=37.0)
    parser.add_argument("--right-z", type=float, default=-22.0)
    args = parser.parse_args()

    hands = install(
        left_rotation=(0, 0, args.left_z),
        right_rotation=(0, 0, args.right_z),
    )
    print("Inspection tool REPL. Both simulated hands hold an item.")
    print("Enter a full tool name, 'state', 'help', or 'quit'.")

    while True:
        try:
            command = input("inspection> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not command:
            continue
        if command in {"quit", "exit"}:
            break
        if command == "help":
            print("\n".join(sorted(TOOL_CALLS)))
            continue
        if command == "state":
            print(json.dumps(hands.state(), indent=2))
            continue
        try:
            result = call_tool(command)
            hands.assert_no_z_commands()
        except (ValueError, AssertionError) as exc:
            print(f"ERROR: {exc}")
            continue
        print(json.dumps({"result": result, "state": hands.state()}, indent=2, default=str))


if __name__ == "__main__":
    main()
