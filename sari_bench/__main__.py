"""``python -m sari_bench <command>``.

    coordinator   run the sandbox registry
    run           run a prompt battery across the fleet
    status        print the current sandbox pool
"""

from __future__ import annotations

import asyncio
import json
import sys

from sari_bench.protocol import DEFAULT_COORDINATOR_PORT

USAGE = __doc__


def _status(argv: list[str]) -> int:
    import argparse

    from sari_bench.client import CoordinatorClient

    parser = argparse.ArgumentParser(prog="sari_bench status")
    parser.add_argument("--coordinator", default=f"ws://localhost:{DEFAULT_COORDINATOR_PORT}")
    parser.add_argument("--json", action="store_true", help="Print the raw pool snapshot.")
    args = parser.parse_args(argv)

    async def fetch() -> list[dict[str, object]]:
        async with CoordinatorClient(args.coordinator) as client:
            return await client.pool()

    pool = asyncio.run(fetch())
    if args.json:
        print(json.dumps(pool, indent=2))
        return 0

    if not pool:
        print("No sandboxes registered.")
        return 0

    print(f"{'SANDBOX':<34} {'ADDRESS':<24} {'STATE':<11} {'LEASE'}")
    for sandbox in pool:
        address = f"{sandbox['host']}:{sandbox['port']}"
        lease = sandbox.get("lease_id") or "-"
        print(f"{sandbox['sandbox_id']:<34} {address:<24} {sandbox['state']:<11} {lease}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]
    if command == "coordinator":
        from sari_bench.coordinator import main as coordinator_main

        return coordinator_main(rest)
    if command == "run":
        from sari_bench.runner import main as runner_main

        return runner_main(rest)
    if command == "status":
        return _status(rest)

    print(f"Unknown command: {command}\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
