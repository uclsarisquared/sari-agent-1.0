"""Wire protocol shared by the coordinator, the benchmark runner, and the Unity sandbox.

Every message is a JSON object with a ``type`` field. The Unity side of this contract lives in
``Assets/Scripts/SocketServers/DistributedBench/BenchCoordinatorClient.cs`` - keep the two in step.

Two routes on the coordinator:

``/sandbox``  sims register and stay connected
    sandbox -> coord   sandbox.hello, sandbox.heartbeat, sandbox.state
    coord -> sandbox   coord.welcome, coord.lease, coord.reset, coord.release

``/bench``  one connection per benchmark worker, at most one lease each
    bench -> coord     bench.acquire, bench.release, bench.status
    coord -> bench     bench.lease, bench.released, bench.pool, bench.sandbox_lost, bench.error
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_VERSION = 1

SANDBOX_ROUTE = "/sandbox"
BENCH_ROUTE = "/bench"

# Sandbox readiness, mirroring the SandboxState enum in SandboxLifecycle.cs. Only READY sandboxes
# are leasable.
STATE_BOOTING = "Booting"
STATE_RESETTING = "Resetting"
STATE_READY = "Ready"
STATE_LEASED = "Leased"

# How long a sandbox may go without a heartbeat before the coordinator drops it. The sim heartbeats
# every 5s from a Unity coroutine, so beats stop whenever the main thread stalls - and a fleet of
# sims all resetting at once on one machine can stall for tens of seconds without anything being
# wrong. Hence the generous margin: this is a liveness check for a sim that is never coming back,
# and a genuinely dead connection is caught much sooner by the websocket keepalive.
HEARTBEAT_TIMEOUT_SECONDS = 45.0

DEFAULT_COORDINATOR_PORT = 9000


def encode(message_type: str, **fields: Any) -> str:
    """Serialises one protocol frame."""
    payload: dict[str, Any] = {"type": message_type, "schema_version": SCHEMA_VERSION}
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False)


def decode(raw: str | bytes) -> dict[str, Any] | None:
    """Parses one protocol frame, returning None for anything that is not a JSON object.

    Malformed frames are never fatal: a sandbox that sends garbage should be ignored, not take
    down the pool for everyone else.
    """
    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return message if isinstance(message, dict) else None


def route_of(websocket: Any, path: str | None = None) -> str | None:
    """Reads the request path across websockets library versions.

    Modern ``websockets`` passes only the connection and exposes the path on ``request``; the
    legacy asyncio implementation passed it as a second handler argument.
    """
    if path is not None:
        return path
    request = getattr(websocket, "request", None)
    if request is not None:
        return getattr(request, "path", None)
    return getattr(websocket, "path", None)


def commands_uri(host: str, port: int) -> str:
    """The sandbox command endpoint an agent connects to for a leased sandbox."""
    return f"ws://{host}:{port}/commands"
