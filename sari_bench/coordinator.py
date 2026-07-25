"""The sandbox registry for Distributed Sari Bench.

Sims connect on ``/sandbox`` and advertise the command port they self-assigned; benchmark workers
connect on ``/bench`` and lease one at a time. The registry's whole job is to make sure a worker
never gets a sandbox that another worker is using, and never gets one that has not been reset since
its last attempt.

Two invariants drive the design:

* **Reset happens on release, not on acquire.** A sandbox handed back after an attempt is taken out
  of the pool, told to reset, and only re-pooled once *it* reports ready. So an acquire always
  yields a pristine environment, and a run that crashed mid-attempt still gets cleaned up - the
  cleanup does not depend on the agent process surviving to ask for it.
* **A lease outlives neither its worker nor its sandbox.** A worker that disconnects has its lease
  reaped; a sandbox that stops heartbeating has its lease holder told ``bench.sandbox_lost`` so the
  attempt can be requeued instead of hanging on a machine that is never coming back.

All state is in memory. A coordinator restart is recoverable without operator action: sims
reconnect on their own backoff, and workers reconnect per attempt.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sari_bench.protocol import (
    BENCH_ROUTE,
    DEFAULT_COORDINATOR_PORT,
    HEARTBEAT_TIMEOUT_SECONDS,
    SANDBOX_ROUTE,
    STATE_READY,
    commands_uri,
    decode,
    encode,
    route_of,
)

# How often the reaper sweeps for dead sandboxes and expired leases.
REAP_INTERVAL_SECONDS = 2.0

# Backstop for a worker whose TCP connection is half-open, so its lease is never held forever.
# Generous by default: a legitimate attempt can legitimately run for tens of minutes.
DEFAULT_LEASE_TTL_SECONDS = 7200.0


@dataclass
class Sandbox:
    sandbox_id: str
    host: str
    port: int
    websocket: Any
    state: str
    store_loaded: bool = True
    v1_compatibility: bool = False
    unity_version: str = ""
    store_name: str = ""
    last_heartbeat: float = field(default_factory=time.monotonic)
    lease_id: str | None = None

    @property
    def leasable(self) -> bool:
        """Ready, idle, and actually serving a store."""
        return self.lease_id is None and self.state == STATE_READY and self.store_loaded

    def describe(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "host": self.host,
            "port": self.port,
            "state": self.state,
            "store_loaded": self.store_loaded,
            "v1_compatibility": self.v1_compatibility,
            "unity_version": self.unity_version,
            "store_name": self.store_name,
            "lease_id": self.lease_id,
            "commands_uri": commands_uri(self.host, self.port),
        }


@dataclass
class Lease:
    lease_id: str
    sandbox_id: str
    holder: Any
    created_at: float = field(default_factory=time.monotonic)


class Coordinator:
    """Sandbox pool plus lease bookkeeping. One instance per coordinator process."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = DEFAULT_COORDINATOR_PORT,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT_SECONDS,
        lease_ttl: float = DEFAULT_LEASE_TTL_SECONDS,
        log: Any = None,
    ) -> None:
        self.host = host
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self.lease_ttl = lease_ttl
        self.log = log or _print_log

        self._sandboxes: dict[str, Sandbox] = {}
        self._leases: dict[str, Lease] = {}
        # Workers waiting for a free sandbox, oldest first, so leases are handed out fairly.
        self._waiters: list[asyncio.Future[Lease]] = []
        self._server: Any = None
        self._reaper: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        import websockets

        self._server = await websockets.serve(self._handle_connection, self.host, self.port)
        self._reaper = asyncio.create_task(self._reap_forever())
        self.log(f"Coordinator listening on ws://{self.host}:{self.port}{SANDBOX_ROUTE} and {BENCH_ROUTE}")

    async def stop(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for waiter in self._waiters:
            if not waiter.done():
                waiter.cancel()
        self._waiters.clear()

    @property
    def bound_port(self) -> int:
        """The port actually bound, which differs from ``port`` when 0 was requested."""
        if self._server is None:
            return self.port
        for socket in getattr(self._server, "sockets", []) or []:
            return socket.getsockname()[1]
        return self.port

    # ------------------------------------------------------------------ routing

    async def _handle_connection(self, websocket: Any, path: str | None = None) -> None:
        route = route_of(websocket, path)
        if route == SANDBOX_ROUTE:
            await self._handle_sandbox(websocket)
        elif route == BENCH_ROUTE:
            await self._handle_bench(websocket)
        else:
            await websocket.close(code=1008, reason=f"Use {SANDBOX_ROUTE} or {BENCH_ROUTE}")

    # ------------------------------------------------------------------ sandbox side

    async def _handle_sandbox(self, websocket: Any) -> None:
        sandbox_id: str | None = None
        try:
            async for raw in websocket:
                message = decode(raw)
                if message is None:
                    continue

                message_type = message.get("type")
                if message_type == "sandbox.hello":
                    sandbox_id = await self._register(websocket, message)
                elif sandbox_id is None:
                    # Everything else is meaningless before we know who is speaking.
                    continue
                elif message_type == "sandbox.heartbeat":
                    sandbox = self._sandboxes.get(sandbox_id)
                    if sandbox is not None:
                        sandbox.last_heartbeat = time.monotonic()
                elif message_type == "sandbox.state":
                    await self._on_sandbox_state(sandbox_id, message)
        except Exception as error:  # noqa: BLE001 - a dropped sim must not kill the coordinator
            self.log(f"Sandbox connection error ({sandbox_id}): {error!r}")
        finally:
            if sandbox_id is not None:
                await self._drop_sandbox(sandbox_id, reason="disconnected")

    async def _register(self, websocket: Any, message: dict[str, Any]) -> str | None:
        sandbox_id = message.get("sandbox_id")
        port = message.get("port")
        if not isinstance(sandbox_id, str) or not sandbox_id or not isinstance(port, int):
            await _send(websocket, encode("coord.error", reason="invalid_hello"))
            return None

        # The sim does not have to know its own reachable address: whatever address it reached us
        # from is, by construction, an address that works. advertised_host overrides that for the
        # cases where it does not - NAT, or a container with a published port.
        advertised = message.get("advertised_host") or ""
        host = advertised or _peer_host(websocket) or "127.0.0.1"

        # A sim that restarted re-registers under the same id; retire the stale entry (and any
        # lease pointing at it) rather than ending up with two rows for one machine.
        if sandbox_id in self._sandboxes:
            await self._drop_sandbox(sandbox_id, reason="re-registered")

        sandbox = Sandbox(
            sandbox_id=sandbox_id,
            host=host,
            port=port,
            websocket=websocket,
            state=str(message.get("state") or STATE_READY),
            store_loaded=bool(message.get("store_loaded", True)),
            v1_compatibility=bool(message.get("v1_compatibility", False)),
            unity_version=str(message.get("unity_version") or ""),
            store_name=str(message.get("store_name") or ""),
        )
        self._sandboxes[sandbox_id] = sandbox

        if not sandbox.store_loaded:
            self.log(
                f"Sandbox {sandbox_id} at {sandbox.host}:{sandbox.port} registered WITHOUT a store "
                "file; it will not be leased until it reports one."
            )
        else:
            self.log(f"Sandbox {sandbox_id} registered at {sandbox.host}:{sandbox.port} ({sandbox.state})")

        await _send(websocket, encode("coord.welcome", sandbox_id=sandbox_id))
        self._fulfil_waiters()
        return sandbox_id

    async def _on_sandbox_state(self, sandbox_id: str, message: dict[str, Any]) -> None:
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            return

        sandbox.state = str(message.get("state") or sandbox.state)
        sandbox.store_loaded = bool(message.get("store_loaded", sandbox.store_loaded))
        sandbox.last_heartbeat = time.monotonic()

        # A sandbox reporting Ready after a release is what actually returns it to the pool: the
        # reset it was told to run has finished.
        if sandbox.state == STATE_READY and sandbox.lease_id is None:
            self._fulfil_waiters()

    async def _drop_sandbox(self, sandbox_id: str, *, reason: str) -> None:
        sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox is None:
            return

        self.log(f"Sandbox {sandbox_id} removed ({reason})")
        if sandbox.lease_id is None:
            return

        # Tell the worker its machine vanished so it can requeue the attempt rather than sit on a
        # subprocess that will never talk to anything again.
        lease = self._leases.pop(sandbox.lease_id, None)
        if lease is not None:
            await _send(
                lease.holder,
                encode(
                    "bench.sandbox_lost",
                    lease_id=lease.lease_id,
                    sandbox_id=sandbox_id,
                    reason=reason,
                ),
            )

    # ------------------------------------------------------------------ bench side

    async def _handle_bench(self, websocket: Any) -> None:
        """One worker, at most one lease. Requests are handled in order on this connection."""
        try:
            async for raw in websocket:
                message = decode(raw)
                if message is None:
                    continue

                message_type = message.get("type")
                if message_type == "bench.acquire":
                    await self._handle_acquire(websocket)
                elif message_type == "bench.release":
                    await self._handle_release(websocket, message)
                elif message_type == "bench.status":
                    await _send(websocket, encode("bench.pool", sandboxes=self.pool_snapshot()))
                else:
                    await _send(
                        websocket,
                        encode("bench.error", reason="unknown_message", message_type=message_type),
                    )
        except Exception as error:  # noqa: BLE001 - one worker must not kill the coordinator
            self.log(f"Bench connection error: {error!r}")
        finally:
            await self._reap_leases_of(websocket, reason="worker_disconnected")
            self._drop_waiters_of(websocket)

    async def _handle_acquire(self, websocket: Any) -> None:
        sandbox = self._take_leasable()
        if sandbox is not None:
            lease = self._claim(sandbox, websocket)
        else:
            # No sandbox free: park until one is. The worker is blocked on our reply, which is
            # exactly the backpressure we want - it stops the fleet running ahead of itself.
            future: asyncio.Future[Lease] = asyncio.get_running_loop().create_future()
            setattr(future, "_bench_holder", websocket)
            self._waiters.append(future)
            try:
                lease = await future
            except asyncio.CancelledError:
                return
            sandbox = self._sandboxes.get(lease.sandbox_id)
            if sandbox is None:
                # The sandbox died between being handed to us and us resuming; _drop_sandbox has
                # already told this worker, so there is nothing left to reply.
                return

        await _send(sandbox.websocket, encode("coord.lease", lease_id=lease.lease_id))
        await _send(
            websocket,
            encode(
                "bench.lease",
                lease_id=lease.lease_id,
                sandbox_id=sandbox.sandbox_id,
                host=sandbox.host,
                port=sandbox.port,
                commands_uri=commands_uri(sandbox.host, sandbox.port),
            ),
        )
        self.log(f"Leased {sandbox.sandbox_id} ({sandbox.host}:{sandbox.port}) as {lease.lease_id}")

    async def _handle_release(self, websocket: Any, message: dict[str, Any]) -> None:
        lease_id = message.get("lease_id")
        lease = self._leases.pop(lease_id, None) if isinstance(lease_id, str) else None
        if lease is None:
            # Releasing an unknown lease is a no-op, not an error: a worker that already had its
            # lease reaped (sandbox died, TTL expired) still runs its release in a finally block.
            await _send(websocket, encode("bench.released", lease_id=lease_id, known=False))
            return

        outcome = message.get("outcome") or "unknown"
        await self._reset_and_repool(lease.sandbox_id, lease.lease_id, outcome)
        await _send(websocket, encode("bench.released", lease_id=lease.lease_id, known=True))

    async def _reset_and_repool(self, sandbox_id: str, lease_id: str, outcome: str) -> None:
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is None:
            return

        sandbox.lease_id = None
        # Deliberately NOT marked ready here. The sandbox re-enters the pool only when it sends
        # sandbox.state{Ready}, i.e. once its reset has actually settled. Marking it ready now
        # would hand a mid-reset environment to the next attempt.
        sandbox.state = "Resetting"
        self.log(f"Released {sandbox_id} (lease {lease_id}, outcome={outcome}); resetting")
        await _send(sandbox.websocket, encode("coord.reset", lease_id=lease_id))

    # ------------------------------------------------------------------ pool helpers

    def _take_leasable(self) -> Sandbox | None:
        for sandbox in self._sandboxes.values():
            if sandbox.leasable:
                return sandbox
        return None

    def _claim(self, sandbox: Sandbox, holder: Any) -> Lease:
        """Marks a sandbox as taken. Must happen synchronously with picking it, so two workers
        cannot both be handed the same sandbox before either has resumed."""
        lease = Lease(lease_id=uuid.uuid4().hex, sandbox_id=sandbox.sandbox_id, holder=holder)
        self._leases[lease.lease_id] = lease
        sandbox.lease_id = lease.lease_id
        return lease

    def _fulfil_waiters(self) -> None:
        """Hands free sandboxes to parked workers, oldest waiter first."""
        while self._waiters:
            # Discard waiters whose worker went away while queued.
            if self._waiters[0].done():
                self._waiters.pop(0)
                continue

            sandbox = self._take_leasable()
            if sandbox is None:
                return

            future = self._waiters.pop(0)
            future.set_result(self._claim(sandbox, getattr(future, "_bench_holder", None)))

    def _drop_waiters_of(self, websocket: Any) -> None:
        remaining: list[asyncio.Future[Lease]] = []
        for future in self._waiters:
            if getattr(future, "_bench_holder", None) is websocket and not future.done():
                future.cancel()
            else:
                remaining.append(future)
        self._waiters = remaining

    async def _reap_leases_of(self, websocket: Any, *, reason: str) -> None:
        for lease in [lease for lease in self._leases.values() if lease.holder is websocket]:
            self._leases.pop(lease.lease_id, None)
            self.log(f"Reaping lease {lease.lease_id} ({reason})")
            await self._reset_and_repool(lease.sandbox_id, lease.lease_id, reason)

    def pool_snapshot(self) -> list[dict[str, Any]]:
        return [sandbox.describe() for sandbox in self._sandboxes.values()]

    # ------------------------------------------------------------------ reaper

    async def _reap_forever(self) -> None:
        while True:
            await asyncio.sleep(REAP_INTERVAL_SECONDS)
            try:
                await self._reap_once()
            except Exception as error:  # noqa: BLE001 - the reaper must never die
                self.log(f"Reaper error: {error!r}")

    async def _reap_once(self) -> None:
        now = time.monotonic()

        for sandbox_id, sandbox in list(self._sandboxes.items()):
            if now - sandbox.last_heartbeat > self.heartbeat_timeout:
                await self._drop_sandbox(sandbox_id, reason="heartbeat_timeout")

        # TTL only catches half-open worker connections; a clean disconnect is reaped immediately
        # in _handle_bench's finally block.
        for lease in list(self._leases.values()):
            if now - lease.created_at > self.lease_ttl:
                self._leases.pop(lease.lease_id, None)
                self.log(f"Lease {lease.lease_id} exceeded its TTL; reclaiming its sandbox")
                await self._reset_and_repool(lease.sandbox_id, lease.lease_id, "lease_ttl")

        self._fulfil_waiters()


async def _send(websocket: Any, payload: str) -> None:
    """Best-effort send: a peer that has gone away is handled by its own connection teardown."""
    try:
        await websocket.send(payload)
    except Exception:  # noqa: BLE001
        return


def _peer_host(websocket: Any) -> str | None:
    address = getattr(websocket, "remote_address", None)
    if isinstance(address, tuple) and address:
        return str(address[0])
    return None


def _print_log(message: str) -> None:
    print(f"[coordinator] {message}", flush=True)


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distributed Sari Bench sandbox registry.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces).")
    parser.add_argument("--port", type=int, default=DEFAULT_COORDINATOR_PORT)
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=HEARTBEAT_TIMEOUT_SECONDS,
        help="Seconds without a heartbeat before a sandbox is dropped from the pool.",
    )
    parser.add_argument(
        "--lease-ttl",
        type=float,
        default=DEFAULT_LEASE_TTL_SECONDS,
        help="Seconds before an unreleased lease is reclaimed.",
    )
    args = parser.parse_args(argv)

    coordinator = Coordinator(
        host=args.host,
        port=args.port,
        heartbeat_timeout=args.heartbeat_timeout,
        lease_ttl=args.lease_ttl,
    )
    await coordinator.start()
    try:
        await asyncio.Future()  # run until interrupted
    except asyncio.CancelledError:
        pass
    finally:
        await coordinator.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
