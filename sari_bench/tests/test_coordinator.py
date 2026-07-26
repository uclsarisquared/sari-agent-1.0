"""Coordinator tests: fake sandboxes and real bench clients over a real loopback websocket.

Everything here is in-process and offline - no sim, no model stack, no subprocess. The cases are
the race-y ones the pool exists to get right:

  1. acquire BLOCKS while the pool is empty, and is satisfied the moment a sandbox registers;
  2. release does NOT re-pool a sandbox - the reset it triggers does, once the sandbox says Ready;
  3. a sandbox that stops heartbeating mid-lease tells its holder, so the attempt can be requeued;
  4. a sandbox evicted for a missed heartbeat is hung up on, so it rejoins instead of stranding;
  5. a worker that disconnects without releasing has its lease reaped and its sandbox reset;
  6. two workers never hold the same sandbox at once.

    python sari_bench/tests/test_coordinator.py
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import websockets

from sari_bench.client import CoordinatorClient
from sari_bench.coordinator import Coordinator
from sari_bench.protocol import SANDBOX_ROUTE, STATE_LEASED, STATE_READY, decode, encode


class FakeSandbox:
    """A sim, minus Unity. Registers, heartbeats, and resets when told to."""

    def __init__(self, sandbox_id: str, port: int, *, auto_ready: bool = True) -> None:
        self.sandbox_id = sandbox_id
        self.port = port
        self.auto_ready = auto_ready
        self.socket: object = None
        self.reset_count = 0
        self.leases: list[str] = []
        self.reset_requested = asyncio.Event()
        # The real sim reconnects off its socket's on-close, so a test that cares about recovery
        # has to be able to see the hang-up the same way the sim would.
        self.hung_up_on = asyncio.Event()
        self._reader: asyncio.Task[None] | None = None
        self._heartbeat: asyncio.Task[None] | None = None

    async def connect(
        self,
        url: str,
        *,
        heartbeat_interval: float = 0.2,
        state: str = STATE_READY,
    ) -> None:
        await self._cancel_tasks()
        self.hung_up_on.clear()
        self.socket = await websockets.connect(f"{url}{SANDBOX_ROUTE}")
        await self.socket.send(
            encode(
                "sandbox.hello",
                sandbox_id=self.sandbox_id,
                port=self.port,
                state=state,
                store_loaded=True,
                v1_compatibility=True,
            )
        )
        self._reader = asyncio.create_task(self._read())
        self._heartbeat = asyncio.create_task(self._beat(heartbeat_interval))

    async def _read(self) -> None:
        socket = self.socket
        try:
            async for raw in socket:
                message = decode(raw) or {}
                if message.get("type") == "coord.lease":
                    self.leases.append(str(message.get("lease_id")))
                elif message.get("type") == "coord.reset":
                    self.reset_count += 1
                    self.reset_requested.set()
                    if self.auto_ready:
                        await self.report_ready()
        except websockets.ConnectionClosed:
            pass
        if socket is self.socket:
            self.hung_up_on.set()

    async def _beat(self, interval: float) -> None:
        socket = self.socket
        while True:
            await asyncio.sleep(interval)
            try:
                await socket.send(encode("sandbox.heartbeat", sandbox_id=self.sandbox_id))
            except websockets.ConnectionClosed:
                return

    async def _cancel_tasks(self) -> None:
        for task in (self._reader, self._heartbeat):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._reader = self._heartbeat = None

    async def report_ready(self) -> None:
        await self.socket.send(
            encode("sandbox.state", sandbox_id=self.sandbox_id, state=STATE_READY, store_loaded=True)
        )

    async def close(self) -> None:
        await self._cancel_tasks()
        if self.socket is not None:
            await self.socket.close()


async def _start_coordinator(**kwargs) -> tuple[Coordinator, str]:
    coordinator = Coordinator(host="127.0.0.1", port=0, log=lambda _message: None, **kwargs)
    await coordinator.start()
    return coordinator, f"ws://127.0.0.1:{coordinator.bound_port}"


async def test_acquire_blocks_until_a_sandbox_registers() -> None:
    coordinator, url = await _start_coordinator()
    try:
        async with CoordinatorClient(url) as client:
            acquire = asyncio.create_task(client.acquire())
            await asyncio.sleep(0.2)
            assert not acquire.done(), "acquire returned with an empty pool"

            sandbox = FakeSandbox("sandbox-a", 51923)
            await sandbox.connect(url)
            lease = await asyncio.wait_for(acquire, timeout=2)

            assert lease.sandbox_id == "sandbox-a"
            assert lease.port == 51923
            assert lease.commands_uri.endswith(":51923/commands")
            assert lease.host in {"127.0.0.1", "::1"}, lease.host
            await sandbox.close()
    finally:
        await coordinator.stop()
    print("ok  acquire blocks on an empty pool and is satisfied on registration")


async def test_acquire_fails_if_coordinator_connection_closes() -> None:
    coordinator, url = await _start_coordinator()
    client = CoordinatorClient(url)
    try:
        await client.connect()
        acquire = asyncio.create_task(client.acquire())
        await asyncio.sleep(0.05)
        # Abort rather than completing a polite close handshake: this is the half-open/drop case
        # that used to strand both the client request and the coordinator waiter.
        client._socket.transport.abort()
        try:
            await asyncio.wait_for(acquire, timeout=1)
        except ConnectionError as error:
            assert "bench.lease" in str(error), error
        else:
            raise AssertionError("acquire remained parked after its coordinator connection closed")
    finally:
        await client.close()
        await coordinator.stop()
    print("ok  a dropped coordinator connection wakes a parked acquire")


async def test_orphaned_lease_is_reset_when_sandbox_registers() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-orphan", 51923)
    try:
        # This is what the sim reports when the coordinator restarted during an attempt: Unity
        # remembers Leased, while the new coordinator necessarily has an empty in-memory lease map.
        await sandbox.connect(url, state=STATE_LEASED)
        await asyncio.wait_for(sandbox.reset_requested.wait(), timeout=2)

        async with CoordinatorClient(url) as client:
            lease = await asyncio.wait_for(client.acquire(), timeout=2)
            assert lease.sandbox_id == "sandbox-orphan"
            assert sandbox.reset_count == 1
    finally:
        await sandbox.close()
        await coordinator.stop()
    print("ok  an orphaned sim-side lease is reset and returned to the pool")


async def test_release_resets_before_repooling() -> None:
    """The point of reset-on-release: the next attempt cannot be handed a dirty environment."""
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51923, auto_ready=False)
    try:
        await sandbox.connect(url)
        async with CoordinatorClient(url) as first:
            lease = await asyncio.wait_for(first.acquire(), timeout=2)
            await first.release(lease, outcome="completed")
            await asyncio.wait_for(sandbox.reset_requested.wait(), timeout=2)
            assert sandbox.reset_count == 1

            async with CoordinatorClient(url) as second:
                pending = asyncio.create_task(second.acquire())
                await asyncio.sleep(0.2)
                assert not pending.done(), "a mid-reset sandbox was handed to the next worker"

                await sandbox.report_ready()
                second_lease = await asyncio.wait_for(pending, timeout=2)
                assert second_lease.sandbox_id == "sandbox-a"
                assert second_lease.lease_id != lease.lease_id
    finally:
        await sandbox.close()
        await coordinator.stop()
    print("ok  release resets first and only re-pools once the sandbox reports ready")


async def test_fleet_resets_are_serialized() -> None:
    coordinator, url = await _start_coordinator()
    sandboxes = [
        FakeSandbox("sandbox-a", 51923, auto_ready=False),
        FakeSandbox("sandbox-b", 51924, auto_ready=False),
    ]
    try:
        for sandbox in sandboxes:
            await sandbox.connect(url)
        async with CoordinatorClient(url) as first, CoordinatorClient(url) as second:
            first_lease = await asyncio.wait_for(first.acquire(), timeout=2)
            second_lease = await asyncio.wait_for(second.acquire(), timeout=2)
            await first.release(first_lease, outcome="completed")
            await second.release(second_lease, outcome="completed")

            await asyncio.sleep(0.1)
            assert sum(sandbox.reset_count for sandbox in sandboxes) == 1
            active = next(sandbox for sandbox in sandboxes if sandbox.reset_count)
            queued = next(sandbox for sandbox in sandboxes if not sandbox.reset_count)

            await active.report_ready()
            await asyncio.wait_for(queued.reset_requested.wait(), timeout=2)
            assert sum(sandbox.reset_count for sandbox in sandboxes) == 2
            await queued.report_ready()
    finally:
        for sandbox in sandboxes:
            await sandbox.close()
        await coordinator.stop()
    print("ok  fleet resets run one at a time instead of stampeding Unity")


async def test_stuck_reset_is_quarantined() -> None:
    coordinator, url = await _start_coordinator(reset_timeout=0.05)
    sandbox = FakeSandbox("sandbox-stuck", 51923, auto_ready=False)
    try:
        await sandbox.connect(url)
        async with CoordinatorClient(url) as client:
            lease = await asyncio.wait_for(client.acquire(), timeout=2)
            await client.release(lease, outcome="completed")
        await asyncio.wait_for(sandbox.hung_up_on.wait(), timeout=3)
        assert not coordinator.pool_snapshot()
    finally:
        await sandbox.close()
        await coordinator.stop()
    print("ok  a reset past its deadline is disconnected and quarantined")


async def test_dead_sandbox_notifies_its_lease_holder() -> None:
    coordinator, url = await _start_coordinator(heartbeat_timeout=0.5)
    sandbox = FakeSandbox("sandbox-a", 51923)
    try:
        await sandbox.connect(url, heartbeat_interval=10.0)  # effectively no heartbeats
        async with CoordinatorClient(url) as client:
            lease = await asyncio.wait_for(client.acquire(), timeout=2)
            lost = await asyncio.wait_for(client.wait_for_sandbox_lost(lease), timeout=5)
            assert lost.sandbox_id == "sandbox-a"
            assert lost.reason == "heartbeat_timeout"
    finally:
        await sandbox.close()
        await coordinator.stop()
    print("ok  a sandbox that stops heartbeating mid-lease tells its holder")


async def test_evicted_sandbox_rejoins_the_pool() -> None:
    """A heartbeat timeout has to be survivable.

    The sim only registers from its socket's on-open and only reconnects when it sees a close, so an
    eviction that leaves the connection open strands it: it keeps heartbeating into a registry with
    no row for it, and nothing ever puts it back. That turned one fleet-wide reset hitch into every
    sandbox in the pool disappearing for good, so the hang-up is the fix being pinned here.
    """
    coordinator, url = await _start_coordinator(heartbeat_timeout=0.5)
    sandbox = FakeSandbox("sandbox-a", 51923)
    try:
        await sandbox.connect(url, heartbeat_interval=10.0)  # stalled, like a sim mid-reset
        await asyncio.wait_for(sandbox.hung_up_on.wait(), timeout=5)
        assert coordinator.pool_snapshot() == [], "an evicted sandbox is still in the pool"

        await sandbox.connect(url)  # what the sim's reconnect backoff does next
        async with CoordinatorClient(url) as client:
            lease = await asyncio.wait_for(client.acquire(), timeout=3)
            assert lease.sandbox_id == "sandbox-a"
    finally:
        await sandbox.close()
        await coordinator.stop()
    print("ok  a sandbox evicted for a missed heartbeat is hung up on and rejoins on reconnect")


async def test_worker_disconnect_reaps_its_lease() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51923)
    try:
        await sandbox.connect(url)
        crashed = CoordinatorClient(url)
        await crashed.connect()
        await asyncio.wait_for(crashed.acquire(), timeout=2)
        await crashed.close()  # a worker process dying without releasing

        await asyncio.wait_for(sandbox.reset_requested.wait(), timeout=3)
        async with CoordinatorClient(url) as client:
            lease = await asyncio.wait_for(client.acquire(), timeout=3)
            assert lease.sandbox_id == "sandbox-a"
    finally:
        await sandbox.close()
        await coordinator.stop()
    print("ok  a worker that dies has its lease reaped and its sandbox reset")


async def test_one_sandbox_is_never_leased_twice() -> None:
    coordinator, url = await _start_coordinator()
    sandbox = FakeSandbox("sandbox-a", 51923)
    try:
        await sandbox.connect(url)
        async with CoordinatorClient(url) as first, CoordinatorClient(url) as second:
            first_wait = asyncio.create_task(first.acquire())
            second_wait = asyncio.create_task(second.acquire())
            done, pending = await asyncio.wait(
                {first_wait, second_wait}, timeout=2, return_when=asyncio.FIRST_COMPLETED
            )
            assert len(done) == 1, "both workers were handed the only sandbox"

            winner = done.pop()
            loser = pending.pop()
            await (first if winner is first_wait else second).release(
                winner.result(), outcome="completed"
            )
            loser_lease = await asyncio.wait_for(loser, timeout=3)
            assert loser_lease.lease_id != winner.result().lease_id
    finally:
        await sandbox.close()
        await coordinator.stop()
    print("ok  a single sandbox is only ever held by one worker at a time")


async def main() -> int:
    for test in (
        test_acquire_blocks_until_a_sandbox_registers,
        test_acquire_fails_if_coordinator_connection_closes,
        test_orphaned_lease_is_reset_when_sandbox_registers,
        test_release_resets_before_repooling,
        test_fleet_resets_are_serialized,
        test_stuck_reset_is_quarantined,
        test_dead_sandbox_notifies_its_lease_holder,
        test_evicted_sandbox_rejoins_the_pool,
        test_worker_disconnect_reaps_its_lease,
        test_one_sandbox_is_never_leased_twice,
    ):
        await test()
    print("\nAll coordinator tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
