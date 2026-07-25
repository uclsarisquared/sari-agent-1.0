"""Coordinator tests: fake sandboxes and real bench clients over a real loopback websocket.

Everything here is in-process and offline - no sim, no model stack, no subprocess. The cases are
the race-y ones the pool exists to get right:

  1. acquire BLOCKS while the pool is empty, and is satisfied the moment a sandbox registers;
  2. release does NOT re-pool a sandbox - the reset it triggers does, once the sandbox says Ready;
  3. a sandbox that stops heartbeating mid-lease tells its holder, so the attempt can be requeued;
  4. a worker that disconnects without releasing has its lease reaped and its sandbox reset;
  5. two workers never hold the same sandbox at once.

    python sari_bench/tests/test_coordinator.py
"""

from __future__ import annotations

import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import websockets

from sari_bench.client import CoordinatorClient
from sari_bench.coordinator import Coordinator
from sari_bench.protocol import SANDBOX_ROUTE, STATE_READY, decode, encode


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
        self._reader: asyncio.Task[None] | None = None
        self._heartbeat: asyncio.Task[None] | None = None

    async def connect(self, url: str, *, heartbeat_interval: float = 0.2) -> None:
        self.socket = await websockets.connect(f"{url}{SANDBOX_ROUTE}")
        await self.socket.send(
            encode(
                "sandbox.hello",
                sandbox_id=self.sandbox_id,
                port=self.port,
                state=STATE_READY,
                store_loaded=True,
                v1_compatibility=True,
            )
        )
        self._reader = asyncio.create_task(self._read())
        self._heartbeat = asyncio.create_task(self._beat(heartbeat_interval))

    async def _read(self) -> None:
        async for raw in self.socket:
            message = decode(raw) or {}
            if message.get("type") == "coord.lease":
                self.leases.append(str(message.get("lease_id")))
            elif message.get("type") == "coord.reset":
                self.reset_count += 1
                self.reset_requested.set()
                if self.auto_ready:
                    await self.report_ready()

    async def _beat(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            await self.socket.send(encode("sandbox.heartbeat", sandbox_id=self.sandbox_id))

    async def report_ready(self) -> None:
        await self.socket.send(
            encode("sandbox.state", sandbox_id=self.sandbox_id, state=STATE_READY, store_loaded=True)
        )

    async def close(self) -> None:
        for task in (self._reader, self._heartbeat):
            if task is not None:
                task.cancel()
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
        test_release_resets_before_repooling,
        test_dead_sandbox_notifies_its_lease_holder,
        test_worker_disconnect_reaps_its_lease,
        test_one_sandbox_is_never_leased_twice,
    ):
        await test()
    print("\nAll coordinator tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
