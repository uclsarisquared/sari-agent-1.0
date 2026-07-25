"""Benchmark-worker side of the coordinator protocol.

One :class:`CoordinatorClient` per worker, one lease at a time. A background reader owns the socket
so the worker can notice ``bench.sandbox_lost`` *while* it is blocked waiting on an agent
subprocess - which is exactly when a sandbox is most likely to die, and the one moment a plain
request/response client would be deaf to it.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from sari_bench.protocol import BENCH_ROUTE, decode, encode


class SandboxLost(Exception):
    """Raised when the leased sandbox stops responding mid-attempt."""

    def __init__(self, sandbox_id: str, reason: str) -> None:
        super().__init__(f"sandbox {sandbox_id} lost: {reason}")
        self.sandbox_id = sandbox_id
        self.reason = reason


class Lease:
    """A leased sandbox and the address an agent should connect to."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.lease_id: str = payload["lease_id"]
        self.sandbox_id: str = payload["sandbox_id"]
        self.host: str = payload["host"]
        self.port: int = payload["port"]
        self.commands_uri: str = payload["commands_uri"]

    def __repr__(self) -> str:
        return f"Lease({self.sandbox_id} @ {self.host}:{self.port})"


class CoordinatorClient:
    """Async client for one benchmark worker."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        if not self.url.endswith(BENCH_ROUTE):
            self.url = f"{self.url}{BENCH_ROUTE}"

        self._socket: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._replies: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._lost: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = asyncio.Event()

    async def __aenter__(self) -> CoordinatorClient:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        import websockets

        self._socket = await websockets.connect(self.url)
        self._reader = asyncio.create_task(self._read_forever())

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None

        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def _read_forever(self) -> None:
        try:
            async for raw in self._socket:
                message = decode(raw)
                if message is None:
                    continue
                if message.get("type") == "bench.sandbox_lost":
                    await self._lost.put(message)
                else:
                    await self._replies.put(message)
        except Exception:  # noqa: BLE001 - surfaced to the worker as a closed connection
            pass
        finally:
            self._closed.set()

    async def _request(self, message: str, expect: str) -> dict[str, Any]:
        await self._socket.send(message)
        while True:
            reply = await self._replies.get()
            if reply.get("type") == expect:
                return reply
            if reply.get("type") == "bench.error":
                raise RuntimeError(f"Coordinator rejected request: {reply.get('reason')}")

    async def acquire(self) -> Lease:
        """Leases a sandbox, waiting for one to free up if the pool is busy."""
        return Lease(await self._request(encode("bench.acquire"), "bench.lease"))

    async def release(self, lease: Lease, outcome: str) -> None:
        """Hands the sandbox back. The coordinator resets it before anyone else gets it."""
        await self._request(
            encode("bench.release", lease_id=lease.lease_id, outcome=outcome),
            "bench.released",
        )

    async def pool(self) -> list[dict[str, Any]]:
        reply = await self._request(encode("bench.status"), "bench.pool")
        return list(reply.get("sandboxes") or [])

    async def wait_for_sandbox_lost(self, lease: Lease) -> SandboxLost:
        """Resolves only if *this* lease's sandbox dies. Race it against the attempt."""
        while True:
            message = await self._lost.get()
            if message.get("lease_id") == lease.lease_id:
                return SandboxLost(
                    str(message.get("sandbox_id") or lease.sandbox_id),
                    str(message.get("reason") or "unknown"),
                )
