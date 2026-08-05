import asyncio
import json
import struct

import websockets

MAGIC = b"LDR1"
HEADER_FORMAT = "<4sHHffffId"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


async def _request_scan_async(uri):
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"command": "RequestLidarScan"}))
        return await ws.recv()


def parse_scan(payload: bytes) -> dict:
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"Payload too short for LiDAR header: {len(payload)} bytes")

    (
        magic,
        channels,
        azimuth_samples,
        min_range,
        max_range,
        azimuth_start_deg,
        azimuth_step_deg,
        sequence,
        timestamp_seconds,
    ) = struct.unpack_from(HEADER_FORMAT, payload, 0)

    if magic != MAGIC:
        raise ValueError(f"Unexpected LiDAR magic: {magic!r}")

    offset = HEADER_SIZE
    vertical_angles = list(struct.unpack_from(f"<{channels}f", payload, offset))
    offset += channels * 4
    range_count = channels * azimuth_samples
    ranges = list(struct.unpack_from(f"<{range_count}f", payload, offset))

    return {
        "channels": channels,
        "azimuth_samples": azimuth_samples,
        "min_range": min_range,
        "max_range": max_range,
        "azimuth_start_deg": azimuth_start_deg,
        "azimuth_step_deg": azimuth_step_deg,
        "sequence": sequence,
        "timestamp_seconds": timestamp_seconds,
        "vertical_angles_deg": vertical_angles,
        "ranges": ranges,
    }


def RequestLidarScan(uri: str = "ws://localhost:8080/commands") -> dict:
    payload = asyncio.get_event_loop().run_until_complete(_request_scan_async(uri))
    return parse_scan(payload)
