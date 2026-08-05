"""Low-overhead supplementary frames for live watching and dense replays.

The agent already saves a frame at the start of every reasoning step.  This recorder only fills
gaps between those frames, so a fast-moving attempt does not pay for redundant screenshots while a
slow model call still leaves the dashboard with a recent view.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

CAPTURE_DIR = "capture"
LATEST_CAPTURE = "latest.json"
CAPTURE_WIDTH = 960
CAPTURE_JPEG_QUALITY = 75
CAPTURE_TIMEOUT_SECONDS = 10.0
# Four observations per second keeps motion legible in the finished-run replay without turning
# every simulator response into a full-resolution artifact.
DEFAULT_INTERVAL_SECONDS = 0.25

_CAPTURE_FRAME = re.compile(r"^frame(\d+)-(\d+)\.jpg$")
_STEP_FRAME = re.compile(r"^step\d+\.png$")


@dataclass
class CaptureStats:
    frames: int = 0
    failures: int = 0


def frame_timestamp_ns(path: Path) -> int:
    """Capture time for a supplementary frame, filesystem publication time otherwise."""
    match = _CAPTURE_FRAME.match(path.name) if path.parent.name == CAPTURE_DIR else None
    if match:
        return int(match.group(2))
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def observation_frames(run_dir: Path) -> list[Path]:
    """All agent step frames and supplementary captures under one attempt."""
    frames: list[Path] = []
    if not run_dir.is_dir():
        return frames

    capture_dir = run_dir / CAPTURE_DIR
    if capture_dir.is_dir():
        frames.extend(
            path for path in capture_dir.iterdir()
            if path.is_file() and _CAPTURE_FRAME.match(path.name)
        )

    for leg_dir in run_dir.iterdir():
        if not leg_dir.is_dir() or not leg_dir.name.startswith("leg"):
            continue
        frames.extend(
            path for path in leg_dir.iterdir()
            if path.is_file() and _STEP_FRAME.match(path.name)
        )
    return frames


def latest_step_frame(run_dir: Path) -> Path | None:
    """Newest agent-owned step frame by publication time."""
    if not run_dir.is_dir():
        return None
    frames: list[Path] = []
    for leg_dir in run_dir.iterdir():
        if not leg_dir.is_dir() or not leg_dir.name.startswith("leg"):
            continue
        frames.extend(
            path for path in leg_dir.iterdir()
            if path.is_file() and _STEP_FRAME.match(path.name)
        )
    return max(frames, key=frame_timestamp_ns) if frames else None


def latest_capture(run_dir: Path) -> Path | None:
    """Newest supplementary frame, using the recorder's O(1) pointer when available."""
    capture_dir = run_dir / CAPTURE_DIR
    pointer = capture_dir / LATEST_CAPTURE
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        name = payload.get("file") if isinstance(payload, dict) else None
        if isinstance(name, str) and _CAPTURE_FRAME.match(name):
            path = capture_dir / name
            if path.is_file():
                return path
    except (OSError, ValueError):
        pass

    if not capture_dir.is_dir():
        return None
    frames = [
        path for path in capture_dir.iterdir()
        if path.is_file() and _CAPTURE_FRAME.match(path.name)
    ]
    return max(frames, key=frame_timestamp_ns) if frames else None


def latest_observation(run_dir: Path) -> Path | None:
    frames = [path for path in (latest_step_frame(run_dir), latest_capture(run_dir)) if path]
    return max(frames, key=frame_timestamp_ns) if frames else None


async def request_screenshot(commands_uri: str) -> bytes:
    """Fetch one native PNG without importing the agent-side ``agent/sim`` package."""
    import websockets

    async with websockets.connect(commands_uri, max_size=None) as websocket:
        await websocket.send(json.dumps({
            "command": "RequestScreenshot",
            "prefix": "",
            "suffix": "",
            "folder_name": "",
            "save_image": False,
        }))
        payload = await websocket.recv()
    if not isinstance(payload, bytes):
        raise TypeError(f"RequestScreenshot returned {type(payload).__name__}, expected bytes")
    return payload


def _save_jpeg(run_dir: Path, image_bytes: bytes, sequence: int, captured_ns: int) -> Path:
    """Downscale once and atomically publish a timestamped archive frame."""
    from PIL import Image

    capture_dir = run_dir / CAPTURE_DIR
    capture_dir.mkdir(parents=True, exist_ok=True)
    final = capture_dir / f"frame{sequence:06d}-{captured_ns}.jpg"

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        if image.width > CAPTURE_WIDTH:
            height = max(1, round(image.height * CAPTURE_WIDTH / image.width))
            image = image.resize((CAPTURE_WIDTH, height), Image.Resampling.LANCZOS)

        fd, temp_name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=capture_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                image.save(handle, format="JPEG", quality=CAPTURE_JPEG_QUALITY)
            os.replace(temp_name, final)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    pointer = capture_dir / LATEST_CAPTURE
    pointer_temp = pointer.with_name(f".{pointer.name}.tmp")
    pointer_temp.write_text(
        json.dumps({"file": final.name, "captured_ns": captured_ns}),
        encoding="utf-8",
    )
    os.replace(pointer_temp, pointer)
    return final


async def record_previews(
    run_dir: Path,
    commands_uri: str,
    interval: float,
    *,
    fetch: Callable[[str], Awaitable[bytes]] = request_screenshot,
    stats: CaptureStats | None = None,
    log: Callable[[str], None] | None = None,
) -> CaptureStats:
    """Fill observation gaps until cancelled.

    Scheduling is based on the newest frame, including the agent's own step frames.  A slow or
    failed capture starts a fresh interval rather than causing catch-up traffic.
    """
    stats = stats or CaptureStats()
    sequence = 1
    capture_dir = run_dir / CAPTURE_DIR
    if capture_dir.is_dir():
        existing = [
            int(match.group(1))
            for path in capture_dir.iterdir()
            if (match := _CAPTURE_FRAME.match(path.name))
        ]
        sequence = max(existing, default=0) + 1
    latest = latest_observation(run_dir)
    latest_ns = frame_timestamp_ns(latest) if latest is not None else 0

    while True:
        step = latest_step_frame(run_dir)
        if step is not None:
            latest_ns = max(latest_ns, frame_timestamp_ns(step))
        if latest_ns:
            age = max(0.0, (time.time_ns() - latest_ns) / 1e9)
            if age < interval:
                await asyncio.sleep(interval - age)
                continue

        fetch_task = asyncio.create_task(fetch(commands_uri))
        try:
            image_bytes = await asyncio.wait_for(fetch_task, CAPTURE_TIMEOUT_SECONDS)
            captured_ns = time.time_ns()
            await asyncio.to_thread(_save_jpeg, run_dir, image_bytes, sequence, captured_ns)
        except asyncio.CancelledError:
            # Python 3.10's wait_for() doesn't retrieve an exception raised by its child while
            # cancellation is in progress. websockets can hit such an exception when recv() is
            # cancelled between fragmented screenshot frames, which would otherwise produce a
            # large "Task exception was never retrieved" traceback during ordinary shutdown.
            outcome = await asyncio.gather(fetch_task, return_exceptions=True)
            error = outcome[0]
            if isinstance(error, Exception) and log is not None:
                detail = " ".join(str(error).splitlines()) or "no details"
                log(f"preview capture cleanup: {type(error).__name__}: {detail}")
            raise
        except Exception:  # A preview must never affect an attempt.
            stats.failures += 1
            await asyncio.sleep(interval)
            continue

        stats.frames += 1
        # Start the next interval after publication. If fetching or JPEG encoding was slower than
        # the requested cadence, measuring from acquisition would immediately start another fetch,
        # producing catch-up traffic and potentially starving the runner's event loop.
        latest_ns = time.time_ns()
        sequence += 1
