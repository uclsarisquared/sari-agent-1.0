"""Supplementary capture, live-frame selection, and dense replay tests."""

from __future__ import annotations

import asyncio
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PIL import Image

from sari_bench import capture, video
from sari_bench.watch import scan


async def _inline_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Keep recorder tests deterministic without leaving Python 3.10's executor mid-shutdown."""
    return function(*args, **kwargs)


def test_default_capture_rate_is_four_frames_per_second() -> None:
    assert capture.DEFAULT_INTERVAL_SECONDS == 0.25
    assert 1 / capture.DEFAULT_INTERVAL_SECONDS == 4
    assert video.DEFAULT_FPS == 4.0
    print("ok  capture and full replay default to four frames per second")


def _png(width: int = 1600, height: int = 900, color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), color).save(output, format="PNG")
    return output.getvalue()


async def test_recorder_fills_gaps_and_publishes_small_jpegs() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run_dir = Path(temp) / "attempt"
        run_dir.mkdir()
        calls = 0

        async def fetch(_uri: str) -> bytes:
            nonlocal calls
            calls += 1
            return _png()

        stats = capture.CaptureStats()
        task = asyncio.create_task(
            capture.record_previews(run_dir, "ws://unused", 0.05, fetch=fetch, stats=stats)
        )
        try:
            for _ in range(100):
                if stats.frames >= 2:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("recorder did not publish two frames")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        frames = sorted((run_dir / capture.CAPTURE_DIR).glob("*.jpg"))
        assert len(frames) == stats.frames
        assert calls in {stats.frames, stats.frames + 1}
        with Image.open(frames[-1]) as image:
            assert image.size == (960, 540), image.size
            assert image.format == "JPEG"
    print("ok  recorder fills gaps with atomic, downscaled JPEGs")


async def test_recent_agent_frame_suppresses_a_redundant_capture() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run_dir = Path(temp) / "attempt"
        step = run_dir / "leg00" / "step01.png"
        step.parent.mkdir(parents=True)
        step.write_bytes(_png(32, 18))
        calls = 0

        async def fetch(_uri: str) -> bytes:
            nonlocal calls
            calls += 1
            return _png(32, 18)

        task = asyncio.create_task(
            capture.record_previews(run_dir, "ws://unused", 0.2, fetch=fetch)
        )
        try:
            await asyncio.sleep(0.06)
            assert calls == 0, "a fresh step frame did not suppress capture"
            for _ in range(50):
                if calls:
                    break
                await asyncio.sleep(0.01)
            assert calls == 1
            # Do not cancel while the JPEG publication is still inside asyncio.to_thread(); Python
            # 3.10 can otherwise wait indefinitely while shutting down its default executor.
            for _ in range(100):
                if capture.latest_capture(run_dir) is not None:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("capture request never finished publishing")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    print("ok  recent agent frames suppress redundant simulator requests")


def test_watch_and_replay_merge_supplementary_frames() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run_dir = Path(temp) / "battery" / "p" / "try01"
        step = run_dir / "leg00" / "step01.png"
        step.parent.mkdir(parents=True)
        step.write_bytes(_png(32, 18, (100, 20, 20)))
        old_ns = time.time_ns() - 2_000_000_000
        os.utime(step, ns=(old_ns, old_ns))
        (run_dir / "leg00.jsonl").write_text(
            json.dumps({"event": "step", "step": 1, "mode": "navigation"}) + "\n",
            encoding="utf-8",
        )
        (run_dir / scan.ATTEMPT_MANIFEST).write_text(
            json.dumps({"started_epoch": (old_ns - 1_000_000_000) / 1e9}),
            encoding="utf-8",
        )

        capture_dir = run_dir / capture.CAPTURE_DIR
        capture_dir.mkdir()
        captured_ns = old_ns + 1_000_000_000
        preview = capture_dir / f"frame000001-{captured_ns}.jpg"
        Image.new("RGB", (32, 18), (20, 100, 20)).save(preview, format="JPEG")

        assert scan._latest_frame(run_dir) == preview
        full = video.collect_frames(run_dir)
        upload = video.collect_frames(run_dir, include_captures=False)
        assert [path for path, _ in full] == [step, preview], full
        assert "live observation" in full[1][1]
        assert [path for path, _ in upload] == [step]
    print("ok  watch and full replay include captures; upload replay remains step-only")


def test_replay_frame_cap_preserves_steps_and_samples_captures() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run_dir = Path(temp) / "attempt"
        capture_dir = run_dir / capture.CAPTURE_DIR
        leg_dir = run_dir / "leg00"
        capture_dir.mkdir(parents=True)
        leg_dir.mkdir()
        base_ns = time.time_ns() - 20_000_000_000
        for sequence in range(10):
            timestamp_ns = base_ns + sequence * 1_000_000_000
            path = capture_dir / f"frame{sequence + 1:06d}-{timestamp_ns}.jpg"
            Image.new("RGB", (32, 18), (sequence, 20, 30)).save(path, format="JPEG")
        for step, offset in ((1, 2), (2, 7)):
            path = leg_dir / f"step{step:02d}.png"
            path.write_bytes(_png(32, 18))
            timestamp_ns = base_ns + offset * 1_000_000_000
            os.utime(path, ns=(timestamp_ns, timestamp_ns))
        (run_dir / "leg00.jsonl").write_text("", encoding="utf-8")
        (run_dir / scan.ATTEMPT_MANIFEST).write_text(
            json.dumps({"started_epoch": base_ns / 1e9}),
            encoding="utf-8",
        )

        all_frames = video.collect_frames(run_dir)
        limited = video.limit_replay_frames(all_frames, 5)
        paths = [path for path, _ in limited]
        assert len(paths) == 5, paths
        assert all(path in paths for path in leg_dir.glob("step*.png"))
        captures = [path for path in paths if path.parent == capture_dir]
        assert len(captures) == 3
        assert captures[0].name.startswith("frame000001-")
        assert captures[-1].name.startswith("frame000010-")
    print("ok  replay caps uniformly sample captures without dropping step frames")


def _minimal_mp4() -> bytes:
    def atom(name: bytes, payload: bytes = b"") -> bytes:
        return struct.pack(">I4s", len(payload) + 8, name) + payload

    return atom(b"ftyp", b"isom\x00\x00\x02\x00") + atom(b"mdat") + atom(b"moov")


def test_render_stages_jpeg_and_publishes_atomically() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run_dir = Path(temp) / "attempt"
        leg_dir = run_dir / "leg00"
        leg_dir.mkdir(parents=True)
        (leg_dir / "step01.png").write_bytes(_png(32, 18))
        (run_dir / "leg00.jsonl").write_text("", encoding="utf-8")
        output = run_dir / video.REPLAY_NAME
        commands: list[tuple[list[str], float | None]] = []
        original_run, original_which = video._run, video.shutil.which

        def fake_run(command: list[str], *, timeout: float | None = None) -> None:
            commands.append((command, timeout))
            Path(command[-1]).write_bytes(_minimal_mp4())

        video._run = fake_run
        video.shutil.which = lambda _name: "/usr/bin/ffmpeg"
        try:
            assert video.render(run_dir, output, timeout=5.0) == output
        finally:
            video._run, video.shutil.which = original_run, original_which

        command, timeout = commands[0]
        assert command[command.index("-i") + 1].endswith("f%05d.jpg"), command
        assert command[command.index("-preset") + 1] == "veryfast"
        assert command[command.index("-threads") + 1] == str(video.ENCODER_THREADS)
        assert timeout is not None and 0 < timeout <= 5.0
        assert output.read_bytes() == _minimal_mp4()
        assert video.is_complete_mp4(output)
        assert not list(run_dir.glob(".*.part.mp4"))
    print("ok  render uses JPEG staging and atomically publishes a complete mp4")


def test_failed_render_preserves_existing_output_and_removes_partial() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run_dir = Path(temp) / "attempt"
        leg_dir = run_dir / "leg00"
        leg_dir.mkdir(parents=True)
        (leg_dir / "step01.png").write_bytes(_png(32, 18))
        (run_dir / "leg00.jsonl").write_text("", encoding="utf-8")
        output = run_dir / video.REPLAY_NAME
        output.write_bytes(b"existing replay")
        original_run, original_which = video._run, video.shutil.which

        def failed_run(command: list[str], *, timeout: float | None = None) -> None:
            Path(command[-1]).write_bytes(b"partial")
            raise subprocess.TimeoutExpired(command, timeout or 0)

        video._run = failed_run
        video.shutil.which = lambda _name: "/usr/bin/ffmpeg"
        try:
            assert video.render(run_dir, output, timeout=5.0) is None
        finally:
            video._run, video.shutil.which = original_run, original_which

        assert output.read_bytes() == b"existing replay"
        assert not list(run_dir.glob(".*.part.mp4"))
        assert not video.is_complete_mp4(output)
    print("ok  a timed-out encode cannot replace a replay or leave a ready-looking partial")


async def main() -> int:
    test_default_capture_rate_is_four_frames_per_second()
    original_to_thread = asyncio.to_thread
    asyncio.to_thread = _inline_to_thread  # type: ignore[assignment]
    try:
        await test_recorder_fills_gaps_and_publishes_small_jpegs()
        await test_recent_agent_frame_suppresses_a_redundant_capture()
    finally:
        asyncio.to_thread = original_to_thread  # type: ignore[assignment]
    test_watch_and_replay_merge_supplementary_frames()
    test_replay_frame_cap_preserves_steps_and_samples_captures()
    test_render_stages_jpeg_and_publishes_atomically()
    test_failed_render_preserves_existing_output_and_removes_partial()
    print("\nAll capture tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
