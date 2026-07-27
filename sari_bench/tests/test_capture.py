"""Supplementary capture, live-frame selection, and dense replay tests."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PIL import Image

from sari_bench import capture, video
from sari_bench.watch import scan


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


async def main() -> int:
    test_default_capture_rate_is_four_frames_per_second()
    await test_recorder_fills_gaps_and_publishes_small_jpegs()
    await test_recent_agent_frame_suppresses_a_redundant_capture()
    test_watch_and_replay_merge_supplementary_frames()
    print("\nAll capture tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
