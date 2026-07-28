"""Turns an attempt's observation frames into a watchable video: ``python -m sari_bench video``.

``run_leg`` saves one ``legNN/stepNN.png`` per timestep and the benchmark runner fills long gaps
with timestamped ``capture/*.jpg`` frames, so this is mostly an ffmpeg invocation. Two choices worth
stating:

* **mp4, not gif.** A 150-frame 1080p gif runs to hundreds of megabytes and cannot be scrubbed;
  h264 is roughly 20x smaller and seekable. ``--gif`` is there for pasting into a chat.
* **Captioned.** A silent video of a store aisle tells you very little. Each frame is stamped with
  the step, mode, action and checkpoint from that step's record in ``legNN.jsonl``, which is what
  makes a death loop legible - you see the same action fire against the same shelf.

Frames are captioned with Pillow (already a dependency) into a temp dir; ffmpeg must be on PATH.

``render_for_upload`` is the same pipeline aimed at a chat attachment: it writes a separate, smaller
``replay.discord.mp4`` at a bitrate computed from the clip's duration, so the watcher can post a replay
with every finish without ever touching the dashboard/CLI ``replay.mp4``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from sari_bench import capture
from sari_bench.watch import scan

_STEP_PNG = re.compile(r"^step(\d+)\.png$")

# Chat-attachment budget. Discord's real webhook cap is 10 MB; 8 leaves room for the embed and for
# libx264 overshooting its target on the last GOP.
DISCORD_BUDGET_BYTES = 8_000_000
BITRATE_HEADROOM = 0.95
MIN_VIDEO_BITRATE = 120_000  # bps floor; below this h264 is unwatchable, so blow the budget instead
RENDER_TIMEOUT_SECONDS = 600.0
STAGING_PROGRESS_FRAMES = 250
STAGING_JPEG_QUALITY = 90
ENCODER_THREADS = 6

# A five-minute clip is long enough to inspect a failed attempt without letting one 90-minute run
# monopolise the serial review worker. Step frames are never discarded; this cap only downsamples
# supplementary captures.
MAX_REPLAY_FRAMES = 1_200

# Full replays use the dense capture stream at four observations per playback second. The bounded
# Discord artifact remains step-only at one frame per second.
DEFAULT_FPS = 4.0
UPLOAD_FPS = 1.0
UPLOAD_WIDTH = 960

# Name the upload copy separately from `replay.mp4`. Dashboard/CLI review owns that one; attachment
# rendering must never quietly re-encode over it at upload quality.
UPLOAD_NAME = "replay.discord.mp4"
REPLAY_NAME = "replay.mp4"


def _steps_by_index(leg_jsonl: Path) -> dict[int, dict[str, Any]]:
    records = scan.read_step_records(leg_jsonl)
    return {int(r["step"]): r for r in records if r.get("event") == "step" and r.get("step")}


def _caption(record: dict[str, Any] | None, leg_name: str, step: int) -> str:
    if not record:
        return f"{leg_name}  step {step}"
    bits = [f"{leg_name}  step {step:>3}"]
    if record.get("mode"):
        bits.append(str(record["mode"]))
    if record.get("actions") is not None:
        bits.append(json.dumps(record["actions"], default=str)[:60])
    if record.get("near_cp") is not None:
        bits.append(f"@{record['near_cp']}")
    if record.get("blocked"):
        bits.append("BLOCKED")
    if record.get("gripped_name"):
        bits.append(f"held={record['gripped_name']}")
    return "   ".join(bits)


def _stamp_frame(source: Path, target: Path, text: str, *, caption: bool = True) -> None:
    from PIL import Image, ImageDraw

    with Image.open(source) as image:
        frame = image.convert("RGB")
        if caption:
            draw = ImageDraw.Draw(frame)
            bar = max(18, frame.height // 26)
            draw.rectangle(
                [(0, frame.height - bar), (frame.width, frame.height)],
                fill=(12, 14, 18),
            )
            draw.text(
                (8, frame.height - bar + max(2, bar // 6)),
                text,
                fill=(235, 238, 242),
            )
        # The capture stream is already JPEG. Recompressing thousands of frames losslessly as PNG
        # was more than ten times slower than the actual h264 encode and consumed gigabytes of /tmp.
        frame.save(target, format="JPEG", quality=STAGING_JPEG_QUALITY)


def collect_frames(run_dir: Path, *, include_captures: bool = True) -> list[tuple[Path, str]]:
    """Every frame of an attempt in play order, with its caption.

    Legacy step-only runs retain their leg/step ordering. When supplementary captures exist, both
    sources are merged by capture/publication time.
    """
    frames: list[tuple[Path, str]] = []
    timed: list[tuple[int, int, Path, str]] = []
    order = 0
    leg_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("leg"))
    for leg_dir in leg_dirs:
        records = _steps_by_index(run_dir / f"{leg_dir.name}.jsonl")
        numbered = sorted(
            ((int(m.group(1)), p) for p in leg_dir.iterdir() if (m := _STEP_PNG.match(p.name))),
        )
        for step, path in numbered:
            text = _caption(records.get(step), leg_dir.name, step)
            frames.append((path, text))
            timed.append((capture.frame_timestamp_ns(path), order, path, text))
            order += 1

    capture_dir = run_dir / capture.CAPTURE_DIR
    captures = (
        [path for path in capture.observation_frames(run_dir) if path.parent == capture_dir]
        if include_captures else []
    )
    if not captures:
        return frames

    manifest = scan._read_json(run_dir / scan.ATTEMPT_MANIFEST)
    started_ns = int(float(manifest.get("started_epoch") or 0) * 1e9)
    for path in captures:
        timestamp_ns = capture.frame_timestamp_ns(path)
        elapsed = max(0.0, (timestamp_ns - started_ns) / 1e9) if started_ns else 0.0
        minutes, seconds = divmod(int(elapsed), 60)
        text = f"live observation   +{minutes:02d}:{seconds:02d}"
        timed.append((timestamp_ns, order, path, text))
        order += 1
    return [(path, text) for _, _, path, text in sorted(timed)]


def limit_replay_frames(
    frames: list[tuple[Path, str]],
    max_frames: int | None,
) -> list[tuple[Path, str]]:
    """Uniformly thin captures to ``max_frames`` while retaining every agent step frame."""
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return frames

    step_indexes = [
        index for index, (path, _) in enumerate(frames)
        if path.parent.name != capture.CAPTURE_DIR
    ]
    capture_indexes = [
        index for index, (path, _) in enumerate(frames)
        if path.parent.name == capture.CAPTURE_DIR
    ]
    capture_budget = max(0, max_frames - len(step_indexes))
    if capture_budget >= len(capture_indexes):
        return frames
    if capture_budget == 1:
        chosen_captures = {capture_indexes[-1]}
    elif capture_budget > 1:
        last = len(capture_indexes) - 1
        chosen_captures = {
            capture_indexes[round(slot * last / (capture_budget - 1))]
            for slot in range(capture_budget)
        }
    else:
        chosen_captures = set()
    selected = set(step_indexes) | chosen_captures
    return [frame for index, frame in enumerate(frames) if index in selected]


def is_complete_mp4(path: Path) -> bool:
    """Cheap structural check that rejects ffmpeg outputs killed before their ``moov`` atom."""
    try:
        size = path.stat().st_size
        if size < 16:
            return False
        found_ftyp = False
        with path.open("rb") as handle:
            offset = 0
            while offset + 8 <= size:
                handle.seek(offset)
                header = handle.read(16)
                if len(header) < 8:
                    return False
                atom_size, atom_type = struct.unpack(">I4s", header[:8])
                header_size = 8
                if atom_size == 1:
                    if len(header) < 16:
                        return False
                    atom_size = struct.unpack(">Q", header[8:16])[0]
                    header_size = 16
                elif atom_size == 0:
                    atom_size = size - offset
                if atom_size < header_size or offset + atom_size > size:
                    return False
                if atom_type == b"ftyp":
                    found_ftyp = True
                if atom_type == b"moov":
                    return found_ftyp
                offset += atom_size
    except OSError:
        return False
    return False


def target_bitrate(frame_count: int, fps: float, budget_bytes: int = DISCORD_BUDGET_BYTES,
                   *, headroom: float = BITRATE_HEADROOM) -> int:
    """Video bitrate in bps that lands `frame_count` frames at `fps` inside `budget_bytes`.

    There is no audio track to subtract. Short clips come out with an absurd ceiling - eight frames at
    6 fps asks for 30 Mbps - and that is correct rather than a bug to cap: 1.3 seconds at 30 Mbps is
    still inside the budget by construction, and capping it would only make short replays uglier.
    """
    duration = max(frame_count / max(fps, 0.1), 1.0)
    return max(MIN_VIDEO_BITRATE, int(budget_bytes * 8 * headroom / duration))


def render(run_dir: Path, out_path: Path, *, fps: float = DEFAULT_FPS, width: int = 1280,
           gif: bool = False, caption: bool = True,
           max_bytes: int | None = None, preset: str = "veryfast",
           include_captures: bool = True,
           max_frames: int | None = None,
           timeout: float = RENDER_TIMEOUT_SECONDS) -> Path | None:
    all_frames = collect_frames(run_dir, include_captures=include_captures)
    frames = limit_replay_frames(all_frames, max_frames)
    if not frames:
        print(f"[sari-bench video] no frames under {run_dir}")
        return None
    if shutil.which("ffmpeg") is None:
        print("[sari-bench video] ffmpeg not found on PATH")
        return None

    started = time.monotonic()
    deadline = started + timeout
    if len(frames) < len(all_frames):
        print(
            f"[sari-bench video] sampling {len(all_frames)} frame(s) down to {len(frames)} "
            f"for {run_dir}",
            flush=True,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_name(
        f".{out_path.stem}.{os.getpid()}.{threading.get_ident()}.part{out_path.suffix}"
    )
    try:
        with tempfile.TemporaryDirectory() as temp:
            staging = Path(temp)
            staged = 0
            for source, text in frames:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("JPEG frame staging", timeout)
                target = staging / f"f{staged:05d}.jpg"
                try:
                    _stamp_frame(source, target, text, caption=caption)
                except Exception as error:  # noqa: BLE001 - one corrupt frame must not lose the clip
                    print(f"[sari-bench video] skipping {source}: {error!r}", flush=True)
                    continue
                staged += 1
                if staged % STAGING_PROGRESS_FRAMES == 0:
                    print(
                        f"[sari-bench video] staged {staged}/{len(frames)} frame(s) "
                        f"in {time.monotonic() - started:.1f}s for {run_dir}",
                        flush=True,
                    )
            if not staged:
                print(f"[sari-bench video] no readable frames under {run_dir}", flush=True)
                return None
            print(
                f"[sari-bench video] staged {staged}/{len(frames)} frame(s) "
                f"in {time.monotonic() - started:.1f}s; encoding {run_dir}",
                flush=True,
            )

            scale = f"scale={width}:-2:flags=lanczos"
            input_pattern = str(staging / "f%05d.jpg")
            if gif:
                palette = staging / "palette.png"
                _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", input_pattern,
                      "-vf", f"{scale},palettegen", str(palette)],
                     timeout=_remaining(deadline, timeout))
                _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", input_pattern,
                      "-i", str(palette), "-lavfi", f"{scale} [x]; [x][1:v] paletteuse",
                      str(part_path)], timeout=_remaining(deadline, timeout))
            elif max_bytes is None:
                _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", input_pattern,
                      "-vf", scale, "-c:v", "libx264", "-preset", preset,
                      "-threads", str(ENCODER_THREADS), "-pix_fmt", "yuv420p", str(part_path)],
                     timeout=_remaining(deadline, timeout))
            else:
                # Single-pass ABR with a hard ceiling and a 2 s VBV buffer: deterministic for a given
                # frame set, and +faststart lets a chat client start playing before the whole download.
                bitrate = target_bitrate(staged, fps, max_bytes)
                _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", input_pattern,
                      "-vf", scale, "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p",
                      "-threads", str(ENCODER_THREADS),
                      "-an", "-b:v", str(bitrate), "-maxrate", str(bitrate),
                      "-bufsize", str(bitrate * 2), "-movflags", "+faststart", str(part_path)],
                     timeout=_remaining(deadline, timeout))
        if out_path.suffix.lower() == ".mp4" and not is_complete_mp4(part_path):
            raise OSError(f"ffmpeg produced an incomplete mp4 at {part_path}")
        os.replace(part_path, out_path)
    except (OSError, subprocess.SubprocessError) as error:
        part_path.unlink(missing_ok=True)
        print(f"[sari-bench video] render failed on {run_dir}: {error!r}", flush=True)
        return None

    size_mb = out_path.stat().st_size / 1e6
    print(
        f"[sari-bench video] {staged} frame(s) -> {out_path} ({size_mb:.1f} MB) "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return out_path


def render_for_upload(run_dir: Path, *, max_bytes: int = DISCORD_BUDGET_BYTES,
                      fps: float = UPLOAD_FPS, width: int = UPLOAD_WIDTH,
                      reuse: bool = True) -> Path | None:
    """`<run_dir>/replay.discord.mp4`, sized for a chat attachment, or None if it cannot be made.

    Never raises. A missing ffmpeg, a dead encode, or a file that still busts the budget all come back
    as None so the caller posts its message without a clip - an attachment is never worth losing the
    notification it was meant to illustrate.
    """
    out_path = run_dir / UPLOAD_NAME
    try:
        if reuse and out_path.is_file():
            size = out_path.stat().st_size
            if 0 < size <= max_bytes and is_complete_mp4(out_path):
                return out_path
        if render(run_dir, out_path, fps=fps, width=width, max_bytes=max_bytes,
                  include_captures=False, max_frames=None) is None:
            return None
        size = out_path.stat().st_size
        if size > max_bytes:
            print(f"[sari-bench video] {out_path} is {size / 1e6:.1f} MB, over budget; skipping upload")
            return None
        return out_path
    except Exception as error:  # noqa: BLE001 - a replay is never worth disturbing the caller
        print(f"[sari-bench video] upload render failed for {run_dir}: {error!r}")
        return None


def _run(command: list[str], *, timeout: float | None = None) -> None:
    subprocess.run(command, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _remaining(deadline: float, configured_timeout: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("render", configured_timeout)
    return remaining


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sari_bench video",
                                     description="Render an attempt's screenshots into a video.")
    parser.add_argument("run_dir", nargs="?", type=Path, default=None,
                        help="One attempt's run dir (…/<prompt_id>/tryNN).")
    parser.add_argument("--battery", type=Path, default=None,
                        help="Render EVERY attempt in this battery dir instead.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help="Images per second (default: 4).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--gif", action="store_true", help="Write a gif instead of an mp4.")
    parser.add_argument("--no-caption", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_REPLAY_FRAMES,
        help=f"Maximum frames per replay (default: {MAX_REPLAY_FRAMES}; 0 keeps every frame).",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (single run only). Default: <run_dir>/replay.mp4")
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")

    suffix = ".gif" if args.gif else ".mp4"
    if args.battery:
        for run_dir in scan.run_dirs_of(args.battery.resolve()):
            render(run_dir, run_dir / f"replay{suffix}", fps=args.fps, width=args.width,
                   gif=args.gif, caption=not args.no_caption,
                   max_frames=args.max_frames or None)
        return 0

    if args.run_dir is None:
        parser.error("give a run dir, or --battery to render all of them")
    run_dir = args.run_dir.resolve()
    out = args.out or (run_dir / f"replay{suffix}")
    return 0 if render(run_dir, out, fps=args.fps, width=args.width, gif=args.gif,
                       caption=not args.no_caption, max_frames=args.max_frames or None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
