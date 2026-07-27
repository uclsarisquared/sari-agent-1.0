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
with every finish without ever touching the uncapped ``replay.mp4`` this CLI writes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
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

# Full replays use the dense capture stream at two observations per playback second. The bounded
# Discord artifact remains step-only at one frame per second.
DEFAULT_FPS = 2.0
UPLOAD_FPS = 1.0
UPLOAD_WIDTH = 960

# Name the upload copy separately from `replay.mp4`. The CLI owns that one and renders it uncapped for
# archive viewing; auto-render must never quietly re-encode over it at attachment quality.
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


def _stamp_frame(source: Path, target: Path, text: str) -> None:
    from PIL import Image, ImageDraw

    with Image.open(source) as image:
        frame = image.convert("RGB")
        draw = ImageDraw.Draw(frame)
        bar = max(18, frame.height // 26)
        draw.rectangle([(0, frame.height - bar), (frame.width, frame.height)], fill=(12, 14, 18))
        draw.text((8, frame.height - bar + max(2, bar // 6)), text, fill=(235, 238, 242))
        frame.save(target)


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
           include_captures: bool = True) -> Path | None:
    frames = collect_frames(run_dir, include_captures=include_captures)
    if not frames:
        print(f"[sari-bench video] no frames under {run_dir}")
        return None
    if shutil.which("ffmpeg") is None:
        print("[sari-bench video] ffmpeg not found on PATH")
        return None

    with tempfile.TemporaryDirectory() as temp:
        staging = Path(temp)
        for index, (source, text) in enumerate(frames):
            target = staging / f"f{index:05d}.png"
            if caption:
                try:
                    _stamp_frame(source, target, text)
                    continue
                except Exception as error:  # noqa: BLE001 - a caption is never worth losing the video
                    print(f"[sari-bench video] caption failed on {source.name}: {error!r}")
            shutil.copyfile(source, target)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        scale = f"scale={width}:-2:flags=lanczos"
        try:
            if gif:
                palette = staging / "palette.png"
                _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(staging / "f%05d.png"),
                      "-vf", f"{scale},palettegen", str(palette)])
                _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(staging / "f%05d.png"),
                      "-i", str(palette), "-lavfi", f"{scale} [x]; [x][1:v] paletteuse",
                      str(out_path)])
            elif max_bytes is None:
                _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(staging / "f%05d.png"),
                      "-vf", scale, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)])
            else:
                # Single-pass ABR with a hard ceiling and a 2 s VBV buffer: deterministic for a given
                # frame set, and +faststart lets a chat client start playing before the whole download.
                bitrate = target_bitrate(len(frames), fps, max_bytes)
                _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(staging / "f%05d.png"),
                      "-vf", scale, "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p",
                      "-an", "-b:v", str(bitrate), "-maxrate", str(bitrate),
                      "-bufsize", str(bitrate * 2), "-movflags", "+faststart", str(out_path)],
                     timeout=RENDER_TIMEOUT_SECONDS)
        except subprocess.SubprocessError as error:
            print(f"[sari-bench video] ffmpeg failed on {run_dir}: {error!r}")
            return None

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    print(f"[sari-bench video] {len(frames)} frame(s) -> {out_path} ({size_mb:.1f} MB)")
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
            if 0 < size <= max_bytes:
                return out_path
        if render(run_dir, out_path, fps=fps, width=width, max_bytes=max_bytes,
                  include_captures=False) is None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sari_bench video",
                                     description="Render an attempt's screenshots into a video.")
    parser.add_argument("run_dir", nargs="?", type=Path, default=None,
                        help="One attempt's run dir (…/<prompt_id>/tryNN).")
    parser.add_argument("--battery", type=Path, default=None,
                        help="Render EVERY attempt in this battery dir instead.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help="Images per second (default: 1, so each image lasts about one second).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--gif", action="store_true", help="Write a gif instead of an mp4.")
    parser.add_argument("--no-caption", action="store_true")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (single run only). Default: <run_dir>/replay.mp4")
    args = parser.parse_args(argv)

    suffix = ".gif" if args.gif else ".mp4"
    if args.battery:
        for run_dir in scan.run_dirs_of(args.battery.resolve()):
            render(run_dir, run_dir / f"replay{suffix}", fps=args.fps, width=args.width,
                   gif=args.gif, caption=not args.no_caption)
        return 0

    if args.run_dir is None:
        parser.error("give a run dir, or --battery to render all of them")
    run_dir = args.run_dir.resolve()
    out = args.out or (run_dir / f"replay{suffix}")
    return 0 if render(run_dir, out, fps=args.fps, width=args.width, gif=args.gif,
                       caption=not args.no_caption) else 1


if __name__ == "__main__":
    raise SystemExit(main())
