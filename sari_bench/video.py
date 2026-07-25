"""Turns an attempt's per-step screenshots into a watchable video: ``python -m sari_bench video``.

``run_leg`` already saves one ``legNN/stepNN.png`` per timestep, zero-padded and in order, so this
is mostly an ffmpeg invocation. Two choices worth stating:

* **mp4, not gif.** A 150-frame 1080p gif runs to hundreds of megabytes and cannot be scrubbed;
  h264 is roughly 20x smaller and seekable. ``--gif`` is there for pasting into a chat.
* **Captioned.** A silent video of a store aisle tells you very little. Each frame is stamped with
  the step, mode, action and checkpoint from that step's record in ``legNN.jsonl``, which is what
  makes a death loop legible - you see the same action fire against the same shelf.

Frames are captioned with Pillow (already a dependency) into a temp dir; ffmpeg must be on PATH.
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

from sari_bench.watch import scan

_STEP_PNG = re.compile(r"^step(\d+)\.png$")


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


def collect_frames(run_dir: Path) -> list[tuple[Path, str]]:
    """Every frame of an attempt in play order, with its caption. Legs concatenate in order, so one
    video covers the whole attempt rather than one per leg."""
    frames: list[tuple[Path, str]] = []
    leg_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("leg"))
    for leg_dir in leg_dirs:
        records = _steps_by_index(run_dir / f"{leg_dir.name}.jsonl")
        numbered = sorted(
            ((int(m.group(1)), p) for p in leg_dir.iterdir() if (m := _STEP_PNG.match(p.name))),
        )
        for step, path in numbered:
            frames.append((path, _caption(records.get(step), leg_dir.name, step)))
    return frames


def render(run_dir: Path, out_path: Path, *, fps: float = 4.0, width: int = 1280,
           gif: bool = False, caption: bool = True) -> Path | None:
    frames = collect_frames(run_dir)
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
        if gif:
            palette = staging / "palette.png"
            _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(staging / "f%05d.png"),
                  "-vf", f"{scale},palettegen", str(palette)])
            _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(staging / "f%05d.png"),
                  "-i", str(palette), "-lavfi", f"{scale} [x]; [x][1:v] paletteuse", str(out_path)])
        else:
            _run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(staging / "f%05d.png"),
                  "-vf", scale, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)])

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    print(f"[sari-bench video] {len(frames)} frame(s) -> {out_path} ({size_mb:.1f} MB)")
    return out_path


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sari_bench video",
                                     description="Render an attempt's screenshots into a video.")
    parser.add_argument("run_dir", nargs="?", type=Path, default=None,
                        help="One attempt's run dir (…/<prompt_id>/tryNN).")
    parser.add_argument("--battery", type=Path, default=None,
                        help="Render EVERY attempt in this battery dir instead.")
    parser.add_argument("--fps", type=float, default=4.0)
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
