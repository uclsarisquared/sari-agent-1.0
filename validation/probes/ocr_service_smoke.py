"""Manual real-model smoke for the central OCR service.

Start ``uv run poe ocr-server`` first, then run:

    uv run python validation/probes/ocr_service_smoke.py receipt.png --clients 20 --pid <server-pid>

The image is sent concurrently by every client. The script verifies that every response agrees and,
when given the daemon PID on Linux, reports its resident memory so the measured ~1.1 GiB baseline can
be checked without loading another model in this process.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parents[2] / "agent"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from vision.ocr_client import check_ocr_health, ocr_lines, resolve_ocr_url


def _rss_mib(pid: int) -> float | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concurrent real-model central OCR smoke test.")
    parser.add_argument("image", type=Path, help="Known receipt PNG/image.")
    parser.add_argument("--ocr-url", default=None)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--pid", type=int, default=None, help="Optional OCR daemon PID for RSS reporting.")
    args = parser.parse_args(argv)
    if args.clients < 1:
        parser.error("--clients must be at least 1")
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")

    url = resolve_ocr_url(args.ocr_url)
    health = check_ocr_health(url)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.clients) as pool:
        results = list(pool.map(lambda _index: ocr_lines(args.image, url), range(args.clients)))
    if any(lines != results[0] for lines in results[1:]):
        raise RuntimeError("concurrent OCR clients returned inconsistent line lists")

    print(f"{args.clients} clients succeeded via {url} ({health['model']})")
    print(f"Recognized {len(results[0])} line(s):")
    for line in results[0]:
        print(f"  {line}")
    if args.pid is not None:
        rss = _rss_mib(args.pid)
        print(f"Daemon RSS: {rss:.1f} MiB" if rss is not None else "Daemon RSS unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
