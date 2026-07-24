"""
get_view.py — Captures the current view from the Unity sandbox.

Usage:
    python get_view.py                        # prints image size, no save
    python get_view.py --save                 # saves to screenshots/view.png
    python get_view.py --save --out my.png    # saves to my.png
    python get_view.py --show                 # opens image in default viewer
"""

import asyncio
import argparse
import sys
from io import BytesIO
from pathlib import Path

import websockets

UNITY_URI = "ws://localhost:8080/commands"


async def _fetch_screenshot(uri: str) -> bytes:
    import json
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({
            "command": "RequestScreenshot",
            "prefix": "",
            "suffix": "",
            "folder_name": "screenshots",
            "save_image": False,
        }))
        return await ws.recv()


def get_current_view(uri: str = UNITY_URI) -> bytes:
    """Return the current Unity sandbox view as raw PNG bytes."""
    return asyncio.get_event_loop().run_until_complete(_fetch_screenshot(uri))


def main():
    parser = argparse.ArgumentParser(description="Capture current Unity sandbox view.")
    parser.add_argument("--save", action="store_true", help="Save image to disk.")
    parser.add_argument("--out", default="screenshots/view.png", help="Output file path (default: screenshots/view.png).")
    parser.add_argument("--show", action="store_true", help="Open image in default viewer.")
    parser.add_argument("--uri", default=UNITY_URI, help="WebSocket URI of the Unity sandbox.")
    args = parser.parse_args()

    print(f"Connecting to {args.uri} ...")
    try:
        image_bytes = get_current_view(args.uri)
    except Exception as e:
        print(f"Failed to get view: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Received {len(image_bytes):,} bytes.")

    if args.save:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(image_bytes)
        print(f"Saved to {out}")

    if args.show:
        try:
            from PIL import Image
            Image.open(BytesIO(image_bytes)).show()
        except ImportError:
            print("Pillow not installed — run: pip install Pillow", file=sys.stderr)


if __name__ == "__main__":
    main()
