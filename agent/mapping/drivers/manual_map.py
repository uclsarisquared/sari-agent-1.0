#!/usr/bin/env python3
"""
Manual-drive mapping test: build the occupancy grid from keyboard-controlled
movement instead of frontier autoexploration.

WASD        - move fwd/bck/left/right (world-space translation, same as env.py)
Arrow keys  - pan left/right, tilt up/down (updates the yaw fed to the LiDAR projection)
ESC         - quit and save the final grid

Run from anywhere; requires SariSandboxV2 in Play mode with the WebSocket
command server running (default ws://localhost:8080/commands).

    python mapping/drivers/manual_map.py

Movement/rotation reuse env.py's _MOVE_*_/_PAN_*_/_TILT_*_ helpers, so the
same TransformAgent world-space-translation caveat from explore.py applies:
WASD steps move along fixed world axes, not the agent's facing direction.
Rotate with the arrow keys to change which way LiDAR scans project (they
turn the same yaw that scan_to_world_points() consumes), but that yaw does
NOT steer WASD's translation direction.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

try:
    from pynput import keyboard
except ImportError:
    print("pynput is not installed. Run:  pip install pynput")
    sys.exit(1)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/drivers
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)
from sim.env import (  # noqa: E402
    TransformAgent,
    _MOVE_FWD_, _MOVE_BCK_, _MOVE_LEFT_, _MOVE_RIGHT_,
    _PAN_LEFT_, _PAN_RIGHT_, _TILT_UP_, _TILT_DOWN_,
)

from occupancy_grid import OccupancyGrid
from lidar_client import RequestLidarScan
from mapping import scan_to_world_points

HOLD_ACTIONS = {
    "w": _MOVE_FWD_,
    "s": _MOVE_BCK_,
    "a": _MOVE_LEFT_,
    "d": _MOVE_RIGHT_,
    "left": _PAN_LEFT_,
    "right": _PAN_RIGHT_,
    "up": _TILT_UP_,
    "down": _TILT_DOWN_,
}

held_keys = set()
held_lock = threading.Lock()
stop_event = threading.Event()

TICK_SECONDS = 0.08  # ~12 Hz, matches keyboard_control.py


def key_to_str(key):
    try:
        ch = key.char
        if ch:
            return ch.lower()
    except AttributeError:
        pass
    name = getattr(key, "name", None)
    if name in ("left", "right", "up", "down"):
        return name
    return None


def on_press(key):
    if key == keyboard.Key.esc:
        stop_event.set()
        return False
    s = key_to_str(key)
    if s in HOLD_ACTIONS:
        with held_lock:
            held_keys.add(s)


def on_release(key):
    s = key_to_str(key)
    if s in HOLD_ACTIONS:
        with held_lock:
            held_keys.discard(s)


def save_snapshot(grid, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"grid_{tag}.npy"), grid.log_odds)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    display = np.full(grid.log_odds.shape, 0.5)
    display[grid.log_odds < grid.FREE_THRESHOLD] = 1.0
    display[grid.log_odds > grid.OCCUPIED_THRESHOLD] = 0.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(display.T, origin="lower", cmap="gray", vmin=0, vmax=1)

    # Crop to the explored region (plus a 1m margin) instead of the full grid extent - most
    # of a 60m grid is still "unknown" gray on any run that hasn't mapped the whole store, so
    # a full-extent plot renders the actual store as a small blob in the middle. Matches the
    # crop build_shelf_graph.py uses, so the occupancy PNG and the graph PNG frame the same area.
    known_cells = np.argwhere(display.T != 0.5)
    if len(known_cells) > 0:
        margin = int(round(1.0 / grid.res))  # 1m padding
        y0, x0 = known_cells.min(axis=0) - margin
        y1, x1 = known_cells.max(axis=0) + margin
        ax.set_xlim(max(0, x0), min(grid.log_odds.shape[0], x1))
        ax.set_ylim(max(0, y0), min(grid.log_odds.shape[1], y1))

    ax.set_title(f"Occupancy grid ({tag})")
    fig.savefig(os.path.join(output_dir, f"grid_{tag}.png"), dpi=150)
    plt.close(fig)


def worker(args, grid, state):
    scans_taken = 0
    ticks_since_scan = 0
    scan_every_ticks = max(1, round(args.scan_interval / TICK_SECONDS))

    while not stop_event.is_set():
        with held_lock:
            active = list(held_keys)

        for s in active:
            try:
                result = HOLD_ACTIONS[s]()
                state["pos"] = result["translation"]
                state["rot"] = result["rotation"]
            except Exception as exc:
                print(f"  [{s.upper()}] ERROR: {exc}")

        ticks_since_scan += 1
        if active and ticks_since_scan >= scan_every_ticks:
            ticks_since_scan = 0
            try:
                scan = RequestLidarScan(args.uri)
                hits = scan_to_world_points(scan, state["pos"], state["rot"][1])
                grid.integrate((state["pos"][0], state["pos"][2]), hits)
                scans_taken += 1
                print(
                    f"[manual_map] scan={scans_taken} pos=({state['pos'][0]:.2f},{state['pos'][2]:.2f}) "
                    f"yaw={state['rot'][1]:.1f} hits={len(hits)}"
                )
                if scans_taken % args.save_every == 0:
                    save_snapshot(grid, args.output_dir, scans_taken)
            except Exception as exc:
                print(f"[manual_map] scan ERROR: {exc}")

        time.sleep(TICK_SECONDS)


def run(args):
    grid = OccupancyGrid(size_m=args.size, resolution=args.resolution)

    initial = TransformAgent((0, 0, 0), (0, 0, 0), uri=args.uri)
    state = {"pos": initial["translation"], "rot": initial["rotation"]}

    print()
    print("╔══════════════════════════════════════════╗")
    print("║   Sari-Agent 1.0  —  Manual Mapping      ║")
    print("╠══════════════════════════════════════════╣")
    print("║  W / S / A / D   Move fwd/bck/left/right ║")
    print("║  ← → ↑ ↓         Pan left/right, tilt    ║")
    print("║  ESC             Quit + save final grid  ║")
    print("╚══════════════════════════════════════════╝")
    print()

    t = threading.Thread(target=worker, args=(args, grid, state), daemon=True)
    t.start()

    print("Listening for keys (ESC to quit)...\n")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    stop_event.set()
    t.join(timeout=2)

    save_snapshot(grid, args.output_dir, "final")
    print(f"\n[manual_map] saved final grid to {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Manual-drive occupancy-grid mapping test.")
    parser.add_argument("--uri", default="ws://localhost:8080/commands")
    parser.add_argument("--size", type=float, default=60.0, help="Grid extent in meters (square)")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid cell size in meters")
    parser.add_argument(
        "--scan-interval", type=float, default=0.4,
        help="Minimum seconds between LiDAR scans while moving/turning",
    )
    parser.add_argument("--save-every", type=int, default=20, help="Save a snapshot every N scans")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_MAPPING_DIR, "output"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
