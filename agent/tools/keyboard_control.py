#!/usr/bin/env python3
"""
Keyboard controller for Sari-Agent 1.0

WASD        - body movement (forward / backward / left / right)
Arrow keys  - pan left/right, tilt up/down
R           - left hand forward
F           - left hand back
T           - right hand forward
G           - right hand back
Q           - toggle left grip
E           - toggle right grip
ESC         - quit
"""

import os
import sys
import asyncio
import threading
import time

try:
    from pynput import keyboard
except ImportError:
    print("pynput is not installed. Run:  pip install pynput")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # agent/ — env lives there
from sim.env import (
    _MOVE_FWD_, _MOVE_BCK_, _MOVE_LEFT_, _MOVE_RIGHT_,
    _PAN_LEFT_, _PAN_RIGHT_, _TILT_UP_, _TILT_DOWN_,
    _XTNFWD_LEFT_, _PLLBCK_LEFT_,
    _XTNFWD_RIGHT_, _PLLBCK_RIGHT_,
    _GRIP_LEFT_, _GRIP_RIGHT_,
)

# ------------------------------------------------------------------
# Action tables
# ------------------------------------------------------------------

# Keys that trigger repeatedly while held
HOLD_ACTIONS: dict[str, tuple] = {
    "w":     (_MOVE_FWD_,      "Move forward"),
    "s":     (_MOVE_BCK_,      "Move backward"),
    "a":     (_MOVE_LEFT_,     "Move left"),
    "d":     (_MOVE_RIGHT_,    "Move right"),
    "r":     (_XTNFWD_LEFT_,   "Left hand forward"),
    "f":     (_PLLBCK_LEFT_,   "Left hand back"),
    "t":     (_XTNFWD_RIGHT_,  "Right hand forward"),
    "g":     (_PLLBCK_RIGHT_,  "Right hand back"),
    "left":  (_PAN_LEFT_,      "Pan left"),
    "right": (_PAN_RIGHT_,     "Pan right"),
    "up":    (_TILT_UP_,       "Tilt up"),
    "down":  (_TILT_DOWN_,     "Tilt down"),
}

# Keys that fire exactly once per press (toggle semantics)
ONESHOT_ACTIONS: dict[str, tuple] = {
    "q": (_GRIP_LEFT_,  "Toggle left grip"),
    "e": (_GRIP_RIGHT_, "Toggle right grip"),
}

# ------------------------------------------------------------------
# Shared state between listener thread and worker thread
# ------------------------------------------------------------------

held_keys:      set[str] = set()
held_lock                = threading.Lock()

oneshot_queue:  list[str] = []
oneshot_lock             = threading.Lock()

stop_event               = threading.Event()


# ------------------------------------------------------------------
# Key helpers
# ------------------------------------------------------------------

def key_to_str(key) -> str | None:
    """Return a canonical string for a key, or None if not mapped."""
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


# ------------------------------------------------------------------
# pynput callbacks
# ------------------------------------------------------------------

def on_press(key):
    if key == keyboard.Key.esc:
        stop_event.set()
        return False  # stops the listener

    s = key_to_str(key)
    if s is None:
        return

    if s in ONESHOT_ACTIONS:
        with oneshot_lock:
            oneshot_queue.append(s)
    elif s in HOLD_ACTIONS:
        with held_lock:
            held_keys.add(s)


def on_release(key):
    s = key_to_str(key)
    if s in HOLD_ACTIONS:
        with held_lock:
            held_keys.discard(s)


# ------------------------------------------------------------------
# Worker thread
# ------------------------------------------------------------------

TICK_SECONDS = 0.08  # ~12 Hz — balances responsiveness and WebSocket load


def worker():
    # Each thread needs its own event loop because env.py uses
    # asyncio.get_event_loop().run_until_complete() internally.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while not stop_event.is_set():
        # --- one-shot actions (grips) ---
        with oneshot_lock:
            shots = list(oneshot_queue)
            oneshot_queue.clear()

        for s in shots:
            fn, label = ONESHOT_ACTIONS[s]
            print(f"  [{s.upper()}] {label} ...", end=" ", flush=True)
            try:
                result = fn()
                gripped = result.get("gripped", "?")
                print(f"gripped={gripped}")
            except Exception as exc:
                print(f"ERROR: {exc}")

        # --- held keys (movement / hand) ---
        with held_lock:
            active = list(held_keys)

        for s in active:
            fn, _label = HOLD_ACTIONS[s]
            try:
                fn()
            except Exception as exc:
                print(f"  [{s.upper()}] ERROR: {exc}")

        time.sleep(TICK_SECONDS)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def print_controls():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║   Sari-Agent 1.0  —  Keyboard Controls   ║")
    print("╠══════════════════════════════════════════╣")
    print("║  W / S / A / D   Move fwd/bck/left/right ║")
    print("║  ← → ↑ ↓         Pan left/right, tilt    ║")
    print("║  R               Left hand forward        ║")
    print("║  F               Left hand back           ║")
    print("║  T               Right hand forward       ║")
    print("║  G               Right hand back          ║")
    print("║  Q               Toggle left grip         ║")
    print("║  E               Toggle right grip        ║")
    print("║  ESC             Quit                     ║")
    print("╚══════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    print_controls()

    # Start the command worker in a background thread
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    print("Listening for keys (ESC to quit)...\n")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    stop_event.set()
    t.join(timeout=2)
    print("\nController stopped.")