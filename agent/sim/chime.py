"""Cross-platform run-completion chime.

Replaces the bare `import winsound` that made every entry point Windows-only: winsound is a
Windows-only stdlib module, imported at module top level, so `run.py` / `subagent_run.py` /
`subtask_agents.py` all died at IMPORT time on macOS and Linux — before any argument parsing,
long before the sim was ever contacted.

Callers invoke this from a `finally:` block, so it must never raise: an exception here would
mask the real error that ended the run. Every failure path is swallowed deliberately.

NOTE: duplicated as agent/chime.py on purpose. The root and agent/ stacks both define
their own env.py and actions.py, so putting the repo root on sys.path to share one copy would
shadow agent's modules with the root's. The duplication is cheaper than that collision.
"""
import subprocess
import sys

# Windows-only default; the historical winsound.Beep(392, 1000) — G4 for one second.
DEFAULT_FREQUENCY_HZ = 392
DEFAULT_DURATION_MS = 1000


def beep(frequency_hz: int = DEFAULT_FREQUENCY_HZ, duration_ms: int = DEFAULT_DURATION_MS) -> None:
    """Sound a completion tone. Silent no-op if the platform offers no way to make noise."""
    try:
        if sys.platform == "win32":
            import winsound  # noqa: PLC0415 - Windows-only, cannot be imported at module scope
            winsound.Beep(frequency_hz, duration_ms)
        elif sys.platform == "darwin":
            # afplay ships with macOS. Fire and forget: a chime is not worth blocking the exit
            # path on, and check=False keeps a missing sound file from raising.
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        else:
            # Terminal bell — honoured by most Linux terminals, ignored harmlessly otherwise.
            print("\a", end="", flush=True)
    except Exception:
        pass
