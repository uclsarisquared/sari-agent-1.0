"""Collapse detection over an attempt's step records.

The point of the dashboard is not to show you eight tiles - it is to tell you which ONE to look at.
A 2-hour attempt that has entered a hallucination loop looks, from the outside, exactly like one
that is working: the process is alive, steps keep arriving, the log keeps growing. What separates
them is *repetition*, and every signal needed to measure it is already in the `step` records
`run_leg` writes (`orchestrator/leg_runner.py`) - no new agent-side instrumentation.

Pure and I/O-free: `score()` takes parsed records and returns a verdict, so it can be tested
offline against the run dirs already sitting in agent/subtask_run_outputs/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# How long without a new step record before an attempt counts as stalled. A normal step costs one
# VLM call plus sim traffic - slow, but nowhere near this. Five minutes of silence means a hung
# request or a wedged sim, not thinking.
STALL_SECONDS = 300.0

# Window of recent steps every repetition signal is measured over.
WINDOW = 10

# Position rounding for the "has it actually moved" test, in metres. The agent pacing back and
# forth in front of one shelf reads as 1-2 distinct cells at this resolution.
POSITION_GRID = 0.5

LEVEL_OK = "ok"
LEVEL_WATCH = "watch"
LEVEL_ALERT = "alert"

# Score at or above which the tile goes red and (if configured) Discord is pinged.
ALERT_AT = 0.6
WATCH_AT = 0.3


@dataclass
class Signal:
    name: str
    weight: float
    detail: str


@dataclass
class HealthReport:
    score: float = 0.0
    level: str = LEVEL_OK
    signals: list[Signal] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return ", ".join(signal.detail for signal in self.signals) or "healthy"

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "summary": self.summary,
            "signals": [{"name": s.name, "weight": s.weight, "detail": s.detail} for s in self.signals],
        }


def _cell(position: Any) -> tuple | None:
    """Rounds a position to a grid cell. Positions arrive as [x, y, z]; y is height, so the
    horizontal cell is (x, z)."""
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    try:
        x, _, z = float(position[0]), float(position[1]), float(position[2])
    except (TypeError, ValueError):
        return None
    return (round(x / POSITION_GRID), round(z / POSITION_GRID))


def score(
    steps: list[dict[str, Any]],
    *,
    seconds_since_last_step: float | None = None,
    max_steps: int | None = None,
    elapsed_fraction: float | None = None,
    halts_refused: int = 0,
) -> HealthReport:
    """Scores one in-flight attempt.

    `steps` is the attempt's `step` records in order. Signals are additive and capped at 1.0 -
    a collapsed run usually trips several at once, and their sum is a better ranking key than any
    single one.
    """
    signals: list[Signal] = []

    if seconds_since_last_step is not None and seconds_since_last_step > STALL_SECONDS:
        signals.append(Signal("stalled", 0.7,
                              f"no step in {seconds_since_last_step / 60:.0f}m"))

    window = steps[-WINDOW:]
    if len(window) >= 6:
        cells = [c for c in (_cell(s.get("pos")) for s in window) if c is not None]
        if cells and len(set(cells)) <= 2:
            signals.append(Signal("spatial_loop", 0.45,
                                  f"{len(set(cells))} distinct position(s) in {len(cells)} steps"))

        actions = [repr(s.get("actions")) for s in window if s.get("actions") is not None]
        if actions:
            most_common = max(set(actions), key=actions.count)
            repeats = actions.count(most_common)
            if repeats >= 4:
                signals.append(Signal("action_loop", 0.4,
                                      f"same action x{repeats} in {len(actions)} steps"))

        modes = [s.get("mode") for s in window if s.get("mode")]
        flips = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
        if flips >= 4:
            signals.append(Signal("mode_thrash", 0.3, f"mode changed {flips}x in {len(modes)} steps"))

        blocked = [bool(s.get("blocked")) for s in window]
        if blocked and sum(blocked) / len(blocked) > 0.5:
            signals.append(Signal("blocked", 0.35,
                                  f"blocked in {sum(blocked)}/{len(blocked)} steps"))

    if halts_refused >= 2:
        signals.append(Signal("refusal_spiral", 0.25, f"{halts_refused} halt(s) refused"))

    # Budget burn is not a fault on its own - it is what turns a mild loop into a lost attempt, so
    # it only carries weight once the attempt is nearly out of road.
    if max_steps and steps:
        used = len(steps) / max_steps
        if used > 0.8:
            signals.append(Signal("step_budget", 0.2, f"{len(steps)}/{max_steps} steps used"))
    if elapsed_fraction is not None and elapsed_fraction > 0.8:
        signals.append(Signal("time_budget", 0.2, f"{elapsed_fraction * 100:.0f}% of time limit used"))

    total = min(1.0, sum(signal.weight for signal in signals))
    level = LEVEL_ALERT if total >= ALERT_AT else LEVEL_WATCH if total >= WATCH_AT else LEVEL_OK
    return HealthReport(score=total, level=level, signals=signals)
