"""Mutable execution context for one leg and its current step."""

from dataclasses import dataclass
from enum import Enum


class StepDisposition(Enum):
    DISPATCH = "dispatch"
    NEXT = "next"
    STOP = "stop"


@dataclass
class LegSession:
    agent: object
    leg: dict
    store_map: object
    artifacts: object
    events: object
    metrics: dict
    state: dict
    visited: set
    grip_tracker: object
    inspection_evidence: object
    completion: object
    augmented_task: str
    semantic_before: object
    started_at: float
    max_steps: int
    max_minutes: float
    leg_idx: int
    inspect_move_left: int = 0
    last_actor_text: str = ""
    last_inspection_result: dict | None = None

    @property
    def text(self):
        return self.leg.get("text") or ""


@dataclass
class StepContext:
    number: int
    stamp: str
    image_b64: str
    guards: object
    target_checkpoints: list | None = None
    off_target: bool = False
    force_manipulate: bool = False
    request: dict | None = None
    response: dict | None = None
    mode: str | None = None
    parsed: dict | None = None
    outcome: object | None = None
    near_checkpoint: object = None
    observation: object = None
