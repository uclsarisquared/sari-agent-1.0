"""Artifacts and structured logging for one leg execution."""

import json
import os
import time

from sim.env import downscale_for_storage


def write_step_output(out_dir, step, response, stamp=""):
    """Write the complete model response associated with a step screenshot."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step{step:02d}{stamp}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            f"=== STEP {step} | mode={response.get('agent_mode')} "
            f"| halt={response.get('halt')} ===\n\n"
        )
        if response.get("nav_note"):
            fh.write(f"--- NAV NOTE ---\n{response['nav_note']}\n\n")
        fh.write(
            f"--- MODE ROUTER (semantic) ---\n"
            f"{response.get('semantic') or '(n/a)'}\n\n"
        )
        fh.write(f"--- VLM ACTOR OUTPUT ---\n{response.get('text') or ''}\n\n")
        fh.write(
            f"--- EPISODIC REFLECTION ---\n"
            f"{response.get('episodic') or '(n/a)'}\n"
        )


class LegArtifacts:
    """Own a leg's JSONL stream and per-step debug files."""

    def __init__(self, log_path, started_at=None):
        self.log_path = log_path
        self.shots_dir = os.path.splitext(log_path)[0] if log_path else None
        self.started_at = time.time() if started_at is None else started_at
        self._log_fh = None

    def __enter__(self):
        if self.shots_dir:
            os.makedirs(self.shots_dir, exist_ok=True)
        if self.log_path:
            self._log_fh = open(self.log_path, "a", encoding="utf-8")
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    def close(self):
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def log(self, record):
        """Append a crash-safe timestamped JSON event without mutating the caller's dict."""
        if not isinstance(record, dict) or not record.get("event"):
            raise ValueError("leg log records require a non-empty 'event' field")
        if not self._log_fh:
            return
        row = {**record, "wall": round(time.time() - self.started_at, 1)}
        self._log_fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._log_fh.flush()

    def save_frame(self, step, stamp, image_bytes):
        """Save a bounded-resolution debug frame; return the written path if enabled."""
        if not self.shots_dir:
            return None
        path = os.path.join(self.shots_dir, f"step{step:02d}{stamp}.png")
        with open(path, "wb") as fh:
            fh.write(downscale_for_storage(image_bytes))
        return path

    def save_response(self, step, stamp, response):
        write_step_output(self.shots_dir, step, response, stamp=stamp)

    def center_dir(self, step, stamp):
        if not self.shots_dir:
            return None
        return os.path.join(self.shots_dir, f"step{step:02d}{stamp}_center")

    def inspection_frames_dir(self, step):
        if not self.shots_dir:
            return None
        return os.path.join(self.shots_dir, f"step{step:02d}_inspection")

    def event_logger(self, leg_idx):
        return LegEventLogger(self.log, leg_idx)


class LegEventLogger:
    """Assemble the stable JSONL event families emitted during one leg."""

    def __init__(self, writer, leg_idx):
        self.writer = writer
        self.leg_idx = leg_idx

    def __call__(self, record):
        """Accept an already assembled event from lower-level dispatch code."""
        self.writer({**record, "leg": self.leg_idx})

    def emit(self, event, *, step=None, **fields):
        if not event:
            raise ValueError("event name must be non-empty")
        reserved = {"event", "leg", "step"}.intersection(fields)
        if reserved:
            raise ValueError(
                f"event fields cannot override reserved keys: {', '.join(sorted(reserved))}"
            )
        row = {"event": event, "leg": self.leg_idx, **fields}
        if step is not None:
            row["step"] = step
        self.writer(row)

    def guard(self, step, backend, verdict, *, guard=None, **context):
        verdict = verdict if isinstance(verdict, dict) else {}
        fields = {
            "backend": backend,
            "match": verdict.get("match"),
            "reason": verdict.get("reason"),
            "conclusive": verdict.get("conclusive"),
            "latency_ms": verdict.get("latency_ms"),
            "reused": verdict.get("reused", False),
            **context,
        }
        if guard is not None:
            fields["guard"] = guard
        self.emit("completion_guard", step=step, **fields)

    def failure(self, event, step, error=None, **fields):
        if error is not None:
            fields["error"] = (
                error if isinstance(error, str)
                else f"{type(error).__name__}: {error}"
            )
        self.emit(event, step=step, **fields)

    def step(self, context, session):
        outcome = context.outcome
        state = session.state
        inspection = outcome.inspection_result
        self.emit(
            "step",
            step=context.number,
            mode=context.mode,
            nav_note=(context.response.get("nav_note") or "")[:200] or None,
            actions=outcome.acted,
            blocked=outcome.blocked_reason or None,
            center=outcome.center_message,
            reach=outcome.last_reach,
            near_cp=context.near_checkpoint,
            pos=state.get("translation"),
            hovered=[state.get("leftHoveredObject"), state.get("rightHoveredObject")],
            gripped=[state.get("leftGrippedState"), state.get("rightGrippedState")],
            gripped_names=dict(session.grip_tracker.names),
            off_target=context.off_target or None,
            checkout=outcome.checkout_result,
            inspection=(
                {key: value for key, value in inspection.items() if key != "steps"}
                if isinstance(inspection, dict)
                else None
            ),
            goal_met=context.observation.met,
            status=(context.parsed.get("notes") or {}).get("status"),
        )
