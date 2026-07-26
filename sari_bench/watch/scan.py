"""Turns a battery output directory into a live view of the fleet, by reading only the filesystem.

The runner and the agents already write everything a dashboard needs, and they flush as they go:

    bench_runs/<battery>/battery.json          the battery's plan (denominators)
    bench_runs/<battery>/attempts.jsonl        finished attempts, appended
    bench_runs/<battery>/<prompt>/try<NN>/
        attempt.json                           this attempt's manifest, incl. pid and deadline
        agent.log                              the agent's stdout
        summary.json                           written at exit
        legNN.jsonl                            one flushed record per step
        legNN/stepNN.png                       the frame that step saw

So the watcher never talks to a runner or an agent, and nothing it does can perturb a battery that
is six hours in. It also means the whole module works retroactively on old run dirs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sari_bench.watch import health

ATTEMPT_MANIFEST = "attempt.json"
BATTERY_MANIFEST = "battery.json"

# The two end reasons that mean the AGENT decided it was done, rather than the harness deciding for
# it: `halt_granted` is a STOP the completion predicate granted, `completed_no_stop` is the backstop
# firing after the predicate held for several steps running. Everything else (step_cap, time_cap,
# halt_forced, errors) is the run being cut off, and there is no claim of completion to review.
#
# These are the only attempts the dashboard offers a human verdict on, because the predicates behind
# them are the ones that grant on unverifiable state - see the `[unverified]` reasons in
# overhaul/orchestrator/subtask_completion.py. A human watching the replay is the check on those.
VERIFIABLE_END_REASONS = frozenset({"halt_granted", "completed_no_stop"})

# Run dirs look like <prompt_id>/try01, plus <prompt_id>/try01.requeue00 for rotated-aside ones.
_TRY_DIR = re.compile(r"^try\d+(\.requeue\d+)?$")
_LEG_JSONL = re.compile(r"^leg(\d+)\.jsonl$")
_STEP_PNG = re.compile(r"^step(\d+)\.png$")


@dataclass
class AttemptView:
    key: str                      # "<prompt_id>/<try dir>", unique within a battery
    prompt_id: str = ""
    attempt: int = 0
    prompt: str = ""
    family: str = ""
    looking_for: str = ""
    run_dir: str = ""

    state: str = "unknown"        # starting | running | finished | requeued | orphaned
    outcome: str = ""
    success: bool = False
    end_reason: str = ""
    exit_code: int | None = None

    sandbox_id: str = ""
    commands_uri: str = ""
    pid: int | None = None
    alive: bool = False
    killed_by: str = ""

    # The human verdict, kept strictly beside `success` and never folded into it. `success` stays the
    # predicate's answer; `verified_success` is a reviewer's. Where they disagree is the signal.
    # `verified_success` is None - not False - until someone actually looks, so "not reviewed" is
    # never read as "reviewed and failed".
    verifiable: bool = False      # finished, and the agent halted of its own accord
    verified: bool = False
    verified_success: bool | None = None
    verified_by: str = ""
    verified_at: str = ""
    verified_note: str = ""

    # Token cost so far (agent_core.token_meter's tokens.json, rewritten every few seconds), so a
    # live attempt's spend is visible while it runs and not only once it exits.
    tokens_in: int = 0
    tokens_out: int = 0

    started_at: str = ""
    elapsed_seconds: float = 0.0
    remaining_seconds: float | None = None
    seconds_since_step: float | None = None

    leg: int | None = None
    leg_type: str = ""
    leg_text: str = ""
    step: int = 0
    max_steps: int | None = None
    mode: str = ""
    actions: Any = None
    status: str = ""
    nav_note: str = ""
    near_cp: Any = None
    pos: Any = None
    blocked: bool = False
    gripped: Any = None
    gripped_name: Any = None
    goal_met: Any = None
    halts_refused: int = 0

    log_bytes: int = 0            # agent.log size; lets the dashboard poll only when it changed
    frame: str = ""               # battery-relative path of the newest screenshot
    health: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatteryView:
    battery_id: str
    path: str
    battery: dict[str, Any]
    attempts: list[dict[str, Any]]
    counts: dict[str, int]
    discovered: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def is_verifiable(state: str, end_reason: str) -> bool:
    """Whether an attempt is eligible for a human verdict.

    One definition, shared by the API guard, the view the dashboard renders from, and the report - so
    the button the reviewer sees and the check the POST handler makes can never drift apart.
    """
    return state == "finished" and end_reason in VERIFIABLE_END_REASONS


def read_step_records(leg_path: Path) -> list[dict[str, Any]]:
    """Parses one legNN.jsonl. Tolerates a torn final line: the file is appended to and flushed
    line-by-line while we read it, so the last record can legitimately be incomplete."""
    records: list[dict[str, Any]] = []
    try:
        with leg_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except OSError:
        return []
    return records


def _leg_files(run_dir: Path) -> list[Path]:
    return sorted(
        (p for p in run_dir.iterdir() if _LEG_JSONL.match(p.name)),
        key=lambda p: int(_LEG_JSONL.match(p.name).group(1)),
    ) if run_dir.is_dir() else []


def _latest_frame(run_dir: Path) -> Path | None:
    """Newest stepNN.png across the attempt's leg dirs, by leg then step - NOT by mtime, which
    reorders under a filesystem with coarse timestamps."""
    best: tuple[int, int] | None = None
    best_path: Path | None = None
    if not run_dir.is_dir():
        return None
    for leg_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("leg")):
        try:
            leg_index = int(leg_dir.name[3:] or 0)
        except ValueError:
            continue
        for frame in leg_dir.iterdir():
            match = _STEP_PNG.match(frame.name)
            if not match:
                continue
            rank = (leg_index, int(match.group(1)))
            if best is None or rank > best:
                best, best_path = rank, frame
    return best_path


def scan_attempt(run_dir: Path, battery_root: Path, now: float) -> AttemptView:
    """Builds one tile's worth of state from a run dir."""
    manifest = _read_json(run_dir / ATTEMPT_MANIFEST)
    view = AttemptView(
        key=f"{run_dir.parent.name}/{run_dir.name}",
        prompt_id=str(manifest.get("prompt_id") or run_dir.parent.name),
        attempt=int(manifest.get("attempt") or 0),
        prompt=str(manifest.get("prompt") or ""),
        family=str(manifest.get("family") or ""),
        looking_for=str(manifest.get("looking_for") or ""),
        run_dir=str(run_dir),
        state=str(manifest.get("state") or "unknown"),
        outcome=str(manifest.get("outcome") or ""),
        success=bool(manifest.get("success")),
        end_reason=str(manifest.get("end_reason") or ""),
        exit_code=manifest.get("exit_code"),
        sandbox_id=str(manifest.get("sandbox_id") or ""),
        commands_uri=str(manifest.get("commands_uri") or ""),
        pid=manifest.get("pid"),
        killed_by=str(manifest.get("killed_by") or ""),
        verified="verified_success" in manifest,
        verified_success=(bool(manifest["verified_success"])
                          if "verified_success" in manifest else None),
        verified_by=str(manifest.get("verified_by") or ""),
        verified_at=str(manifest.get("verified_at") or ""),
        verified_note=str(manifest.get("verified_note") or ""),
        started_at=str(manifest.get("started_at") or ""),
        max_steps=manifest.get("max_steps"),
    )

    tokens = _read_json(run_dir / "tokens.json")
    view.tokens_in = int(tokens.get("tokens_in") or manifest.get("tokens_in") or 0)
    view.tokens_out = int(tokens.get("tokens_out") or manifest.get("tokens_out") or 0)

    started = manifest.get("started_epoch")
    deadline = manifest.get("deadline_epoch")
    if view.state == "finished":
        view.elapsed_seconds = float(manifest.get("wall_seconds") or 0.0)
    elif isinstance(started, (int, float)):
        view.elapsed_seconds = round(now - float(started), 1)
    if view.state != "finished" and isinstance(deadline, (int, float)):
        view.remaining_seconds = round(float(deadline) - now, 1)

    if view.state in {"starting", "running"}:
        view.alive = _pid_alive(view.pid)
        if not view.alive and view.pid:
            # The manifest says live but the process is gone: the runner died before it could close
            # the attempt out. Say so rather than showing a tile frozen forever at its last step.
            view.state = "orphaned"

    try:
        view.log_bytes = (run_dir / "agent.log").stat().st_size
    except OSError:
        view.log_bytes = 0

    # After the orphan downgrade, so a tile whose runner died can never be offered for review.
    view.verifiable = is_verifiable(view.state, view.end_reason)

    legs = _leg_files(run_dir)
    steps: list[dict[str, Any]] = []
    if legs:
        records = read_step_records(legs[-1])
        steps = [r for r in records if r.get("event") == "step"]
        for record in records:
            event = record.get("event")
            if event == "leg_start":
                view.leg = record.get("leg")
                view.leg_type = str(record.get("type") or "")
                view.leg_text = str(record.get("text") or "")
            elif event == "halt_request" and not record.get("granted"):
                view.halts_refused += 1
        if steps:
            last = steps[-1]
            view.step = int(last.get("step") or len(steps))
            view.mode = str(last.get("mode") or "")
            view.actions = last.get("actions")
            view.status = str(last.get("status") or "")
            view.nav_note = str(last.get("nav_note") or "")
            view.near_cp = last.get("near_cp")
            view.pos = last.get("pos")
            view.blocked = bool(last.get("blocked"))
            view.gripped = last.get("gripped")
            view.gripped_name = last.get("gripped_name")
            view.goal_met = last.get("goal_met")
        try:
            view.seconds_since_step = round(now - legs[-1].stat().st_mtime, 1)
        except OSError:
            view.seconds_since_step = None

    frame = _latest_frame(run_dir)
    if frame is not None:
        view.frame = str(frame.relative_to(battery_root))

    if view.state in {"starting", "running"}:
        elapsed_fraction = None
        limit = manifest.get("time_limit_minutes")
        if isinstance(limit, (int, float)) and limit > 0:
            elapsed_fraction = view.elapsed_seconds / (float(limit) * 60.0)
        view.health = health.score(
            steps,
            seconds_since_last_step=view.seconds_since_step,
            max_steps=view.max_steps,
            elapsed_fraction=elapsed_fraction,
            halts_refused=view.halts_refused,
        ).as_dict()
    else:
        view.health = health.HealthReport().as_dict()

    return view


def find_batteries(root: Path) -> list[Path]:
    """Battery dirs under `root`, newest first.

    A battery is any directory holding a battery.json, or - for dirs written before manifests
    existed - one holding <prompt>/try<NN> run dirs.
    """
    if not root.is_dir():
        return []
    found = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / BATTERY_MANIFEST).exists() or any(
            _TRY_DIR.match(sub.name) for prompt in child.iterdir() if prompt.is_dir()
            for sub in prompt.iterdir() if sub.is_dir()
        ):
            found.append(child)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def run_dirs_of(battery: Path) -> list[Path]:
    return sorted(
        sub
        for prompt in battery.iterdir() if prompt.is_dir()
        for sub in prompt.iterdir() if sub.is_dir() and _TRY_DIR.match(sub.name)
    )


def scan_battery(battery: Path, now: float, *, discovered: list[Path] | None = None) -> BatteryView:
    """Full state for one battery: its plan, every attempt, and the tally."""
    attempts = [scan_attempt(run_dir, battery, now) for run_dir in run_dirs_of(battery)]
    # Worst-first. With eight concurrent attempts you want to look at one tile, not scan eight, so
    # the ranking is the feature: live attempts sort by collapse score, finished ones sink.
    attempts.sort(
        key=lambda a: (
            a.state == "finished",
            -float(a.health.get("score") or 0.0),
            a.prompt_id,
            a.attempt,
        )
    )

    counts: dict[str, int] = {}
    for attempt in attempts:
        bucket = attempt.outcome if attempt.state == "finished" else attempt.state
        counts[bucket] = counts.get(bucket, 0) + 1
        if attempt.success:
            counts["success"] = counts.get("success", 0) + 1
        if attempt.verified:
            counts["verified"] = counts.get("verified", 0) + 1
            key = "verified_success" if attempt.verified_success else "verified_fail"
            counts[key] = counts.get(key, 0) + 1
            if bool(attempt.verified_success) != attempt.success:
                counts["disagree"] = counts.get("disagree", 0) + 1
        elif attempt.verifiable:
            counts["awaiting_verdict"] = counts.get("awaiting_verdict", 0) + 1

    return BatteryView(
        battery_id=battery.name,
        path=str(battery),
        battery=_read_json(battery / BATTERY_MANIFEST),
        attempts=[attempt.as_dict() for attempt in attempts],
        counts=counts,
        discovered=[{"id": p.name, "path": str(p)} for p in (discovered or [])],
    )
