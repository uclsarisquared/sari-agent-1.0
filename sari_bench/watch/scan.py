"""Turns a battery output directory into a live view of the fleet, by reading only the filesystem.

The runner and the agents already write everything a dashboard needs, and they flush as they go:

    bench_runs/<battery>/battery.json          the battery's plan (denominators)
    bench_runs/<battery>/attempts.jsonl        canonical finished-attempt index
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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sari_bench import capture
from sari_bench.storage import RUNNER_LOCK
from sari_bench.watch import health

ATTEMPT_MANIFEST = "attempt.json"
BATTERY_MANIFEST = "battery.json"
ATTEMPTS_INDEX = "attempts.jsonl"
# The agent's final user-facing answer, written once at exit by orchestrator.task_response. Read
# straight off disk like everything else here, so it also appears for old run dirs and for attempts
# whose summary.json never made it into the attempts index.
RESPONSE_FILE = "response.txt"
# It is three sentences by contract, and the dashboard ships it in every state poll for every
# attempt. A cap keeps one malformed run from bloating the payload for the whole battery.
RESPONSE_MAX_CHARS = 2000

# How stale the battery's own bookkeeping may be before a runner holding no lock - a resumed or
# partial run - stops counting as live.
LIVE_GRACE_SECONDS = 120.0

# Run dirs look like <prompt_id>/try01, plus <prompt_id>/try01.requeue00 for rotated-aside ones.
_TRY_DIR = re.compile(r"^try\d+(\.requeue\d+)?$")
_LEG_JSONL = re.compile(r"^leg(\d+)\.jsonl$")
_STEP_PNG = re.compile(r"^step(\d+)\.png$")


@dataclass
class AttemptView:
    key: str                      # "<prompt_id>/<try dir>", unique within a battery
    run_id: str = ""              # unique execution identity; changes when a try is retried
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
    stop_reason: str = ""
    stop_requested_at: str = ""
    stop_requested_by: str = ""
    winning_attempt_key: str = ""
    retry_state: str = ""
    retry_error: str = ""

    # The human verdict, kept strictly beside `success` and never folded into it. `success` stays the
    # predicate's answer; `verified_success` is a reviewer's. Where they disagree is the signal.
    # `verified_success` is None - not False - until someone actually looks, so "not reviewed" is
    # never read as "reviewed and failed".
    #
    # `verified_verdict` is the reviewer's full answer and has four values, because "the harness
    # broke" is not the same finding as "the agent failed the task": an INVALID run is one nobody
    # should count in either direction. It carries `verified_success = None` for exactly the reason
    # an unreviewed attempt does - no reader may total it as a failure. ALREADY_SUCCESSFUL is the
    # same exclusion for the opposite cause: the try was halted because the prompt was already won.
    verifiable: bool = False      # finished and eligible for a human verdict
    verified: bool = False
    # "pass" | "fail" | "invalid" | "already_successful", "" when unreviewed
    verified_verdict: str = ""
    verified_success: bool | None = None
    verified_by: str = ""
    verified_at: str = ""
    verified_note: str = ""

    # Token cost so far (agent_core.token_meter's tokens.json, rewritten every few seconds), so a
    # live attempt's spend is visible while it runs and not only once it exits.
    tokens_in: int = 0
    tokens_out: int = 0
    # The same spend split by which reasoner made the call: role -> {tokens_in, tokens_out, calls}.
    # Empty for an attempt run before roles existed (or one whose agent died before its first
    # tokens.json write), which readers must show as "unknown", never as a battery of zeroes - the
    # tokens_in/tokens_out above are still right in that case and would contradict them.
    tokens_by_role: dict[str, Any] = field(default_factory=dict)

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

    # The agent's own answer to the prompt, once it has written one. "" while it runs, and "" for a
    # run that died before finalizing - which a reader must show as absent, never as an empty answer.
    response: str = ""

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
    discovered: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


# The reasoner names agent_core.token_meter bills calls to, in the order a reader should see them:
# roughly the order a task moves through them, so a column of these reads as the agent's own
# pipeline rather than as alphabet soup. A role the meter reports but this list does not know is
# still shown - appended at the end - so adding a reasoner agent-side needs no dashboard release.
ROLE_ORDER = ("decomposer", "resolver", "actor", "semantic", "episodic", "advisor",
              "perception", "guard", "findings", "responder", "unattributed")


def normalize_by_role(raw: Any) -> dict[str, dict[str, int]]:
    """Coerces a tokens.json ``by_role`` block into role -> {tokens_in, tokens_out, calls} ints.

    Defensive because it parses a file another process is rewriting under it, and because run dirs
    predating per-role accounting have no block at all. Returns {} rather than a zero-filled skeleton
    for those: "this run did not record roles" and "this run spent nothing on any role" are different
    findings, and only the empty dict can say the first.
    """
    if not isinstance(raw, dict):
        return {}
    rows: dict[str, dict[str, int]] = {}
    for name, row in raw.items():
        if not isinstance(row, dict):
            continue
        try:
            rows[str(name)] = {field_name: int(row.get(field_name) or 0)
                               for field_name in ("tokens_in", "tokens_out", "calls")}
        except (TypeError, ValueError):
            continue
    return rows


def sorted_roles(names: Any) -> list[str]:
    """Role names in ROLE_ORDER, unknown ones alphabetically after them."""
    known = [name for name in ROLE_ORDER if name in names]
    return known + sorted(name for name in names if name not in ROLE_ORDER)


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


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _process_start_ticks(pid: int) -> str:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        return stat[stat.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return ""


def agent_is_alive(manifest: dict[str, Any], pid: int | None) -> bool:
    """Whether the process this manifest recorded is still the process running under `pid`.

    A pid on its own proves nothing after the fact: across a reboot, or once the number has wrapped,
    it names a stranger's process just as readily as the agent's. That matters in both directions -
    a tile would read as live forever, and `kill` would signal whoever inherited the number. The
    runner stamps the boot id and the process's start ticks for exactly this, and manifests written
    before it did are trusted as they were.
    """
    if not pid or not _pid_alive(pid):
        return False
    recorded_boot = str(manifest.get("runner_boot_id") or "")
    if recorded_boot and recorded_boot != _boot_id():
        return False
    recorded_start = str(manifest.get("process_start_ticks") or "")
    return not recorded_start or recorded_start == _process_start_ticks(int(pid))


def is_verifiable(state: str, _end_reason: str) -> bool:
    """Whether an attempt is eligible for a human verdict.

    One definition, shared by the API guard, the view the dashboard renders from, and the report - so
    the button the reviewer sees and the check the POST handler makes can never drift apart.
    """
    return state == "finished"


# "already_successful" is the fourth answer and, like "invalid", scores nothing - but for the
# opposite reason. The run is not broken: it was halted because another try of the same prompt had
# already been judged a success, so it never got a task to fail at. Excluded, not failed.
VERDICTS = ("pass", "fail", "invalid", "already_successful")
ALREADY_SUCCESSFUL = "already_successful"
# The verdicts that write no `verified_success` at all, because neither True nor False is honest.
EXCLUDED_VERDICTS = frozenset({"invalid", ALREADY_SUCCESSFUL})
AUTO_INVALID_OUTCOMES = frozenset({"agent_error"})


def verdict_of(manifest: dict[str, Any]) -> str:
    """The reviewer's verdict on one attempt: "pass", "fail", "invalid", "already_successful", or ""
    for unreviewed.

    One definition, shared by the dashboard's view and the report, so the two surfaces can never
    disagree about what a reviewer said.

    `verified_verdict` is authoritative when present. Attempts judged before it existed carry only
    `verified_success`, so that is the fallback - and an invalid verdict deliberately writes no
    `verified_success` at all, which keeps any reader that predates this function from silently
    totalling an excluded run as a human-confirmed failure.
    """
    recorded = str(manifest.get("verified_verdict") or "")
    if recorded in VERDICTS:
        return recorded
    if "verified_success" in manifest:
        return "pass" if manifest["verified_success"] else "fail"
    return ""


def effective_verdict(manifest: dict[str, Any]) -> str:
    """Explicit human verdict, or the watcher's automatic classification for runs nobody need judge.

    An explicit verdict always wins, so a reviewer can still inspect an agent_error and deliberately
    mark it pass or fail. Without one, agent_error behaves exactly like pressing E: it is invalid and
    excluded from review arithmetic rather than being mistaken for a task failure.

    A try the runner or the watcher halted because a sibling had already been verified successful is
    classified the same way pressing A would: `already_successful`, excluded. Nobody has to review a
    cancellation the harness performed on its own bookkeeping, so those cells leave the review queue
    instead of sitting in it as halts awaiting a verdict that would mean nothing.
    """
    explicit = verdict_of(manifest)
    if explicit:
        return explicit
    if manifest.get("end_reason") == ALREADY_SUCCESSFUL:
        return ALREADY_SUCCESSFUL
    return "invalid" if manifest.get("outcome") in AUTO_INVALID_OUTCOMES else ""


def read_response(run_dir: Path, max_chars: int = RESPONSE_MAX_CHARS) -> str:
    """The agent's final response for one attempt, or "" if it never wrote one.

    Written with an atomic replace at the very end of a run, so there is no torn-read case to defend
    against - only a missing file, which is the normal state for every attempt still in flight.
    """
    try:
        text = (run_dir / RESPONSE_FILE).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


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
    """Newest live observation.

    Agent frames retain their leg/step ordering for compatibility with coarse filesystems. A
    supplementary capture wins only when its embedded capture timestamp is newer.
    """
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
    capture_dir = run_dir / capture.CAPTURE_DIR
    newest_capture = capture.latest_capture(run_dir) if capture_dir.is_dir() else None

    if newest_capture is None:
        return best_path
    if best_path is None:
        return newest_capture
    return (
        newest_capture
        if capture.frame_timestamp_ns(newest_capture) >= capture.frame_timestamp_ns(best_path)
        else best_path
    )


def scan_attempt(run_dir: Path, battery_root: Path, now: float) -> AttemptView:
    """Builds one tile's worth of state from a run dir."""
    manifest = _read_json(run_dir / ATTEMPT_MANIFEST)
    outcome = str(manifest.get("outcome") or "")
    verdict = effective_verdict(manifest)
    view = AttemptView(
        key=f"{run_dir.parent.name}/{run_dir.name}",
        run_id=str(manifest.get("run_id") or manifest.get("started_at") or ""),
        prompt_id=str(manifest.get("prompt_id") or run_dir.parent.name),
        attempt=int(manifest.get("attempt") or 0),
        prompt=str(manifest.get("prompt") or ""),
        family=str(manifest.get("family") or ""),
        looking_for=str(manifest.get("looking_for") or ""),
        run_dir=str(run_dir),
        state=str(manifest.get("state") or "unknown"),
        outcome=outcome,
        # Repair old affected manifests at read time as well as preventing new ones in the runner.
        success=bool(manifest.get("success")) if outcome == "completed" else False,
        end_reason=str(manifest.get("end_reason") or ""),
        exit_code=manifest.get("exit_code"),
        sandbox_id=str(manifest.get("sandbox_id") or ""),
        commands_uri=str(manifest.get("commands_uri") or ""),
        pid=manifest.get("pid"),
        killed_by=str(manifest.get("killed_by") or ""),
        stop_reason=str(manifest.get("stop_reason") or ""),
        stop_requested_at=str(manifest.get("stop_requested_at") or ""),
        stop_requested_by=str(manifest.get("stop_requested_by") or ""),
        winning_attempt_key=str(manifest.get("winning_attempt_key") or ""),
        verified=bool(verdict),
        verified_verdict=verdict,
        verified_success=({"pass": True, "fail": False}.get(verdict)),
        verified_by=str(manifest.get("verified_by") or ""),
        verified_at=str(manifest.get("verified_at") or ""),
        verified_note=str(manifest.get("verified_note") or ""),
        started_at=str(manifest.get("started_at") or ""),
        max_steps=manifest.get("max_steps"),
    )

    tokens = _read_json(run_dir / "tokens.json")
    view.tokens_in = int(tokens.get("tokens_in") or manifest.get("tokens_in") or 0)
    view.tokens_out = int(tokens.get("tokens_out") or manifest.get("tokens_out") or 0)
    view.tokens_by_role = normalize_by_role(tokens.get("by_role") or manifest.get("tokens_by_role"))

    started = manifest.get("started_epoch")
    deadline = manifest.get("deadline_epoch")
    if view.state == "finished":
        view.elapsed_seconds = float(manifest.get("wall_seconds") or 0.0)
    elif isinstance(started, (int, float)):
        view.elapsed_seconds = round(now - float(started), 1)
    if view.state != "finished" and isinstance(deadline, (int, float)):
        view.remaining_seconds = round(float(deadline) - now, 1)

    if view.state in {"starting", "running"}:
        view.alive = agent_is_alive(manifest, view.pid)
        if not view.alive and view.pid:
            # The manifest says live but the process is gone: the runner died before it could close
            # the attempt out. Say so rather than showing a tile frozen forever at its last step.
            view.state = "orphaned"

    view.response = read_response(run_dir)

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


def _flocked(path: Path) -> bool:
    """Whether some process holds an flock on `path`, asked without taking one.

    Trying for the lock would answer the same question and is what the runner itself does - but the
    runner asks non-blockingly and aborts if it loses, so a watcher holding that lock for even a
    microsecond could stop a battery from starting. /proc/locks is the read-only way to ask: one line
    per held lock, carrying the owner's device and inode.
    """
    try:
        stat = path.stat()
    except OSError:
        return False
    wanted = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    try:
        with open("/proc/locks", encoding="utf-8") as handle:
            for line in handle:
                fields = line.split()
                # "<n>: FLOCK ADVISORY WRITE <pid> <maj>:<min>:<inode> <start> <end>"
                if len(fields) > 5 and fields[1] == "FLOCK" and fields[5] == wanted:
                    return True
    except OSError:
        return False  # no procfs: the recency fallback is the only signal left
    return False


def battery_live(battery: Path, now: float | None = None) -> bool:
    """Whether a runner is still working on this battery.

    The exact signal is `.runner.lock`, which a full battery runner flocks for its whole life. A
    resumed or partial runner is started without `initialize_battery` and so holds nothing, hence the
    recency fallback on the battery's own bookkeeping. Deliberately O(1) per battery - this runs for
    every battery on every state poll, so nothing here may walk the run dirs.
    """
    if _flocked(battery / RUNNER_LOCK):
        return True

    now = time.time() if now is None else now
    newest = 0.0
    for name in (BATTERY_MANIFEST, ATTEMPTS_INDEX, ""):
        try:
            newest = max(newest, (battery / name if name else battery).stat().st_mtime)
        except OSError:
            continue
    return bool(newest) and (now - newest) < LIVE_GRACE_SECONDS


def describe_batteries(paths: list[Path], now: float | None = None) -> list[dict[str, Any]]:
    """The picker's view of every battery on disk: what to call it, and whether it is running."""
    now = time.time() if now is None else now
    described = []
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        described.append({
            "id": path.name,
            "path": str(path),
            "live": battery_live(path, now),
            "mtime": mtime,
        })
    return described


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
            key = {
                "pass": "verified_success",
                "fail": "verified_fail",
                ALREADY_SUCCESSFUL: "verified_already_successful",
            }.get(attempt.verified_verdict, "verified_invalid")
            counts[key] = counts.get(key, 0) + 1
            # An invalid run is excluded, not disagreed with: the reviewer threw the attempt out
            # rather than ruling against the predicate, so it is no evidence either way.
            if attempt.verified_success is not None and attempt.verified_success != attempt.success:
                counts["disagree"] = counts.get("disagree", 0) + 1
        elif attempt.verifiable:
            counts["awaiting_verdict"] = counts.get("awaiting_verdict", 0) + 1

    return BatteryView(
        battery_id=battery.name,
        path=str(battery),
        battery=_read_json(battery / BATTERY_MANIFEST),
        attempts=[attempt.as_dict() for attempt in attempts],
        counts=counts,
        discovered=describe_batteries(discovered or [], now),
    )
