"""Discord webhook notifications, driven off the watcher's poll loop.

This lives in the watcher rather than the runner on purpose. The runner is a scheduler: giving it an
outbound HTTP dependency puts a network call on the path that releases a lease. The watcher already
computes health, already sees the whole battery, and is the one place that can hold cooldown state -
so it is where notification belongs.

Everything here is best-effort and swallows its own errors. A webhook that 429s, a DNS failure, or
a missing URL must never disturb the poll loop, let alone a battery.

Configure with SARI_BENCH_DISCORD_WEBHOOK in api.env (or the environment).
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

WEBHOOK_ENV = "SARI_BENCH_DISCORD_WEBHOOK"

# One alert per (attempt, kind); a flapping collapse score must not spam the channel.
COOLDOWN_SECONDS = 900.0

# A text post should fail fast; an 8 MB replay upload needs room to actually finish. The long timeout
# is only ever paid on the replay worker thread, which is off every hot path.
POST_TIMEOUT_SECONDS = 10.0
UPLOAD_TIMEOUT_SECONDS = 120.0

# Discord rejects a webhook attachment over 10 MB. Stay clear of the wall rather than eat a 413.
MAX_ATTACHMENT_BYTES = 9_500_000

# One retry on a rate limit, honouring Retry-After up to this many seconds.
MAX_RETRY_AFTER_SECONDS = 30.0

# Discord embed colours.
_RED = 0xE0553F
_AMBER = 0xE0A23F
_GREEN = 0x4FA96B
_BLUE = 0x4F7FA9

# Every finish is announced, successes included. This reverses an earlier decision to post failures
# only: back then a success was a bare line of metrics, which really was noise at 3 tries x 20 prompts.
# Now each message carries the attempt's replay, and a clip of a win is worth watching too.
_HARNESS_FAILURES = {"agent_error", "harness_timeout", "sandbox_lost", "harness_error"}


class Discord:
    """Fail-soft Discord webhook client with per-key cooldowns."""

    def __init__(self, webhook_url: str | None = None, *, enabled: bool = True) -> None:
        self.webhook_url = webhook_url or os.getenv(WEBHOOK_ENV, "")
        self.enabled = bool(enabled and self.webhook_url)
        self._last_sent: dict[str, float] = {}
        self._seen_finished: set[str] = set()
        self._alerted: set[str] = set()

    # -- transport ------------------------------------------------------------------------

    def _post(self, payload: dict[str, Any], attachment: Path | None = None) -> None:
        if not self.enabled:
            return
        try:
            usable = _usable_attachment(attachment)
            if usable is not None:
                body, content_type = _multipart(payload, usable)
                timeout = UPLOAD_TIMEOUT_SECONDS
            else:
                body, content_type = json.dumps(payload).encode("utf-8"), "application/json"
                timeout = POST_TIMEOUT_SECONDS
            request = urllib.request.Request(
                self.webhook_url, data=body, headers={"Content-Type": content_type}, method="POST"
            )
            self._send(request, timeout, retry=True)
        except (urllib.error.URLError, OSError, ValueError) as error:  # noqa: BLE001
            print(f"[sari-bench watch] discord notify failed: {error!r}", flush=True)

    def _send(self, request: urllib.request.Request, timeout: float, *, retry: bool) -> None:
        """One attempt, plus one retry if Discord asks us to wait.

        Worth the retry because these notifications do arrive in bursts - reopening the dashboard after
        it has been closed a while flushes every finish that piled up - and a dropped burst is exactly
        what the operator asked not to miss. Beyond one retry we give up: a poll loop that keeps
        hammering a rate-limited webhook is how you get muted for good.
        """
        try:
            urllib.request.urlopen(request, timeout=timeout).close()
        except urllib.error.HTTPError as error:
            if not (retry and error.code == 429):
                raise
            delay = _retry_after(error)
            print(f"[sari-bench watch] discord rate limited, retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
            self._send(request, timeout, retry=False)

    def _cooled(self, key: str) -> bool:
        now = time.monotonic()
        if now - self._last_sent.get(key, -1e9) < COOLDOWN_SECONDS:
            return False
        self._last_sent[key] = now
        return True

    # -- events ---------------------------------------------------------------------------

    def battery_started(self, view: dict[str, Any], fleet_size: int) -> None:
        battery = view.get("battery") or {}
        if not self._cooled(f"start:{view.get('battery_id')}"):
            return
        self._post({
            "embeds": [{
                "title": f"Battery started: {view.get('battery_id')}",
                "color": _BLUE,
                "fields": [
                    _field("Prompts", len(battery.get("prompts") or [])),
                    _field("Attempts planned", battery.get("planned_attempts", "?")),
                    _field("Concurrency", battery.get("concurrency", "?")),
                    _field("Arm", battery.get("arm", "?")),
                    _field("Time limit", f"{battery.get('time_limit_minutes', '?')} min"),
                    _field("Sandboxes in pool", fleet_size),
                ],
            }]
        })

    def collapse(self, attempt: dict[str, Any], frame: Path | None) -> None:
        """Fires once per attempt when its score first crosses into alert.

        The frame is the payload that matters: a still of the agent nose-to-nose with a shelf tells
        you in one glance what no metric does, and it is what lets you decide to kill from a phone.
        """
        key = attempt.get("key", "")
        if key in self._alerted or not self._cooled(f"collapse:{key}"):
            return
        self._alerted.add(key)
        health = attempt.get("health") or {}
        embed = {
            "title": f"Possible collapse: {attempt.get('prompt_id')} try {attempt.get('attempt')}",
            "description": attempt.get("prompt", "")[:400],
            "color": _RED,
            "fields": [
                _field("Signals", health.get("summary", "?"), inline=False),
                _field("Score", f"{health.get('score', 0):.2f}"),
                _field("Step", f"{attempt.get('step')} / {attempt.get('max_steps')}"),
                _field("Elapsed", _minutes(attempt.get("elapsed_seconds"))),
                _field("Remaining", _minutes(attempt.get("remaining_seconds"))),
                _field("Mode", attempt.get("mode") or "?"),
                _field("Sandbox", attempt.get("sandbox_id") or "?"),
            ],
        }
        if frame is not None and frame.exists():
            embed["image"] = {"url": f"attachment://{frame.name}"}
        self._post({"embeds": [embed]}, attachment=frame)

    def attempt_finished(self, attempt: dict[str, Any], video: Path | None = None) -> None:
        """Announces one halt, with its replay clip when the watcher managed to render one.

        The clip is deliberately not referenced as ``attachment://`` the way `collapse` references its
        frame: that only renders for stills. An unreferenced video attachment shows up as a player
        under the embed, which also means a missing clip needs no change to the payload.
        """
        key = attempt.get("key", "")
        if key in self._seen_finished:
            return
        self._seen_finished.add(key)
        if not self._cooled(f"finished:{key}"):
            return
        title, color = _finish_style(attempt)
        self._post({
            "embeds": [{
                "title": f"{title}: {attempt.get('prompt_id')} try {attempt.get('attempt')}",
                "description": attempt.get("prompt", "")[:400],
                "color": color,
                "fields": [
                    _field("Outcome", attempt.get("outcome", "")),
                    _field("Success", attempt.get("success")),
                    _field("End reason", attempt.get("end_reason") or "-"),
                    _field("Exit code", attempt.get("exit_code")),
                    _field("Wall", _minutes(attempt.get("elapsed_seconds"))),
                    _field("Sandbox", attempt.get("sandbox_id") or "?"),
                ],
            }]
        }, attachment=video)

    def suppress_finished(self, keys: Iterable[str]) -> None:
        """Marks halts as already-announced without sending anything.

        The watcher uses this on its first pass over a battery so that restarting it mid-run does not
        replay every finish that happened while it was down.
        """
        self._seen_finished.update(keys)

    def battery_finished(self, view: dict[str, Any], attachments: list[Path] | None = None) -> None:
        battery_id = view.get("battery_id")
        if not self._cooled(f"done:{battery_id}"):
            return
        counts = view.get("counts") or {}
        attempts = view.get("attempts") or []
        successes = sum(1 for a in attempts if a.get("success"))
        lines = [
            f"{name}: {count}" for name, count in sorted(counts.items()) if name != "success"
        ]
        self._post({
            "embeds": [{
                "title": f"Battery complete: {battery_id}",
                "color": _GREEN if successes else _AMBER,
                "description": "\n".join(lines) or "no attempts recorded",
                "fields": [
                    _field("Succeeded", f"{successes}/{len(attempts)}"),
                    _field("Arm", (view.get("battery") or {}).get("arm", "?")),
                ],
            }]
        }, attachment=(attachments or [None])[0])


def _finish_style(attempt: dict[str, Any]) -> tuple[str, int]:
    """Title verb and colour for a halt.

    `completed` only says the agent exited cleanly; `success` is the goal check. An attempt that ran to
    the end and missed the goal is the normal, interesting case - amber, not red, which is reserved for
    the harness itself going wrong.
    """
    outcome = attempt.get("outcome", "")
    if attempt.get("success"):
        return "Attempt succeeded", _GREEN
    if outcome == "operator_kill":
        return "Attempt killed", _BLUE
    if outcome in _HARNESS_FAILURES:
        return "Attempt failed", _RED
    return "Attempt finished, goal not met", _AMBER


def _usable_attachment(path: Path | None) -> Path | None:
    """The file if it can actually be uploaded, else None so the caller falls back to a text post."""
    if path is None or not path.is_file():
        return None
    size = path.stat().st_size
    if size == 0 or size > MAX_ATTACHMENT_BYTES:
        print(f"[sari-bench watch] skipping attachment {path.name} ({size / 1e6:.1f} MB)", flush=True)
        return None
    return path


def _retry_after(error: urllib.error.HTTPError) -> float:
    header = (error.headers or {}).get("Retry-After", "")
    try:
        delay = float(header)
    except (TypeError, ValueError):
        delay = 1.0
    return min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS)


def _field(name: str, value: Any, *, inline: bool = True) -> dict[str, Any]:
    return {"name": str(name), "value": f"`{value}`" if value not in (None, "") else "`-`",
            "inline": inline}


def _minutes(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    return f"{seconds / 60:.1f} min"


def _multipart(payload: dict[str, Any], attachment: Path) -> tuple[bytes, str]:
    """Builds a multipart/form-data body so an embed can carry a screenshot or a replay clip.

    `mimetypes` picks the part's content type, so an .mp4 arrives as video/mp4 and Discord gives it a
    player. Note this buffers the file twice - once in `read_bytes`, once in the join - so an 8 MB clip
    peaks around 24 MB. Fine on a fleet host, but it is why the attachment cap is not generous.
    """
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(attachment.name)[0] or "application/octet-stream"
    parts: list[bytes] = []

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="payload_json"\r\n')
    parts.append(b"Content-Type: application/json\r\n\r\n")
    parts.append(json.dumps(payload).encode("utf-8") + b"\r\n")

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="files[0]"; filename="{attachment.name}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    parts.append(attachment.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
