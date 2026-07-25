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
from pathlib import Path
from typing import Any

WEBHOOK_ENV = "SARI_BENCH_DISCORD_WEBHOOK"

# One alert per (attempt, kind); a flapping collapse score must not spam the channel.
COOLDOWN_SECONDS = 900.0

# Discord embed colours.
_RED = 0xE0553F
_AMBER = 0xE0A23F
_GREEN = 0x4FA96B
_BLUE = 0x4F7FA9

# Outcomes worth a message. Successes are not: at 3 tries x 20 prompts they are just noise.
_FAILURE_OUTCOMES = {"agent_error", "harness_timeout", "sandbox_lost", "harness_error"}


class Discord:
    """Fail-soft Discord webhook client with per-key cooldowns."""

    def __init__(self, webhook_url: str | None = None, *, enabled: bool = True) -> None:
        self.webhook_url = webhook_url or os.getenv(WEBHOOK_ENV, "")
        self.enabled = bool(enabled and self.webhook_url)
        self._last_sent: dict[str, float] = {}
        self._seen_finished: set[str] = set()
        self._alerted: set[str] = set()

    # -- transport ------------------------------------------------------------------------

    def _post(self, payload: dict[str, Any], image: Path | None = None) -> None:
        if not self.enabled:
            return
        try:
            if image is not None and image.exists():
                body, content_type = _multipart(payload, image)
            else:
                body, content_type = json.dumps(payload).encode("utf-8"), "application/json"
            request = urllib.request.Request(
                self.webhook_url, data=body, headers={"Content-Type": content_type}, method="POST"
            )
            urllib.request.urlopen(request, timeout=10).close()
        except (urllib.error.URLError, OSError, ValueError) as error:  # noqa: BLE001
            print(f"[sari-bench watch] discord notify failed: {error!r}", flush=True)

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
        self._post({"embeds": [embed]}, image=frame)

    def attempt_finished(self, attempt: dict[str, Any]) -> None:
        key = attempt.get("key", "")
        if key in self._seen_finished:
            return
        self._seen_finished.add(key)
        outcome = attempt.get("outcome", "")
        if outcome not in _FAILURE_OUTCOMES:
            return
        if not self._cooled(f"finished:{key}"):
            return
        self._post({
            "embeds": [{
                "title": f"Attempt failed: {attempt.get('prompt_id')} try {attempt.get('attempt')}",
                "description": attempt.get("prompt", "")[:400],
                "color": _AMBER,
                "fields": [
                    _field("Outcome", outcome),
                    _field("End reason", attempt.get("end_reason") or "-"),
                    _field("Exit code", attempt.get("exit_code")),
                    _field("Wall", _minutes(attempt.get("elapsed_seconds"))),
                    _field("Sandbox", attempt.get("sandbox_id") or "?"),
                ],
            }]
        })

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
        }, image=(attachments or [None])[0])


def _field(name: str, value: Any, *, inline: bool = True) -> dict[str, Any]:
    return {"name": str(name), "value": f"`{value}`" if value not in (None, "") else "`-`",
            "inline": inline}


def _minutes(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    return f"{seconds / 60:.1f} min"


def _multipart(payload: dict[str, Any], image: Path) -> tuple[bytes, str]:
    """Builds a multipart/form-data body so an embed can carry its screenshot."""
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    parts: list[bytes] = []

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="payload_json"\r\n')
    parts.append(b"Content-Type: application/json\r\n\r\n")
    parts.append(json.dumps(payload).encode("utf-8") + b"\r\n")

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="files[0]"; filename="{image.name}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    parts.append(image.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
