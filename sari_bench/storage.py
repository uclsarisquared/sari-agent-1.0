"""Small, process-safe persistence helpers shared by the runner and watcher."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ATTEMPTS_LOCK = ".attempts.lock"
BATTERY_LOCK = ".battery.lock"
RUNNER_LOCK = ".runner.lock"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a JSON object with rename, so polling readers never see a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temp, path)


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Hold an advisory process lock for the duration of the context."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def edit_json_locked(path: Path, lock_name: str = BATTERY_LOCK) -> Iterator[dict[str, Any]]:
    """Read-modify-write one JSON object while holding its sibling lock."""
    with file_lock(path.parent / lock_name):
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not a JSON object")
        yield payload
        write_json_atomic(path, payload)


def read_jsonl_rows(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    """Read JSON-object lines, optionally rejecting malformed content."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            if strict:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            continue
        if not isinstance(row, dict):
            if strict:
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            continue
        rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False, default=str)}\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temp, path)


def canonical_attempt_rows(
    output_dir: Path,
    *,
    strict: bool = False,
    rewrite: bool = False,
) -> list[dict[str, Any]]:
    """Return the latest row per logical attempt, optionally compacting the file."""
    path = output_dir / "attempts.jsonl"
    with file_lock(output_dir / ATTEMPTS_LOCK):
        rows = read_jsonl_rows(path, strict=strict)
        latest: dict[tuple[str, int], dict[str, Any]] = {}
        order: list[tuple[str, int]] = []
        for index, row in enumerate(rows, 1):
            prompt_id = str(row.get("prompt_id") or "")
            try:
                attempt = int(row.get("attempt"))
            except (TypeError, ValueError) as error:
                if strict:
                    raise ValueError(
                        f"{path}: row {index} has an invalid attempt number"
                    ) from error
                continue
            if not prompt_id or attempt < 1:
                if strict:
                    raise ValueError(f"{path}: row {index} has no valid logical attempt key")
                continue
            key = (prompt_id, attempt)
            if key not in latest:
                order.append(key)
            latest[key] = row
        canonical = [latest[key] for key in order]
        if rewrite and (rows != canonical or (path.exists() and not path.read_text().endswith("\n"))):
            write_jsonl_atomic(path, canonical)
        return canonical


def upsert_attempt_row(output_dir: Path, row: dict[str, Any]) -> None:
    """Replace one logical attempt row atomically, preserving other rows' order."""
    prompt_id = str(row.get("prompt_id") or "")
    attempt = int(row.get("attempt") or 0)
    if not prompt_id or attempt < 1:
        raise ValueError("attempt result has no valid prompt_id/attempt key")

    path = output_dir / "attempts.jsonl"
    with file_lock(output_dir / ATTEMPTS_LOCK):
        rows = read_jsonl_rows(path)
        replacement = (prompt_id, attempt)
        kept: list[dict[str, Any]] = []
        replaced = False
        for existing in rows:
            try:
                existing_attempt = int(existing.get("attempt") or 0)
            except (TypeError, ValueError):
                existing_attempt = 0
            key = (str(existing.get("prompt_id") or ""), existing_attempt)
            if key == replacement:
                if not replaced:
                    kept.append(row)
                    replaced = True
                continue
            kept.append(existing)
        if not replaced:
            kept.append(row)
        write_jsonl_atomic(path, kept)


def purge_attempt_rows(output_dir: Path, prompt_id: str, attempt: int) -> None:
    path = output_dir / "attempts.jsonl"
    with file_lock(output_dir / ATTEMPTS_LOCK):
        rows = read_jsonl_rows(path)
        kept: list[dict[str, Any]] = []
        for row in rows:
            try:
                row_attempt = int(row.get("attempt") or 0)
            except (TypeError, ValueError):
                row_attempt = 0
            if str(row.get("prompt_id") or "") == prompt_id and row_attempt == attempt:
                continue
            kept.append(row)
        if path.exists():
            write_jsonl_atomic(path, kept)
