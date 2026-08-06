"""Semantic-memory journal and rendering policies."""

from __future__ import annotations

import difflib

from agent_core.context_policy import ContextPolicy, validate_context_policy


class SemanticLog:
    """Immutable base memory plus accepted learned entries.

    Retention is a render-time concern, so a leg marker can still recover entries that A1/A2c no
    longer expose to prompts.
    """

    def __init__(self, base_text: str, policy: ContextPolicy) -> None:
        self._base_text = base_text
        self._policy = validate_context_policy(policy)
        self._entries: list[tuple[str, str]] = []

    @property
    def base_text(self) -> str:
        return self._base_text

    @property
    def entries(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._entries)

    def append(self, tag: str, text: str) -> bool:
        threshold = self._policy.semantic_dedupe
        if threshold is not None:
            recent = self._entries[-self._policy.semantic_dedupe_window :]
            if any(
                difflib.SequenceMatcher(None, text, existing).ratio() > threshold
                for _existing_tag, existing in recent
            ):
                return False
        self._entries.append((tag, text))
        return True

    def render(self) -> str:
        keep_last = self._policy.semantic_keep_last
        entries = self._entries if keep_last is None else self._entries[-keep_last:]
        # ``[-0:]`` means all entries in Python, while A1 intentionally means none.
        if keep_last == 0:
            entries = []
        return self._base_text + self._format(entries)

    def mark(self) -> int:
        return len(self._entries)

    def since(self, mark: int) -> str:
        if not isinstance(mark, int) or mark < 0 or mark > len(self._entries):
            raise ValueError(f"invalid semantic log mark: {mark!r}")
        return self._format(self._entries[mark:])

    @staticmethod
    def _format(entries: list[tuple[str, str]]) -> str:
        return "".join(f"{tag}: {text}\n" for tag, text in entries)
