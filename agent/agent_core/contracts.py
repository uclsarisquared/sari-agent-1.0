"""Typed response contracts and pure routing helpers for the embodied agent."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Optional

from loguru import logger

from toolset.actions_str import (
    INSPECTION_ACTIONS,
    MANIPULATION_ACTIONS,
    NAVIGATION_ACTIONS,
    PERCEPTION_ACTIONS,
)


JSON_BLOCK_PATTERN = re.compile(r"```\s*json\s*([\s\S]*?)\s*```", re.DOTALL)


class AgentMode(str, Enum):
    """The high-level execution routes a semantic decision may select."""
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    STOP = "STOP"


@dataclass(frozen=True)
class SemanticDecision:
    """Validated semantic learner output used to route and guide one timestep."""
    new_semantic_memory: str = ""
    recall: str = ""
    mode: AgentMode = AgentMode.NAVIGATION
    next_action: Optional[str] = None
    reported_answer: Any = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticDecision":
        raw_mode = value.get("mode", AgentMode.NAVIGATION.value)
        try:
            mode = AgentMode(raw_mode)
        except (TypeError, ValueError):
            logger.warning(f"[learner] unsupported mode {raw_mode!r}; using navigation fallback")
            mode = AgentMode.NAVIGATION
        return cls(
            new_semantic_memory=value.get("new_semantic_memory", ""),
            recall=value.get("recall", ""),
            mode=mode,
            next_action=value.get("next_action"),
            reported_answer=value.get("reported_answer", ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "new_semantic_memory": self.new_semantic_memory,
            "recall": self.recall,
            "mode": self.mode.value,
            "next_action": self.next_action,
            "reported_answer": self.reported_answer,
        }


@dataclass(frozen=True)
class EpisodicReflection:
    """Compact hindsight record generated from recent actor interaction."""
    dense_summary: str = ""
    what_worked: str = ""
    what_to_avoid: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodicReflection":
        return cls(
            dense_summary=value.get("dense_summary", ""),
            what_worked=value.get("what_worked", ""),
            what_to_avoid=value.get("what_to_avoid", ""),
        )


@dataclass(frozen=True)
class NavigationResult:
    """Navigation feedback passed to the actor as text and an optional fresh frame."""
    note: str = ""
    image_bytes: Optional[bytes] = None


_SEMANTIC_FALLBACK = SemanticDecision().as_dict()
_EPISODIC_FALLBACK = EpisodicReflection().__dict__


def extract_json(pattern: re.Pattern[str], text: str) -> str:
    """Extract a fenced JSON-like payload, falling back to the raw reply."""
    match = re.search(pattern, text)
    return match[1] if match else text.strip()


def safe_literal_dict(
    pattern: re.Pattern[str],
    text: str,
    fallback: Mapping[str, Any],
    *,
    tag: str = "parse",
) -> dict[str, Any]:
    """Parse the endpoint's legacy Python-literal dictionary without aborting a step."""
    raw = extract_json(pattern, text)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return {**fallback, **parsed}
        logger.warning(f"[{tag}] reply parsed but was not a dict; using fallback")
    except (SyntaxError, ValueError, TypeError) as error:
        logger.warning(
            f"[{tag}] unparseable reply ({type(error).__name__}: {error}); using fallback"
        )
    return dict(fallback)


def parse_semantic_response(pattern: re.Pattern[str], text: str) -> dict[str, Any]:
    parsed = safe_literal_dict(pattern, text, _SEMANTIC_FALLBACK, tag="learner")
    if "mode" not in parsed:
        logger.warning("[learner] reply missing 'mode'; using navigation fallback")
        parsed["mode"] = AgentMode.NAVIGATION.value
    return SemanticDecision.from_mapping(parsed).as_dict()


def parse_semantic_decision(pattern: re.Pattern[str], text: str) -> SemanticDecision:
    return SemanticDecision.from_mapping(parse_semantic_response(pattern, text))


def parse_episodic_reflection(pattern: re.Pattern[str], text: str) -> EpisodicReflection:
    parsed = safe_literal_dict(pattern, text, _EPISODIC_FALLBACK, tag="episodic")
    return EpisodicReflection.from_mapping(parsed)


def stop_response(decision: SemanticDecision | Mapping[str, Any], raw_text: str) -> dict[str, Any]:
    answer = (
        decision.reported_answer
        if isinstance(decision, SemanticDecision)
        else decision.get("reported_answer", "")
    )
    return {
        "halt": True,
        "text": "STOP action received, terminating execution...",
        "agent_mode": AgentMode.STOP.value,
        "reported_answer": answer if isinstance(answer, str) else "",
        "semantic": raw_text,
    }


def resolve_agent_mode(
    agent_mode: str | AgentMode,
    force_navigate: bool = False,
    force_manipulate: bool = False,
    inspect_mode: Optional[str] = None,
) -> str:
    mode = agent_mode.value if isinstance(agent_mode, AgentMode) else agent_mode
    if mode == AgentMode.STOP.value:
        return mode
    if inspect_mode == "held":
        return AgentMode.MANIPULATION.value
    if inspect_mode == "visual":
        return AgentMode.PERCEPTION.value
    if force_navigate:
        return (
            AgentMode.NAVIGATION.value
            if mode in (AgentMode.PERCEPTION.value, AgentMode.MANIPULATION.value)
            else mode
        )
    if force_manipulate and mode == AgentMode.PERCEPTION.value:
        return AgentMode.MANIPULATION.value
    return mode


def available_actions(agent_mode: str | AgentMode, held_item_inspection: bool = False) -> str:
    mode = agent_mode.value if isinstance(agent_mode, AgentMode) else agent_mode
    if mode == AgentMode.PERCEPTION.value:
        return f"{PERCEPTION_ACTIONS}\n\n"
    if mode == AgentMode.NAVIGATION.value:
        return f"{NAVIGATION_ACTIONS}\n\n"
    if mode == AgentMode.MANIPULATION.value:
        actions = INSPECTION_ACTIONS if held_item_inspection else MANIPULATION_ACTIONS
        return f"{actions}\n\n"
    raise ValueError(f"unsupported agent mode: {mode!r}")


def reach_move_steps(last_reach: Any) -> Optional[int]:
    if not isinstance(last_reach, str) or not last_reach.startswith("MOVE"):
        return None
    match = re.search(r"move_forward\s+(\d+)", last_reach)
    return int(match.group(1)) if match else None
