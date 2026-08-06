"""Compatibility façade for the decomposed embodied-agent runtime.

New code should import from the focused modules. Existing runners and tests may
continue importing the historical names from ``agent_core.agent``.
"""

from agent_core.actors import (
    SemanticEpisodicAssociativeLearner,
    VLMAgent,
)
from agent_core.contracts import (
    available_actions as _available_actions,
    parse_semantic_response as _parse_semantic_response,
    resolve_agent_mode as _resolve_agent_mode,
    stop_response as _stop_response,
)
from agent_core.llm import (
    BaseAgent,
    LLMConfig,
    OpenRouterConfig,
    agent_vlm_config,
    call_with_api_retries,
)
from agent_core.runtime import EmbodiedAgent


__all__ = [
    "BaseAgent",
    "EmbodiedAgent",
    "LLMConfig",
    "OpenRouterConfig",
    "SemanticEpisodicAssociativeLearner",
    "VLMAgent",
    "_available_actions",
    "_parse_semantic_response",
    "_resolve_agent_mode",
    "_stop_response",
    "agent_vlm_config",
    "call_with_api_retries",
]
