"""LLM-backed actor and associative learner implementations."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from PIL import Image

from agent_core import token_meter
from agent_core.context import SemanticLog
from agent_core.context_policy import ContextPolicy, validate_context_policy
from agent_core.llm import BaseAgent, LLMConfig, build_content
from agent_core.sys_inst import SYS_INST_VLM_LEAN


class AssociativeLearner(BaseAgent):
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        super().__init__(config)
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key, max_retries=0)

    def generate_content(
        self,
        system_instruction: str,
        image: Optional[Image.Image],
        text: str,
    ) -> str:
        content = build_content(image, text) if image else [{"type": "text", "text": text}]
        return self._api_call_with_retry(
            self.client,
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content},
            ],
        )


SemanticEpisodicAssociativeLearner = AssociativeLearner


class VLMAgent(BaseAgent):
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        context_policy: ContextPolicy = ContextPolicy(),
    ) -> None:
        super().__init__(config)
        self.context_policy = validate_context_policy(context_policy)
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key, max_retries=0)
        self.history: List[Dict[str, Any]] = []
        # Retained here for compatibility. MemoryRuntime owns mutation/persistence at runtime.
        self.episodic_memory = ""
        self.semantic_log = SemanticLog("", self.context_policy)
        logger.info(f"VLMAgent initialized with model: {self.config.model_id}")

    def reset_history(self) -> None:
        self.history = []

    def send_message(self, content: list) -> str:
        self.history.append({"role": "user", "content": content})
        messages = [
            {"role": "system", "content": SYS_INST_VLM_LEAN},
            *self._outbound_history(),
        ]
        with token_meter.role(token_meter.ROLE_ACTOR):
            reply = self._api_call_with_retry(self.client, messages)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _outbound_history(self) -> list[dict[str, Any]]:
        keep = self.context_policy.actor_image_history
        if keep is None:
            return self.history

        user_indices = [
            index for index, message in enumerate(self.history) if message.get("role") == "user"
        ]
        newest = set(user_indices[-keep:])
        outbound: list[dict[str, Any]] = []
        for index, message in enumerate(self.history):
            content = message.get("content")
            if index not in newest and message.get("role") == "user" and isinstance(content, list):
                filtered = [
                    copy.deepcopy(part)
                    for part in content
                    if not (isinstance(part, dict) and part.get("type") == "image_url")
                ]
                outbound.append({**message, "content": filtered})
            else:
                outbound.append(message)
        return outbound

    def get_history_text(self, n: int = 8) -> str:
        result = ""
        for message in self.history[-n:]:
            role = message["role"]
            content = message["content"]
            if isinstance(content, list):
                text = " ".join(part["text"] for part in content if part.get("type") == "text")
            else:
                text = content
            result += f"{role.upper()}: {text}\n"
        return result.strip()
