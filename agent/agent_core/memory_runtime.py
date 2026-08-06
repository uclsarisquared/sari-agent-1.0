"""Attempt-local semantic and episodic memory lifecycle."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from loguru import logger

from agent_core.context import SemanticLog
from agent_core.context_policy import ContextPolicy
from agent_core.memory import base_semantic_memory


class MemoryRuntime:
    """Manage one run's semantic and episodic memory state and artifacts."""
    def __init__(
        self,
        vlm_agent=None,
        *,
        context_policy: ContextPolicy = ContextPolicy(),
        map_output_dir: Optional[str] = None,
        run_dir: Optional[str] = None,
        leg: Optional[int] = None,
    ) -> None:
        self.vlm_agent = vlm_agent
        self.context_policy = context_policy
        self.map_output_dir = map_output_dir
        active_run_dir = run_dir or os.environ.get("SARI_RUN_DIR")
        self.run_dir = os.path.abspath(active_run_dir) if active_run_dir else None
        self.leg = leg

    def reset_semantic(self) -> None:
        self.vlm_agent.semantic_log = SemanticLog(
            base_semantic_memory(self.map_output_dir), self.context_policy
        )
        logger.info("Base semantic memory set for VLMAgent.")

    def set_episodic(self, episodic_memory: str) -> None:
        self.vlm_agent.episodic_memory = episodic_memory
        logger.info(f"Episodic memory updated: {episodic_memory}")

    def semantic_tag(self, timestep: int) -> str:
        if self.leg is not None:
            return f"@ leg {self.leg} step {timestep}"
        return f"@ timestep {timestep}"

    def artifact_path(self, name: str) -> str:
        return os.path.join(self.run_dir, name) if self.run_dir else name

    @staticmethod
    def write_text_atomic(path: str, content: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def persist(self) -> None:
        self.write_text_atomic(
            self.artifact_path("semantic_memory.txt"), self.vlm_agent.semantic_log.render()
        )
        self.write_text_atomic(
            self.artifact_path("episodic_memory.txt"), self.vlm_agent.episodic_memory
        )
