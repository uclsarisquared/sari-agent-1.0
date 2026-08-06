"""Compatibility exports for the agent runtime's Markdown prompt assets."""

from agent_core.prompt_loader import load_prompt, render_prompt
from toolset.actions_str import ATOMIC_ACTIONS


AGENT_STATE_DOC = load_prompt("runtime/agent_state")

SYS_INST_ASSOCIATIVE_SEMANTIC = render_prompt(
    "runtime/semantic_associative",
    AGENT_STATE_DOC=AGENT_STATE_DOC,
)

SYS_INST_ASSOCIATIVE_EPISODIC = load_prompt("runtime/episodic_associative")

SYS_INST_VLM_LEAN = render_prompt(
    "runtime/actor",
    AGENT_STATE_DOC=AGENT_STATE_DOC,
    ATOMIC_ACTIONS=ATOMIC_ACTIONS,
)
