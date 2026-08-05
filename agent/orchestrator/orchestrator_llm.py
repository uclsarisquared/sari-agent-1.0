"""Model configuration and orchestrator-level LLM calls."""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent_core import token_meter
from agent_core.agent import call_with_api_retries, agent_vlm_config
from agent_core.context_policy import ContextPolicy
from agent_core.models import agent_model
from orchestrator.subtask_completion import TYPED_DECOMPOSER_SYSTEM, parse_decomposition

load_dotenv(Path(__file__).resolve().parent.parent.parent / "config.env")


ORCHESTRATOR_MODEL = agent_model()  # $SARI_MODEL in config.env (OpenRouter retired 2026-07-19)

# Every reasoner runs on the OpenAI API compatible endpoint from config.env (OpenRouter fully
# retired 2026-07-21). agent_vlm_config carries the load-bearing enable_thinking=False +
# max_tokens cap - see agent.agent_vlm_config. Mirrors eval_pickup.py / env_simulation.py. The
# orchestrator LLM below (_llm_client) already targets the same endpoint.
VLM_CONFIG = agent_vlm_config(temperature=0.5)
ASSOCIATIVE_CONFIG = agent_vlm_config(temperature=0.3)


# ---------------------------------------------------------------------------
# Orchestrator LLM helpers
# ---------------------------------------------------------------------------

def _llm_client() -> OpenAI:
    from agent_core.agent import _endpoint_creds
    host, key = _endpoint_creds()
    return OpenAI(base_url=f"http://{host}:8000/v1", api_key=key, max_retries=0)


def _llm_call(client: OpenAI, system: str, user: str, role: str) -> str:
    """One orchestrator-level completion. `role` is which reasoner to bill it to - this helper serves
    the decomposer, findings reporter, and final responder, which are separately measurable, so the
    caller must say which one it is rather than letting them pool into one unreadable number."""
    with token_meter.role(role):
        resp = call_with_api_retries(
            lambda: client.chat.completions.create(
                model=ORCHESTRATOR_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.3,
                timeout=120,
            )
        )
    return resp.choices[0].message.content


def decompose_task(client: OpenAI, task: str) -> list:
    """Phase 6.3: returns a list of TYPED subtask dicts ({"type", "text", ...}), not free strings, so
    each leg's completion is checked by a code-side predicate keyed on its type instead of by grepping
    its prose (the pre-6.3 keyword guards). The type vocabulary is closed (pickup|checkout|compare|
    goto); any untypeable element degrades to `{"type": "unknown"}` inside parse_decomposition, which
    run_leg then handles with the OLD keyword guards. The A/B that validated this prompt lives in
    tests/ab_decompose.py (11/11 clean on the four-family battery, 2026-07-23)."""
    raw = _llm_call(client, TYPED_DECOMPOSER_SYSTEM, f"Task: {task}", token_meter.ROLE_DECOMPOSER)
    subtasks = parse_decomposition(raw, task)
    if any(s.get("type") == "unknown" for s in subtasks):
        print("[WARN] Decomposition had untypeable element(s) — those legs fall back to keyword guards "
              "(logged as `untyped`).")
    return subtasks


def generate_findings_summary(
    client: OpenAI,
    completed_subtask: str,
    final_state: dict,
    new_semantic_entries: str,
    context_policy: ContextPolicy = ContextPolicy(),
) -> str:
    """
    Comprehensive summary of everything the agent found/learned during a subtask.
    Passed to the orchestrator so all future subtask agents receive accumulated context.
    """
    if context_policy.findings_max_chars is None:
        system = (
            "You are a findings reporter for an Embodied AI Agent in a 3D convenience "
            "store simulation. After a subtask completes, produce a comprehensive findings "
            "summary for future agent instances. Include ALL of the following:\n"
            "  1. POSITION: Current agent position in plain English (near which shelf/counter).\n"
            "  2. HANDS: What each hand is holding (gripped items, or empty).\n"
            "  3. OBJECTS LOCATED: Every object/item seen and its approximate shelf or position.\n"
            "  4. NAVIGATION INSIGHTS: Which paths/routes worked; where the agent got stuck or lost.\n"
            "  5. SEMANTIC LEARNINGS: Key facts about the store environment learned this subtask.\n"
            "  6. WHAT TO AVOID: Any approaches that failed or cost unnecessary time.\n"
            "  7. UPCOMING TASK PREP: Specific observations that will help with future subtasks.\n"
            "Be comprehensive and factual. Future agents cannot re-explore what you already found, "
            "so document every useful detail."
        )
    else:
        system = (
            "Write a concise factual handoff for the next store-agent subtask. State only the "
            "current position, what each hand holds, useful object locations/routes, failed "
            "approaches to avoid, and facts that directly prepare the remaining task. Use compact "
            "sentences and no preamble."
        )
    user = (
        f"Completed subtask: {completed_subtask}\n\n"
        f"Final agent state:\n{json.dumps(final_state, indent=2, default=str)}\n\n"
        f"New semantic memory entries learned during this subtask:\n{new_semantic_entries}"
    )
    findings = _llm_call(client, system, user, token_meter.ROLE_FINDINGS)
    if context_policy.findings_max_chars is not None:
        findings = findings[: context_policy.findings_max_chars]
    return findings


def _generate_findings_if_enabled(
    policy: ContextPolicy,
    client: OpenAI,
    completed_subtask: str,
    final_state: dict,
    new_semantic_entries: str,
) -> str | None:
    """Return a retained handoff, or no work at all for A3."""
    if not policy.findings_enabled:
        return None
    return generate_findings_summary(
        client,
        completed_subtask=completed_subtask,
        final_state=final_state,
        new_semantic_entries=new_semantic_entries,
        context_policy=policy,
    )
