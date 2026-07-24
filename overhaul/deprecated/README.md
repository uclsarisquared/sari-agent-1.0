# deprecated/

Superseded code, kept for reference (git history has it too, but these carry design context worth
keeping greppable). Nothing imports anything in here.

- **`subagent_run.py`** (moved 2026-07-24) — the OpenRouter-era multi-step orchestrator: hardcodes
  `openrouter.ai` + `OPENROUTER_API_KEY` + `google/gemini-3.1-pro-preview`. OpenRouter was retired
  for agent calls when its credits ran out (agent runtime is the UCL vLLM server now), so this
  cannot run as-is. Superseded by `subtask_agents.py` (typed-subtask orchestrator, UCL qwen). Its
  decompose→run→handoff structure is the ancestor of subtask_planning/subtask_agents.
