# deprecated/

Superseded code, kept for reference (git history has it too, but these carry design context worth
keeping greppable). Nothing imports anything in here.

- **`subagent_run.py`** (moved 2026-07-24) — the OpenRouter-era multi-step orchestrator: hardcodes
  `openrouter.ai` + `OPENROUTER_API_KEY` + `google/gemini-3.1-pro-preview`. OpenRouter was retired
  for agent calls when its credits ran out (the agent runtime now targets the OpenAI API
  compatible endpoint in `config.env`), so this cannot run as-is. Superseded by the current
  typed-subtask orchestrator (launched through `../run_agent.py`). Its decompose→run→handoff
  structure is the ancestor of the current planning and orchestration modules.
