# legacy/ — the v1 root stack (deprecated)

The original two-process agent, moved here from the repo root 2026-07-24. **Not the current
agent** — that lives in `agent/` (entry: `run_agent.py`). Kept for reference and
comparison; do not build on it.

How it worked: `run.py` polls the sim and POSTs screenshots to `server.py` (LitServe on `:8005`),
which asks a VLM for raw `MOVE_FWD` / `PAN_LEFT` / `GRIP_*` actions with durations. Open-ended VLM
navigation, no map, no A* — precisely the measured failure mode the overhaul replaced. Likely
non-functional today: `server.py`/`openrouter.py` call OpenRouter, retired for agent calls.

```bash
uv run python legacy/server.py inf_base   # LitServe on :8005 (or inf_super)
uv run python legacy/run.py "your task"   # polls the sim, posts to /predict
```

The snapshot carries its own `env.py`, `actions.py`, `chime.py`, `functions.py`,
`comprehension.py`, `reference.py`, `ClientGUI.py`, `fixed_reqs.txt`, and the May-2025 root
`subagent_run.py`. `openrouter.py` creates `base_semantic_memory.txt` at runtime. These deliberately
shadow nothing in `agent/` — both stacks define their own `env.py`/`actions.py`, which is why the
code was never shared via `sys.path`.
