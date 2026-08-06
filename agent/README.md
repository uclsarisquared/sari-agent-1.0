# agent/ — layout

Reorganized 2026-07-24: the runtime was split into component packages, and everything imported by
the live agent lives in a named folder. All imports are package-qualified
(`from sim.env import ...`); the flat-module era (`from env import ...`) is gone. Run everything
from `agent/`; `run_agent.py` is the stable, human-facing command-line entry point.

## Component packages (the agent runtime)

| Package | Files | Role |
|---|---|---|
| `sim/` | `env.py`, `hand_reset.py`, `chime.py` | Unity WebSocket bridge: commands, screenshots, hand reset, run beep |
| `agent_core/` | `runtime.py`, `navigation.py`, `hands.py`, `memory_runtime.py`, `actors.py`, `llm.py`, `contracts.py`, `agent.py` | Composed embodied runtime; navigation/hand/memory services; LLM clients and typed response contracts. `agent.py` is the legacy import façade |
| `prompts/` | Markdown prompt assets grouped by runtime role | Canonical reusable production LLM instructions and templates |
| `toolset/` | `actions.py`, `actions_str.py` | The agent's toolset: the atomic-action vocabulary the VLM sees (`actions_str`) and the wrappers that bind each action to sim/vision/manip code (`actions`) |
| `vision/` | `perception.py`, `md_tools.py`, `annotation_tools.py` | Detection/centring/OCR/scan, moondream pointing, bbox annotation |
| `manip/` | `manipulation.py` | Reach/place envelopes, hand poses, grab primitive |
| `nav/` | `store_map.py`, `locate_task.py` | Checkpoint-graph navigation, checkout macros, item resolver |
| `orchestrator/` | `cli.py`, `orchestration.py`, `leg_runner.py`, `subtask_planning.py`, `subtask_completion.py` | Long-horizon typed-subtask orchestrator: decompose → run legs → judge → retry. `subtask_agents.py` remains as a compatibility import façade |

Run the agent:

```bash
python run_agent.py "find and pick up Pepero"
```

Cross-package facts worth knowing: mapping keeps FLAT imports (`from capture_walk import ...`) with its
files in category subfolders — `mapping/_bootstrap.py` puts those dirs on `sys.path`, and
agent consumers (`nav.store_map`, `nav.locate_task`, `agent_core.memory_gen`) import it;
mapping scripts import `sim.env` / `nav.store_map` back.
Root-level runtime state stays at the root because the code writes it CWD-relative:
`episodic_memory.txt`, `semantic_memory.txt` (written by `agent_core.memory_runtime`).

## Supporting folders

- **`../validation/`** — all maintained offline tests, probes, calibration tools, evals, and
  supervised acceptance checks. Generated evidence goes to its gitignored `artifacts/` directory.
- **`tools/`** — human-driven utilities: `keyboard_control.py` (WASD driving).
- **`deprecated/`** — superseded code kept for reference; see its README. Currently
  `subagent_run.py` (OpenRouter-era orchestrator, replaced by the current typed-subtask
  orchestrator).
- **`mapping/`** — the mapping + annotation pipeline (own README, `plans/`, frozen `output/`).
  Files live in category subfolders (`core/`, `graph/`, `drivers/`, `capture/`, `annotate/`, and
  `scoring/`) but imports stay flat via `_bootstrap.py`; `pipeline_app.py` is the root GUI entry.
- **`logs/`, `screenshots/`, `subtask_run_outputs/`** — run outputs, written by the runtime with
  these exact paths. Don't relocate without editing the writers.

New runtime modules go in the matching package with package-qualified imports. New validation
scripts follow the taxonomy in `../validation/README.md`.

🎮 = needs the Unity sim in Play mode (ws://localhost:8080). Run everything from `agent/`.
