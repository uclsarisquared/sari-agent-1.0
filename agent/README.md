# agent/ — layout

Reorganized 2026-07-24: the runtime was split into component packages and everything nothing
imports lives in a named folder. All imports are package-qualified (`from sim.env import ...`);
the flat-module era (`from env import ...`) is gone. Run everything from `agent/` — entry
points carry a `sys.path` shim so `python orchestrator/subtask_agents.py "task"` works directly.

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
| `orchestrator/` | `subtask_agents.py` (**CURRENT entry**), `subtask_planning.py`, `subtask_completion.py` | Long-horizon typed-subtask orchestrator: decompose → run legs → judge → retry |
| `evals/` | `eval_pickup.py` (Phase 4.2 A/B; imported by the orchestrator), `env_simulation.py` (legacy single-task loop; VMap_Plan.md builds on it) | Eval harnesses / legacy entry |

Run the agent:

```bash
python orchestrator/subtask_agents.py "find and pick up Pepero"
```

Cross-package facts worth knowing: `orchestrator.subtask_agents` imports `evals.eval_pickup`
(for `return_to_start`); slamtest keeps FLAT imports (`from capture_walk import ...`) with its
files in category subfolders — `slamtest/_bootstrap.py` puts those dirs on `sys.path`, and
agent consumers (`nav.store_map`, `nav.locate_task`, `agent_core.memory_gen`) import it;
slamtest scripts import `sim.env` / `nav.store_map` back.
Root-level runtime state stays at the root because the code writes it CWD-relative:
`episodic_memory.txt`, `semantic_memory.txt` (written by `agent_core.memory_runtime`).

## Leaf folders (nothing imports these)

- **`tests/`** — offline unit checks, no sim: `test_plan_reach.py`, `test_plan_place.py`
  (frozen-envelope geometry tables), `center_offline_check.py` (centring math A/B on saved PNGs).
- **`probes/`** — 🎮 interactive calibration probes and their offline fitters: `reach_probe.py` /
  `fit_envelope.py`, `place_probe.py` / `fit_place_envelope.py`, `center_live_test.py`,
  `probe_translation.py`. Fitters read the CSVs the probes write under `slamtest/output/`.
- **`gates/`** — 🎮 phase-gate and smoke harnesses: `gate_checkout.py` (Phase 6.2 5-run gate),
  `smoke_checkout.py` (one-item checkout chain dry-run).
- **`tools/`** — human-driven utilities: `keyboard_control.py` (WASD driving).
- **`deprecated/`** — superseded code kept for reference; see its README. Currently
  `subagent_run.py` (OpenRouter-era orchestrator, replaced by `orchestrator/subtask_agents.py`).
- **`slamtest/`** — the mapping + annotation pipeline (own README, `plans/`, `tests/`, frozen
  `output/` — now also home to the phase-6 measurement evidence `gate6_1_out/`, `step0_out/`).
  Files live in category subfolders (`core/`, `graph/`, `drivers/`, `capture/`, `annotate/`,
  `scoring/`, `app/`) but imports stay flat via `_bootstrap.py`.
- **`logs/`, `screenshots/`, `subtask_run_outputs/`** — run outputs, written by the runtime with
  these exact paths. Don't relocate without editing the writers.

New leaf scripts go in the matching folder WITH the two-line `sys.path` shim
(`_ROOT = dirname(dirname(abspath(__file__)))`); new runtime modules go in the matching package
with package-qualified imports.

🎮 = needs the Unity sim in Play mode (ws://localhost:8080). Run everything from `agent/`.
