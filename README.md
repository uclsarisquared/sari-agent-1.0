Virtual Embodied Agent operating in a 3D Convenience Store Environment
========================

An embodied agent that navigates and manipulates objects in a Unity grocery-store sim. The store
is mapped offline into an **LLM-consumable checkpoint graph** — each shelf node knows what products
are on it — so the agent can resolve *"find and pick up Pepero"* without a VLM ever doing spatial
reasoning.

> **The graph owns spatial truth. The VLM only judges what is directly in front of it. The agent
> verifies on arrival.**

Open-ended VLM navigation is this agent's primary measured failure mode — it collides with walls and
burns its budget on global path planning it cannot do from a first-person view. So navigation and
geometry are deterministic (A*, LiDAR, the skeleton graph) and the VLM is scoped narrowly.

The Unity project is a **separate repo** (`SariSandboxV2`). It is the sim and the ground-truth
product catalog; this repo is the agent/mapping side.

---

## Prerequisites

- **Python 3.x**, `pip install -r requirements.txt` (the `overhaul/` stack has its own
  `overhaul/requirements.txt`)
- **The Unity sim running in Play mode**, exposing its WebSocket command server at
  `ws://localhost:8080/commands`. Nothing that touches the environment works without it.
- **Windows.** Both entry points `import winsound` at module top level, so they fail at import on
  macOS/Linux regardless of configuration.
- **`api.env` at the repo root** — see [Configuration](#configuration).

## Configuration

Copy the template and fill it in:

```bash
cp api.env.example api.env
```

`api.env` lives at the **repo root** and is gitignored. Every module resolves it from `__file__`,
so it loads regardless of which directory you run from — you do not need to `cd` anywhere
particular for config to be found.

The two variables that matter most are **`UCL_BASE_URL`** and **`UCL_API`**: the vLLM server
(`Qwen/Qwen3.6-27B`) that serves all per-step agent calls. `UCL_BASE_URL` is a *bare host* — the
code appends `:8000/v1` itself. The server 401s without a bearer key. If these are unset the agent
raises `UCL_BASE_URL/UCL_API not found` on startup.

`api.env.example` documents every variable, including which are legacy (`OPENROUTER_API_KEY` —
credits exhausted, agent calls moved to the UCL server) and which have no reader in the codebase
at all (`REPLICATE_API_TOKEN`).

### A split worth not crossing

| | Backend | Why |
|---|---|---|
| **Annotator** (offline, pipeline step 5) | `claude -p`, Sonnet, medium effort — pinned | Annotation quality is measured and frozen on Sonnet |
| **Agent runtime** (per-step VLM calls) | UCL vLLM, `Qwen/Qwen3.6-27B` | Agent behaviour is measured on Qwen |

Also note `claude -p` bills a claude.ai subscription while the `anthropic` SDK path bills API
credits — different accounts. Don't switch silently.

---

## Running the agent

The current stack lives in `overhaul/`. **The map is already built and frozen** — you do not need
to run the mapping pipeline first.

```bash
cd overhaul
python subagent_run.py "find and pick up Pepero"
```

`subagent_run.py` is the orchestrator: it decomposes the task into subtasks, runs each through an
agent instance, judges completion, retries failures with context, and generates a handoff summary
between subtasks. No separate server process is needed — it calls the UCL server directly.

Before a run, sanity-check that the map artifacts are present:

```bash
ls overhaul/slamtest/output
```

### Legacy root stack

The original two-process setup. `server.py` uses OpenRouter, which is retired for agent calls, so
this path is likely dead — kept for reference.

```bash
python server.py inf_base      # LitServe on :8005 (or inf_super)
python run.py "your task"      # polls the sim, posts to /predict
```

---

## The mapping pipeline

This is the **offline build step** that produces the map the agent navigates on. It is a one-time
build, not part of a normal agent run.

Run from `overhaul/`. 🎮 = needs the sim in Play mode; the rest are offline.

| # | Phase | Command | Produces |
|---|---|---|---|
| 1 | Map 🎮 | `python slamtest/explore.py` | `grid_final.npy/.png`, `topology_final.json` |
| 2 | Shelf graph | `python slamtest/build_shelf_graph.py slamtest/output` | `topology_final_shelf.json` + graph PNG |
| 3 | Reachability | `python slamtest/audit_standability.py slamtest/output --topology-tag final_shelf` | prints |
| 4 | Capture 🎮 | `python slamtest/capture_walk.py slamtest/output --limit 0 --angles 2` | `captures/cp<id>_primary.png`, `_crouch.png` |
| 5 | Annotate | `python slamtest/annotate_pass.py slamtest/output` | `annotations_*.json`, `products_*.json`, `semantic_map_*.txt` |

**Current state:** phases 1–2 are shipped. `topology_final_shelf.json` holds **55 checkpoints** —
39 shelf nodes, 15 base junction/end/doorway nodes, and 1 landmark (the checkout counter).
`annotations_final_shelf.json` is current: 39 records, 290 products.

Step 5 is **offline over saved PNGs**, so prompts and models can be iterated freely without
re-driving the sim. Prefer that over re-capturing.

### ⚠️ Do not run `explore.py` casually

`--clear-output` defaults to **true**. It wipes `output/` and regenerates the map with **new
checkpoint IDs**, invalidating every capture and annotation keyed to the old ones. The frozen map is
the working baseline; everything downstream of it is deterministic and safely regenerable *from it*.

Other standing constraints:

- **Never use Unity's `isColliding`** for safety or obstacle marking — it is unreliable. Use LiDAR
  (`swept_clearance_ahead` / `voxel.integrate`).
- **Map quality bar:** a good `grid_final.png` has **no grey holes** in the store interior.

---

## Repo layout

| Path | What |
|---|---|
| `overhaul/` | Current agent stack. Start at `subagent_run.py` → `subtask_agents.py` → `agent.py`. |
| `overhaul/slamtest/` | Mapping + annotation pipeline (LiDAR, occupancy grid, topology, capture, annotate). |
| `overhaul/env.py`, `actions.py` | Unity WebSocket bridge and the action vocabulary. |
| `overhaul/CLAUDE.md` | **Design rationale, measured findings, and open threads. Read before changing anything.** |
| `run.py`, `server.py`, `openrouter.py` | Legacy root stack. |

## Working conventions

This project has repeatedly burned hypotheses that sounded obviously right:

- **Measure, don't assume.** The fridge read badly and it wasn't the glass, wasn't the prompt, and
  wasn't the model — it was effective resolution after the vision encoder's downscale.
- **A/B on the identical input before claiming a prompt change worked.** Two reasonable-looking
  prompt changes measurably made things *worse* and were reverted.
- **Use negative controls.** `guided_json` looked like it worked for three runs — because conforming
  output is also what an *ignored* schema produces.
- **Record measured findings in the code.** Docstrings here carry *why*, including dead ends, on
  purpose. Read the `MEASURED - DO NOT RE-ATTEMPT` block atop `slamtest/annotator_sys_inst.py`
  before touching annotator prompts.
