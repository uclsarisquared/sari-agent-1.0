

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/9b52e148-3998-420e-ba2a-cb15b0011681">
    <img width="180" height="180" alt="Sarilogofilled" src="https://github.com/user-attachments/assets/9b52e148-3998-420e-ba2a-cb15b0011681" />
  </picture>
</p>

<h1 align="center">Sari Agent²</h1>

<p align="center">
  <em>The 2nd-generation embodied agent for Sari Sandbox.</em>
</p>

An embodied agent that navigates and manipulates objects in Sari Sandbox 1.0/2.0. The store
is mapped offline into an **LLM-consumable checkpoint graph** — each shelf node knows what products
are on it — so the agent can resolve *"find and pick up Pepero"* without a VLM ever doing spatial
reasoning.

The Unity project is a **separate repo** (`SariSandboxV2`). It is the sim and the ground-truth
product catalog; this repo is the agent, mapping, and benchmarking side.

## Quickstart

### One-time setup

```bash
uv sync
cp config.env.example config.env  # add OPENAI_API_URL and OPENAI_API_KEY
ls overhaul/slamtest/output       # confirm the map artifacts exist
```

### Solo agent
Start Unity in Play mode, then run these in separate terminals:

```bash
# Terminal 1
uv run poe ocr-server

# Terminal 2
uv run python overhaul/orchestrator/subtask_agents.py \
    --config runconfig.toml --task "find and pick up Pepero"
```

### Sari Bench battery
Run in order:

```bash
# Terminal 1
uv run poe ocr-server

# Terminal 2
uv run poe dbench-coordinator

# Terminal 3 (repeat on each Unity host)
SARI_BENCH_COORDINATOR=ws://<runner-host>:9000/sandbox ./SariSandbox

# Terminal 4
uv run poe dbench-run            # reads [bench] from runconfig.toml

# Optional Terminal 5
uv run poe dbench-watch
```

```text
SOLO
Task ──▶ Agent ──ws──▶ Unity
           ├──HTTP──▶ OCR daemon
           └──HTTP──▶ OpenAI endpoint

SARI BENCH
Prompt battery ──▶ Runner ──▶ Agent subprocesses ──ws──▶ Unity fleet
                     │                 │
                     │                 ├──HTTP──▶ OCR daemon
                     │                 └──HTTP──▶ OpenAI endpoint
                     └──ws──▶ Coordinator ◀──ws── Unity fleet
```

The live entrypoint is `overhaul/orchestrator/subtask_agents.py`, not legacy `run.py`. OCR defaults
to `http://127.0.0.1:9100`; override it with `--ocr-url` or `SARI_OCR_URL`.

## Configuration

Runtime and experiment settings live in the checked-in
[`runconfig.toml`](runconfig.toml). It documents every standalone-agent and distributed-bench
option in place:

```bash
# Standalone agent; the prompt remains easy to change.
uv run python overhaul/orchestrator/subtask_agents.py \
    --config runconfig.toml --task "find and pick up Pepero"

# Distributed battery; prompts, tries, caps, arm, coordinator, and output settings come from [bench].
uv run python -m sari_bench run --config runconfig.toml
```

The standalone entrypoint reads `[agent]`, `[limits]`, `[environment]`, and `[output]`. The battery
runner reads `[bench]`. Explicit flags override TOML values, so a one-off ablation can use
`--config runconfig.toml --arm vlm` without editing the file. Relative paths are resolved from the
config file rather than the current working directory. Unknown sections, misspelled options,
invalid choices, and invalid numeric ranges fail before a run starts.

Credentials and host-specific environment values remain in the gitignored `config.env`. Copy its
template and fill it in:

```bash
cp config.env.example config.env
```

`config.env` lives at the **repo root** and is gitignored. Every module resolves it from `__file__`,
so it loads regardless of which directory you run from — you do not need to `cd` anywhere
particular for config to be found.

The two variables that matter most are **`OPENAI_API_URL`** and **`OPENAI_API_KEY`**: the vLLM server
(`Qwen/Qwen3.6-27B`) that serves all per-step agent calls. `OPENAI_API_URL` is a *bare host* — the
code appends `:8000/v1` itself. The server 401s without a bearer key. If these are unset the agent
raises `OPENAI_API_URL/OPENAI_API_KEY not found` on startup.

`config.env.example` documents every variable, including which are legacy (`OPENROUTER_API_KEY` —
credits exhausted, agent calls moved to the UCL server) and which have no reader in the codebase
at all (`REPLICATE_API_TOKEN`).

### Annotator Setup

| | Backend | Why |
|---|---|---|
| **Annotator** (offline, pipeline step 5) | `claude -p`, Sonnet, medium effort — pinned | Annotation quality is measured and frozen on Sonnet |
| **Agent runtime** (per-step VLM calls) | OpenAI-compatible VLM endpoint | Agent behaviour is measured on `Qwen3.6-27B` |

---

## Running the agent

There are two agent stacks in this repo. **Only the `overhaul/` one is live**; the `legacy/` stack
(`run.py` / `server.py`, formerly at the repo root) is deprecated and receives no further
development. `sari_bench/` is a third piece — a fleet harness that runs the current agent at scale,
not an alternative agent.

| | Entrypoint | Status |
|---|---|---|
| **Current agent** (map-based) | `uv run python overhaul/orchestrator/subtask_agents.py --config runconfig.toml --task "<task>"` | Live — all development happens here |
| **Legacy agent** (open-ended VLM) | `uv run python legacy/server.py inf_base` + `uv run python legacy/run.py "<task>"` | **Deprecated**, kept for reference only |
| **Distributed Sari Bench** | `python -m sari_bench coordinator/run/watch ...` | Live — runs the current agent across a fleet |

### Current agent

Lives in `overhaul/`. It navigates on the frozen checkpoint graph with deterministic A*, and calls
the VLM only for local judgements. **The map is already built and frozen** — you do not need to run
the mapping pipeline first.

```bash
cd overhaul
uv run python orchestrator/subtask_agents.py "find and pick up Pepero"
```

`subtask_agents.py` is the orchestrator: it decomposes the task into typed subtasks (legs), runs
each through an agent instance, judges completion, retries failed legs with the failure reason in
context, and hands off state between legs. It calls the UCL model server directly and uses the
separately started runner-local OCR daemon for receipt recognition. (Its OpenRouter-era ancestor
`subagent_run.py` is kept in `overhaul/deprecated/`.)

Before a run, sanity-check that the map artifacts are present:

```bash
ls overhaul/slamtest/output
```

#### `subtask_agents.py` flags

Run `uv run python overhaul/orchestrator/subtask_agents.py --help` for the authoritative list;
the table below shows built-in defaults used when no config supplies a value:

| Flag | Default | What |
|---|---|---|
| `--config` | — | TOML run configuration. Explicit CLI flags override it. |
| `task` (positional) | — | The long-horizon task, e.g. `"find and pick up Pepero"` (or use `--task`). Prompted interactively if neither is given. |
| `--task` | — | Same as the positional arg; takes precedence if both are given. |
| `--arm` | `graph` | Navigation arm: `graph` (measured-better deterministic navigator), `vlm`, or `graph-advised` (drives each graph hop through a per-hop advisor VLM). |
| `--max-steps` | `150` | Per-leg step cap. |
| `--max-minutes` | `40.0` | Per-leg wall-clock cap. |
| `--out` | `<run-dir>/summary.json` | Where to write the run summary. |
| `--run-dir` | — | Run directory for logs/screenshots. |
| `--resolver-backend` | `qwen` | Backend for subtask resolution: `qwen` or `claude-cli`. |
| `--completion-guard` | `deterministic` | Pickup target-grounding backend: `deterministic` or `vlm`. |
| `--output-dir` | `$SARI_MAP_DIR`, else `slamtest/output` | slamtest output dir to load the map from (topology/annotations/grid). |
| `--leg-retries` | `1` | How many times to retry a failed leg with the failure reason in context before aborting the task. `0` restores abort-on-first-failure. |
| `--reset-start` | off | Drive to the fixed spawn pose once before starting (eval-reproducibility). A plain run starts from the agent's current pose. |
| `--restart-env` | off | Hard-reset the store to its initial state before starting (items back on shelves, checkouts undone, agent to spawn) so a fresh task doesn't inherit the last run's grabbed/checked-out items. Unlike `--reset-start`, which only moves the agent. |
| `--ws-uri` | `$SARI_WS_URI`, else `ws://localhost:8080/commands` | Sandbox command endpoint. Sets `SARI_WS_URI` for this process. Distributed Sari Bench passes the URI of the sandbox it leased for the attempt, which is how several agents run against one machine at once. |
| `--ocr-url` | `$SARI_OCR_URL`, else `http://127.0.0.1:9100` | Shared OCR service base URL. Its `/health` must pass before the first simulator command. |

### Distributed Sari Bench

`sari_bench/` runs a prompt battery across a fleet of Sandbox instances — several attempts per
prompt, a guaranteed-clean environment for every attempt, a live dashboard, and CSV/mp4 reporting.
It drives the same `orchestrator/subtask_agents.py` entrypoint as a subprocess per attempt; it does
not reimplement the agent. See **[`sari_bench/README.md`](sari_bench/README.md)** for the coordinator
/ sandbox / runner setup, the watch dashboard, and failure-handling semantics.

```bash
uv run poe ocr-server  # separate terminal
python -m sari_bench coordinator --port 9000
python -m sari_bench run --config runconfig.toml
```

### Legacy stack (`legacy/`) — deprecated

**`legacy/run.py` does *not* run the current agent.** It is the original two-process setup:
`run.py` polls the sim and POSTs to `server.py` (LitServe on `:8005`), which asks a VLM for
raw `MOVE_FWD` / `PAN_LEFT` / `GRIP_*` actions with durations. It has no map, no checkpoint graph
and no A* — it is precisely the open-ended VLM navigation the overhaul replaced.

It is also likely non-functional: `server.py` calls OpenRouter, which is retired for agent calls.
Kept for reference and comparison only; do not build on it.

```bash
uv run python legacy/server.py inf_base      # LitServe on :8005 (or inf_super)
uv run python legacy/run.py "your task"      # polls the sim, posts to /predict
```

---

## Mapping Pipeline

This is the **offline build step** that produces the map the agent navigates on. It is a one-time
build, not part of a normal agent run.

Run from `overhaul/`. 🎮 = needs the sim in Play mode; the rest are offline.

| # | Phase | Command | Produces |
|---|---|---|---|
| 1 | Map 🎮 | `uv run python slamtest/explore.py` | `grid_final.npy/.png`, `topology_final.json` |
| 2 | Shelf graph | `uv run python slamtest/build_shelf_graph.py slamtest/output` | `topology_final_shelf.json` + graph PNG |
| 3 | Reachability | `uv run python slamtest/audit_standability.py slamtest/output --topology-tag final_shelf` | prints |
| 4 | Capture 🎮 | `uv run python slamtest/capture_walk.py slamtest/output --limit 0 --angles 2` | `captures/cp<id>_primary.png`, `_crouch.png` |
| 5 | Annotate | `uv run python slamtest/annotate_pass.py slamtest/output` | `annotations_*.json`, `products_*.json`, `semantic_map_*.txt` |

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
| `overhaul/` | Current agent stack. Start at `orchestrator/subtask_agents.py` → `agent_core/agent.py`. See `overhaul/README.md` for its internal layout (component packages `sim/`, `agent_core/`, `toolset/`, `vision/`, `manip/`, `nav/`, `orchestrator/`, `evals/`, plus `tests/`, `probes/`, `gates/`, `tools/`, `deprecated/`). |
| `overhaul/slamtest/` | Mapping + annotation pipeline (LiDAR, occupancy grid, topology, capture, annotate). |
| `sari_bench/` | Distributed Sari Bench — coordinator, sandbox fleet pool, runner, live watch dashboard, report/video tooling. See `sari_bench/README.md`. |
| `overhaul/sim/env.py`, `overhaul/toolset/` | Unity WebSocket bridge and the action vocabulary. |
| `overhaul/CLAUDE.md` | **Design rationale, measured findings, and open threads. Read before changing anything.** |
| `legacy/` | **Deprecated** v1 stack (`run.py`, `server.py`, `openrouter.py`, its own `env.py`/`actions.py`) — open-ended VLM, no map. Not the current agent. |
| `experiments/` | Dormant explorations: the SariVoxeLLMap/Depth-Anything heightmap plan (`VMap_Plan.md`) and monocular-depth tests (May 2026). Superseded in practice by slamtest's LiDAR mapping. |
| `docs/` | The SARI paper PDF and historical sketches. |
| `legacy/chime.py`, `overhaul/chime.py` | Cross-platform completion beep. Duplicated on purpose — both stacks define their own `env.py`/`actions.py`, so sharing one copy via `sys.path` would shadow `overhaul`'s modules with the legacy stack's. |
| `pyproject.toml` | Dependency source of truth (uv). The `requirements.txt` files are legacy freezes. |
