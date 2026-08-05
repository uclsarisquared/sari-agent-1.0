# Sari — LiDAR mapping + VLM store annotation

## What this is

A pipeline that maps a Unity grocery-store sim and annotates it into an **LLM-consumable map**: a
checkpoint graph where each shelf node knows what products are on it, so an agent can answer *"find
and pick up Pepero"* without a VLM ever doing spatial reasoning.

The Unity project is a **separate repo** (e.g. `SariSandboxV2`) at a path that differs per machine —
set `SARI_SANDBOX_DIR` in `config.env` to point at it (see `agent/sim/sim_paths.py`; catalog
grounding and the offline scoring/reconciliation scripts under `slamtest/scoring/` read it). It is
the sim and the ground-truth product catalog; this repo is the agent/mapping side.

## The principle that explains most decisions here

> **The graph owns spatial truth. The VLM only judges what is directly in front of it. The agent
> verifies on arrival.**

The navigation-failure diagnosis (formerly `NavReasonPlan.md`, removed 2026-07-19 — its
load-bearing quote survives verbatim in `slamtest/plans/phase4.1_navigation_ablation.md`, and the
file itself is in git history) documents open-ended VLM navigation as this agent's primary failure mode — it
collides with walls and burns its budget on global path planning it cannot do from a first-person
view. So: navigation and geometry are deterministic (A*, LiDAR, the skeleton graph); the VLM is
scoped to "what is on this shelf"; anything it can't resolve at build time is left for the agent to
resolve up close at pickup time. When a design question comes up, this principle usually answers it.

## Pipeline

Run from `agent/`. 🎮 = needs the sim in Play mode; the rest are offline.

| # | Phase | Command | Produces |
|---|---|---|---|
| 1 | Map 🎮 | `python slamtest/drivers/explore.py` | `grid_final.npy/.png`, `topology_final.json` |
| 2 | Shelf graph | `python slamtest/graph/build_shelf_graph.py slamtest/output` | `topology_final_shelf.json` + graph PNG |
| 3 | Reachability | `python slamtest/graph/audit_standability.py slamtest/output --topology-tag final_shelf` | prints |
| 4 | Capture 🎮 | `python slamtest/capture/capture_walk.py slamtest/output --limit 0 --angles 2` | `output/captures/cp<id>_primary.png`, `_crouch.png` |
| 5 | Annotate | `python slamtest/annotate/annotate_pass.py slamtest/output` | `annotations_*.json`, `products_*.json`, `semantic_map_*.txt` |

Step 5 is **offline over saved PNGs** — prompts and models can be iterated freely without re-driving
the sim. Prefer that over re-capturing.

## Layout (reorganized 2026-07-24 — see README.md for the full map)

The runtime is split into component packages with package-qualified imports: `sim/` (env,
hand_reset, chime), `agent_core/` (agent, sys_inst, memory, memory_gen), `toolset/` (actions,
actions_str — everything that defines the agent's action/tool vocabulary),
`vision/` (perception, md_tools, annotation_tools), `manip/` (manipulation), `nav/` (store_map,
locate_task), `orchestrator/` (subtask_agents — the CURRENT entry — plus subtask_planning,
subtask_completion), `evals/` (eval_pickup, env_simulation). The agent runs as
`python orchestrator/subtask_agents.py "<task>"`. slamtest keeps FLAT imports
(`from capture_walk import ...`) even though its files live in category subfolders — importing
`slamtest/_bootstrap.py` puts the category dirs on `sys.path` (agent consumers get it via
`nav.store_map`); slamtest scripts import `sim.env` / `nav.store_map` back. `plan6/` was dissolved
2026-07-24 (its tests → `tests/`, probes → `probes/`, outputs → `slamtest/output/`;
CHECKLIST.md deleted on user request, in git history). Leaf scripts nothing imports live in `tests/` (offline),
`probes/` (🎮 calibration + fitters), `gates/` (🎮 gates/smokes), `tools/`, `deprecated/` — each
carries a two-line `sys.path` shim pointing at the agent root. New code follows the same rules:
runtime → the matching package, package-qualified imports; leaf scripts → the matching folder,
with the shim. `episodic_memory.txt` / `semantic_memory.txt` stay at the root (written
CWD-relative by `agent_core.agent`; run everything from `agent/`).

## Standing constraints — ask before violating

- **Do not run `explore.py` casually.** `--clear-output` defaults to **true**: it wipes `output/`
  and regenerates the map with **new checkpoint IDs**, invalidating every capture and annotation
  keyed to the old ones. The frozen map is the working baseline. Everything downstream of it is
  deterministic and safely regenerable from it.
- **Never use Unity's `isColliding` for safety or obstacle marking** — it is unreliable. Use LiDAR
  (`swept_clearance_ahead` / `voxel.integrate`).
- **`claude -p` bills the claude.ai subscription** (`claude auth login`, defaults to `--claudeai`).
  The `anthropic` SDK path bills API credits instead — a different account. Don't switch silently.
  Never pass `--bare` to `claude -p` here: it forces `ANTHROPIC_API_KEY` and refuses the OAuth login.
- **The ANNOTATOR defaults to `claude -p`, sonnet, medium effort** — the measured/frozen quality
  baseline. It is no longer *locked* to claude: as of 2026-07-20 (user directive) `annotate_pass.py`
  takes `--backend {claude-cli,qwen}`, and the pipeline app exposes the choice. claude-cli stays the
  DEFAULT (unchanged baseline); `qwen` (the same endpoint, `annotate_qwen.py`) is selectable for a
  fully self-hosted run or to A/B annotation quality on identical captures. If you switch the
  annotator to qwen for a real pass, A/B qwen-vs-sonnet on the reviewed captures (cp015/cp017/cp067)
  first — quality was only ever measured on sonnet. The AGENT RUNTIME (per-step VLM/learner calls)
  runs against an **OpenAI API compatible endpoint** (`$OPENAI_API_URL/v1`, bearer
  `$OPENAI_API_KEY`) — OpenRouter was retired for agent calls when its credits ran out.
- **Model ids are config, not code.** `$SARI_MODEL` (and `$SARI_ANNOTATOR_MODEL` for the qwen
  annotator backend) in the repo-root `config.env` select what the endpoint is asked for; code
  reads them via `agent_core/models.py`, which holds the ONLY hardcoded default in the tree
  (`Qwen/Qwen3.6-27B`, the measured baseline). Don't reintroduce model literals at call sites.
- **Map quality bar:** a good `grid_final.png` has **no grey holes** in the store interior.

## How to work here — this matters more than it sounds

This project has repeatedly burned hypotheses that sounded obviously right. The habits that actually
worked:

- **Measure, don't assume.** Nearly every "obvious" cause this project chased was wrong: the fridge
  read badly and it wasn't the glass, wasn't the prompt, and wasn't the model — it was effective
  resolution after the vision encoder's downscale.
- **A/B on the identical input before claiming a prompt change worked.** Two prompt changes were
  made, looked reasonable, measurably made things *worse*, and were reverted. Re-run the same image.
- **Use negative controls.** `guided_json` looked like it was working for three runs — because
  conforming output is also what an *ignored* schema produces. Only a deliberately-absurd schema
  (only legal answer: `"banana"`) exposed that vLLM was ignoring it entirely.
- **Record measured findings in the code**, not just in chat. Docstrings here carry *why*, including
  dead ends, on purpose.
- **Report honestly.** If a test contradicts something claimed earlier in the session, say so plainly
  and correct the record.

## Read before touching

- **Annotator prompts** → the `MEASURED - DO NOT RE-ATTEMPT` block at the top of
  `slamtest/annotate/annotator_sys_inst.py`. It records what was tried, what it cost, and why the current
  wording is what it is. Several intuitive "improvements" are documented there as *failures*.
- **Node spacing / reading distance** → the constants in `slamtest/graph/shelf_coverage.py`; each carries
  its measured history and the trap that bites when tuning it.
- **Phase design** → `slamtest/plans/phase*.md`.

## Current state (2026-07)

- **Phases 1–2: shipped.** Frozen map + `topology_final_shelf.json` = **55 checkpoints**: **39 shelf
  nodes** (2.0m interval, **1.0m reading distance**, no stretch left uncovered), 15 base
  junction/end/doorway nodes, and **1 landmark** — the checkout counter, found by the
  unobserved-structure pass and docked 0.20m off its face so the agent can drop items on it.
- **`annotations_final_shelf.json` is CURRENT** — 39 records regenerated 2026-07-17 against *this*
  graph, 290 products. Each record carries graph-derived **`route_hints`**: per neighbour, the
  nearest shelf/landmark reachable through it and the hop count, or `null` for a dead end (102
  routed, 1 dead end). These cost nothing — `annotate_pass --resume` recomputes them, plus the
  graph-owned `neighbors` cache, from the topology without re-annotating anything.
- **Phase 3: working end to end.** Three VLM backends share one set of prompts and images —
  `annotate_probe.py` (Qwen/vLLM), `annotate_probe_claude.py` (SDK), **`annotate_claude_cli.py`
  (`claude -p`, the default — Sonnet at medium effort)**.
- **Validated against the Unity catalog:** ~97–98% of annotated items are real products; ~1%
  invented. Recall is the weak axis (~34% of the 250-SKU catalog), concentrated in the
  **refrigerated categories** (Juice 0/16, Dairies 2/20) where labels are unreadable through glass.

### Open threads

1. **The annotator's view labels no longer match what capture sends.** The multi-view contract
   itself IS wired — `annotate_pass.view_paths()` ships the crouch shot as an item-bearing view (not
   context), and the `items` rule reads from ALL views and dedups the overlap. But the wording drifted
   when crouch replaced the down-pitch:
     - the base prompt names the views **STRAIGHT / DOWN / UP** and `items` singles out *"the DOWN
       view in particular"* for bottom rows — while capture actually sends **CROUCHED**. The prompt
       advertises a view that never arrives and receives a label it never defines.
     - `sign_text`, in *both* overlays, still says *"the PRIMARY image"* — a term the base prompt
       stopped introducing when it switched to named views. That rule currently has no referent.
     - the module docstring (`PRIMARY vs CONTEXT`, and the `CALLER CONTRACT` block citing *"items
       come ONLY from the PRIMARY image"*) still describes the superseded contract.
   Low-risk to fix, but it is a prompt change: A/B on identical captures before claiming it helped.
2. **Snap-to-catalog reconciler** (highest value, designed not built): fuzzy-match each annotated
   name to a canonical SKU from `SariSandboxV2/Assets/Resources/Data/PriceData.json` +
   `Categories.json`. Fixes misreads (`Picattos`→`Piattos`, `Jackerel`→`Mackerel`), collapses
   name fragmentation, resolves collisions, and flags brand-only reads as `variant_uncertain`.
3. **The landmark has no capture, so no prose.** `captures/` covers cp15–cp53 only (39 × primary +
   crouch); cp54 was spliced into the graph *after* that walk. It routes correctly — cp32 reaches it
   in 1 hop as `kind: landmark`, via the topology-kind fallback, without needing an annotation — but
   its `holds` is empty and it has no `semantic_summary`, so it is a reachable dot rather than "the
   checkout counter". One capture fixes it: `capture_walk --kind landmark --ids 54 --angles 1` 🎮,
   which also sidesteps the pending `SetCrouch` C# recompile (crouch only fires at `--angles 2`).
4. **Route hints are persisted only for annotated nodes.** The 15 base nodes get no record, so the
   JSON holds no hints for them — and cp11–14 are *blind* (no shelf/landmark neighbour at all).
   `walk_map` recomputes hints live so the terminal walker covers them; a Phase-4 agent reading only
   `annotations_*.json` does not. The fix would be stub records for base nodes, which also changes
   what `semantic_map_*.txt` renders — a contract call, not a mechanical one.
5. Cereal taxonomy gap (the enum has no Cereal; the store files them under `Biscuit`, so the model
   returns `other`). Refrigerated recall needs tiling or the closer/crouched capture to pay off.
