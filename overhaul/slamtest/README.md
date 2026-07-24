# Building the annotated store map — run order

This is the pipeline that turns the Unity grocery sim into an **LLM-consumable map**: a checkpoint
graph where each shelf node knows what products are on it. Run everything from `overhaul/`.

> **Principle:** the graph owns spatial truth, the VLM only judges what is directly in front of it,
> the agent verifies on arrival. Navigation/geometry are deterministic (A*, LiDAR, skeleton graph);
> the VLM is scoped to "what is on this shelf".

🎮 = the Unity `SariSandboxV2` scene must be in **Play mode** with the WebSocket command server up
(`ws://localhost:8080/commands`). The rest are **offline** and safe to re-run.

---

## The order

```bash
# 1. Map the store 🎮  — LiDAR frontier exploration → occupancy grid + topology graph
python slamtest/drivers/explore.py

# 2. Shelf graph (offline) — thin the grid to a skeleton, add shelf/landmark checkpoints
python slamtest/graph/build_shelf_graph.py slamtest/output

# 3. Reachability audit (offline) — flag checkpoints an agent can't actually stand at
python slamtest/graph/audit_standability.py slamtest/output --topology-tag final_shelf

# 4. Capture 🎮 — drive to every checkpoint, save one PNG per view
python slamtest/capture/capture_walk.py slamtest/output --limit 0 --angles 2

# 5. Annotate (offline) — VLM reads each capture → durable, queryable map
python slamtest/annotate/annotate_pass.py slamtest/output

# 5b. (optional, offline) — snap product names to canonical Unity catalog SKUs
python slamtest/scoring/reconcile_products.py
```

| # | Stage | Sim? | Produces (in `slamtest/output/`) |
|---|---|:---:|---|
| 1 | `explore.py` | 🎮 | `grid_final.npy` / `grid_final.png`, `topology_final.json` |
| 2 | `build_shelf_graph.py` | — | `topology_final_shelf.json` + graph PNG |
| 3 | `audit_standability.py` | — | prints (no file) — at-risk checkpoints |
| 4 | `capture_walk.py` | 🎮 | `captures/cp<id>_primary.png`, `captures/cp<id>_crouch.png` |
| 5 | `annotate_pass.py` | — | `annotations_final_shelf.json`, `products_final_shelf.json`, `semantic_map_final_shelf.txt` |
| 5b | `reconcile_products.py` | — | `products_final_shelf_reconciled.json` + report |

Stages 2–5 chain on the default tags (`final` → `final_shelf`), so the bare commands above just
work as long as everything lives in one output directory.

---

## ⚠️ Before you run step 1 — the frozen-baseline trap

`slamtest/output/` holds the **frozen working map** (55 checkpoints). `explore.py` defaults to
`--clear-output` (**wipes the output dir**) and mints **new checkpoint IDs**, which invalidates every
existing capture and annotation keyed to the old IDs. Two ways to go from scratch safely:

**A — new map, keep the baseline untouched (recommended).** Explore into a fresh directory and point
every downstream stage at it:

```bash
python slamtest/drivers/explore.py            --output-dir slamtest/output_runs/mymap
python slamtest/graph/build_shelf_graph.py  slamtest/output_runs/mymap
python slamtest/graph/audit_standability.py slamtest/output_runs/mymap --topology-tag final_shelf
python slamtest/capture/capture_walk.py       slamtest/output_runs/mymap --limit 0 --angles 2
python slamtest/annotate/annotate_pass.py      slamtest/output_runs/mymap
python slamtest/scoring/reconcile_products.py --output-dir slamtest/output_runs/mymap   # optional
```

**B — replace the frozen baseline in place.** Just run the commands in [The order](#the-order). This
overwrites `slamtest/output/` and renumbers everything — only do it if you intend to discard the
current baseline. (The Tkinter `pipeline_app.py` hard-refuses to explore into `slamtest/output` for
exactly this reason.)

A good `grid_final.png` has **no grey holes** in the store interior — if it does, re-run step 1
(raise `--max-steps`) before continuing.

---

## Prerequisites

- **Steps 1 & 4** need the sim in Play mode (`ws://localhost:8080/commands`).
- **Step 5** defaults to `--backend claude-cli` (`claude -p`, Sonnet, medium effort) — the
  measured/frozen quality baseline. Requires `claude auth login` (bills the claude.ai subscription).
  `--backend qwen` runs fully self-hosted on the UCL vLLM server instead (creds in repo-root
  `api.env`); if you switch, A/B qwen-vs-sonnet on cp015/cp017/cp067 first — quality was only ever
  measured on Sonnet.
- **Step 5b** is pure string matching against the pinned Unity catalog
  (`SariSandboxV2/Assets/Resources/Data/`) — no model, no sim, deterministic.

## Useful flags

- `explore.py --max-steps 20` — quick smoke test; `--no-clear-output` to append instead of wipe.
- `capture_walk.py --ids 15 67` / `--kind shelf` — capture a subset; `--limit 0` = all,
  `--angles 2` = primary + crouch (default 1 = primary only).
- `annotate_pass.py --resume` — skip checkpoints already annotated; `--ids …` for a subset;
  `--skip-classify` to treat every shelf-kind node as a real shelf.

## One-window alternative

`python slamtest/app/pipeline_app.py` runs stages 1→5 from a single Tkinter window (subprocesses of the
same CLIs, so no defaults drift). It protects the frozen baseline and shows the map building live.

## Deeper docs

- Annotator prompt history (measured dead ends) → top of `annotator_sys_inst.py`.
- Node spacing / reading distance constants → `shelf_coverage.py`.
- Phase design → `plans/phase*.md`.
