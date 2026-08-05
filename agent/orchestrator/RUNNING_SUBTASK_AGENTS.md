# Running `subtask_agents.py`

Long-horizon task orchestrator: decomposes a natural-language store task into typed
subtask legs (`pickup` / `goto` / `compare` / `checkout` / `inspect`) and runs each leg as its own
self-contained embodied-agent loop, with shared semantic/episodic memory and a findings
summary carried forward between legs.

## Prerequisites

- The Unity sim must be in **Play mode**.
- The OpenAI API compatible endpoint must be reachable (credentials and `$SARI_MODEL` loaded
  from `config.env` at the repo root — three parents up from `orchestrator/`).
- Run from the `agent` directory:

```bash
cd C:/Sari/sari-agent-1.0/agent
```

## Basic usage

```bash
python orchestrator/subtask_agents.py --config ../runconfig.toml \
  "get the green Piattos and bring it to the checkout counter"
```

If no task is given on the command line, it prompts interactively (`Task: `).
The root `runconfig.toml` documents the standalone settings under `[agent]`, `[limits]`,
`[environment]`, and `[output]`, plus distributed-runner settings under `[bench]`. Explicit flags
override configured values. Paths in TOML are resolved relative to that file.

## Arguments

All arguments are optional except the task itself.

| Argument | Default | Meaning |
|---|---|---|
| `--config PATH` | none | TOML run configuration; explicit CLI flags override it |
| `task` (positional) or `--task "..."` | interactive prompt | The long-horizon task in plain English |
| `--arm {vlm, graph, graph-advised}` | `graph` | Navigation arm. `graph` is the measured-better graph navigator; `vlm` is the old VLM-navigation control arm; `graph-advised` drives each graph hop through a per-hop advisor VLM |
| `--max-steps N` | `150` | Step cap **per leg** (not per task) |
| `--max-minutes M` | `40` | Wall-clock cap per leg, in minutes |
| `--leg-retries N` | `1` | How many times a failed leg is retried (with the failure reason fed into the retry's context) before the whole task aborts; `0` restores abort-on-first-failure |
| `--resolver-backend {qwen, claude-cli}` | `qwen` | Backend for the plan-time map target resolver |
| `--completion-guard {deterministic,vlm}` | `deterministic` | Optional pickup, compare, and unknown completion backend. `inspect` completion is VLM-verified in both modes |
| `--output-dir DIR` | `slamtest/output` | Which slamtest map (topology / annotations / grid) to load — defaults to the frozen baseline map |
| `--run-dir DIR` | auto | Directory for this run's logs and per-step screenshots |
| `--out PATH` | `<run-dir>/summary.json` | Where the summary JSON is written |
| `--reset-start` | off | Drive to the fixed spawn pose once before starting. Eval-reproducibility machinery only — a plain run starts from wherever the agent currently is |
| `--restart-env` | off | Hard-reset the **store** in Unity first (`ResetEnvironment`: items back on shelves, prior checkouts undone, agent to spawn). Use it so a fresh task doesn't inherit the last run's grabbed / checked-out items. Unlike `--reset-start`, this resets the environment, not just the agent's pose |

## Examples

```bash
# Default graph arm, explicit caps
python orchestrator/subtask_agents.py --task "get the green Piattos and bring it to the checkout counter" --arm graph --max-steps 150 --max-minutes 40

# Control arm (old VLM navigation)
python orchestrator/subtask_agents.py --task "..." --arm vlm

# Per-hop advisor-VLM drive
python orchestrator/subtask_agents.py --task "..." --arm graph-advised

# Fresh store state (recommended between unrelated runs)
python orchestrator/subtask_agents.py "pick up 2 Jin Ramen" --restart-env

# Eval-reproducible start from spawn
python orchestrator/subtask_agents.py --task "..." --reset-start
```

## Behaviour notes

- **STOP is a request, not an end.** Each leg ends when the agent emits STOP *and* a
  code-side completion predicate (keyed on the leg's type) grants it. Refused STOPs are
  capped (`halt_forced` after the cap); a goal that measurably holds for several steps
  without a STOP ends the leg as success anyway (completion backstop).
- **Self-correction:** a pickup leg holding a verifiably wrong item auto-releases it once
  per leg and resets the refusal budget; a failed leg is retried per `--leg-retries` with
  the failure reason in context.
- **Inspection is read-only and fail-closed:** an `inspect` leg must report a non-empty
  answer that the image-bound VLM guard conclusively verifies against the actor's frame
  **plus every label the leg has already inspected**. Each successful `inspect_held_item` run files
  its winning frame in a per-leg evidence ledger (`inspection_evidence`, logged as
  `inspection_evidence_recorded`), and the guard replays the whole ledger with the current frame —
  a two-item comparison is only verifiable across frames, since one held item faces the camera at a
  time. With an item in **each** hand, a STOP is refused deterministically (no VLM call) until every
  held item has been inspected, and the refusal names the hand still to read.
  For an already-held item, the actor sees only the restricted `inspect_held_item` macros—not raw
  presentation, movement, or rotation tools. It can auto-select a held hand or explicitly choose
  `inspect_held_item_left` / `inspect_held_item_right` when both hands carry items. The macro
  presents the selected item, runs a fresh isolated VLM
  visibility check, then performs eight 45-degree X turns with a check after every turn. If still
  unresolved, deterministic Y turns check the top, default, and bottom views. It returns control as
  soon as the requested nutritional/expiration label directly faces the camera and is legible.
  Every presentation, turn, fresh-frame verdict, and final result is written to the leg JSONL; the
  associated frames are saved under the step's inspection directory.
  These controls never change grip state. Every
  inspect exit restores the inspected hand's exact pre-inspection orientation by replaying every
  recorded local-axis turn in reverse with its sign inverted, then restores both hands to canonical
  REST translation without any Euler feedback. Grip/grab,
  release-toggle, checkout, and body-translation attempts are blocked and logged as scope violations.
- **Outputs:** per-leg JSONL logs, per-step screenshots and full agent output under the
  run dir, plus `summary.json` with per-leg metrics (`end_reason`, timings, LLM calls),
  planned type counts, and `unknown_subtask_rate`.
