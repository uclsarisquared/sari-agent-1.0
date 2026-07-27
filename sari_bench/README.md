# Distributed Sari Bench

Run a prompt battery across a fleet of Sari Sandbox instances, several attempts per prompt, with a
guaranteed-clean environment for every attempt.

```
  machine A            machine B                      coordinator machine
┌────────────┐       ┌────────────┐                 ┌──────────────────────┐
│ Sandbox #1 │──ws──▶│            │────────ws──────▶│  sari_bench          │
│ :51923     │       │ Sandbox #2 │                 │  coordinator :9000   │
└────────────┘       │ :49110     │                 │   /sandbox  /bench   │
      ▲              └────────────┘                 └──────────┬───────────┘
      │                     ▲                                  │ lease
      │  ws commands        │                                  ▼
      └─────────────────────┴───────────  subprocess:  orchestrator/subtask_agents.py
                                                        --ws-uri ws://A:51923/commands
```

## Running it

**1. Coordinator** (anywhere the sandboxes and the runner can both reach):

```bash
python -m sari_bench coordinator --port 9000
```

**2. Sandboxes.** Build the bench player once — *Build > Distributed Sari Bench Player* in the
editor, or `-executeMethod DistributedBenchBuild.BuildFromCommandLine` in batch mode. Then launch as
many as the machine can carry:

```bash
SARI_BENCH_COORDINATOR=ws://coordinator-host:9000/sandbox ./SariSandbox &
SARI_BENCH_COORDINATOR=ws://coordinator-host:9000/sandbox ./SariSandbox &
```

Each one picks its own free port, binds it on all interfaces, and registers. Set
`SARI_BENCH_ADVERTISED_HOST` if the address the coordinator sees is not the address agents should
dial (NAT, or a container with a published port). Check the pool with:

```bash
python -m sari_bench status --coordinator ws://coordinator-host:9000
```

You do not strictly need the bench build: setting `SARI_BENCH_COORDINATOR` makes any build join a
fleet, which is the convenient way to test with an editor session. The define only matters for a
player you want to be a fleet member with no environment set.

**3. Runner:**

```bash
python -m sari_bench run \
    --prompts sari_bench/prompts/example_battery.json \
    --tries 3 --time-limit 120 --per-leg-minutes 40 \
    --coordinator ws://coordinator-host:9000
```

With no `--concurrency` flag, the runner fills every registered, store-loaded sandbox and adds
workers automatically when more sandboxes join during the battery. Pass `--concurrency N` to keep
an explicit upper bound.

The `uv run poe dbench` shortcut waits up to 30 seconds for at least one usable sandbox to
register, then exits with launch instructions instead of leaving every prompt at
`waiting for a sandbox`. Use the `status` command above to inspect the pool.

Results land in `bench_runs/<timestamp>/`: `battery.json` (the plan), `summary.json` (per-prompt
success rates and outcome counts), `attempts.jsonl` (atomically updated as attempts finish, with one
canonical row per prompt/attempt, so an interrupted battery is still usable), and
`<prompt_id>/try<NN>/` holding `attempt.json` (that attempt's
manifest — sandbox, pid, deadline, outcome), the orchestrator's own `summary.json`, per-leg JSONL,
screenshots, and the agent's stdout in `agent.log`.

### Restarting a battery

Reusing a non-empty `--output-dir` is rejected unless resume is explicit:

```bash
python -m sari_bench run \
    --prompts sari_bench/prompts/example_battery.json \
    --tries 3 --time-limit 120 --per-leg-minutes 40 \
    --completion-guard vlm \
    --coordinator ws://coordinator-host:9000 \
    --output-dir bench_runs/20260727_120000 \
    --resume
```

Resume requires the prompts and attempt-shaping settings to match the original `battery.json`;
operational settings such as coordinator and concurrency may change. Finished logical attempts are
not run again. A try left in flight is stopped if its orphan subprocess can be safely identified,
its stale lease is defensively released, and its partial `tryNN` directory is deleted before that
try starts cleanly. There is no mid-step continuation.

Legacy duplicate rows in `attempts.jsonl` are compacted to the latest row for each prompt/attempt,
and a finished `attempt.json` whose aggregate row was never published is recovered during resume.
Human-verified winners in `battery.json` are preserved, so their remaining siblings stay skipped.
A fully completed battery resumes without needing a live coordinator or sandbox.

`--completion-guard {deterministic,vlm}` is passed to every orchestrator subprocess and recorded in
`battery.json` plus each `attempt.json`. It defaults to `deterministic`; watcher-triggered retries
preserve the original battery setting.

The watcher is independent of this scheduler behavior: `python -m sari_bench watch --run-dir
<battery>` can visualize a completed battery without running or resuming anything.

**Token usage** is recorded per attempt: `tokens_in` (prompt) / `tokens_out` (completion) on every
row of `attempts.jsonl`, in each attempt's `attempt.json`, summed per prompt and battery-wide in the
battery `summary.json`, and per leg in the orchestrator's own `summary.json`. The counts come from
`agent_core/token_meter.py`, which patches the OpenAI SDK once so **every** reasoner is counted —
actor, semantic/episodic learner, advisor, decomposer, resolver, perception, plus the SDK's internal
retries — not just the calls someone remembered to instrument. Moondream (grab-time pointing) reports
no usage and is therefore not counted; its qwen fallback is.

The agent rewrites `<prompt_id>/try<NN>/tokens.json` every few seconds, so an attempt that is
SIGKILLed on the harness timeout — usually the most expensive kind — still accounts for what it
burned, even though it never wrote a `summary.json`.

**`--time-limit` and `--per-leg-minutes` are different clocks.** `--time-limit` bounds the whole
attempt and is enforced by the harness (SIGTERM, then SIGKILL). `--per-leg-minutes` is the agent's
own `--max-minutes`, which is **per leg**; it defaults to `--time-limit`, which is only sensible for
single-leg tasks. Set it lower for a long attempt limit — otherwise a 5-leg task can never hit its
own `time_cap`, so every overrun arrives as a SIGKILLed `harness_timeout` with no `summary.json` and
no per-leg detail.

**4. Watch it live** (on the runner's machine — screenshots are written to its local disk):

```bash
python -m sari_bench watch --discord            # newest battery under bench_runs/
python -m sari_bench watch --run-dir bench_runs/20260725_170000   # or pin one
```

Then <http://127.0.0.1:8900>. See [Watching a battery](#watching-a-battery).

The sim must be running with **`sariSandboxV1CompatibilityLayer` ON** — `overhaul/sim/env.py`
parses the V1 text protocol.

## Prompt batteries

Same schema as `overhaul/tests/decompose_battery.json`, so existing files work unchanged. A bare
list, a list of bare strings, or `{"prompts": [...]}` are all accepted.

```json
{"prompts": [
  {"id": "pickup_01", "family": "pickup", "prompt": "Pick up a bottle of soy sauce",
   "looking_for": "soy sauce held in either hand"}
]}
```

## What guarantees a clean environment

The coordinator resets a sandbox **on release**, not on acquire, and only returns it to the pool
once the sandbox itself reports ready. Three things follow:

- an acquire never hands out a mid-reset environment, and never waits on one;
- an attempt whose agent crashed still gets cleaned up, because the cleanup does not depend on the
  agent asking for it;
- `ResetEnvironment` in a current sim only acks once items are back on shelves *and at rest*, with
  the agent returned to its spawn pose, hands and grip cleared, and the basket lowered. Older sims
  acked in the same frame, before Unity had even run the deferred destroys.
- resets are issued one at a time across the fleet, avoiding a same-host teardown/rebuild stampede;
  a reset that does not report `Ready` within three minutes is disconnected and quarantined.

An agent that connects while its sandbox is still booting or resetting has its commands **parked**
sim-side and answered when the sandbox is ready — it waits rather than seeing a garbled reply.
`subtask_agents.py` also calls `env.wait_for_ready()` before any sim traffic, which additionally
retries the connection itself for a sandbox that is not listening yet.

## Failure handling

| What happened | Recorded outcome | Sandbox |
|---|---|---|
| Agent exited 0 | `completed` (success read from its summary.json) | reset, re-pooled |
| Agent exited non-zero | `agent_error`, always `success: false`; automatically invalid/excluded in watch UI | reset, re-pooled |
| Attempt overran `--time-limit` | `harness_timeout` (SIGTERM then SIGKILL to the process group) | reset, re-pooled |
| Killed from the dashboard | `operator_kill` | reset, re-pooled |
| Sibling stopped after a human-verified success | `operator_kill`, end reason `already_successful` | reset, re-pooled |
| Sibling not yet spawned after a human-verified success | `skipped`, end reason `already_successful` | unchanged |
| Sandbox stopped heartbeating | attempt **requeued** (up to 3x), then `sandbox_lost` | hung up on, rejoins when the sim reconnects |
| Runner died holding a lease | — | lease reaped, reset, re-pooled |
| Runner died mid-attempt | shown `orphaned`; `orphaned`/`operator_kill` once closed out from the dashboard | released on close-out |

A sandbox-loss requeue reuses its `<prompt_id>/try<NN>` path, so the dead attempt's dir is **rotated
aside** to `try<NN>.requeue<KK>` before the replacement starts. Without that the orchestrator's
append-mode `legNN.jsonl` interleaves two attempts' steps into one file and their `stepNN.png`
frames overwrite each other, which misreports timesteps for both.

## Watching a battery

`python -m sari_bench watch` serves a live dashboard. It coordinates with the runner through durable
filesystem stamps and never sends sandbox commands itself. Kill requests and successful human
verdicts can stop work; ordinary observation, replay, and fail/invalid verdicts do not perturb the
battery.
The watcher can be restarted mid-run freely.

The runner keeps each live tile fresh during long model calls by filling gaps between the agent's
own step screenshots. `--capture-interval 0.25` is the default (4 frames/second); a recent step frame suppresses the
extra request, and `--capture-interval 0` disables supplementary capture. These frames are stored as
960px JPEGs under each attempt's `capture/` directory and are included in full dashboard/CLI
replays. The dashboard's visible live cards also refresh at 4 FPS; its heavier state and log polling
remains every 2 seconds, and hidden/overview tabs do not poll live frames. Discord's size-bounded
clip remains step-only.

Run it **beside the runner**. Screenshots and step logs are written by the agent subprocesses the
runner spawns, so they are on the runner's local disk; the coordinator is allowed to be a third
machine. The watcher opens one read-only `/bench` connection to show the sandbox pool — it only ever
sends `bench.status`, never `bench.acquire`, so it cannot take a sandbox from a worker. `--no-pool`
skips it entirely.

By default it **auto-discovers** the newest battery under `bench_runs/` and follows the fleet from
battery to battery without a restart; `--run-dir` pins one. Either way it logs which battery it
picked, and the runner logs whether it created a fresh output dir or is reusing one that already
holds results.

Tiles are sorted **worst-first** by a collapse score, which is the actual feature — with eight
concurrent attempts you want to look at one tile, not scan eight. Every signal comes from the `step`
records `run_leg` already flushes, so there is no new agent-side instrumentation:

| Signal | What it measures |
|---|---|
| `stalled` | no new step record in 5 minutes — a hung VLM call or a wedged sim |
| `spatial_loop` | ≤2 distinct 0.5 m cells over the last 10 steps — pacing in front of one shelf |
| `action_loop` | the same action fired 4+ times in the window |
| `mode_thrash` | the mode router flip-flopping |
| `blocked` | blocked on >50% of recent steps — bumping a wall |
| `refusal_spiral` | halts requested and refused, with no `goal_met` |
| `step_budget` / `time_budget` | >80% of the cap burned |

**Kill** on a tile SIGTERMs the agent's process group and then gets out of the way: the runner's own
`process.wait()` returns non-zero, it records `operator_kill`, and its `finally` releases the lease
so the coordinator resets and re-pools the sandbox. There is no second code path, and the watcher
never has to talk to the coordinator about it.

That relies on the runner still being there. When it is not - it crashed, was SIGKILLed, its terminal
closed - its attempt keeps `state: running` and a pid that no longer exists, which the dashboard shows
as **orphaned**. On those tiles the kill button reads **close out**: since nothing will ever write the
attempt's closing record, the watcher writes it instead, recording only what is knowable - it stopped,
with no result of its own (`orphaned`, or `operator_kill` if a kill was what stranded it), end reason
`runner_gone`, no exit code, and `closed_out_by: watcher` so the row is never mistaken for the
runner's. Tokens come from the agent's last `tokens.json`, the lease is released best-effort, and the
attempt becomes an ordinary finished one: on the spine in `attempts.jsonl`, and reviewable - usually
as `invalid`. A runner that is merely slow to finalize keeps the last word; close-out waits for it
first and defers to whatever it writes.

**Retry** replaces one logical prompt/try in place. It stops a live agent through that same cleanup
path, deletes `tryNN` and all of its `tryNN.requeue*` history, removes the old aggregate result, and
leases a fresh sandbox for the same prompt and try number. The watcher owns the replacement runner,
so retry remains available after the original battery runner exits. This is deliberately destructive:
the dashboard confirms before deleting logs, frames, replay clips, verdicts, and report data.

`--host` defaults to `127.0.0.1`. Binding `0.0.0.0` exposes the destructive kill and retry endpoints
to the network; there is no auth.

### Human-verified outcomes

`success` is whatever `subtask_completion.py` decided, and several of its predicates grant on state
they cannot ground — they say so themselves, in reasons like `goto granted [unverified]: checkpoint
info unavailable`. `predicate_unknown` is a keyword guard. So the headline success rate is a number
the harness cannot fully stand behind.

A halted attempt's tile therefore carries **▶ replay · ✓ success · ✗ fail · ⊘ invalid**. The watcher
queues its full replay as soon as the run finishes; press ▶ to play it in a modal and judge the run
against what you just watched. Discord's separate, size-bounded attachment is rendered by the same
one-at-a-time worker.

* **Every finished attempt is judgeable.** Verdict controls are available regardless of end reason,
  including forced halts, caps, errors, and administrative skips. A still-running attempt remains
  ineligible because its outcome can change. The server re-checks this; the button is not the only
  gate.
* **⊘ invalid is a third answer, not a soft fail.** It is for a run the harness broke rather than one
  the agent lost: a sandbox that never came up, a crashed capture, a prompt that never reached the
  agent. An invalid try is excluded from the battery's arithmetic entirely — it leaves the
  reliability denominator, it cannot fail a prompt on its own (a row of nothing but ⊘ reads as
  undispatched, not as a loss), and it cancels no siblings, because it decided nothing. Its cell is
  gray `E`, deliberately the same family as the killed cells: both are runs nothing is scored from.
  Token totals still include it — the tokens were spent.
  `agent_error` receives this invalid classification automatically unless a reviewer explicitly
  overrides it with pass or fail; its underlying benchmark `success` remains false.
* **Stored beside `success`, never over it.** The stamp is `verified_verdict` (`pass` / `fail` /
  `invalid`) / `verified_success` / `verified_by` / `verified_at` / `verified_note` in
  `attempt.json`. An invalid verdict writes **no `verified_success` at all**, so any reader that
  predates the third verdict falls back to "unreviewed" rather than to "a human said it failed". This is the same honest-scoring rule
  `gates/gate_checkout.py` follows: measured and verified are logged separately and never promoted,
  because *a measured pass with a verified fail is the discrepancy the whole exercise exists to
  surface*. A card where the two disagree says so without re-highlighting the completed card; the
  human badge is green for ✓ and red for ✗.
* **A verified success stops sibling tries.** Confirming ✓ success durably cancels only other tries
  of that exact prompt ID (never other prompts with the same `family`). Running siblings finish as
  `operator_kill`; queued siblings are recorded as zero-runtime `skipped` attempts. Both carry
  `end_reason=already_successful` and the winning attempt key, so the planned denominator and report
  coverage remain intact.
* **Verdict metadata is correctable; cancellation is not.** Pressing the other button overwrites and
  `clear` removes the review stamp, but neither action restarts, requeues, or restores siblings
  stopped by an earlier successful verdict.
* Clips are queued **on finish**, on the same one-at-a-time worker Discord uses, and reused once
  written. Requesting an older missing replay from the modal queues it as a fallback.
* The header tracks review progress: `N reviewed (M✓/K✗/J⊘, D disagree) · P awaiting review`.
* **The overview tab reviews from the keyboard.** Hover any try cell and press <kbd>P</kbd>,
  <kbd>F</kbd> or <kbd>E</kbd> to mark it pass, fail or invalid; the cell flashes to acknowledge the
  keystroke, because the verdict itself only repaints a poll later. Clicking a cell opens the same
  replay modal, with that attempt's **complete** log rather than its tail.

None of this needs Discord — `--no-replay` is the only flag that turns clip rendering off, and
without `ffmpeg` on PATH the verdict buttons still work, you just cannot watch the run first.

### Discord

`--discord` with `SARI_BENCH_DISCORD_WEBHOOK` set in `api.env` posts: battery started, **collapse
alerts with the offending screenshot attached** (once per attempt, 15-minute cooldown), **every halt
with a replay clip attached**, and battery complete. Notification lives in the watcher rather than
the runner deliberately: the runner is a scheduler, and an outbound HTTP call has no business on the
path that releases a lease. Every send is fail-soft — a webhook that 429s can never disturb the poll
loop (it gets one `Retry-After` retry, then is dropped).

When an agent halts, the watcher renders that attempt's screenshots into a `replay.discord.mp4` and
attaches it to the finish message, so the channel tells you *what happened* and not just that
something did. Notes:

* **All outcomes, not just failures.** An earlier version posted failures only, because a success was
  a bare line of metrics and that really was noise at 3 tries × 20 prompts. A clip of a win is worth
  watching, so every finish is announced now.
* **A separate file from `replay.mp4`.** The clip is capped at 8 MB (960 px, bitrate computed from its
  duration) because that is what a webhook will carry; the `video` CLI below writes an uncapped
  `replay.mp4` for archive viewing. On 300 frames of worst-case footage those are 7.5 MB and 178 MB
  respectively, which is why auto-render never writes over the CLI's copy.
* **Rendered on a worker thread**, one encode at a time. The notify diff runs on the thread serving
  `/api/state`, so an inline ffmpeg pass would hang the dashboard for as long as the encode takes,
  and eight simultaneous finishes would mean eight ffmpegs competing with the agents still running.
* **Restarting the watcher mid-battery does not replay the run into the channel.** Finishes already on
  disk at startup are marked announced silently; `--replay-backfill` opts into posting them.
* `--no-replay` turns clips off — both the attachment here and the dashboard's replay — and posts
  finishes as text. Without `ffmpeg` on PATH you get the same thing, plus a warning at startup.

## Stats and replays

```bash
python -m sari_bench report                       # newest battery -> attempts.csv + legs.csv
python -m sari_bench video --battery bench_runs/20260725_170000   # replay.mp4 per attempt
```

`report` writes **two** CSVs, joined on `run_dir`: attempts and legs are different grains, and
folding a variable number of legs into an attempt row either truncates or explodes. It reads
`attempts.jsonl` plus each attempt's own `summary.json` — both written incrementally — so it is
runnable mid-battery, and it folds in attempts that started but never closed out (recording their
collapse score, which is what makes "how many were in a death loop when I killed them" answerable
afterwards).

`attempts.csv` carries the human verdict beside the predicate's: `verified_verdict`,
`verified_success`, `verdict_agrees` and `success_final` (the human's call where there is one, the
predicate's otherwise — group by this).
`verified_success` is **blank, not `False`, where nobody has looked**, so an unreviewed attempt is
never counted as one a human failed. A run marked `invalid` leaves all three of `verified_success`,
`verdict_agrees` and `success_final` blank for the same reason — it drops out of every grouping
instead of landing in the failure bucket — and `verified_verdict` is where you find it.
The closing line reports how many verdicts agreed and how many
did not, which is the number to watch: it is a direct measurement of how much the completion
predicates can be trusted.

`video` chronologically merges `legNN/stepNN.png` with supplementary `capture/*.jpg` frames into an
mp4, captioning steps with their mode, action and checkpoint and gap captures with elapsed time. mp4
rather than gif: a long 1080p gif runs to hundreds of megabytes and cannot be scrubbed (`--gif` is
there for pasting into chat). Needs `ffmpeg` on PATH. This writes `replay.mp4` uncapped, for watching;
the watcher's step-only 8 MB `replay.discord.mp4` is a separate file and neither overwrites the other.

## Tests

Offline, in-process, no sim and no model stack:

```bash
python sari_bench/tests/test_coordinator.py   # pool, leases, reaping, reset gating
python sari_bench/tests/test_runner.py        # lease -> spawn -> release against a stub agent
python sari_bench/tests/test_watch.py         # scanning, collapse scoring, discovery, HTTP, report
python overhaul/tests/test_token_meter.py     # token counting, against the real SDK on loopback
```
