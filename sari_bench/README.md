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
    --tries 3 --time-limit 120 --per-leg-minutes 40 --concurrency 4 \
    --coordinator ws://coordinator-host:9000
```

Results land in `bench_runs/<timestamp>/`: `battery.json` (the plan), `summary.json` (per-prompt
success rates and outcome counts), `attempts.jsonl` (written as attempts finish, so an interrupted
battery is still usable), and `<prompt_id>/try<NN>/` holding `attempt.json` (that attempt's
manifest — sandbox, pid, deadline, outcome), the orchestrator's own `summary.json`, per-leg JSONL,
screenshots, and the agent's stdout in `agent.log`.

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

An agent that connects while its sandbox is still booting or resetting has its commands **parked**
sim-side and answered when the sandbox is ready — it waits rather than seeing a garbled reply.
`subtask_agents.py` also calls `env.wait_for_ready()` before any sim traffic, which additionally
retries the connection itself for a sandbox that is not listening yet.

## Failure handling

| What happened | Recorded outcome | Sandbox |
|---|---|---|
| Agent exited 0 | `completed` (success read from its summary.json) | reset, re-pooled |
| Agent exited non-zero | `agent_error` | reset, re-pooled |
| Attempt overran `--time-limit` | `harness_timeout` (SIGTERM then SIGKILL to the process group) | reset, re-pooled |
| Killed from the dashboard | `operator_kill` | reset, re-pooled |
| Sandbox stopped heartbeating | attempt **requeued** (up to 3x), then `sandbox_lost` | dropped from the pool |
| Runner died holding a lease | — | lease reaped, reset, re-pooled |

A requeued attempt reuses its `<prompt_id>/try<NN>` path, so the dead attempt's dir is **rotated
aside** to `try<NN>.requeue<KK>` before the replacement starts. Without that the orchestrator's
append-mode `legNN.jsonl` interleaves two attempts' steps into one file and their `stepNN.png`
frames overwrite each other, which misreports timesteps for both.

## Watching a battery

`python -m sari_bench watch` serves a live dashboard. It reads the filesystem and nothing else: it
never talks to a sandbox, and the only things it ever writes into a run dir are the `killed_by` stamp
and a human verdict — neither of which an agent reads. So it cannot perturb a battery that is hours
in, and it can be restarted mid-run freely.

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

`--host` defaults to `127.0.0.1`. Binding `0.0.0.0` exposes the kill endpoint to the network; there
is no auth.

### Human-verified outcomes

`success` is whatever `subtask_completion.py` decided, and several of its predicates grant on state
they cannot ground — they say so themselves, in reasons like `goto granted [unverified]: checkpoint
info unavailable`. `predicate_unknown` is a keyword guard. So the headline success rate is a number
the harness cannot fully stand behind.

A halted attempt's tile therefore carries **▶ replay · ✓ success · ✗ fail**. Press ▶ and the watcher
renders that attempt's frames into the same ≤8 MB clip Discord gets, plays it in a modal, and you
judge the run against what you just watched.

* **Only where the agent stopped on its own** — `end_reason` of `halt_granted` (it said STOP and the
  predicate granted) or `completed_no_stop` (the backstop fired). A run the harness cut off at its
  step or time cap never claimed to be done, so there is nothing to confirm or deny. The server
  re-checks this; the button is not the only gate.
* **Stored beside `success`, never over it.** The stamp is `verified_success` / `verified_by` /
  `verified_at` / `verified_note` in `attempt.json`. This is the same honest-scoring rule
  `gates/gate_checkout.py` follows: measured and verified are logged separately and never promoted,
  because *a measured pass with a verified fail is the discrepancy the whole exercise exists to
  surface*. A card where the two disagree is outlined in amber and says so.
* **Correctable.** Pressing the other button overwrites; `clear` removes the stamp entirely, so the
  attempt reads as never reviewed rather than as a verdict of fail.
* Clips are rendered **on demand**, on the same one-at-a-time worker Discord uses, and reused if that
  path already made one. Nothing is encoded for attempts nobody opens.
* The header tracks review progress: `N reviewed (M✓/K✗, J disagree) · P awaiting review`.

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

`attempts.csv` carries the human verdict beside the predicate's: `verified_success`, `verdict_agrees`
and `success_final` (the human's call where there is one, the predicate's otherwise — group by this).
`verified_success` is **blank, not `False`, where nobody has looked**, so an unreviewed attempt is
never counted as one a human failed. The closing line reports how many verdicts agreed and how many
did not, which is the number to watch: it is a direct measurement of how much the completion
predicates can be trusted.

`video` stitches `legNN/stepNN.png` into an mp4, captioning each frame with that step's mode,
action and checkpoint. mp4 rather than gif: a 150-frame 1080p gif runs to hundreds of megabytes and
cannot be scrubbed (`--gif` is there for pasting into chat). Needs `ffmpeg` on PATH. This writes
`replay.mp4` uncapped, for watching; the watcher's 8 MB `replay.discord.mp4` is a separate file and
neither overwrites the other.

## Tests

Offline, in-process, no sim and no model stack:

```bash
python sari_bench/tests/test_coordinator.py   # pool, leases, reaping, reset gating
python sari_bench/tests/test_runner.py        # lease -> spawn -> release against a stub agent
python sari_bench/tests/test_watch.py         # scanning, collapse scoring, discovery, HTTP, report
python overhaul/tests/test_token_meter.py     # token counting, against the real SDK on loopback
```
