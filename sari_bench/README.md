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
    --tries 3 --time-limit 40 --concurrency 4 \
    --coordinator ws://coordinator-host:9000
```

Results land in `bench_runs/<timestamp>/`: `summary.json` (per-prompt success rates and outcome
counts), `attempts.jsonl` (written as attempts finish, so an interrupted battery is still usable),
and `<prompt_id>/try<NN>/` holding the orchestrator's own `summary.json`, per-leg JSONL,
screenshots, and the agent's stdout in `agent.log`.

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
| Sandbox stopped heartbeating | attempt **requeued** (up to 3x), then `sandbox_lost` | dropped from the pool |
| Runner died holding a lease | — | lease reaped, reset, re-pooled |

## Tests

Offline, in-process, no sim and no model stack:

```bash
python sari_bench/tests/test_coordinator.py   # pool, leases, reaping, reset gating
python sari_bench/tests/test_runner.py        # lease -> spawn -> release against a stub agent
```
