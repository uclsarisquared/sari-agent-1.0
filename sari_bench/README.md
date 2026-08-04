# Distributed Sari Bench

Run a prompt battery across a fleet of Sari Sandbox instances. Each attempt leases one sandbox,
runs the current agent, records its artifacts, and resets the sandbox before returning it to the
pool.

## Quick start

Configure `[bench]` in the root `runconfig.toml`, then start these processes in separate terminals:

```bash
# 1. Shared OCR service on the runner machine
uv run poe ocr-server

# 2. Coordinator reachable by the runner and every sandbox host
uv run poe dbench-coordinator

# 3. One or more sandbox players (repeat on every Unity host)
SARI_BENCH_COORDINATOR=ws://<coordinator-host>:9000/sandbox ./SariSandbox

# 4. Battery runner
uv run poe dbench-run

# 5. Optional live dashboard at http://127.0.0.1:8900
uv run python -m sari_bench watch
```

Build the player from Unity with **Build > Distributed Sari Bench Player**. For editor testing, set
`SARI_BENCH_COORDINATOR` before starting Play mode. The sim must have
`sariSandboxV1CompatibilityLayer` enabled.

Check registered sandboxes with:

```bash
python -m sari_bench status --coordinator ws://<coordinator-host>:9000
```

The runner fills all ready sandboxes by default. Set `concurrency` in `runconfig.toml` to cap it.
Command-line flags override the config file.

## Results

Runs are written to `bench_runs/<timestamp>[_<name>]/`. The main files are:

- `summary.json` — battery and per-prompt results
- `attempts.jsonl` — one row per attempt
- `<prompt_id>/try<NN>/` — logs, screenshots, tokens, response, and attempt details

Useful follow-up commands:

```bash
python -m sari_bench report                       # export CSV summaries
python -m sari_bench video --battery <run-dir>    # render attempt replays
python -m sari_bench watch --run-dir <run-dir>    # inspect a specific run
```

To continue an interrupted battery, keep the same prompt and attempt settings, set its existing
directory as `output_dir`, and enable `resume` in `runconfig.toml`. The harness restarts incomplete
attempts; it does not resume them mid-step.

The dashboard can grade, kill, and retry attempts. It has no authentication, so keep its default
localhost binding unless the network is trusted.
