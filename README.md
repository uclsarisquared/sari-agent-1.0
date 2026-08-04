<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/9b52e148-3998-420e-ba2a-cb15b0011681">
    <img width="180" height="180" alt="Sari Agent logo" src="https://github.com/user-attachments/assets/9b52e148-3998-420e-ba2a-cb15b0011681">
  </picture>
</p>

<h1 align="center">Sari Agent²</h1>

The current embodied agent for Sari Sandbox V2. It uses a checkpoint graph for navigation and
vision models for local perception, manipulation, and task reasoning. The Unity simulator lives in
the separate `SariSandboxV2` repository.

## Quick start

Requirements:
* [uv](https://docs.astral.sh/uv/)
* Sari Sandbox 1.0/Sari Sandbox² running in Unity Play mode or a build
  * If on Sandbox², `sariSandboxV1CompatibilityLayer` must be enabled
* Python is pinned to 3.10.13.

```bash
uv sync
cp config.env.example config.env
```

Set `OPENAI_API_URL` (a bare host with no scheme or port) and `OPENAI_API_KEY` in `config.env`.
`MDREAM_API_KEY` enables the primary grab-pointing service; the runtime falls back to Qwen if it is
unavailable.

The checked-in [`runconfig.toml`](runconfig.toml) already points at the complete map in
`overhaul/slamtest/output_runs/run_0724_164652`. Start these in separate terminals:

```bash
# Terminal 1: required shared OCR service
uv run poe ocr-server

# Terminal 2: run one task
uv run python overhaul/orchestrator/subtask_agents.py \
  --config runconfig.toml --task "find and pick up Pepero"
```

The simulator WebSocket defaults to `ws://localhost:8080/commands`; override `ws_uri` in
`runconfig.toml` when needed. Run artifacts default to `overhaul/subtask_run_outputs/`.

## Configuration

- `runconfig.toml` controls the agent, limits, map, outputs, and benchmark. CLI flags override it.
- `config.env` holds credentials and machine-specific paths and is gitignored.
- `config.env.example` documents every supported environment variable.
- `--help` on the agent or `python -m sari_bench <command> --help` is the authoritative flag list.

## Distributed benchmark

Edit `[bench]` in `runconfig.toml`, then run:

```bash
uv run poe ocr-server
uv run poe dbench-coordinator
SARI_BENCH_COORDINATOR=ws://<coordinator-host>:9000/sandbox ./SariSandbox
uv run poe dbench-run
uv run python -m sari_bench watch  # http://127.0.0.1:8900
```

See [`sari_bench/README.md`](sari_bench/README.md) for the concise fleet, resume, and reporting
guide.

## Repository map

- `overhaul/` — live agent; entrypoint: `orchestrator/subtask_agents.py`
- `overhaul/slamtest/` — map building and annotation; normal runs use the shipped map
- `sari_bench/` — distributed runner, dashboard, reports, and replay tooling
- `analysis/` — benchmark and ablation reports
- `legacy/` — deprecated first-generation agent, retained only for reference

For internals, see [`overhaul/README.md`](overhaul/README.md). For rebuilding the map, follow
[`overhaul/slamtest/README.md`](overhaul/slamtest/README.md); do not overwrite the working map
casually because checkpoint IDs key its captures and annotations.
