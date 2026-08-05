"""Command-line interface for the long-horizon orchestrator."""

import argparse
import os
import sys

from sari_runconfig import RunConfigError, load_run_config, normalize_value
from agent_core.context_policy import CONTEXT_POLICY_NAMES
from orchestrator.orchestration import orchestrate

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(argv)
    config = None
    if config_args.config:
        try:
            config = load_run_config(config_args.config)
        except RunConfigError as error:
            config_parser.error(str(error))

    def configured(section, key, fallback=None):
        return config.get(section, key, fallback) if config else fallback

    ap = argparse.ArgumentParser(description="Long-horizon typed-subtask orchestrator.")
    ap.add_argument(
        "--config",
        help="TOML run configuration. Explicit command-line flags override configured values.",
    )
    ap.add_argument("task", nargs="?", default=None,
                    help="the long-horizon task (or use --task)")
    ap.add_argument("--task", dest="task_opt", default=None, help="the long-horizon task")
    ap.add_argument("--arm", choices=["vlm", "graph", "graph-advised"],
                    default=configured("agent", "arm", "graph"),
                    help="navigation arm (default graph - the measured-better navigator; "
                         "graph-advised drives each graph hop through a per-hop advisor VLM)")
    ap.add_argument(
        "--context-policy",
        choices=CONTEXT_POLICY_NAMES,
        default=configured("agent", "context_policy", "baseline"),
        help="named context-window policy (default baseline)",
    )
    ap.add_argument("--max-steps", type=int, default=configured("limits", "max_steps", 0),
                    help="per-leg step cap; 0 = NO LIMIT (default)")
    ap.add_argument("--max-minutes", type=float,
                    default=configured("limits", "max_minutes", 0.0),
                    help="per-leg wall-clock cap in minutes; 0 = NO LIMIT (default)")
    ap.add_argument("--out", default=configured("output", "summary"),
                    help="summary.json path (default: <run-dir>/summary.json)")
    ap.add_argument("--run-dir", default=configured("output", "run_dir"),
                    help="EXACT directory for this run's outputs (per-leg JSONL + screenshots + "
                         "summary.json). Default: an auto-named <MMDD_HHMMSS>_<arm> dir under "
                         "--runs-dir.")
    ap.add_argument("--runs-dir", default=configured("output", "runs_dir"),
                    help="base directory the auto-named per-run folder is created under "
                         "(default: agent/subtask_run_outputs/). Ignored when --run-dir pins an "
                         "exact directory.")
    # `type` runs before `choices`, so the deprecated 'qwen' spelling is rewritten to 'endpoint'
    # (with a stderr warning) instead of being rejected. It stays out of `choices` deliberately:
    # accepted, not advertised.
    ap.add_argument("--resolver-backend", choices=["endpoint", "claude-cli"],
                    type=lambda v: normalize_value("agent", "resolver_backend", v,
                                                   source="--resolver-backend"),
                    default=configured("agent", "resolver_backend", "endpoint"),
                    help="plan-time map target resolver. 'endpoint' (default) uses the configured "
                         "OpenAI-compatible endpoint on $SARI_MODEL - the same model as the rest of "
                         "the run; 'claude-cli' shells out to `claude -p` instead. ('qwen' is a "
                         "DEPRECATED alias for 'endpoint'.)")
    ap.add_argument("--completion-guard", choices=["deterministic", "vlm"],
                    default=configured("agent", "completion_guard", "deterministic"),
                    help="optional pickup/compare/unknown completion backend "
                         "(default deterministic; inspect is always VLM-verified)")
    ap.add_argument("--output-dir", default=configured("environment", "map_dir"),
                    help="slamtest output dir to load the map from (topology/annotations/grid). "
                         "Default: $SARI_MAP_DIR, else slamtest/output (StoreMap's "
                         "DEFAULT_OUTPUT_DIR).")
    ap.add_argument("--leg-retries", type=int, default=configured("agent", "leg_retries", 1),
                    help="how many times to RETRY a failed leg with the failure reason in context "
                         "before aborting the task (orchestrator-level self-correction; 0 restores "
                         "the old abort-on-first-failure behaviour)")
    ap.add_argument("--reset-start", action=argparse.BooleanOptionalAction,
                    default=configured("environment", "reset_start", False),
                    help="drive to the fixed spawn pose once before starting (eval-reproducibility; "
                         "OFF by default - a plain run starts from the agent's current pose)")
    ap.add_argument("--restart-env", action=argparse.BooleanOptionalAction,
                    default=configured("environment", "restart_env", False),
                    help="hard-reset the STORE to its initial state before starting (Unity's "
                         "ResetEnvironment: items back on shelves, prior checkouts undone, agent to "
                         "spawn). OFF by default - use it so a fresh task doesn't inherit the last "
                         "run's grabbed/checked-out items. (Unlike --reset-start, which only moves "
                         "the agent.)")
    ap.add_argument("--ws-uri", default=configured("environment", "ws_uri"),
                    help="sandbox command endpoint, e.g. ws://host:51923/commands. Sets SARI_WS_URI "
                         "for this process. Default: $SARI_WS_URI, else ws://localhost:8080/commands. "
                         "Distributed Sari Bench passes the URI of the sandbox it leased for this "
                         "attempt, which is how several agents run against one machine at once.")
    ap.add_argument(
        "--ocr-url",
        default=configured("environment", "ocr_url"),
        help="OCR service base URL. Resolution: this flag, $SARI_OCR_URL, then "
             "http://127.0.0.1:9100.",
    )
    args = ap.parse_args(argv)

    # Must be set before anything reads it. sim.env resolves the default per call, not at import,
    # so setting it here still takes effect in the already-imported module.
    if args.ws_uri:
        os.environ["SARI_WS_URI"] = args.ws_uri

    task = args.task_opt or args.task or configured("agent", "task") or input("Task: ")
    orchestrate(task, arm=args.arm, caps=(max(0, args.max_steps), max(0.0, args.max_minutes)),
                out=args.out, run_dir=args.run_dir, runs_dir=args.runs_dir,
                resolver_backend=args.resolver_backend,
                reset_start=args.reset_start, restart_env=args.restart_env,
                leg_retries=max(0, args.leg_retries), output_dir=args.output_dir,
                completion_guard=args.completion_guard, ocr_url=args.ocr_url,
                context_policy=args.context_policy)


