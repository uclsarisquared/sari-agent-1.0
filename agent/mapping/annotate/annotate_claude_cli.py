"""`claude -p` annotation backend - the third and cheapest way to run the annotator's VLM step.

There are now three backends for the SAME prompts (annotator_sys_inst.py) and the SAME captured
images, which makes them a controlled comparison with one variable, the model:

  * annotate_probe.py        - Qwen 3.6-27B via the private vLLM server (raw HTTP, OpenAI shape)
  * annotate_probe_claude.py - Claude via the anthropic SDK (needs an API key / Console billing)
  * annotate_claude_cli.py   - Claude via `claude -p` (THIS file) - rides the claude.ai / Max-plan
                               login, so it bills against the subscription, not per-call API credits

Why a subprocess to `claude` instead of the SDK: the SDK path authenticates to the Anthropic API
(Console billing, a card on file). `claude -p` with `claude auth login --claudeai` authenticates
to the Claude subscription instead - the same plan a chat/Claude-Code user already pays for. For
running many hundreds of annotations off one map, that is the difference between "free under the
plan" and "metered API spend".

HOW THE IMAGE GETS IN: `claude -p` has no image flag. Instead the prompt names the image's
absolute path and Claude's own Read tool loads it (Read views PNGs, not just text). `--tools Read`
restricts the session to exactly that one tool, so nothing else (Bash, Write, WebFetch) is
reachable - the annotation step can only look, never act.

MEASURED SETUP NOTES (all verified 2026-07, don't silently "improve" them):
  * `--tools Read` + `--permission-mode acceptEdits` runs unattended with zero permission prompts
    and zero hangs. `--tools Read` alone bounds the blast radius; acceptEdits auto-approves the
    read so no TTY prompt is waited on.
  * NOT `--bare`: --bare forces ANTHROPIC_API_KEY and explicitly refuses the claude.ai OAuth we
    rely on - it would defeat the entire point of this backend.
  * `--system-prompt` REPLACES Claude Code's default system prompt, giving the model the same
    clean slate the Qwen probe gets. (`--append-system-prompt` would instead bolt our contract
    onto Claude Code's coding-agent persona - wrong for a one-shot annotator.)
  * The envelope's `structured_output` field is already a parsed dict when `--json-schema` is
    passed; `result` carries the same JSON as a string. We prefer structured_output and fall back
    to parsing result.
  * `--json-schema` uses the same structured-output subset as the API: minItems/maxItems on the
    shelf_type array are rejected, so we strip them (the prompt text still states the 1-2 rule).

    python mapping/annotate/annotate_claude_cli.py mapping/output/captures12/cp067_primary.png
    python mapping/annotate/annotate_claude_cli.py mapping/output/captures/cp028_primary.png --classify
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/annotate
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from annotator_sys_inst import (  # noqa: E402
    SYS_INST_CLASSIFY, CLASSIFY_SCHEMA,
    build_annotation_instructions, schema_for, effective_kind,
)

DEFAULT_MODEL = "sonnet"
DEFAULT_EFFORT = "medium"
DEFAULT_TIMEOUT_S = 240.0


def _strip_unsupported(schema):
    """Claude's structured-output JSON-Schema subset rejects minItems/maxItems (used on the
    shelf_type array). Drop them for the wire; the prompt still states the 1-2 constraint, so only
    server-side enforcement of it is lost, not the instruction. Deep-copies - never mutates the
    shared schema dict from annotator_sys_inst."""
    schema = json.loads(json.dumps(schema))

    def _walk(node):
        if isinstance(node, dict):
            node.pop("minItems", None)
            node.pop("maxItems", None)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(schema)
    return schema


class ClaudeCliError(RuntimeError):
    """`claude -p` returned is_error, non-zero, or unparseable output. Carries the envelope (or
    raw stdout/stderr) so the caller can log what actually happened rather than a bare message."""

    def __init__(self, message, *, envelope=None, stdout=None, stderr=None, returncode=None):
        super().__init__(message)
        self.envelope = envelope
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def annotate(image_path, system, schema, *, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT,
             timeout=DEFAULT_TIMEOUT_S, extra_views=None):
    """Run one annotation through `claude -p` and return (result_dict, envelope).

    result_dict is the schema-shaped annotation (from the envelope's structured_output). envelope
    is the full result JSON - carries usage, total_cost_usd (equivalent, not billed on Max), and
    timing. Raises ClaudeCliError on any failure.

    extra_views: optional [(label, path)] of ADDITIONAL views of the same shelf from the same spot
    - e.g. [("CROUCHED", ...)] from capture_walk's crouch capture. `image_path` is the STANDING
    view. These are NOT "context": every view is item-bearing, because they are one shelf at
    different camera heights and the CROUCHED view reaches the bottom rows the STANDING view clips.
    The prompt tells the model to enumerate across all of them and de-duplicate the overlap.
    """
    claude = shutil.which("claude") or "claude"
    lines = [f"STANDING view: {os.path.abspath(image_path)}"]
    for label, path in (extra_views or []):
        lines.append(f"{label} view: {os.path.abspath(path)}")
    prompt = "\n".join(lines)

    cmd = [
        claude, "-p", prompt,
        "--system-prompt", system,
        "--tools", "Read",                    # only tool reachable: view images, nothing else
        "--permission-mode", "acceptEdits",   # auto-approves the read; no TTY prompt to hang on
        "--output-format", "json",
        "--json-schema", json.dumps(_strip_unsupported(schema)),
        "--model", model,
        "--effort", effort,
        "--no-session-persistence",           # don't litter session files per annotation
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ClaudeCliError(f"claude -p timed out after {timeout:.0f}s", stdout=e.stdout, stderr=e.stderr)
    except FileNotFoundError:
        raise ClaudeCliError("`claude` not on PATH - is Claude Code installed? (claude auth status)")

    # Decode explicitly as UTF-8: product names carry accents/non-ASCII, and Windows' default
    # locale codec would mangle them.
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        raise ClaudeCliError("claude -p output was not JSON", stdout=stdout, stderr=stderr,
                             returncode=proc.returncode)

    if envelope.get("is_error") or proc.returncode != 0:
        # The "Not logged in · Please run /login" case lands here (is_error true, result = message).
        raise ClaudeCliError(f"claude -p reported an error: {envelope.get('result')!r}",
                             envelope=envelope, returncode=proc.returncode)

    result = envelope.get("structured_output")
    if result is None:
        try:
            result = json.loads(envelope.get("result", ""))
        except json.JSONDecodeError:
            raise ClaudeCliError("no structured_output and result was not JSON",
                                 envelope=envelope, returncode=proc.returncode)
    return result, envelope


def _build_request(args):
    """Resolve which prompt + schema + label this invocation uses (classify vs annotate, by kind)."""
    if args.classify:
        return SYS_INST_CLASSIFY, CLASSIFY_SCHEMA, "classify"
    kind = "non_shelf" if args.kind == "non_shelf" else effective_kind(args.kind)
    return build_annotation_instructions(kind), schema_for(kind), f"annotate:{kind}"


def main():
    p = argparse.ArgumentParser(description="Annotate one capture via `claude -p` (rides the claude.ai / Max-plan login).")
    p.add_argument("image", help="STANDING view (a PNG from the capture walk)")
    p.add_argument("--down", default=None, help="DOWN-pitch view of the same shelf (cp<id>_down.png)")
    p.add_argument("--up", default=None, help="UP-pitch view of the same shelf (cp<id>_up.png)")
    p.add_argument("--classify", action="store_true", help="Run Stage 1 (shelf/non_shelf) instead of Stage 2")
    p.add_argument("--kind", default="shelf", help="Stage-2 effective kind: shelf | junction | end | doorway | non_shelf")
    p.add_argument("--model", default=DEFAULT_MODEL, help="claude model or alias (default: sonnet)")
    p.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = p.parse_args()

    system, schema, label = _build_request(args)
    print(f"[claude-cli] {label}  model={args.model} effort={args.effort}  {os.path.basename(args.image)}")
    try:
        views = [(lbl, pth) for lbl, pth in (("DOWN", args.down), ("UP", args.up)) if pth]
        result, envelope = annotate(args.image, system, schema, model=args.model,
                                    effort=args.effort, timeout=args.timeout,
                                    extra_views=views)
    except ClaudeCliError as e:
        print(f"[claude-cli] FAILED: {e}")
        if e.stderr:
            print(e.stderr[:1500])
        sys.exit(1)

    u = envelope.get("usage", {})
    print(f"[claude-cli] {envelope.get('duration_ms', 0)/1000:.1f}s  "
          f"turns={envelope.get('num_turns')}  "
          f"in={u.get('input_tokens')} out={u.get('output_tokens')} "
          f"cache_read={u.get('cache_read_input_tokens')}  "
          f"cost_equiv=${envelope.get('total_cost_usd', 0):.4f}")
    print("-" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
