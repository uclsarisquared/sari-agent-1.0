"""First-contact probe: can the Qwen server actually annotate one of our captures?

This is NOT the Phase 3 annotator. It is deliberately the smallest thing that answers the
questions blocking it, one image at a time, with no navigation and no graph:

  1. Does the server accept our images at all, in the qwen chat-vision data-URI form?
  2. Does vLLM's guided_json actually enforce the schemas in annotator_sys_inst.py?
  3. Is Qwen3.6-27B good enough at THIS store's content to be worth building on - does it read
     Tostitos off a shelf, and does it correctly refuse to see products on a bare wall?

Question 3 is the real one. 1 and 2 are plumbing; 3 decides whether Phase 3 is viable at all,
and the two most informative inputs are already on disk from the capture walk:

    python slamtest/annotate_probe.py slamtest/output/captures/cp067_primary.png
        -> a dense chips+noodles shelf. Expect a 2-value shelf_type and a real item list.

    python slamtest/annotate_probe.py slamtest/output/captures/cp028_primary.png --classify
        -> a bare wall the topology mislabelled "shelf". Expect "non_shelf". This is the case
           Stage 1 exists for; ~40% of shelf checkpoints look like this.

Server: vLLM serving Qwen/Qwen3.6-27B over Chat Completions at <UCL_BASE_URL>:8000/v1, no key.
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# Repo-root api.env (overhaul/slamtest/ -> repo root is three parents up), resolved from
# __file__ so the UCL creds load regardless of CWD - this module is the shared UCL resolver for
# the slamtest tools (vlm_planner, explore_vlm import resolve_api_key/resolve_base_url from here),
# and several of them are run standalone without agent.py's loader ever executing.
load_dotenv(Path(__file__).resolve().parent.parent.parent / "api.env")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from annotator_sys_inst import (  # noqa: E402
    SYS_INST_CLASSIFY, CLASSIFY_SCHEMA,
    build_annotation_instructions, schema_for, effective_kind,
)

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_PORT = 8000


def image_content_block(model, mime_type, base64_data):
    """Vendored verbatim from SariAgent/sari_agent/openai_client.py.

    Vendored rather than imported: SariAgent lives in a different repo (SariSandboxMY), and a
    cross-repo sys.path hack for five lines is a worse dependency than a copy. The phase3 doc
    scopes the reuse to exactly this helper - not SariAgent's streaming/tool-loop harness, since
    annotation is single-turn.
    """
    url = f"data:{mime_type};base64,{base64_data}"
    if "qwen" in model.lower():
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "input_image", "image_url": url}


def resolve_api_key(explicit=None):
    """The qwen server REQUIRES a bearer key since 2026-07 (measured: /v1/models returns 401
    without it - the old "vLLM ignores it" era is over). $UCL_API first; conda-meta/state
    fallback because invoking sari_env_old's python.exe directly skips the env-var hooks.
    Resolved at call time, never hardcoded, so a rotated key is picked up automatically.
    Read as UCL_API_KEY first (api.env's spelling) then UCL_API (legacy env / conda state)."""
    if explicit not in (None, "", "none"):
        key = explicit
    else:
        key = os.environ.get("UCL_API_KEY") or os.environ.get("UCL_API")
    if not key:
        try:
            import json as _json
            with open(r"C:/Sari/sari_env_old/conda-meta/state", encoding="utf-8") as f:
                ev = _json.load(f).get("env_vars", {})
            key = ev.get("UCL_API_KEY") or ev.get("UCL_API")
        except OSError:
            pass
    return key or "none"


def resolve_base_url(explicit):
    """UCL_BASE_URL is a bare host (e.g. "202.92.159.240"); the qwen server is :8000/v1.
    Same env-then-conda-state resolution as resolve_api_key, for the same reason: invoking
    sari_env_old's python.exe directly skips the activation hooks that set the vars."""
    raw = explicit or os.environ.get("UCL_BASE_URL")
    if not raw:
        try:
            import json as _json
            with open(r"C:/Sari/sari_env_old/conda-meta/state", encoding="utf-8") as f:
                raw = _json.load(f).get("env_vars", {}).get("UCL_BASE_URL")
        except OSError:
            pass
    if not raw:
        sys.exit("no --base-url, no $UCL_BASE_URL, and no sari_env_old conda state to read")
    raw = raw.strip().rstrip("/")
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    if ":" not in raw.split("//", 1)[1]:
        raw = f"{raw}:{DEFAULT_PORT}"
    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"
    return raw


def post_chat(base, payload, api_key, timeout):
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # vLLM puts the real complaint (bad schema, unsupported field) in the body, not the status.
        sys.exit(f"HTTP {e.code} from server:\n{e.read().decode()[:2000]}")


def main():
    p = argparse.ArgumentParser(description="Probe the Qwen server with one captured image.")
    p.add_argument("image", help="PNG from the capture walk")
    p.add_argument("--base-url", default=None, help="Default: $UCL_BASE_URL, +:8000/v1")
    p.add_argument("--api-key", default=None,
                   help="Bearer for the qwen server (default: $UCL_API, then sari_env_old's "
                        "conda state). The server 401s without it - measured 2026-07-19.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--classify", action="store_true",
                   help="Run Stage 1 (shelf/non_shelf) instead of Stage 2 annotation")
    p.add_argument("--kind", default="shelf",
                   help="Stage-2 effective kind: shelf | junction | end | doorway | non_shelf")
    p.add_argument("--guided", action=argparse.BooleanOptionalAction, default=True,
                   help="Send the schema as vLLM guided_json. --no-guided to see what the model "
                        "emits unconstrained (i.e. whether the prompt alone holds the contract)")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                   help="Let Qwen3.x reason before answering. OFF by default: the schema already "
                        "does the structuring, so thinking buys nothing here and a measured run "
                        "spent its whole budget on it (and looped) without reaching an answer.")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()

    base = resolve_base_url(args.base_url)
    with open(args.image, "rb") as f:
        raw = f.read()
    mime = mimetypes.guess_type(args.image)[0] or "image/png"

    if args.classify:
        system, schema, label = SYS_INST_CLASSIFY, CLASSIFY_SCHEMA, "classify"
    else:
        kind = effective_kind(args.kind) if args.kind != "non_shelf" else "non_shelf"
        system, schema, label = build_annotation_instructions(kind), schema_for(kind), f"annotate:{kind}"

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": "PRIMARY image:"},
                image_content_block(args.model, mime, base64.b64encode(raw).decode()),
            ]},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        # Qwen3.x reasons unless told not to. This is the documented off-switch and vLLM passes
        # it through to the chat template.
        "chat_template_kwargs": {"enable_thinking": args.think},
    }
    if args.guided:
        # MUST be response_format. `guided_json` is the older vLLM spelling and this build
        # SILENTLY IGNORES it - measured with a negative-control schema whose only legal label was
        # "banana": with guided_json the model answered "non_shelf" (its own answer), with
        # response_format it answered "banana". Silent, because the prompt alone already yields
        # conforming JSON, so an ignored schema looks exactly like a working one.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": label.replace(":", "_"), "schema": schema, "strict": True},
        }

    print(f"[probe] {label}  guided={args.guided}  {os.path.basename(args.image)} "
          f"({len(raw)/1e6:.1f}MB)  -> {base}")
    t0 = time.time()
    resp = post_chat(base, payload, resolve_api_key(args.api_key), args.timeout)
    dt = time.time() - t0

    choice = resp["choices"][0]
    message = choice["message"]
    text = message.get("content")
    usage = resp.get("usage", {})
    print(f"[probe] {dt:.1f}s  prompt={usage.get('prompt_tokens','?')} "
          f"completion={usage.get('completion_tokens','?')} "
          f"finish={choice.get('finish_reason')}")
    print("-" * 80)

    if not text:
        # content=None is not a bug to route around - it says the model never reached an answer.
        # Qwen3.x reasons by default and vLLM splits that into reasoning_content, so a thinking
        # run that hits max_tokens returns exactly this. Show what it DID emit.
        print(f"[probe] content is empty. message keys: {sorted(message)}")
        # vLLM names this `reasoning` on this build; other builds use `reasoning_content`.
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            print(f"[probe] reasoning_content ({len(reasoning)} chars) - tail:")
            print(reasoning[-1500:])
        else:
            print(json.dumps(resp, indent=2)[:3000])
        return

    try:
        print(json.dumps(json.loads(text), indent=2))
    except json.JSONDecodeError:
        # Worth seeing verbatim: unparseable output IS the finding when --no-guided.
        print("[probe] NOT VALID JSON - raw output follows:")
        print(text)


if __name__ == "__main__":
    main()
