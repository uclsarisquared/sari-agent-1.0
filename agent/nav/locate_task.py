"""Phase 4 locate agent: resolve -> drive -> verify. The first end-to-end consumer of the map.

    python locate_task.py "Locate the Coke Zero in the store."

The division of labour is the whole architecture (`phase4_agent_integration.md`):

  RESOLVER (LLM, text-only, ONCE)  reads the whole product index and names the target + which
        checkpoints might hold it. Semantic matching is deliberately its job - "Coke Zero" shares
        zero tokens with the index's "Coca-Cola", so a deterministic matcher misses (measured:
        StoreMap.search("coke zero") == []), while brand knowledge is exactly what an LLM has.
  CODE (deterministic)             orders the candidate visits by graph hops, drives with the
        frozen-map executor, takes level-camera screenshots. Zero LLM calls in transit.
  VERIFIER (VLM, per arrival)      answers one scoped question from one image: "is the target
        visible here?" It is told what to look for, but NOT what the index claims is on this
        shelf - a verifier that knows the expected answer is a rubber stamp, not a check.

Backends (--backend):
  claude-cli  (default) `claude -p` - same billing path as annotate_pass: the claude.ai
              subscription, NOT API credits. Do not switch silently (CLAUDE.md).
  qwen        the OpenAI API compatible endpoint (Chat Completions at $OPENAI_API_URL:8000/v1);
              model id from $SARI_MODEL. Credentials come from the repo-root config.env.

Outputs land in --run-dir: every screenshot, every verifier verdict, locate_report.json.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Repo-root config.env (agent/ -> repo root is two parents up), resolved from __file__ so the
# qwen backend's endpoint creds load when this is run standalone (agent.py loads its own copy when
# it imports this module for graph-nav, so this is a harmless no-op in that path).
load_dotenv(Path(__file__).resolve().parent.parent.parent / "config.env")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OVERHAUL_DIR = os.path.dirname(_THIS_DIR)  # agent/ — packages + slamtest live under here
_SLAM_DIR = os.path.join(_OVERHAUL_DIR, "slamtest")
for _p in (_OVERHAUL_DIR, _SLAM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: F401,E402  (slamtest category dirs -> flat imports keep working)
import _bootstrap  # noqa: F401,E402 - slamtest category dirs onto sys.path (flat-import contract)

from nav.store_map import StoreMap, NavSession  # noqa: E402
from agent_core.models import agent_model  # noqa: E402
from annotate_claude_cli import _strip_unsupported, ClaudeCliError  # noqa: E402

RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "target_name": {"type": "string",
                        "description": "Canonical name of what the task wants, as the store would label it"},
        "target_appearance": {"type": "string",
                              "description": "What to look for visually, specific enough to tell the target from lookalikes (colors, label text, packaging)"},
        "candidates": {"type": "array", "items": {"type": "integer"},
                       "description": "Checkpoint ids that might hold the target. Empty if unresolvable."},
        "tier": {"type": "string", "enum": ["name", "attribute", "category", "unresolvable"],
                 "description": "name = matched index rows; attribute = property-based task, rows marked by world knowledge; category = routed by category; unresolvable = task is nonsensical for a store"},
        "reasoning": {"type": "string"},
    },
    "required": ["target_name", "target_appearance", "candidates", "tier", "reasoning"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "target_visible": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "where": {"type": ["string", "null"],
                  "description": "Where in the frame the target is (e.g. 'right fridge, second shelf, third bottle'), null if not visible"},
        "evidence": {"type": "string",
                     "description": "What in the image supports the verdict - legible text, colors, packaging"},
        "seen_instead": {"type": "array", "items": {"type": "string"},
                         "description": "Closely related products that ARE visible (helps the caller decide whether to keep looking)"},
    },
    "required": ["target_visible", "confidence", "where", "evidence", "seen_instead"],
}

# MEASURED 2026-07-23 (A/B, 12 eval tasks, qwen, resolve-only - slamtest/plans/ResolverResolved.md
# + slamtest/output/resolver_ab/): the previous two-tier prompt STARVED 4/12 tasks (empty or
# singleton candidates) because most eval tasks are attribute-shaped ("with stevia", "imported",
# "< 30 pesos") and neither the name nor category tier fires for a property. Starvation is what
# strands the graph arm: with 0-1 candidates, navigation mode has nowhere new to go and the VLM
# actor falls back to manual body-moves (pickup run 0722_160423, stuck at cp48 for 12 steps).
# This ATTRIBUTE-tier version measured 1/12 starved - and that one (Bingo Corned Beef -> [50]) is
# a correct singleton - with no regression on any name/category task. Generosity is the point:
# a wrong candidate costs one wasted visit, and the on-site verifier is the ground truth.
RESOLVER_SYS = (
    "You are the task resolver for a store agent. You receive a task and the store's product "
    "index - every product a mapping pass observed, with the checkpoint id where it was seen. "
    "Checkpoints are standing positions in front of shelves; the agent will drive to your "
    "candidates and visually verify, so a wrong candidate costs one wasted visit, not a wrong "
    "outcome.\n\n"
    "Rules:\n"
    "1. NAME TIER first: find index rows that plausibly ARE the target. Use world knowledge for "
    "brand/nickname matching (e.g. 'Coke' is Coca-Cola). The index was written by a vision model "
    "that often records only the brand - a row naming the brand without the exact variant is "
    "still a candidate, and telling variants apart is the on-site verifier's job, not yours.\n"
    "2. ATTRIBUTE TIER when the task names a PROPERTY rather than a product - an ingredient "
    "('with stevia'), weight, price, origin ('imported'), packaging ('in a wrapper'), size, or "
    "location ('near the checkout'). Use world knowledge to mark EVERY index row that plausibly "
    "satisfies the property and return all their checkpoints. Prefer breadth: 4-8 candidates, "
    "most-plausible first. Being unsure which items satisfy the property is NORMAL here - the "
    "on-site verifier is the ground truth, your job is only to pick which shelves are worth a "
    "look. Do not return fewer than 3 candidates on this tier unless the index genuinely offers "
    "fewer plausible rows.\n"
    "3. CATEGORY TIER only if neither of the above applies: pick checkpoints whose category "
    "should hold the target. The category->checkpoint map is provided.\n"
    "4. UNRESOLVABLE only if the task is nonsensical for a grocery store (e.g. 'find the "
    "forklift'). Uncertainty about an attribute or a name is NOT unresolvable - guess broadly "
    "instead; empty candidates strand the agent.\n"
    "5. candidates: every checkpoint id whose rows (or category) support the target, "
    "most-plausible first. Do NOT order by store position - you do not know the store's "
    "geometry, and routing is not your job.\n"
    "6. target_appearance: describe what distinguishes the target from its nearest lookalikes "
    "on a shelf (for Coke Zero vs regular Coca-Cola: black label/cap vs red)."
)

VERIFY_CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "category_present": {"type": "boolean",
                             "description": "Does this shelf hold the requested KIND of product?"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "where": {"type": ["string", "null"],
                  "description": "Where the matching products are in the frame, null if absent"},
        "evidence": {"type": "string",
                     "description": "Which visible products establish (or rule out) the category"},
        "examples_seen": {"type": "array", "items": {"type": "string"},
                          "description": "Identifiable products of the requested category that are visible"},
    },
    "required": ["category_present", "confidence", "where", "evidence", "examples_seen"],
}

VERIFIER_CATEGORY_SYS = (
    "You verify whether a store shelf holds a certain KIND of product (a category - 'chips', "
    "'canned goods', 'soda'), from up to two views taken at the same spot: STANDING, and "
    "CROUCHED (same shelf, lower rows seen face-on). Answer STRICTLY from the images.\n\n"
    "Rules:\n"
    "1. category_present=true only if you can actually see products of that kind - this is a "
    "judgment about the SHELF, not about any single product, so a shelf half-full of the "
    "category counts. A single ambiguous item does not.\n"
    "2. Never guess through glare or unreadable packaging; unreadable means not confirmed.\n"
    "3. examples_seen: name the identifiable products that establish the category. An empty "
    "list with category_present=true is a contradiction - do not produce it.\n"
    "4. This store's 3-D assets do not always match real-world packaging colors - trust "
    "legible label TEXT and product shape over expected color schemes."
)

VERIFIER_SYS = (
    "You verify whether a specific product is visible on the store shelf in front of you. You "
    "get up to two views from the SAME spot: STANDING, and CROUCHED (same shelf, lower rows "
    "seen face-on - bottom rows are dim and oblique in the standing view, so check the crouched "
    "view for them). Answer STRICTLY from the images; the target counts as visible if it is "
    "identifiable in EITHER view.\n\n"
    "Rules:\n"
    "1. target_visible=true only if you can actually see it - legible label, or unambiguous "
    "trade-dress (shape + colors). If the packaging colors match but you cannot confirm the "
    "variant, that is target_visible=false with the lookalike listed in seen_instead.\n"
    "2. Never guess. Glass doors, glare and distance make labels unreadable; unreadable means "
    "not confirmed.\n"
    "3. seen_instead: list close relatives of the target that ARE identifiable (same brand, "
    "other variant; same category, other brand). This is used to decide whether looking harder "
    "here is worthwhile, so it matters as much as the verdict.\n"
    "4. where: precise enough that someone could walk up and point at it.\n"
    "5. This store's 3-D assets do not always match real-world packaging colors - trust legible "
    "label TEXT over an expected color scheme when they disagree."
)


# ---------------------------------------------------------------------------- backends

def _meter(model, body):
    """Reports one raw-HTTP completion to the token meter.

    This backend posts with ``requests``, so agent_core.token_meter's OpenAI-SDK patch never sees
    it - which means the advisor's and the resolver's tokens were missing from every run's totals
    until this call existed, not merely unattributed. The import is lazy and swallowed: slamtest
    tools import this module on sys.paths where agent_core is not present, and accounting is never
    a reason to fail a call that already succeeded.
    """
    try:
        from agent_core import token_meter
        token_meter.record_external(model, (body or {}).get("usage"))
    except Exception:  # noqa: BLE001
        pass


def claude_json(system, prompt, schema, image_paths=(), model="sonnet", effort="medium",
                timeout=240.0):
    """One `claude -p` call -> (dict, envelope). Images ride via absolute paths in the prompt +
    the Read tool (annotate_claude_cli's measured setup, same flags, same reasons).

    The prompt goes in via STDIN, not argv. MEASURED 2026-07-19: the reconciled index pushed
    the resolver prompt to ~44k chars and Windows' ~32k CreateProcess argv limit killed every
    call with WinError 206 ("filename or extension is too long" - the error names argv, not a
    file). The qwen backend was immune (HTTP POST), which is exactly the kind of asymmetric
    backend failure that would have skewed an A/B silently if the error were less loud."""
    claude = shutil.which("claude") or "claude"
    lines = [prompt]
    for label, path in image_paths:
        lines.append(f"{label} image: {os.path.abspath(path)}")
    cmd = [
        claude, "-p",
        "--system-prompt", system,
        "--tools", "Read",
        "--permission-mode", "acceptEdits",
        "--output-format", "json",
        "--json-schema", json.dumps(_strip_unsupported(schema)),
        "--model", model,
        "--effort", effort,
        "--no-session-persistence",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                          input="\n".join(lines).encode("utf-8"))
    stdout = proc.stdout.decode("utf-8", errors="replace")
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        raise ClaudeCliError("claude -p output was not JSON", stdout=stdout,
                             stderr=proc.stderr.decode("utf-8", errors="replace"),
                             returncode=proc.returncode)
    if envelope.get("is_error") or proc.returncode != 0:
        raise ClaudeCliError(f"claude -p error: {envelope.get('result')!r}", envelope=envelope)
    result = envelope.get("structured_output")
    if result is None:
        result = json.loads(envelope.get("result", ""))
    return result, envelope


def qwen_json(system, prompt, schema, image_paths=(), model=None, timeout=180.0,
              base_url=None, api_key=None):
    """One Chat Completions call against the configured endpoint -> (dict, envelope-ish).
    `model` defaults to $SARI_MODEL. MEASURED (see CLAUDE.md): vLLM has silently ignored
    guided_json before - the schema is stated in the prompt AND parsed defensively, never
    trusted to be enforced server-side. The endpoint requires a bearer key
    (verified 2026-07-19: /v1/models returns 401 without it) - $OPENAI_API_KEY, alongside
    $OPENAI_API_URL in the repo-root config.env."""
    import requests
    model = model or agent_model()
    host = base_url or os.environ.get("OPENAI_API_URL")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not host:
        raise RuntimeError("qwen backend needs --base-url or $OPENAI_API_URL "
                           "(set it in the repo-root config.env)")
    if not host.startswith("http"):
        host = f"http://{host}"
    if ":8000" not in host:
        host = f"{host}:8000"
    url = f"{host}/v1/chat/completions"

    content = [{"type": "text", "text": prompt + "\n\nReply with ONLY a JSON object matching:\n"
                + json.dumps(schema)}]
    for label, path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = requests.post(url, json=payload, timeout=timeout, headers=headers)
    resp.raise_for_status()
    body = resp.json()
    _meter(model, body)
    text = body["choices"][0]["message"]["content"]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError(f"qwen reply had no JSON object: {text[:200]!r}")
    return json.loads(text[start:end + 1]), body


def make_backend(args):
    if args.backend == "claude-cli":
        return lambda system, prompt, schema, images=(): claude_json(
            system, prompt, schema, images, model=args.model_name or "sonnet",
            effort=args.effort)
    return lambda system, prompt, schema, images=(): qwen_json(
        system, prompt, schema, images, model=args.model_name or agent_model(),
        base_url=args.base_url)


# ---------------------------------------------------------------------------- the task

def resolve(call, sm, task):
    cats = sm.categories()
    prompt = (
        f"## TASK\n{task}\n\n"
        f"## CATEGORY -> CHECKPOINTS\n"
        + "\n".join(f"{c}: {sorted(ids)}" for c, ids in sorted(cats.items()))
        + f"\n\n## PRODUCT INDEX ({len(sm.products)} observations)\n{sm.index_text()}"
    )
    result, envelope = call(RESOLVER_SYS, prompt, RESOLVE_SCHEMA)
    return result, envelope


def zoom_tiles(image_path, out_dir, grid=(2, 2), overlap=0.15, scale=2):
    """Cut a frame into overlapping tiles and upscale each - digital close-in. This is the
    'tiling' answer to the measured refrigerated-label failure: the variant word on a bottle is
    a few pixels after the vision encoder's downscale, and magnifying the crop is the only move
    that adds effective resolution without fighting the executor's 0.65 m safety margin."""
    from PIL import Image
    im = Image.open(image_path)
    w, h = im.size
    base = os.path.splitext(os.path.basename(image_path))[0]
    cols, rows = grid
    tw, th = w / cols, h / rows
    ox, oy = tw * overlap, th * overlap
    paths = []
    for r in range(rows):
        for c in range(cols):
            box = (max(0, int(c * tw - ox)), max(0, int(r * th - oy)),
                   min(w, int((c + 1) * tw + ox)), min(h, int((r + 1) * th + oy)))
            tile = im.crop(box)
            tile = tile.resize((tile.width * scale, tile.height * scale), Image.LANCZOS)
            p = os.path.join(out_dir, f"{base}_tile{r}{c}.png")
            tile.save(p)
            paths.append(p)
    return paths


def near_miss(target_name, seen_instead):
    """Did the verifier see a same-brand/same-family lookalike? Token overlap, deterministic -
    'Coca-Cola Light' vs 'Coca-Cola (regular, red label)' shares coca/cola. Triggers the zoom
    pass only when looking harder at THIS shelf is actually promising."""
    toks = {t for t in target_name.lower().replace("-", " ").split() if len(t) > 2}
    for s in seen_instead:
        stoks = set(s.lower().replace("-", " ").split())
        if toks & stoks:
            return True
    return False


def order_candidates(sm, candidates, from_cp):
    """Valid shelf checkpoints only, nearest-by-graph first. Spatial ordering is code's job."""
    valid = [c for c in dict.fromkeys(candidates) if c in sm.by_id
             and c in sm.shelf_checkpoints()]
    return sorted(valid, key=lambda c: (sm.hops(from_cp, c) is None,
                                        sm.hops(from_cp, c) or 0))


def verify(call, task, resolution, cp_info, views):
    """views: [(label, path)] - STANDING and (usually) CROUCHED from the same spot.

    The resolver's tier picks the QUESTION: an item-shaped task asks "is this product
    visible?"; a category-shaped task ("find a shelf with chips") asks "does this shelf hold
    this kind?" - different verdict, different evidence fields, so a different prompt+schema
    rather than one prompt doing both badly. Category verdicts are normalised to the item
    schema's field names so the caller's loop handles both identically.

    The attribute tier (added 2026-07-23, see RESOLVER_SYS) routes to the CATEGORY question:
    "a product with stevia" / "an imported drink" is a KIND of product, not a specific one,
    so "does this shelf hold products like this" is the answerable question. Reasoned, not
    measured - the eval graph arm never calls verify() (arm B has no verifier), so this path
    only runs in standalone locate_task sessions; A/B it there before tuning further."""
    if resolution.get("tier") in ("category", "attribute"):
        prompt = (
            f"## LOOKING FOR SHELVES HOLDING\n{resolution['target_name']}\n"
            f"## WHAT THAT KIND LOOKS LIKE\n{resolution['target_appearance']}\n"
            f"## ORIGINAL TASK\n{task}\n"
            "Judge only from the attached images."
        )
        v, env = call(VERIFIER_CATEGORY_SYS, prompt, VERIFY_CATEGORY_SCHEMA, views)
        return ({"target_visible": v["category_present"], "confidence": v["confidence"],
                 "where": v["where"], "evidence": v["evidence"],
                 "seen_instead": v["examples_seen"]}, env)
    prompt = (
        f"## LOOKING FOR\n{resolution['target_name']}\n"
        f"## EXPECTED APPEARANCE\n{resolution['target_appearance']}\n"
        f"## ORIGINAL TASK\n{task}\n"
        "Judge only from the attached images."
    )
    # Deliberately NOT telling the verifier what the index says is on this shelf - a verifier
    # that knows the expected answer is a rubber stamp.
    return call(VERIFIER_SYS, prompt, VERIFY_SCHEMA, views)


def main():
    p = argparse.ArgumentParser(description="Locate a product in the store via the frozen map.")
    p.add_argument("task", help='e.g. "Locate the Coke Zero in the store."')
    p.add_argument("--backend", choices=["claude-cli", "qwen"], default="claude-cli")
    p.add_argument("--model-name", default=None)
    p.add_argument("--effort", default="medium")
    p.add_argument("--base-url", default=None, help="qwen backend host (default $OPENAI_API_URL)")
    p.add_argument("--max-visits", type=int, default=4)
    p.add_argument("--uri", default=None, help="sim websocket (default from executor args)")
    p.add_argument("--run-dir", default=None,
                   help="default: slamtest/output/locate_runs/<timestamp>")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve only - no sim, no driving, no verification")
    args = p.parse_args()

    run_dir = args.run_dir or os.path.join(
        _SLAM_DIR, "output", "locate_runs", datetime.now().strftime("%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    call = make_backend(args)
    sm = StoreMap()
    t0 = time.time()

    print(f"[locate] task: {args.task}")
    print(f"[locate] backend: {args.backend}  run dir: {run_dir}")

    # -- resolve (LLM, once) --
    resolution, r_env = resolve(call, sm, args.task)
    print(f"[locate] resolved: {resolution['target_name']!r} tier={resolution['tier']} "
          f"candidates={resolution['candidates']}")
    print(f"[locate]   appearance: {resolution['target_appearance']}")
    report = {"task": args.task, "backend": args.backend, "resolution": resolution,
              "visits": [], "outcome": None}

    if resolution["tier"] == "unresolvable" or not resolution["candidates"]:
        report["outcome"] = "unresolvable"
        print("[locate] UNRESOLVABLE - neither name nor category tier applies. Honest exit.")
        _write(report, run_dir, t0)
        return

    if args.dry_run:
        report["outcome"] = "dry_run"
        _write(report, run_dir, t0)
        return

    # -- drive + verify --
    nav = NavSession(sm, uri=args.uri)
    try:
        drive_and_verify(call, sm, nav, args.task, resolution, report, run_dir,
                         max_visits=args.max_visits)
    finally:
        nav.close()
    _write(report, run_dir, t0)


def drive_and_verify(call, sm, nav, task, resolution, report, run_dir, max_visits=4,
                     zoom=True, tag=""):
    """The visit loop, callable with a live NavSession so a trial harness can run many tasks
    through ONE session (the agent keeps driving from wherever the last task left it - which is
    also the realistic condition). Mutates and returns `report`. `tag` prefixes screenshot
    names so tasks in a shared run_dir don't overwrite each other's frames."""
    x, z, yaw, start_cp = nav.where()
    print(f"[locate] agent at ({x:.2f},{z:.2f}) yaw={yaw:.0f}, nearest cp{start_cp}")
    ordered = order_candidates(sm, resolution["candidates"], start_cp)[:max_visits]
    print(f"[locate] visit order (graph-nearest first): {ordered}")

    for cp_id in ordered:
        info = sm.checkpoint(cp_id)
        print(f"[locate] -> cp{cp_id} {info['holds']} ...")
        if not nav.goto(cp_id):
            print(f"[locate]    could not reach cp{cp_id} - executor refused; skipping")
            report["visits"].append({"checkpoint": cp_id, "reached": False})
            continue
        views = [
            ("STANDING", nav.screenshot(
                os.path.join(run_dir, f"{tag}visit_cp{cp_id:03d}.png"))),
            ("CROUCHED", nav.screenshot(
                os.path.join(run_dir, f"{tag}visit_cp{cp_id:03d}_crouch.png"), crouched=True)),
        ]
        verdict, v_env = verify(call, task, resolution, info, views)
        visit = {"checkpoint": cp_id, "reached": True,
                 "screenshots": [os.path.basename(p) for _, p in views],
                 "verdict": verdict}
        vis = verdict["target_visible"]
        print(f"[locate]    visible={vis} ({verdict['confidence']}) - {verdict['evidence'][:100]}")
        if verdict["seen_instead"]:
            print(f"[locate]    seen instead: {', '.join(verdict['seen_instead'][:5])}")

        # Same-brand lookalike seen but variant unconfirmed -> the variant word is probably
        # just below legibility at this distance. Re-verify on magnified tiles of the SAME
        # frames before moving on (digital close-in; no sim motion). NAME tier only: a
        # category/attribute verdict ("no chips / nothing stevia-like on this shelf") is about
        # the shelf's kind, which magnification cannot change - and near_miss's token overlap
        # is meaningless against attribute target names like "Imported Drink" anyway.
        if (not vis and zoom and resolution.get("tier") not in ("category", "attribute")
                and near_miss(resolution["target_name"], verdict["seen_instead"])):
            print(f"[locate]    near miss (same brand seen) - zooming in ...")
            tiles = []
            for label, p in views:
                tiles += [(f"{label} ZOOMED CROP", t) for t in zoom_tiles(p, run_dir)]
            z_prompt_note = ("These are overlapping MAGNIFIED CROPS of the two views you "
                             "may have seen before, same spot. Variant text is small - "
                             "read labels carefully.")
            verdict2, _ = call(VERIFIER_SYS,
                               f"## LOOKING FOR\n{resolution['target_name']}\n"
                               f"## EXPECTED APPEARANCE\n{resolution['target_appearance']}\n"
                               f"{z_prompt_note}", VERIFY_SCHEMA, tiles)
            visit["zoom_verdict"] = verdict2
            vis = verdict2["target_visible"]
            verdict = verdict2 if vis else verdict
            print(f"[locate]    zoomed: visible={vis} ({verdict2['confidence']}) - "
                  f"{verdict2['evidence'][:100]}")
        report["visits"].append(visit)
        if vis:
            wx, wz = info["world_xz"]
            report["outcome"] = "found"
            report["found"] = {"checkpoint": cp_id, "world_xz": [wx, wz],
                               "where": verdict["where"]}
            print(f"\n[locate] FOUND at checkpoint {cp_id} ({wx:.2f}, {wz:.2f}): "
                  f"{verdict['where']}")
            break
    else:
        report["outcome"] = "not_found"
        print("\n[locate] NOT FOUND at any candidate - reporting honestly "
              "(candidates exhausted, see per-visit verdicts).")
    return report


def _write(report, run_dir, t0):
    report["elapsed_s"] = round(time.time() - t0, 1)
    path = os.path.join(run_dir, "locate_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[locate] report -> {path}  ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
