"""The annotation pass: turn a directory of captured checkpoint images into durable, queryable
map annotations. This is the write side Phase 3 item 8 called for - the piece between "the model
can read a shelf" (the probes) and "we have a queryable map" (this).

Deliberately OFFLINE. capture_walk.py already drove the sim and saved one PNG per checkpoint;
this reads those PNGs and the frozen topology, runs the two-stage VLM annotation, and writes the
results. No sim, no re-walk - so prompts and models can be iterated freely without re-driving the
store, and a good capture set is annotated as many times as it takes to get the prompts right.

Two stages per checkpoint, exactly as annotator_sys_inst.effective_kind() defines:
  * A kind="shelf" checkpoint (geometry says "a shelf might be here") is CLASSIFIED first from its
    primary image - shelf vs. bare wall. ~40% are walls by construction, and a wall annotated as a
    shelf would get "list the products" pointed at nothing. The classifier is the filter.
  * The effective kind (topology kind reconciled with the classifier verdict) picks the Stage-2
    overlay: the shelf overlay enumerates items; the non-shelf overlay describes the spot and is
    forbidden from inventing products.

Three outputs, one per job the design assigns (see plans/phase3.1_semantic_product_layer.md):
  * annotations_<tag>.json - the per-checkpoint sidecar, keyed by checkpoint id. The machine record:
    every checkpoint's effective kind, classifier label, and full annotation. Written incrementally
    after each checkpoint, so an interrupted pass keeps what it finished and --resume continues it.
  * products_<tag>.json - the ONE flat product index. Every item from every shelf, each row tagged
    with its checkpoint_id. This is what answers "find and pick up Pepero" without the agent
    knowing Pepero's category or checking each checkpoint. No dedup, no merge tier - two shelves
    holding Pepero are two rows, and the agent picks the nearest (the flat-index decision).
  * semantic_map_<tag>.txt - a first-cut rendered document in the shape of overhaul's
    semantic_memory.txt: per-checkpoint prose from the VLM, plus connectivity and facing taken from
    the graph (never the VLM - the graph owns spatial truth). The assembler, not a writer.

Backend is annotate_claude_cli (claude -p on the claude.ai / Max-plan login). Swap --model /
--effort to compare; the prompts and schema are the same ones the probes use.

    # annotate every shelf checkpoint that has a capture in captures12/:
    python slamtest/annotate/annotate_pass.py slamtest/output --capture-dir slamtest/output/captures12

    # resume an interrupted pass (skip checkpoints already written):
    python slamtest/annotate/annotate_pass.py slamtest/output --capture-dir slamtest/output/captures12 --resume

    # a quick subset, no Stage-1 classify (treat every shelf-kind as a real shelf):
    python slamtest/annotate/annotate_pass.py slamtest/output --ids 15 67 --skip-classify
"""
import argparse
import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # slamtest/annotate
_SLAM_DIR = os.path.dirname(_THIS_DIR)                         # slamtest
if _SLAM_DIR not in sys.path:
    sys.path.insert(0, _SLAM_DIR)
import _bootstrap  # noqa: F401,E402  (overhaul root + all slamtest category dirs)

from annotator_sys_inst import (  # noqa: E402
    SYS_INST_CLASSIFY, CLASSIFY_SCHEMA, SHELF_KIND,
    build_annotation_instructions, schema_for, effective_kind,
)
from annotate_claude_cli import (  # noqa: E402
    annotate as annotate_claude, ClaudeCliError,
    DEFAULT_MODEL as CLAUDE_DEFAULT_MODEL, DEFAULT_EFFORT,
)
from annotate_qwen import (  # noqa: E402
    annotate as annotate_qwen, QwenAnnotateError, DEFAULT_MODEL as QWEN_DEFAULT_MODEL,
)
from topology import route_hints  # noqa: E402


INTERESTING_KINDS = ("shelf", "landmark")

# Annotator backends. claude-cli is the DEFAULT: annotation quality was measured/frozen on sonnet,
# so the baseline is unchanged unless the user opts in. qwen was un-pinned as a CHOICE 2026-07-20
# (user directive) - selectable for a fully self-hosted run (no claude.ai subscription / no CLI) or
# to A/B annotation quality on identical captures. Both expose the same annotate() signature.
ANNOTATE_BACKENDS = {
    "claude-cli": (annotate_claude, CLAUDE_DEFAULT_MODEL),
    "qwen": (annotate_qwen, QWEN_DEFAULT_MODEL),
}
# One "backend call failed for this checkpoint" signal, whichever backend raised it.
AnnotateError = (ClaudeCliError, QwenAnnotateError)


def add_route_hints(annotations, topology):
    """Annotate each record with what lies THROUGH each of its neighbours.

    Runs after the whole pass, not per-checkpoint, because a hint depends on how every OTHER
    checkpoint was classified - a shelf the classifier rejected as a bare wall is not a destination
    worth routing toward, and that verdict may not exist yet while the pass is still walking.

    This is what makes a non-shelf checkpoint navigable. 11 of this store's 55 checkpoints have no
    interesting neighbour at all, so their adjacency reads as a list of blanks; the hint replaces
    each blank with the nearest real destination through it, or marks it a dead end. Taken from the
    graph, never from an image - a camera at a junction sees a corridor, while the graph knows what
    the corridor reaches and how far."""
    neighbors_by_id = {c["id"]: c.get("neighbors", []) for c in topology["checkpoints"]}
    kind_by_id = {c["id"]: c.get("kind") for c in topology["checkpoints"]}

    def effective(cp_id):
        rec = annotations.get(str(cp_id))
        return rec["effective_kind"] if rec else kind_by_id.get(cp_id)

    interesting = lambda i: effective(i) in INTERESTING_KINDS
    for cid, rec in annotations.items():
        hints = {}
        for via, hit in route_hints(neighbors_by_id, int(cid), interesting).items():
            if hit is None:
                hints[str(via)] = None          # dead end - nothing reachable that way
                continue
            tid, hops = hit
            trec = annotations.get(str(tid))
            hints[str(via)] = {
                "to": tid,
                "hops": hops,
                "kind": effective(tid),
                "holds": (trec or {}).get("annotation", {}).get("shelf_type") or [],
            }
        rec["route_hints"] = hints
    return annotations


def refresh_graph_fields(annotations, topology):
    """Re-copy the graph-owned fields into every record, including resumed ones.

    `neighbors` is cached in the record so Phase 4 can read one checkpoint without loading the whole
    graph - but a cache diverges. Splicing the checkout-counter landmark in gave cp32 a new
    neighbour, and a --resume pass skips the record holding the old list, so the rendered map (which
    reads the graph) said "connects to 3, 6, 54" while the record still said "3, 6". An agent
    trusting the record would not have known the counter was one hop away - exactly the thing the
    landmark exists to prevent.

    So the copy is refreshed here, alongside the other derived outputs, instead of only at
    annotation time. The graph owns connectivity; anything cached from it is rebuilt every run."""
    nb_by_id = {c["id"]: c.get("neighbors", []) for c in topology["checkpoints"]}
    for cid, rec in annotations.items():
        rec["neighbors"] = nb_by_id.get(int(cid), [])
    return annotations


def primary_path(capture_dir, cp_id):
    return os.path.join(capture_dir, f"cp{cp_id:03d}_primary.png")


def view_paths(capture_dir, cp_id):
    """The extra PITCH views of the same shelf: [(label, path)] for whichever of down/up exist.

    These are NOT context images - capture_walk shoots them at the same spot and yaw as the
    primary, just lower, so they show THIS shelf's bottom rows face-on (which a standing frame
    clips at a 1.0m reading distance). Every one is item-bearing; annotate() and the prompt both
    treat them that way. Missing files are simply skipped, so an --angles 1 capture set still
    annotates fine."""
    views = []
    for label, suffix in (("CROUCHED", "crouch"),):
        p = os.path.join(capture_dir, f"cp{cp_id:03d}_{suffix}.png")
        if os.path.exists(p):
            views.append((label, p))
    return views


def select_checkpoints(topology, args):
    cps = topology["checkpoints"]
    if args.kind != "all":
        kinds = {k.strip() for k in args.kind.split(",")}
        cps = [c for c in cps if c.get("kind") in kinds]
    if args.ids:
        wanted = set(args.ids)
        cps = [c for c in cps if c["id"] in wanted]
    cps.sort(key=lambda c: c["id"])
    if args.limit:
        cps = cps[: args.limit]
    return cps


def classify(primary, args, annotate_fn):
    """Stage 1: shelf vs non_shelf from the primary image. Returns the label string, or None if
    the call failed (caller falls back to trusting the geometry)."""
    result, _env = annotate_fn(primary, SYS_INST_CLASSIFY, CLASSIFY_SCHEMA,
                               model=args.model, effort=args.effort, timeout=args.timeout)
    return result.get("label")


def annotate_checkpoint(cp, primary, views, args, annotate_fn):
    """Full two-stage annotation for one checkpoint. Returns the sidecar record dict.

    `annotate_fn` is the selected backend's annotate() (claude-cli or qwen); both share one
    signature so this body is backend-agnostic."""
    topo_kind = cp.get("kind", "?")

    # Stage 1 only runs on shelf-kind checkpoints (the only ones with shelf-vs-wall ambiguity).
    # It uses the STANDING view alone - shelf-vs-wall is obvious at a level frame, and the extra
    # views would only add cost to a binary the classifier already gets right.
    label = None
    if topo_kind == SHELF_KIND and not args.skip_classify:
        label = classify(primary, args, annotate_fn)

    ekind = effective_kind(topo_kind, label)
    system = build_annotation_instructions(ekind)
    schema = schema_for(ekind)

    # The crouch view exists only to read a shelf's bottom rows face-on. The non-shelf overlay
    # describes the place and is forbidden from enumerating products, so a second image would be
    # pure cost there. When Stage 1 rules this out as a shelf (or the node is a landmark, which is
    # non-shelf by construction), annotate from the STANDING primary alone - the crouch is not sent.
    stage2_views = views if ekind == SHELF_KIND else []

    result, env = annotate_fn(primary, system, schema, model=args.model, effort=args.effort,
                              timeout=args.timeout, extra_views=stage2_views)

    return {
        "id": cp["id"],
        "topology_kind": topo_kind,
        "classify_label": label,          # None = classify skipped or not applicable
        "effective_kind": ekind,
        # Straight from the topology, never from the VLM - the graph owns connectivity, and Phase 4
        # needs it here so an agent reading one checkpoint's record can see where it can go next
        # without loading the whole graph.
        "neighbors": cp.get("neighbors", []),
        "backend": args.backend,
        "model": args.model,
        "effort": args.effort if args.backend == "claude-cli" else None,
        "cost_equiv_usd": env.get("total_cost_usd"),
        "views": ["STANDING"] + [lbl for lbl, _ in stage2_views],
        "annotation": result,             # the schema-shaped VLM output
    }


def flatten_products(annotations):
    """Collapse every shelf checkpoint's items into ONE flat list, each row tagged with its
    checkpoint_id. Non-shelf checkpoints contribute nothing (they have no items field). No dedup -
    the same product at two checkpoints is two rows on purpose."""
    products = []
    for rec in annotations.values():
        if rec["effective_kind"] != SHELF_KIND:
            continue
        ann = rec.get("annotation", {})
        shelf_type = ann.get("shelf_type") or []
        for item in ann.get("items", []):
            products.append({
                "name": item.get("name"),
                "variant": item.get("variant"),
                "price": item.get("price"),
                "appearance": item.get("appearance"),
                # category is required per-item now; fall back to the shelf's dominant type, then
                # "other", so a malformed item never lands in the index without a category.
                "category": item.get("category") or (shelf_type[0] if shelf_type else "other"),
                "checkpoint_id": rec["id"],
            })
    return products


def render_semantic_map(annotations, topology):
    """First-cut assembler: the semantic_memory.txt-shaped doc. Per-checkpoint PROSE comes from the
    VLM (semantic_summary); CONNECTIVITY and POSITION come from the graph, never the VLM - the graph
    owns spatial truth. This is the join, not a new annotation."""
    by_id = {c["id"]: c for c in topology["checkpoints"]}
    lines = ["# Semantic map (rendered from annotations + graph)", ""]
    for cp_id in sorted(annotations, key=int):
        rec = annotations[cp_id]
        cp = by_id.get(int(cp_id), {})
        ann = rec.get("annotation", {})
        summary = ann.get("semantic_summary", "").strip()
        wx = cp.get("world_xz")
        pos = f" at ({wx[0]:.2f}, {wx[1]:.2f})" if wx else ""
        header = f"## Checkpoint {cp_id} [{rec['effective_kind']}]{pos}"
        lines.append(header)
        if summary:
            lines.append(summary)
        st = ann.get("shelf_type")
        if st:
            lines.append(f"Holds: {', '.join(st)}.")
        neighbors = cp.get("neighbors")
        if neighbors:
            lines.append(f"Connects to: {', '.join(str(n) for n in neighbors)}.")
        lines.append("")
    return "\n".join(lines)


def run(args):
    # Resolve the backend: its annotate() and its default model. --model overrides the default,
    # so `--backend qwen` without --model uses Qwen/Qwen3.6-27B, not sonnet.
    annotate_fn, default_model = ANNOTATE_BACKENDS[args.backend]
    if not args.model:
        args.model = default_model
    if args.backend == "qwen" and args.base_url:
        # Only the qwen backend accepts base_url; inject it so classify/annotate_checkpoint stay
        # backend-agnostic (they call annotate_fn with the shared signature only).
        _base_fn = annotate_fn
        annotate_fn = lambda *a, **k: _base_fn(*a, base_url=args.base_url, **k)

    topo_path = os.path.join(args.output_dir, f"topology_{args.topology_tag}.json")
    with open(topo_path, encoding="utf-8") as f:
        topology = json.load(f)

    capture_dir = args.capture_dir or os.path.join(args.output_dir, "captures")
    ann_path = os.path.join(args.output_dir, f"annotations_{args.out_tag}.json")
    prod_path = os.path.join(args.output_dir, f"products_{args.out_tag}.json")
    map_path = os.path.join(args.output_dir, f"semantic_map_{args.out_tag}.txt")

    annotations = {}
    if args.resume and os.path.exists(ann_path):
        with open(ann_path, encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"[annotate_pass] resuming - {len(annotations)} checkpoint(s) already annotated")

    targets = select_checkpoints(topology, args)
    eff = args.effort if args.backend == "claude-cli" else "n/a"
    print(f"[annotate_pass] {len(targets)} target checkpoint(s); captures from {capture_dir}; "
          f"backend={args.backend} model={args.model} effort={eff}")

    done = skipped = failed = 0
    for cp in targets:
        cid = str(cp["id"])
        if args.resume and cid in annotations:
            skipped += 1
            continue
        primary = primary_path(capture_dir, cp["id"])
        if not os.path.exists(primary):
            print(f"[annotate_pass] id={cp['id']:3d} no capture ({os.path.basename(primary)}) - skipping")
            skipped += 1
            continue

        views = view_paths(capture_dir, cp["id"])
        try:
            rec = annotate_checkpoint(cp, primary, views, args, annotate_fn)
        except AnnotateError as e:
            print(f"[annotate_pass] id={cp['id']:3d} FAILED: {e}")
            failed += 1
            continue

        annotations[cid] = rec
        # Write after every checkpoint so an interrupted pass keeps its progress and --resume works.
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        ann = rec["annotation"]
        if rec["effective_kind"] == SHELF_KIND:
            detail = f"{ann.get('shelf_type')} {len(ann.get('items', []))} item(s)"
        else:
            detail = "non-shelf"
        lbl = rec["classify_label"]
        print(f"[annotate_pass] id={cp['id']:3d} kind={rec['topology_kind']:8s} "
              f"classify={lbl or '-':9s} -> {rec['effective_kind']:9s} {detail} "
              f"(${rec['cost_equiv_usd'] or 0:.3f})")
        done += 1

    # Derived outputs - rebuilt in full each run from the (possibly resumed) annotations.
    refresh_graph_fields(annotations, topology)
    add_route_hints(annotations, topology)
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    products = flatten_products(annotations)
    with open(prod_path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(render_semantic_map(annotations, topology))

    total_cost = sum(r.get("cost_equiv_usd") or 0 for r in annotations.values())
    cost_note = ("subscription-covered on Max" if args.backend == "claude-cli"
                 else "self-hosted qwen - no per-call billing")
    print(f"[annotate_pass] done: {done} annotated, {skipped} skipped, {failed} failed")
    print(f"[annotate_pass] {len(annotations)} checkpoints, {len(products)} products, "
          f"~${total_cost:.2f} equiv ({cost_note})")
    print(f"[annotate_pass]   {ann_path}")
    print(f"[annotate_pass]   {prod_path}")
    print(f"[annotate_pass]   {map_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Annotate captured checkpoints and write durable map outputs.")
    p.add_argument("output_dir", help="Directory holding topology_<tag>.json; outputs written here")
    p.add_argument("--capture-dir", default=None, help="Where the cp<id>_primary.png live (default: <output_dir>/captures)")
    p.add_argument("--topology-tag", default="final_shelf")
    p.add_argument("--out-tag", default="final_shelf", help="Suffix for the output files")
    p.add_argument("--kind", default="shelf,landmark",
                   help="Comma-separated kinds to annotate (default: shelf,landmark; 'all' for every kind). "
                        "A landmark gets the non-shelf overlay, which describes the place instead of "
                        "enumerating products - right for a checkout counter.")
    p.add_argument("--ids", type=int, nargs="*", default=None, help="Only these checkpoint ids")
    p.add_argument("--limit", type=int, default=0, help="At most this many checkpoints (0 = all)")
    p.add_argument("--resume", action="store_true", help="Skip checkpoints already in annotations_<tag>.json")
    p.add_argument("--skip-classify", action="store_true", help="Skip Stage 1; treat every shelf-kind checkpoint as a real shelf")
    p.add_argument("--backend", choices=list(ANNOTATE_BACKENDS), default="claude-cli",
                   help="Annotation model backend. claude-cli (default): `claude -p` sonnet, the "
                        "measured/frozen quality baseline. qwen: the self-hosted vLLM server - no "
                        "claude.ai subscription or `claude` CLI needed. Un-pinned 2026-07-20.")
    p.add_argument("--model", default=None,
                   help="Model or alias. Default depends on --backend: 'sonnet' for claude-cli, "
                        f"'{QWEN_DEFAULT_MODEL}' for qwen.")
    p.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"],
                   help="claude-cli only; ignored by the qwen backend.")
    p.add_argument("--base-url", default=None, help="qwen backend host (default $OPENAI_API_URL:8000/v1)")
    p.add_argument("--timeout", type=float, default=240.0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
