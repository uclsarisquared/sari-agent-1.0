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
    python slamtest/annotate_pass.py slamtest/output --capture-dir slamtest/output/captures12

    # resume an interrupted pass (skip checkpoints already written):
    python slamtest/annotate_pass.py slamtest/output --capture-dir slamtest/output/captures12 --resume

    # a quick subset, no Stage-1 classify (treat every shelf-kind as a real shelf):
    python slamtest/annotate_pass.py slamtest/output --ids 15 67 --skip-classify
"""
import argparse
import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from annotator_sys_inst import (  # noqa: E402
    SYS_INST_CLASSIFY, CLASSIFY_SCHEMA, SHELF_KIND,
    build_annotation_instructions, schema_for, effective_kind,
)
from annotate_claude_cli import annotate, ClaudeCliError, DEFAULT_MODEL, DEFAULT_EFFORT  # noqa: E402


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
        cps = [c for c in cps if c.get("kind") == args.kind]
    if args.ids:
        wanted = set(args.ids)
        cps = [c for c in cps if c["id"] in wanted]
    cps.sort(key=lambda c: c["id"])
    if args.limit:
        cps = cps[: args.limit]
    return cps


def classify(primary, args):
    """Stage 1: shelf vs non_shelf from the primary image. Returns the label string, or None if
    the call failed (caller falls back to trusting the geometry)."""
    result, _env = annotate(primary, SYS_INST_CLASSIFY, CLASSIFY_SCHEMA,
                            model=args.model, effort=args.effort, timeout=args.timeout)
    return result.get("label")


def annotate_checkpoint(cp, primary, views, args):
    """Full two-stage annotation for one checkpoint. Returns the sidecar record dict."""
    topo_kind = cp.get("kind", "?")

    # Stage 1 only runs on shelf-kind checkpoints (the only ones with shelf-vs-wall ambiguity).
    # It uses the STRAIGHT view alone - shelf-vs-wall is obvious at a level frame, and the extra
    # views would only add cost to a binary the classifier already gets right.
    label = None
    if topo_kind == SHELF_KIND and not args.skip_classify:
        label = classify(primary, args)

    ekind = effective_kind(topo_kind, label)
    system = build_annotation_instructions(ekind)
    schema = schema_for(ekind)

    result, env = annotate(primary, system, schema, model=args.model, effort=args.effort,
                           timeout=args.timeout, extra_views=views)

    return {
        "id": cp["id"],
        "topology_kind": topo_kind,
        "classify_label": label,          # None = classify skipped or not applicable
        "effective_kind": ekind,
        # Straight from the topology, never from the VLM - the graph owns connectivity, and Phase 4
        # needs it here so an agent reading one checkpoint's record can see where it can go next
        # without loading the whole graph.
        "neighbors": cp.get("neighbors", []),
        "model": args.model,
        "effort": args.effort,
        "cost_equiv_usd": env.get("total_cost_usd"),
        "views": ["STRAIGHT"] + [lbl for lbl, _ in views],
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
    print(f"[annotate_pass] {len(targets)} target checkpoint(s); captures from {capture_dir}; "
          f"model={args.model} effort={args.effort}")

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
            rec = annotate_checkpoint(cp, primary, views, args)
        except ClaudeCliError as e:
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
    products = flatten_products(annotations)
    with open(prod_path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(render_semantic_map(annotations, topology))

    total_cost = sum(r.get("cost_equiv_usd") or 0 for r in annotations.values())
    print(f"[annotate_pass] done: {done} annotated, {skipped} skipped, {failed} failed")
    print(f"[annotate_pass] {len(annotations)} checkpoints, {len(products)} products, "
          f"~${total_cost:.2f} equiv (subscription-covered on Max)")
    print(f"[annotate_pass]   {ann_path}")
    print(f"[annotate_pass]   {prod_path}")
    print(f"[annotate_pass]   {map_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Annotate captured checkpoints and write durable map outputs.")
    p.add_argument("output_dir", help="Directory holding topology_<tag>.json; outputs written here")
    p.add_argument("--capture-dir", default=None, help="Where the cp<id>_primary.png live (default: <output_dir>/captures)")
    p.add_argument("--topology-tag", default="final_shelf")
    p.add_argument("--out-tag", default="final_shelf", help="Suffix for the output files")
    p.add_argument("--kind", default="shelf", help="Which checkpoint kind to annotate (default: shelf; 'all' for every kind)")
    p.add_argument("--ids", type=int, nargs="*", default=None, help="Only these checkpoint ids")
    p.add_argument("--limit", type=int, default=0, help="At most this many checkpoints (0 = all)")
    p.add_argument("--resume", action="store_true", help="Skip checkpoints already in annotations_<tag>.json")
    p.add_argument("--skip-classify", action="store_true", help="Skip Stage 1; treat every shelf-kind checkpoint as a real shelf")
    p.add_argument("--model", default=DEFAULT_MODEL, help="claude model or alias (default: sonnet)")
    p.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--timeout", type=float, default=240.0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
