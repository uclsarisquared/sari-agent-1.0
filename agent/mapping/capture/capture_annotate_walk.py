"""Fused capture+annotate walk: annotate each checkpoint WHILE the walk drives to the next.

capture_walk 🎮 and annotate_pass are pipeline stages 4 and 5, normally run back to back. But the
walk blocks on the sim websocket (LiDAR + steps) and the annotator blocks on `claude -p`
subprocesses or HTTP - neither wants the CPU - so running them sequentially wastes the whole
annotate stage's wall clock. This driver fuses them: the walk loop is the producer, annotate_pass's
checkpoint pool is the consumer, and a checkpoint is submitted the moment its shots are on disk.
Total time drops from walk+annotate to roughly walk + the last checkpoint's annotation tail.

Design notes (why it looks the way it does):
  * IN-PROCESS, not a file watcher. Completion is program order (capture_walk's on_captured hook),
    so there is no "is the PNG fully written?" race, no done-sentinel protocol, and pipeline_app
    keeps its one-stage-at-a-time model by treating this as a single stage.
  * The walk thread is the ONLY writer of annotations_<tag>.json. Futures are harvested on the
    walk thread - opportunistically after each checkpoint, then a blocking drain when the walk
    ends - preserving annotate_pass's single-writer incremental-write semantics without locks.
  * The fan-in (route hints, product index, semantic map) still runs ONCE at the end: a route hint
    depends on how every other checkpoint classified, so it cannot stream. It runs in a finally,
    so an interrupted walk still leaves consistent (partial) outputs.
  * The OFFLINE annotate_pass stays the primary prompt-iteration tool (saved PNGs, no sim); this
    is the fast path for full-pipeline runs. Same flags, same output files, and an interrupted
    fused run is finished by `annotate_pass --resume` - or by re-running fused with --resume,
    which also skips the WALK for already-annotated checkpoints.

    # full store, standing+crouch, annotating as it walks:
    python mapping/capture/capture_annotate_walk.py mapping/output --limit 0 --angles 2

    # resume an interrupted fused run (skips finished checkpoints on both sides):
    python mapping/capture/capture_annotate_walk.py mapping/output --limit 0 --angles 2 --resume
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/capture
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

import capture_walk  # noqa: E402
from annotate_pass import (  # noqa: E402
    AnnotateError, DEFAULT_JOBS, add_annotator_args, annotate_checkpoint,
    resolve_annotate_fn, write_derived_outputs,
)
from annotator_sys_inst import SHELF_KIND  # noqa: E402


def build_parser():
    # Capture owns navigation AND selection (--kind/--ids/--limit pick what gets WALKED; whatever
    # gets captured gets annotated). The annotator contributes only its own flags - the two sets
    # are disjoint by construction (see add_annotator_args's docstring).
    p = capture_walk.build_parser()
    p.description = ("Fused stages 4+5: walk the checkpoints and annotate each one while driving "
                     "to the next. Needs the sim in Play mode.")
    return add_annotator_args(p)


def run(args):
    annotate_fn = resolve_annotate_fn(args)
    with open(os.path.join(args.output_dir, f"topology_{args.topology_tag}.json"),
              encoding="utf-8") as f:
        topology = json.load(f)

    ann_path = os.path.join(args.output_dir, f"annotations_{args.out_tag}.json")
    prod_path = os.path.join(args.output_dir, f"products_{args.out_tag}.json")
    map_path = os.path.join(args.output_dir, f"semantic_map_{args.out_tag}.txt")

    annotations = {}
    if args.resume and os.path.exists(ann_path):
        with open(ann_path, encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"[fused] resuming - {len(annotations)} checkpoint(s) already annotated "
              f"(skipped by the walk too)")

    jobs = args.jobs or DEFAULT_JOBS[args.backend]
    eff = args.effort if args.backend == "claude-cli" else "n/a"
    print(f"[fused] annotating DURING the walk: backend={args.backend} model={args.model} "
          f"effort={eff} jobs={jobs}")

    pool = ThreadPoolExecutor(max_workers=jobs)
    pending = {}   # future -> checkpoint dict
    stats = {"done": 0, "failed": 0}

    def collect(fut):
        """Fold one finished future into the annotations dict + incremental JSON. Walk thread only."""
        cp = pending.pop(fut)
        try:
            rec = fut.result()
        except AnnotateError as e:
            print(f"[fused] id={cp['id']:3d} annotate FAILED: {e}")
            stats["failed"] += 1
            return
        annotations[str(cp["id"])] = rec
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        ann = rec["annotation"]
        detail = (f"{ann.get('shelf_type')} {len(ann.get('items', []))} item(s)"
                  if rec["effective_kind"] == SHELF_KIND else "non-shelf")
        print(f"[fused] id={cp['id']:3d} annotated -> {rec['effective_kind']:9s} {detail} "
              f"(${rec['cost_equiv_usd'] or 0:.3f})")
        stats["done"] += 1

    def on_captured(cp, shots):
        # shots is capture_plan order: primary first, then crouch when --angles 2. Building views
        # from what was JUST written (not re-globbing the dir) keeps stale files from older runs
        # out of this run's annotations.
        views = [("CROUCHED", shots[1])] if len(shots) > 1 else []
        pending[pool.submit(annotate_checkpoint, cp, shots[0], views, args, annotate_fn)] = cp
        # Opportunistic harvest at walk cadence: keeps the incremental JSON current (bounded
        # staleness of ~one checkpoint's drive time) without any cross-thread writes.
        for fut in [f for f in pending if f.done()]:
            collect(fut)

    skip = {int(cid) for cid in annotations} if args.resume else ()
    try:
        capture_walk.run(args, on_captured=on_captured, skip_ids=skip)
    finally:
        # Drain + fan-in even if the walk died mid-run (sim hiccup, Ctrl-C): every annotation
        # already paid for lands in the JSON, and the derived outputs stay consistent with it.
        # A Ctrl-C here still waits on in-flight backend calls (bounded by --timeout).
        if pending:
            print(f"[fused] walk done - waiting on {len(pending)} in-flight annotation(s)")
        for fut in as_completed(list(pending)):
            collect(fut)
        pool.shutdown(wait=True)

        products = write_derived_outputs(
            annotations, topology, ann_path, prod_path, map_path,
            topo_path=os.path.join(args.output_dir, f"topology_{args.topology_tag}.json"),
            output_dir=args.output_dir)
        total_cost = sum(r.get("cost_equiv_usd") or 0 for r in annotations.values())
        print(f"[fused] done: {stats['done']} annotated, {stats['failed']} failed; "
              f"{len(annotations)} checkpoints, {len(products)} products, ~${total_cost:.2f} equiv")
        print(f"[fused]   {ann_path}")
        print(f"[fused]   {prod_path}")
        print(f"[fused]   {map_path}")


def main():
    args = build_parser().parse_args()
    if args.capture_dir is None:
        args.capture_dir = os.path.join(args.output_dir, "captures")
    run(args)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
