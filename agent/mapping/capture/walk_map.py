"""Interactive terminal walk of the annotated checkpoint graph - the human-driven version of what
a Phase-4 agent will do.

You stand at a checkpoint; it tells you what is in front of you and where you can go; you pick a
neighbour and it drives there. That is the whole loop, and it is the cheapest way to find out
whether the map is actually usable as a map: whether the summaries describe the thing you are
looking at, whether the connectivity matches the store you are standing in, and whether "go to the
shelf next door" resolves to the shelf next door.

Movement reuses capture_walk's goto() - the same A* + swept-clearance executor explore.py uses -
so what you can reach here is exactly what the annotator could reach.

    python mapping/capture/walk_map.py mapping/output
    python mapping/capture/walk_map.py mapping/output --annotations-tag final_shelf --start 50

Needs the sim in Play mode. Reads a frozen map; never re-maps and never re-annotates.

ON STALE ANNOTATIONS: annotations are keyed by checkpoint id, and an id only means anything inside
the topology it was generated from. Rebuilding the shelf graph (a different interval, a different
reading distance) RENUMBERS every checkpoint, so yesterday's annotations silently describe today's
other shelf. This module refuses to join them quietly - see check_alignment().
"""
import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # mapping/capture
_MAPPING_DIR = os.path.dirname(_THIS_DIR)                         # mapping
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

from sim.env import SetHandsActive  # noqa: E402
from explore import step_agent  # noqa: E402
from frontier_planner import _inflate_occupied  # noqa: E402
from topology import route_hints  # noqa: E402
import capture_walk  # noqa: E402
from capture_walk import load_grid, perpendicular_yaw, goto, face  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def load_annotations(path):
    """Returns (annotations_by_id_str, note). Missing file is fine - the graph is still walkable,
    you just get no summaries. Falls back to cp1252 for files written before annotate_pass pinned
    its encoding, which on Windows silently wrote non-ASCII product names in the locale codec."""
    if not os.path.exists(path):
        return {}, f"no annotations at {os.path.basename(path)} - walking the bare graph"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except UnicodeDecodeError:
        with open(path, encoding="cp1252") as f:
            return json.load(f), "annotations read as cp1252 (written before the encoding fix)"


def check_alignment(topology, annotations):
    """Do these annotations actually belong to this topology? Returns a list of complaints.

    Checkpoint ids are positional, not stable: they are assigned while sweeping the skeleton, so
    changing the checkpoint interval or the reading distance renumbers everything. An annotation
    joined to the wrong graph doesn't error - it describes a different shelf, convincingly. These
    are the cheap tells."""
    by_id = {c["id"] for c in topology["checkpoints"]}
    problems = []

    orphans = sorted(int(i) for i in annotations if int(i) not in by_id)
    if orphans:
        problems.append(
            f"{len(orphans)} annotated checkpoint(s) do not exist in this topology "
            f"(ids {orphans[:6]}{'...' if len(orphans) > 6 else ''}) - the annotations were "
            f"generated from a DIFFERENT graph, so every id here means something else now")

    # Newer annotations record the neighbours they were written against; if those disagree with the
    # graph, the id has been reused for another checkpoint.
    topo_neighbors = {c["id"]: sorted(c.get("neighbors", [])) for c in topology["checkpoints"]}
    drifted = [int(i) for i, rec in annotations.items()
               if rec.get("neighbors") is not None and int(i) in topo_neighbors
               and sorted(rec["neighbors"]) != topo_neighbors[int(i)]]
    if drifted:
        problems.append(f"{len(drifted)} checkpoint(s) have annotation neighbours that disagree "
                        f"with the graph (ids {sorted(drifted)[:6]}) - stale annotations")
    return problems


def summary_of(rec):
    if not rec:
        return None
    return (rec.get("annotation", {}).get("semantic_summary") or "").strip() or None


def snippet(rec, width=44):
    s = summary_of(rec)
    if not s:
        kind = (rec or {}).get("effective_kind")
        return "(not a shelf)" if kind and kind != "shelf" else "(not annotated)"
    # ASCII "..." rather than U+2026: this prints to a Windows console whose codepage is usually
    # cp1252, where a real ellipsis renders as a replacement char.
    return s if len(s) <= width else s[: width - 3] + "..."


def wrap(text, width=72, indent="   "):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(indent + line)
    return "\n".join(out)


def effective_kind_of(cp_id, by_id, annotations):
    """What a checkpoint IS, after the classifier's verdict. Falls back to the topology kind for
    structural nodes, which are never annotated."""
    rec = annotations.get(str(cp_id))
    if rec:
        return rec.get("effective_kind")
    return by_id[cp_id].get("kind") if cp_id in by_id else None


def show(cp, rec, by_id, annotations):
    """The arrival report: what am I looking at, and where can I go from here."""
    wx = cp.get("world_xz") or (0, 0)
    print("\n" + RULE)
    print(f" CHECKPOINT {cp['id']}   [{cp.get('kind')}]   world ({wx[0]:.2f}, {wx[1]:.2f})")
    print(THIN)

    ann = (rec or {}).get("annotation", {})
    if rec and rec.get("effective_kind") == "shelf":
        st = ann.get("shelf_type")
        if st:
            print(f" Holds   : {', '.join(st)}")
        items = ann.get("items", [])
        print(f" Items   : {len(items)}" + ("   ('i' to list)" if items else ""))
    elif rec:
        print(f" Kind    : {rec.get('effective_kind')} (no products here)")

    s = summary_of(rec)
    print(" Summary :")
    print(wrap(s, indent="   ") if s else "   (not annotated)")

    print(THIN)
    print(" Adjacent:")
    neighbors_by_id = {c["id"]: c.get("neighbors", []) for c in by_id.values()}
    interesting = lambda i: effective_kind_of(i, by_id, annotations) in ("shelf", "landmark")
    hints = route_hints(neighbors_by_id, cp["id"], interesting)
    for nid in cp.get("neighbors", []):
        n = by_id.get(nid)
        if n is None:
            print(f"   [{nid:>3}]  (not in topology)")
            continue
        nwx = n.get("world_xz") or (0, 0)
        d = math.hypot(nwx[0] - wx[0], nwx[1] - wx[1])
        nrec = annotations.get(str(nid))
        if interesting(nid):
            detail = snippet(nrec)
        else:
            # Nothing to see AT this neighbour - so say what lies THROUGH it instead of the
            # useless "(not annotated)". This is graph truth, not a guess from an image.
            hit = hints.get(nid)
            if hit is None:
                detail = "-> dead end"
            else:
                tid, hops = hit
                trec = annotations.get(str(tid))
                what = (trec or {}).get("annotation", {}).get("shelf_type") or []
                kind = effective_kind_of(tid, by_id, annotations)
                label = ", ".join(what) if what else (kind or "?")
                detail = f"-> {label} ({hops} hop{'s' if hops != 1 else ''}, cp{tid})"
        print(f"   [{nid:>3}]  {n.get('kind','?'):<9} {d:4.1f}m   {detail}")
    print(RULE)


def list_items(rec):
    items = (rec or {}).get("annotation", {}).get("items", [])
    if not items:
        print("   (no items)")
        return
    for it in items:
        bits = [it.get("name", "?")]
        if it.get("variant"):
            bits.append(f"({it['variant']})")
        if it.get("price"):
            bits.append(f"@{it['price']}")
        print(f"   - {' '.join(bits):<46} {it.get('category','')}")


def nearest_checkpoint(cps, pos):
    return min(cps, key=lambda c: math.hypot(c["world_xz"][0] - pos[0], c["world_xz"][1] - pos[2]))


def arrive(args, grid, inflated, cp, pos, rot):
    """Drive to a checkpoint and face its main orientation. Returns (pos, rot, ok)."""
    pos, rot, ok = goto(args, grid, inflated, tuple(cp["world_xz"]), pos, rot)
    if not ok:
        return pos, rot, False
    # The "main orientation" is only defined for shelf checkpoints: it's the perpendicular Phase 2
    # already computed (cell -> shelf_cell). A junction faces nothing in particular.
    if cp.get("shelf_cell"):
        pos, rot = face(args, pos, rot, perpendicular_yaw(cp))
    return pos, rot, True


def build_parser():
    # Reuse capture_walk's parser so the executor knobs (step size, safety margin, body radius,
    # nudge/escape) stay identical to the ones the annotator and explore.py drive with - copying
    # them here would let the two drift apart silently.
    p = capture_walk.build_parser()
    p.description = "Interactively walk the annotated checkpoint graph from the terminal."
    p.add_argument("--annotations-tag", default="final_shelf",
                   help="Loads annotations_<tag>.json for the summaries (default: final_shelf)")
    p.add_argument("--start", type=int, default=None,
                   help="Checkpoint to start at (default: whichever is nearest the agent now)")
    return p


def main():
    args = build_parser().parse_args()
    grid = load_grid(args.output_dir, args.grid_tag, args.resolution)
    with open(os.path.join(args.output_dir, f"topology_{args.topology_tag}.json"),
              encoding="utf-8") as f:
        topology = json.load(f)
    annotations, note = load_annotations(
        os.path.join(args.output_dir, f"annotations_{args.annotations_tag}.json"))

    if note:
        print(f"[walk_map] note: {note}")
    for problem in check_alignment(topology, annotations):
        print(f"[walk_map] *** STALE ANNOTATIONS: {problem}")
    if check_alignment(topology, annotations):
        print("[walk_map] *** Summaries below may describe a DIFFERENT shelf than the one you are\n"
              "[walk_map] *** standing at. Re-run capture_walk + annotate_pass against "
              f"topology_{args.topology_tag} to fix.\n")

    cps = [c for c in topology["checkpoints"] if c.get("world_xz")]
    by_id = {c["id"]: c for c in topology["checkpoints"]}
    inflated = _inflate_occupied(grid, args.body_radius)

    pos, rot, _ = step_agent((0, 0, 0), (0, 0, 0), args.uri)
    if args.stow_hands:
        SetHandsActive(False, uri=args.uri)

    try:
        cur = by_id.get(args.start) if args.start is not None else nearest_checkpoint(cps, pos)
        if cur is None:
            print(f"[walk_map] no checkpoint {args.start} in topology_{args.topology_tag}")
            return
        print(f"[walk_map] walking to checkpoint {cur['id']}...")
        pos, rot, ok = arrive(args, grid, inflated, cur, pos, rot)
        if not ok:
            print(f"[walk_map] could not reach checkpoint {cur['id']} - try another --start")
            return

        while True:
            rec = annotations.get(str(cur["id"]))
            show(cur, rec, by_id, annotations)
            choice = input(" move to id | i=items | r=re-orient | q=quit > ").strip().lower()

            if choice in ("q", "quit", "exit"):
                break
            if choice == "i":
                list_items(rec)
                input(" (enter to continue) ")
                continue
            if choice == "r":
                pos, rot, _ = arrive(args, grid, inflated, cur, pos, rot)
                continue
            if not choice:
                continue

            try:
                target = int(choice)
            except ValueError:
                print(" ? enter one of the adjacent ids, or i / r / q")
                continue
            if target not in cur.get("neighbors", []):
                print(f" ? {target} is not adjacent to {cur['id']}. "
                      f"Adjacent: {cur.get('neighbors', [])}")
                continue

            nxt = by_id[target]
            print(f"[walk_map] -> {target} ...")
            pos, rot, ok = arrive(args, grid, inflated, nxt, pos, rot)
            if not ok:
                # Adjacency is a graph edge; reachability is the executor's call. They can disagree
                # (see audit_standability) - so say so and stay put rather than pretending.
                print(f"[walk_map] could not reach {target} - the edge exists but the executor "
                      f"would not go. Staying at {cur['id']}.")
                pos, rot, _ = arrive(args, grid, inflated, cur, pos, rot)
                continue
            cur = nxt
    except (KeyboardInterrupt, EOFError):
        print("\n[walk_map] interrupted")
    finally:
        if args.stow_hands:
            SetHandsActive(True, uri=args.uri)
        print("[walk_map] done")


if __name__ == "__main__":
    main()
