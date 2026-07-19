"""Phase 4 library API over the frozen slamtest artifacts.

This is the read side of `phase4_agent_integration.md`: everything an agent needs to answer
"where is product X and how do I stand in front of it" WITHOUT a VLM doing spatial reasoning.
The graph owns spatial truth; this module is how the agent reads the graph.

Reuses, never copies:
  - executor knobs come from `capture_walk.build_parser()` parsed with defaults - the CLI and the
    library CANNOT drift because there is only one definition of the defaults
  - driving is `capture_walk.goto()` + `face()` + `pitch_to()` - the same A* + swept-LiDAR
    executor that produced every capture the annotations were made from
  - staleness refusal is `walk_map.check_alignment()` - but promoted from a warning to a raised
    error, because a library caller has no console to read the warning on and stale annotations
    describe a DIFFERENT shelf, convincingly

Offline half (StoreMap) needs no sim; live half (NavSession) needs Play mode.

    from store_map import StoreMap, NavSession
    sm = StoreMap()
    rows = sm.search("coca")             # deterministic tier; LLM resolver is the primary tier
    nav = NavSession(sm)                 # sim must be in Play mode
    ok = nav.goto(15)                    # drive + face the shelf perpendicular
    png = nav.screenshot("cp15.png")     # level-camera capture of what the agent now faces
    nav.close()
"""
import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SLAM_DIR = os.path.join(_THIS_DIR, "slamtest")
for _p in (_THIS_DIR, _SLAM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_walk  # noqa: E402
from capture_walk import (  # noqa: E402
    load_grid, goto, face, pitch_to, perpendicular_yaw, PITCH_LEVEL_DEG,
)
from frontier_planner import _inflate_occupied  # noqa: E402
from topology import route_hints  # noqa: E402
from explore import step_agent  # noqa: E402
from walk_map import load_annotations, check_alignment, effective_kind_of  # noqa: E402
from env import SetHandsActive, SetCrouch, RequestScreenshot  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(_SLAM_DIR, "output")


class StaleMapError(RuntimeError):
    """Annotations and topology disagree - ids would describe different shelves than they claim."""


def executor_args(output_dir=DEFAULT_OUTPUT_DIR, **overrides):
    """The executor's knobs (step size, safety margin, body radius, nudge/escape...) as an
    argparse Namespace with capture_walk's defaults. Parsing the real parser instead of copying
    values is the anti-drift property: there is exactly one definition of every default."""
    args = capture_walk.build_parser().parse_args([output_dir])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown executor knob: {k}")
        setattr(args, k, v)
    return args


class StoreMap:
    """The frozen artifacts, joined and queryable. Offline - never touches the sim."""

    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR, topology_tag="final_shelf",
                 annotations_tag="final_shelf", grid_tag="final", resolution=0.1,
                 use_reconciled=True):
        self.output_dir = output_dir
        self.grid_tag = grid_tag
        self.resolution = resolution

        with open(os.path.join(output_dir, f"topology_{topology_tag}.json"), encoding="utf-8") as f:
            self.topology = json.load(f)
        self.annotations, note = load_annotations(
            os.path.join(output_dir, f"annotations_{annotations_tag}.json"))
        if note:
            print(f"[store_map] note: {note}")

        problems = check_alignment(self.topology, self.annotations)
        if problems:
            raise StaleMapError(
                "annotations do not belong to this topology:\n  " + "\n  ".join(problems))

        # Prefer the reconciled index when the reconciler has run: same rows, enriched with
        # canonical catalog SKUs (reconcile_products.py). Falling back to the raw file keeps
        # this loader working on a fresh annotation pass that hasn't been reconciled yet.
        reconciled = os.path.join(output_dir, f"products_{annotations_tag}_reconciled.json")
        products_path = reconciled if (use_reconciled and os.path.exists(reconciled)) \
            else os.path.join(output_dir, f"products_{annotations_tag}.json")
        with open(products_path, encoding="utf-8") as f:
            self.products = json.load(f)
        self.reconciled = products_path == reconciled

        self.by_id = {c["id"]: c for c in self.topology["checkpoints"]}
        self._neighbors = {c["id"]: c.get("neighbors", []) for c in self.topology["checkpoints"]}

    # ---- checkpoint queries -------------------------------------------------

    def checkpoint(self, cp_id):
        """One checkpoint, graph + annotation joined. Raises KeyError on unknown id."""
        cp = self.by_id[cp_id]
        rec = self.annotations.get(str(cp_id), {})
        ann = rec.get("annotation", {}) or {}
        return {
            "id": cp_id,
            "kind": cp.get("kind"),
            "effective_kind": effective_kind_of(cp_id, self.by_id, self.annotations),
            "world_xz": tuple(cp.get("world_xz") or ()),
            "neighbors": list(cp.get("neighbors", [])),
            "summary": (ann.get("semantic_summary") or "").strip() or None,
            "holds": ann.get("shelf_type") or [],
            "items": ann.get("items") or [],
            "route_hints": rec.get("route_hints") or {},
        }

    def shelf_checkpoints(self):
        return [i for i in self.by_id
                if effective_kind_of(i, self.by_id, self.annotations) == "shelf"]

    def category_checkpoints(self, category):
        """Shelf checkpoints whose shelf_type includes `category` - the fallback routing tier."""
        out = []
        for i in self.shelf_checkpoints():
            if category in self.checkpoint(i)["holds"]:
                out.append(i)
        return out

    def counter_checkpoint(self):
        """The landmark node (cp54, the checkout counter). Topology-kind fallback on purpose -
        the landmark has no annotation yet (open thread: its capture)."""
        for c in self.topology["checkpoints"]:
            if c.get("kind") == "landmark":
                return c["id"]
        return None

    def nearest_checkpoint(self, pos_xz):
        """Nearest checkpoint to a world (x, z). The whole localizer - the map and the sim share
        one absolute frame (verified against Store 2 v2.json, see phase4_agent_integration.md)."""
        return min(
            (c for c in self.topology["checkpoints"] if c.get("world_xz")),
            key=lambda c: math.hypot(c["world_xz"][0] - pos_xz[0], c["world_xz"][1] - pos_xz[1]),
        )["id"]

    def hops(self, a, b):
        """BFS hop count between checkpoints, or None if disconnected. Used to order candidate
        visits by graph distance - a spatial judgment, so it is code's job, never the resolver
        LLM's (the graph owns spatial truth)."""
        if a == b:
            return 0
        from collections import deque
        seen, q = {a}, deque([(a, 0)])
        while q:
            cur, d = q.popleft()
            for n in self._neighbors.get(cur, []):
                if n == b:
                    return d + 1
                if n not in seen:
                    seen.add(n)
                    q.append((n, d + 1))
        return None

    def hints_from(self, cp_id):
        """Live route hints (graph-computed, covers base nodes the JSON has no records for)."""
        interesting = lambda i: effective_kind_of(i, self.by_id, self.annotations) in ("shelf", "landmark")
        return route_hints(self._neighbors, cp_id, interesting)

    # ---- product queries ----------------------------------------------------

    def search(self, text):
        """Deterministic name tier: case-insensitive substring/token overlap. This is NOT the
        primary resolver - 'Coke Zero' does not match 'Coca-Cola' by tokens, and per Phase 3.1 the
        semantic matching intelligence lives in the LLM consumer. This tier exists for tests and
        for exact hits that shouldn't cost an LLM call."""
        t = text.lower().strip()
        toks = set(t.replace("-", " ").split())
        rows = []
        for r in self.products:
            name = r["name"].lower()
            ntoks = set(name.replace("-", " ").split())
            if t in name or name in t or (toks & ntoks):
                rows.append(r)
        return rows

    def index_text(self):
        """The whole product index as compact lines for an LLM resolver prompt. 290 rows ~ 15k
        tokens as raw JSON; this halves it and keeps every field the resolver needs.

        When the reconciled index is loaded, each snapped row also carries its canonical
        catalog id ('= RITZ_ORIGINAL_3PACK') or variant family ('~ COCACOLA_{4 variants}') -
        the resolver can then match a task's formal name against the canonical id even when
        the VLM-read name is a misread ('Ritz Riginal')."""
        lines = []
        for r in sorted(self.products, key=lambda r: (r["category"], r["name"])):
            bits = [f"cp{r['checkpoint_id']}", r["category"], r["name"]]
            if r.get("sku"):
                bits.append(f"= {r['sku']}")
            elif r.get("sku_candidates"):
                bits.append(f"~ one of {len(r['sku_candidates'])}: "
                            + "|".join(r["sku_candidates"][:4]))
            if r.get("variant"):
                bits.append(f"variant={r['variant']}")
            if r.get("appearance"):
                bits.append(f"looks: {r['appearance']}")
            lines.append(" | ".join(bits))
        return "\n".join(lines)

    def categories(self):
        cats = {}
        for i in self.shelf_checkpoints():
            for c in self.checkpoint(i)["holds"]:
                cats.setdefault(c, []).append(i)
        return cats


class NavSession:
    """The live half: drives the real agent along the frozen map. Needs Play mode.

    Owns the executor state (grid, inflated mask, live pose) and the hands/pitch hygiene that
    capture_walk learned the hard way: hands stowed while walking, camera re-levelled before any
    screenshot (the agent has been found sitting at 16 deg pitch from manual sessions)."""

    def __init__(self, store_map: StoreMap, uri=None, stow_hands=True, **knob_overrides):
        self.sm = store_map
        self.args = executor_args(store_map.output_dir, **knob_overrides)
        if uri:
            self.args.uri = uri
        self.grid = load_grid(store_map.output_dir, store_map.grid_tag, store_map.resolution)
        self.inflated = _inflate_occupied(self.grid, self.args.body_radius)
        self.pos, self.rot, _ = step_agent((0, 0, 0), (0, 0, 0), self.args.uri)
        self._stowed = False
        if stow_hands:
            SetHandsActive(False, uri=self.args.uri)
            self._stowed = True

    def where(self):
        """(world_x, world_z, yaw_deg) and the nearest checkpoint id."""
        return (self.pos[0], self.pos[2], self.rot[1],
                self.sm.nearest_checkpoint((self.pos[0], self.pos[2])))

    def goto(self, cp_id, face_shelf=True):
        """Drive to a checkpoint; at shelf/landmark nodes, face the perpendicular Phase 2
        computed. Returns True iff arrived - a False is an executor refusal, not an exception,
        because adjacency is a graph edge and reachability is the executor's call."""
        cp = self.sm.by_id[cp_id]
        target = tuple(cp["world_xz"])
        self.pos, self.rot, ok = goto(self.args, self.grid, self.inflated,
                                      target, self.pos, self.rot)
        if not ok:
            return False
        if face_shelf and cp.get("shelf_cell"):
            self.pos, self.rot = face(self.args, self.pos, self.rot, perpendicular_yaw(cp))
        return True

    def screenshot(self, out_path, crouched=False):
        """Capture what the agent faces right now, camera forced level first.

        crouched=True halves the view height for the shot and ALWAYS stands back up before
        returning - walking while crouched breaks the LiDAR clearance gate (a crouched scan
        reports the floor as an obstacle; capture_walk documents this as a safety property).
        Crouch is the measured answer to bottom shelf rows: rows are numbered bottom-up in the
        store JSON, and row 1 - second from the floor - is dim and oblique in a standing frame
        but face-on and legible crouched (verified live on shelf 7's lone Coca-Cola Light)."""
        self.pos, self.rot = pitch_to(self.args, self.rot, PITCH_LEVEL_DEG)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        try:
            if crouched:
                SetCrouch(True, uri=self.args.uri)
            result = RequestScreenshot(save_image=False, uri=self.args.uri)
            with open(out_path, "wb") as f:
                f.write(result["image"])
        finally:
            if crouched:
                SetCrouch(False, uri=self.args.uri)
        return out_path

    def close(self):
        if self._stowed:
            SetHandsActive(True, uri=self.args.uri)
            self._stowed = False
