"""Phase 4 library API over the frozen mapping artifacts.

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

    from nav.store_map import StoreMap, NavSession
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
_AGENT_DIR = os.path.dirname(_THIS_DIR)  # agent/ — packages + mapping live under here
_MAPPING_DIR = os.path.join(_AGENT_DIR, "mapping")
for _p in (_AGENT_DIR, _MAPPING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: F401,E402 - mapping category dirs onto sys.path (flat-import contract)

import capture_walk  # noqa: E402
from capture_walk import (  # noqa: E402
    load_grid, goto, face, pitch_to, perpendicular_yaw, PITCH_LEVEL_DEG,
)
from frontier_planner import _inflate_occupied  # noqa: E402
from topology import route_hints  # noqa: E402
from explore import step_agent  # noqa: E402
from walk_map import load_annotations, check_alignment, effective_kind_of  # noqa: E402
from sim.env import (  # noqa: E402
    SetHandsActive, SetCrouch, RequestScreenshot, RequestLidarCenter,
    move_forward, move_backward, move_left, move_right,
)
from mapping import normalize_deg  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(_MAPPING_DIR, "output")


def default_output_dir():
    """The map dir a caller who names none gets: $SARI_MAP_DIR, else the frozen mapping/output.

    The env var is how a HARNESS points a whole process tree at a captured map dir without every
    call site having to thread `--output-dir`: Sari Bench sets it per attempt (sari_bench/runner
    `_spawn_agent`), and it reaches the StoreMap() calls scattered across probes, gates and evals
    that take no dir argument at all. Resolved per call, not bound at import, so setting the var
    early in main() also works (same contract as SARI_WS_URI in sim/env.py).
    """
    return os.environ.get("SARI_MAP_DIR") or DEFAULT_OUTPUT_DIR


class StaleMapError(RuntimeError):
    """Annotations and topology disagree - ids would describe different shelves than they claim."""


def executor_args(output_dir=None, **overrides):
    """The executor's knobs (step size, safety margin, body radius, nudge/escape...) as an
    argparse Namespace with capture_walk's defaults. Parsing the real parser instead of copying
    values is the anti-drift property: there is exactly one definition of every default."""
    args = capture_walk.build_parser().parse_args([output_dir or default_output_dir()])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown executor knob: {k}")
        setattr(args, k, v)
    return args


class StoreMap:
    """The frozen artifacts, joined and queryable. Offline - never touches the sim."""

    def __init__(self, output_dir=None, topology_tag="final_shelf",
                 annotations_tag="final_shelf", grid_tag="final", resolution=0.1,
                 use_reconciled=True):
        output_dir = output_dir or default_output_dir()
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

    def hop_path(self, a, b):
        """BFS shortest checkpoint path a -> b INCLUSIVE of both ends, or None if disconnected.
        The per-hop companion to hops(): the graph-advised navigator (agent._advised_goto) needs
        the route's NEXT HOP as advice each step, and path[1] is exactly that. Same BFS as hops()
        with parent tracking - a spatial judgment, so it is code's job, never a VLM's."""
        if a == b:
            return [a]
        from collections import deque
        prev, q = {a: None}, deque([a])
        while q:
            cur = q.popleft()
            for n in self._neighbors.get(cur, []):
                if n in prev:
                    continue
                prev[n] = cur
                if n == b:
                    path = [b]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(n)
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
        computed, then re-level the camera. Returns True iff arrived - a False is an executor
        refusal, not an exception, because adjacency is a graph edge and reachability is the
        executor's call."""
        cp = self.sm.by_id[cp_id]
        target = tuple(cp["world_xz"])
        self.pos, self.rot, ok = goto(self.args, self.grid, self.inflated,
                                      target, self.pos, self.rot)
        if not ok:
            return False
        if face_shelf and cp.get("shelf_cell"):
            self.pos, self.rot = face(self.args, self.pos, self.rot, perpendicular_yaw(cp))
        # Re-level the camera on arrival: a move-to-checkpoint should leave the agent looking
        # STRAIGHT ahead, not tilted up/down from a prior crouch/grab/manual pitch (agents have
        # been found sitting at 16 deg). This matters most for the graph-nav dispatcher
        # (agent._graph_navigate), which grabs its post-arrival frame with a raw RequestScreenshot
        # that does NOT pass through screenshot()'s levelling. Absolute pitch_to, so it never drifts.
        self.pos, self.rot = pitch_to(self.args, self.rot, PITCH_LEVEL_DEG)
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


# ===== Phase 6.2: deterministic "go to the checkout counter" =====================================
def go_to_counter(nav, resync=True):
    """Drive a live NavSession to the checkout counter (the cp54 landmark) and face it - the
    fixed-target navigation mirror of perception.center_to_counter, and the 'go to the counter'
    primitive the place subtask needs. The counter is a KNOWN singleton landmark, so this needs NO
    resolver LLM: it looks cp54 up directly (StoreMap.counter_checkpoint) and A*-drives there.

    Deterministic-over-topology by design (phase6 plan, "cp54 standoff"): prefer this approach step
    (code, regenerable, A/B-able) over re-docking cp54 in the frozen topology (a data migration with
    route-hint/annotation fallout).

    resync (default True) re-reads NavSession's cached pose from the sim first - the phase4.2 desync
    landmine: VLM/manipulation/manual-WASD actions move the agent behind NavSession's back, so the
    A* start would otherwise be stale. Set False only if the caller just synced.

    CARRY-SAFETY IS THE CALLER'S JOB (as in agent._graph_navigate): assert the REST hand pose before
    calling this so a held item rides the drive. NavSession must have been built stow_hands=False
    (the default stows hands, which drops a carried item and undoes Phase 6.1). This function does not
    touch the hands - it must not bypass the agent's state-tracked _set_hand_pose.

    Returns {'arrived': bool, 'checkpoint': id|None, 'x', 'z', 'yaw'} (or a 'reason' if there is no
    landmark node in the graph at all)."""
    cp = nav.sm.counter_checkpoint()
    if cp is None:
        return {"arrived": False, "checkpoint": None,
                "reason": "no landmark/counter checkpoint in the graph"}
    if resync:
        nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
    arrived = nav.goto(cp)
    x, z, yaw, _near = nav.where()
    return {"arrived": bool(arrived), "checkpoint": cp, "x": x, "z": z, "yaw": yaw}


# ===== Phase 6.2 (restructured): square onto the barcode scan pad (the "last metre") =============
# The `aligned` verdict is on the LATERAL offset to the pad, not the raw yaw angle. Measured 2026-07-23
# from 2 live checkout runs: the sweep scanned at 0.033 m (yaw 2.2 deg) and 0.047 m (yaw 3.2 deg) lateral.
# The lateral strafe moves in 0.10 m steps and rounds, so it cannot converge finer than 0.05 m - which is
# also above both measured-OK offsets. So 0.05 m ties the verdict to the strafe's own resolution.
ALIGN_LATERAL_TOL_M = 0.05


def align_to_scanner(nav, target_slant=0.85, max_lateral_iters=3, max_advance_iters=6, resync=True,
                     debug_dir=None):
    """Square the agent onto the checkout scan pad from the counter checkpoint - the visual-servo
    'last metre' the occupancy grid is too coarse to place (0.1 m cells vs a centimetre-scale pad).
    Phase 6.2, promoted from place_probe's `align` + `adv`, which centred + aligned well enough to
    scan on both live chain runs (2026-07-22).

    Deterministic-over-topology by design (the frozen-map-preferred Option A): all existing
    primitives - center_to_scanner (perception) + strafe/advance (env) + the graph's perpendicular -
    so this is regenerable and A/B-able rather than a re-dock of cp54 in the frozen topology. Splice a
    scan-dock node only if this measures flaky (the same trigger the phase6 plan set for cp54).

    PRECONDITION: at/near the counter (call go_to_counter first) and CARRYING (hands at REST, active).
    Like go_to_counter, this does NOT touch the hands - the caller keeps the item riding at REST.

    Two loops:
      1. Lateral: center_to_scanner -> yaw error vs cp54's perpendicular (from the graph) -> strafe
         slant*sin(yaw_off) -> re-face the perpendicular -> re-centre. <= max_lateral_iters; converged
         at <= one pan step (2.5 deg) of residual yaw.
      2. Advance: close the pad's centre-LiDAR slant to `target_slant` (SCAN_REACH_M, measured
         comfortable at 0.84-0.89 m), re-centring the scanner after each move.

    Returns {'aligned': bool, 'slant': float|None, 'residual_yaw': float, 'checkpoint': id|None,
    'iters': int, 'reason': str} - surfaced as `last_align`. `aligned` iff the pad was detected, the
    yaw residual is within one pan step, AND the final slant <= target_slant + a small margin (the
    item is within the hand's reach of the pad). An honest False tells the caller to re-approach or
    fall back, never a silent miss."""
    from vision.perception import center_to_scanner   # lazy: perception pulls the OpenAI client (live-only)

    cp_id = nav.sm.counter_checkpoint()
    if cp_id is None:
        return {"aligned": False, "slant": None, "residual_yaw": 0.0, "checkpoint": None,
                "iters": 0, "reason": "no landmark/counter checkpoint in the graph"}
    if resync:
        nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
    perp = perpendicular_yaw(nav.sm.by_id[cp_id])

    detected = False
    yaw_off = 0.0
    iters = 0
    # ---- Loop 1: lateral alignment --------------------------------------------------------------
    for iters in range(1, max_lateral_iters + 1):
        res = center_to_scanner(debug_dir=debug_dir)
        if res.get("outcome") == "detection_error":
            return {"aligned": False, "slant": None, "residual_yaw": yaw_off,
                    "checkpoint": cp_id, "iters": iters,
                    "reason": "scan-pad detector returned malformed output after retries; "
                    "visibility is unknown (not a not-detected result)"}
        if res.get("outcome") == "not_detected" or res.get("box") is None:
            return {"aligned": False, "slant": None, "residual_yaw": yaw_off, "checkpoint": cp_id,
                    "iters": iters, "reason": "scan pad not detected - re-approach the counter or "
                    "pan it into view (center_to_scanner failed)"}
        detected = True
        nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
        yaw_off = normalize_deg(nav.rot[1] - perp)
        s = RequestLidarCenter()
        if not s.get("hit"):
            return {"aligned": False, "slant": None, "residual_yaw": yaw_off, "checkpoint": cp_id,
                    "iters": iters, "reason": "no LiDAR hit at the pad centre after centring"}
        lateral = float(s["distance"]) * math.sin(math.radians(yaw_off))
        steps = round(abs(lateral) / 0.10)
        if abs(yaw_off) <= 2.5 or steps == 0:
            break                                  # converged
        nav.pos, nav.rot = face(nav.args, nav.pos, nav.rot, perp)
        (move_right if lateral > 0 else move_left)(steps)

    # ---- Loop 2: advance to the scan-reach slant ------------------------------------------------
    slant = None
    for _ in range(max_advance_iters):
        s = RequestLidarCenter()
        if not s.get("hit"):
            center_to_scanner(debug_dir=debug_dir)   # re-centre and re-read
            continue
        slant = float(s["distance"])
        theta = math.radians(float(s["pitch_deg"]))
        vertical = slant * math.sin(theta)
        gap = slant * math.cos(theta)
        if target_slant <= vertical:
            break                                  # target is below the vertical drop - as close as moving gets
        needed = math.sqrt(target_slant ** 2 - vertical ** 2)
        steps = round((gap - needed) / 0.10)
        if steps == 0:
            break                                  # at the scan-reach slant
        (move_forward if steps > 0 else move_backward)(min(abs(steps), 10))
        center_to_scanner(debug_dir=debug_dir)     # a move de-centres the pad

    # Final read for the honest verdict - re-measure BOTH the slant and the yaw/lateral from the ACTUAL
    # final pose (the advance loop may have drifted laterally). The verdict is on the LATERAL offset,
    # not the raw yaw angle: what the scan needs is the item transiting the pad, which is a lateral
    # distance, and at close range a few degrees of yaw is only centimetres. CALIBRATED from 2 live
    # runs (2026-07-23): the sweep SCANNED at lateral 0.033 m (yaw -2.2 deg) AND 0.047 m (yaw +3.2 deg);
    # the old `abs(yaw) <= 2.5` verdict wrongly failed the 3.2 deg run even though it scanned. The
    # strafe cannot converge finer than ~0.05 m anyway (half a 0.10 m step - round(x/0.10)==0 <=> x<0.05),
    # so ALIGN_LATERAL_TOL_M ties the verdict to the strafe's own resolution AND the measured-OK bound.
    nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
    yaw_off = normalize_deg(nav.rot[1] - perp)
    s = RequestLidarCenter()
    slant = float(s["distance"]) if s.get("hit") else slant
    lateral = abs(slant * math.sin(math.radians(yaw_off))) if slant is not None else None
    within_reach = slant is not None and slant <= target_slant + 0.05
    lateral_ok = lateral is not None and lateral <= ALIGN_LATERAL_TOL_M
    aligned = bool(detected and lateral_ok and within_reach)
    if aligned:
        reason = (f"squared on the pad (lateral {lateral * 1000:.0f} mm, residual yaw {yaw_off:+.1f} deg, "
                  f"slant {slant:.2f} m)")
    elif not within_reach:
        reason = (f"aligned laterally but pad still {slant:.2f} m away (> {target_slant} m reach) - "
                  "advance closer, or the dock geometry blocks it (consider a spliced scan dock)")
    else:
        lat_str = "n/a" if lateral is None else f"{lateral * 1000:.0f} mm"
        reason = (f"lateral offset {lat_str} > {ALIGN_LATERAL_TOL_M * 1000:.0f} mm "
                  f"(residual yaw {yaw_off:+.1f} deg) - re-approach the counter, or splice a scan-dock "
                  "node (Option B)")
    print(f"[align_to_scanner] aligned={aligned} lateral={None if lateral is None else round(lateral, 3)} "
          f"residual_yaw={yaw_off:+.1f} slant={slant} iters={iters}")
    return {"aligned": aligned, "slant": slant, "residual_yaw": yaw_off, "lateral": lateral,
            "checkpoint": cp_id, "iters": iters, "reason": reason}


# ===== Phase 6.2 (restructured): the deterministic checkout macro ================================
def checkout_held_item(nav, hand="auto", drive=True, bag_if_unscanned=False, debug_dir=None):
    """Check out the item currently in hand: drive to the checkout (optional), align on the scan pad,
    scan the held item, then bag it in the tray - the ONE deterministic macro, NO LLM inside. This is
    the single tool the 6.3 typed `checkout`/`place` subtask dispatches to; the VLM never sequences the
    mechanical steps (CLAUDE.md doctrine: geometry is deterministic; the VLM judges only what is in
    front of it). Composes the four built + calibrated primitives, validated end-to-end by two live
    smoke runs (2026-07-23):

        [go_to_counter] -> align_to_scanner -> (baseline receipt) -> scan_held_item -> place_held_item

    The POS-screen baseline is read AFTER aligning (only then is the screen close/square enough to OCR -
    smoke finding), then the pad is re-acquired (reading the screen yaws the body to it) so the sweep
    goes the right way; this makes `scanned` an honest receipt DELTA, not an absolute read.

    PRECONDITION: holding an item (refused otherwise). NavSession must be stow_hands=False (carry-safe);
    this sets REST before the drive and each primitive owns its own GRAB/REST around the hand.

    bag_if_unscanned (default False): if the sweep does NOT register a scan, the item is NOT bagged - an
    unscanned item is task-incomplete, and dropping it anyway would hide the failure. The macro returns
    scanned=False, still holding, for the caller to retry / re-align. Set True only to force a bag.

    Returns {success, scanned, placed, aligned, steps{counter,align,scan,place}, reason}. success =
    scanned AND placed - both MEASURED (OCR delta / released-under-a-placeable-verdict). Honest scoring
    only: the human/gate owns `scanned_verified` (beep) and `placed_verified` (in-tray screenshot);
    this macro never promotes measured to verified."""
    from vision.perception import (scan_held_item, place_held_item, center_to_screen,
                            center_to_scanner, read_text_in_box)
    from manip.manipulation import set_hand_pose, resolve_release_hand
    from sim.env import TransformHands

    steps = {"counter": None, "align": None, "scan": None, "place": None}

    def _out(scanned, placed, aligned, reason):
        return {"success": bool(scanned and placed), "scanned": scanned, "placed": placed,
                "aligned": aligned, "steps": steps, "reason": reason, "hand": hand}

    # 'auto' resolves ONCE here to the hand that IS holding (left-first when both hold; name 'right'
    # to check out the other item) and is threaded through the whole chain, so the scan and the place
    # act on the SAME hand even if the other one grabs/releases mid-macro. Subsumes the holding guard.
    hand, refuse_reason = resolve_release_hand(hand)
    if hand is None:
        return _out(False, False, False, refuse_reason + " - nothing to check out")

    # 1. Drive to the checkout (carry-safe: REST first, then go_to_counter resyncs + faces cp54).
    if drive:
        set_hand_pose("rest", hand=hand)
        res = go_to_counter(nav)
        steps["counter"] = res
        if not res.get("arrived"):
            return _out(False, False, False,
                        f"could not reach the counter: {res.get('reason', 'path blocked')}")

    # 2. Align on the scan pad.
    al = align_to_scanner(nav, debug_dir=debug_dir)
    steps["align"] = al
    aligned = bool(al.get("aligned"))

    # 3. Baseline receipt from the ALIGNED pose (screen legible now), then re-acquire the pad so the
    #    sweep goes toward it (center_to_screen yaws the body to the screen; this is rotation-only, so
    #    the slant align set is preserved).
    sres = center_to_screen(debug_dir=debug_dir)
    baseline = read_text_in_box(sres.get("box")) if sres.get("box") else []
    center_to_scanner(debug_dir=debug_dir)

    # 4. Scan the held item (item stays in hand).
    sc = scan_held_item(hand=hand, baseline=baseline, debug_dir=debug_dir)
    steps["scan"] = sc
    scanned = bool(sc.get("scanned"))
    if not sc.get("still_holding"):
        return _out(scanned, False, aligned,
                    "the grip opened during the sweep - item dropped, aborting before bagging")

    # 5. Bag it - but not an UNSCANNED item unless forced (dropping it would hide the scan failure).
    if not scanned and not bag_if_unscanned:
        return _out(False, False, aligned,
                    "item did not scan - not bagging an unscanned item (still holding); "
                    "re-run or check alignment")
    pl = place_held_item(hand=hand, debug_dir=debug_dir)
    steps["place"] = pl
    placed = bool(pl.get("placed"))

    # 6. Back off the counter face once the item is dropped. The agent finishes the macro parked hard
    #    against the checkout, which the graph/A* treats as a wall - the next leg reads that as a
    #    blocked path and can stall. Stepping back 5 units (~0.5 m, body-relative; still facing the
    #    counter) clears the agent off the obstacle so the following leg plans freely.
    if placed:
        from sim.env import move_backward
        move_backward(5)

    reason = ("checked out: scanned and bagged" if (scanned and placed)
              else f"scanned={scanned} placed={placed} - {pl.get('reason', 'see steps')}")
    return _out(scanned, placed, aligned, reason)


def checkout_held_items(nav, drive=True, bag_if_unscanned=False, debug_dir=None):
    """Check out EVERY item currently in hand in ONE fused pass - the both-hands sibling of
    checkout_held_item. Drive + align ONCE, sweep-scan each held hand (verifying each off the POS
    receipt) BEFORE bagging any, then bag each scanned item:

        [go_to_counter] -> align_to_scanner -> baseline receipt
                        -> sweep LEFT  -> read screen  (scanned_left  = receipt delta)
                        -> sweep RIGHT -> read screen  (scanned_right = delta vs the post-left receipt)
                        -> bag LEFT -> bag RIGHT

    ORDER RATIONALE (user, 2026-07-23): both SWEEPS happen before either DROP. A sweep is REVERSIBLE
    (the grip never opens - a held item keeps its Barcode trigger live, so it scans without dropping);
    the bag is the ONE irreversible release. Front-loading both scans means a missed sweep is caught
    while BOTH items are still in hand, so a single re-align retries the pair - versus scan-drop-scan-
    drop (what two back-to-back checkout_held_item calls do), which commits the first drop before the
    second scan is even known. One drive + one align also serves both items instead of re-approaching
    the counter per hand.

    Baseline threading: the receipt ACCUMULATES on the POS screen, so each sweep's `scanned` is a delta
    against the receipt AFTER the previous sweep - scan_held_item returns `receipt`; we thread it as the
    next sweep's baseline (baseline=None only for the first, a fresh checkout). scan_held_item ends by
    yawing the body to the SCREEN to read it, so we center_to_scanner again before the next sweep - the
    same pad<->screen dance checkout_held_item does around its single scan.

    DEGRADES to checkout_held_item when only ONE hand holds (nothing to fuse - that macro already
    drives/aligns/scans/bags a single hand); REFUSES with nothing held. Composes only the four measured
    Phase 6.2 primitives (align/scan/place + the screen reads), so it adds NO new geometry or
    calibration over the single macro - UNMEASURED live only in that two sweeps + two drops now run off
    one stance; validate it with a two-item run before trusting it
    (validation/acceptance/checkout.py and checkout_smoke.py).

    PRECONDITION: holding item(s) (refused otherwise). NavSession must be stow_hands=False (carry-safe);
    each primitive owns its own GRAB/REST around its hand, and this RESTs every held hand before driving.

    bag_if_unscanned (default False): an item whose sweep did not register is NOT bagged (dropping it
    would hide the scan miss) - it stays in hand, and the checkout completion predicate, seeing a hand
    still gripping, sends the agent to re-run checkout. Set True only to force a bag.

    Returns {success, scanned, placed, aligned, steps, reason, hand, per_hand}. To stay compatible with
    predicate_checkout WITHOUT changing it, top-level `scanned`/`placed` are the AND across the held
    hands (ALL scanned / ALL bagged) and `hand` is None - so the predicate's single-hand consistency
    check is skipped and its still-gripping guard (any hand left holding => refuse) does the work.
    `per_hand` carries the per-item {scanned, placed} detail. success = every held item scanned AND
    bagged (both MEASURED - OCR receipt delta / released-under-a-placeable-verdict). Honest scoring
    only: the gate owns scanned_verified (beep) and placed_verified (in-tray screenshot)."""
    from vision.perception import (scan_held_item, place_held_item, center_to_screen,
                            center_to_scanner, read_text_in_box)
    from manip.manipulation import set_hand_pose, hand_grip_states
    from sim.env import move_backward

    steps = {"counter": None, "align": None, "scans": {}, "places": {}}
    hands = [h for h in ("left", "right") if hand_grip_states()[h]]

    def _out(all_scanned, all_placed, aligned, reason, scanned=None, placed=None):
        per = {h: {"scanned": (scanned or {}).get(h), "placed": (placed or {}).get(h)} for h in hands}
        return {"success": bool(all_scanned and all_placed), "scanned": bool(all_scanned),
                "placed": bool(all_placed), "aligned": aligned, "steps": steps, "hand": None,
                "per_hand": per, "reason": reason}

    if not hands:
        return _out(False, False, False, "no hand is holding an item - nothing to check out")
    if len(hands) == 1:
        # Nothing to fuse: the single macro already drives/aligns/scans/bags one hand, and its
        # {scanned, placed, hand} shape is exactly what the predicate wants for a lone item.
        return checkout_held_item(nav, hand=hands[0], drive=drive,
                                  bag_if_unscanned=bag_if_unscanned, debug_dir=debug_dir)

    # 1. Drive to the checkout ONCE (carry-safe: REST every held hand first so the items ride the drive).
    if drive:
        for h in hands:
            set_hand_pose("rest", hand=h)
        res = go_to_counter(nav)
        steps["counter"] = res
        if not res.get("arrived"):
            return _out(False, False, False,
                        f"could not reach the counter: {res.get('reason', 'path blocked')}")

    # 2. Align on the scan pad ONCE - both sweeps go from this stance.
    al = align_to_scanner(nav, debug_dir=debug_dir)
    steps["align"] = al
    aligned = bool(al.get("aligned"))

    # 3. Baseline receipt from the ALIGNED pose (screen legible now), then re-acquire the pad so the
    #    first sweep goes toward it (reading the screen yaws the body; this is rotation-only).
    sres = center_to_screen(debug_dir=debug_dir)
    baseline = read_text_in_box(sres.get("box")) if sres.get("box") else []
    center_to_scanner(debug_dir=debug_dir)

    # 4. Sweep-scan EACH held hand before bagging any. scan_held_item ends on a screen read, so its
    #    returned receipt is the running baseline; re-centre the pad before the next sweep.
    scanned = {}
    for i, h in enumerate(hands):
        sc = scan_held_item(hand=h, baseline=baseline, debug_dir=debug_dir)
        steps["scans"][h] = sc
        scanned[h] = bool(sc.get("scanned"))
        baseline = sc.get("receipt") or baseline
        if i < len(hands) - 1:
            center_to_scanner(debug_dir=debug_dir)   # re-face the pad for the next sweep

    # 5. Bag each item - but never an UNSCANNED one unless forced (dropping it hides the scan miss),
    #    and skip a hand that dropped mid-sweep (place_held_item would just refuse an empty hand).
    placed = {}
    grips_now = hand_grip_states()
    for h in hands:
        if (not scanned[h] and not bag_if_unscanned) or not grips_now[h]:
            placed[h] = False
            continue
        pl = place_held_item(hand=h, debug_dir=debug_dir)
        steps["places"][h] = pl
        placed[h] = bool(pl.get("placed"))

    # 6. Back off the counter face once something is dropped (same wall-clearance step as the single
    #    macro: the agent finishes parked against the checkout, which A* reads as a blocked path).
    if any(placed.values()):
        move_backward(5)

    all_scanned = all(scanned.get(h) for h in hands)
    all_placed = all(placed.get(h) for h in hands)
    reason = ("checked out both items: scanned and bagged" if (all_scanned and all_placed)
              else f"scanned={scanned} placed={placed} - some item did not scan/bag; still holding it")
    return _out(all_scanned, all_placed, aligned, reason, scanned=scanned, placed=placed)
