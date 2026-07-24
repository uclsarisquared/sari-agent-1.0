"""VLM navigation planners.

WHY THIS EXISTS
---------------
CLAUDE.md's governing principle - "the graph owns spatial truth; the VLM only judges what is
directly in front of it" - rests on a claim NavReasonPlan.md asserts but never measured:

    "Navigation is the primary failure mode of the agent: it collides with walls, takes
     suboptimal paths, and the VLM wastes its reasoning budget on global path planning it
     fundamentally cannot do from a first-person view. The root cause is that no component
     has a global view of the store."                              - NavReasonPlan.md:5

That last sentence is exactly why the claim cannot be cited as evidence for A* as it stands. It
attributes the failure to MISSING INFORMATION (no global view), not to the VLM's spatial reasoning.
An examiner can fairly reply: of course a blind navigator loses to one holding a map - that
compares "map" against "no map", not "A*" against "VLM". Confounded, and proves nothing.

So this module deliberately refuses to build the strawman. The VLM planner here is handed STRICTLY
MORE information than A* ever gets:

  * the same OccupancyGrid A* plans on, rendered top-down as an image (walls, free space, unknown)
  * the SAME grid again as a coarse ASCII map, because a text grid sidesteps the vision encoder's
    downscale entirely (see the resolution trap in `render_map_png`)
  * every frontier cluster's exact world (x, z), size, distance and relative bearing, as text -
    so it never has to visually localise a target to aim at one
  * its own pose, and the running history of where its own proposals got blocked
  * the first-person camera view - information A* has no way to consume at all

and is asked for EXACTLY the contract A* satisfies: the next straight-line-reachable waypoint
toward a frontier. Same inputs (plus extras), same output type, same safety layer, same
termination rule, same loop. Only the planner object differs.

If the VLM still underperforms under those conditions, "it just needed the map" is no longer an
available objection, and the measured gap is attributable to the spatial reasoning itself. If it
does NOT underperform, that is a genuine finding too, and better learned here than at a defence.

WHAT IS MEASURED, AND THE ONE METRIC THAT MATTERS
-------------------------------------------------
Every proposal is validated against the grid and recorded, never silently corrected - correcting it
would launder the failure this exists to quantify. The decisive metric is `crosses_occupied`:
the straight segment from the agent to the VLM's waypoint passes through cells the grid ALREADY
knows are occupied. A* cannot emit such a waypoint - `simplify_path`/`_line_of_sight` structurally
forbid it - so this is a clean, quantitative statement of "proposed a route through a known wall,
while holding a map of that wall." It is the number to put in the thesis.

Nothing here can drive the agent into a shelf: explore.py's `swept_clearance_ahead` LiDAR cap
applies identically to both arms and is what actually keeps the body safe. A bad waypoint therefore
costs a wasted/blocked step, not a collision - which is precisely what makes it measurable.

DELIBERATE FAIRNESS CHOICES (all reversible, so the generosity itself is testable)
---------------------------------------------------------------------------------
  * Frontier detection is SHARED, not reimplemented: both arms call grid.frontiers() +
    cluster_frontiers() with the same parameters. The contested claim is about navigation, so
    frontier *detection* must not be allowed to vary between arms.
  * Termination is SHARED: a run ends when no clusters remain / max_steps / the same circuit
    breaker. The model's own `exploration_complete` is advisory and only LOGGED (as
    premature_done). If it ended runs, the arms would stop on different criteria and their
    coverage numbers would not be comparable - and "it quit at step 3" would waste a live run.
  * The VLM is re-asked every step by default (--vlm-replan-every 1): maximum responsiveness, and
    the most expensive setting. Cost is reported honestly rather than bought with capability.
  * On an unusable reply we retry, then coast on the last good waypoint, then hold. We NEVER fall
    back to A* - that would contaminate the arm being measured.

MEASURED SERVER LESSONS INHERITED FROM annotate_probe.py - do not re-litigate
----------------------------------------------------------------------------
  * Schema MUST be sent as `response_format`. This vLLM build SILENTLY IGNORES `guided_json`,
    and silently is the whole problem: the prompt alone already yields conforming JSON, so an
    ignored schema is indistinguishable from a working one until you negative-control it.
  * `enable_thinking` defaults OFF. A measured probe run spent its entire token budget reasoning,
    looped, and never reached an answer. The schema does the structuring instead; the `reasoning`
    field below captures the model's own stated rationale in one short string (which is also the
    qualitative evidence worth quoting in the writeup - the model narrating why it is about to
    walk into a shelf).
"""
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # slamtest/drivers
_SLAM_DIR = os.path.dirname(_THIS_DIR)                         # slamtest
if _SLAM_DIR not in sys.path:
    sys.path.insert(0, _SLAM_DIR)
import _bootstrap  # noqa: F401,E402  (overhaul root + all slamtest category dirs)

from sim.env import RequestScreenshot, TransformAgent  # noqa: E402

from frontier_planner import (  # noqa: E402
    FrontierPlanner,
    NavCommand,
    PlannerState,
    _inflate_occupied,
    _line_of_sight,
    astar,
    cluster_frontiers,
    simplify_path,
)
from mapping import normalize_deg  # noqa: E402

# image_content_block/resolve_base_url are imported rather than re-vendored: they encode measured
# server facts (the qwen data-URI block shape; UCL_BASE_URL being a bare host that needs :8000/v1)
# and a second copy in this directory would drift from the probe's. post_chat is deliberately NOT
# imported - it sys.exit()s on any HTTPError, which is right for a one-shot probe and catastrophic
# for a 300-step live run, where one transient blip would kill the run and lose the map. Hence the
# retrying _post_chat below.
from annotate_probe import image_content_block, resolve_api_key, resolve_base_url  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"

# Distance sanity bound on a proposed waypoint, in meters. Not a correction - a classification.
# A* waypoints are line-of-sight hops off a simplified path and are never absurd; a VLM asked for
# "the next waypoint" that answers with a point 40m away has not understood the question, and that
# is worth counting separately from a merely-wrong-but-plausible answer.
MAX_SANE_WAYPOINT_M = 12.0

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "One or two sentences: why this frontier and why this waypoint is "
                           "reachable in a straight line from the current position.",
        },
        "target_frontier_id": {
            "type": "integer",
            "description": "id of the frontier from the FRONTIERS list you are heading toward.",
        },
        "waypoint_x": {"type": "number", "description": "World X of the next waypoint, meters."},
        "waypoint_z": {"type": "number", "description": "World Z of the next waypoint, meters."},
        "exploration_complete": {
            "type": "boolean",
            "description": "True only if you believe nothing worth mapping remains.",
        },
    },
    "required": ["reasoning", "target_frontier_id", "waypoint_x", "waypoint_z",
                 "exploration_complete"],
    "additionalProperties": False,
}

CLUSTER_CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "target_frontier_id": {"type": "integer"},
        "exploration_complete": {"type": "boolean"},
    },
    "required": ["reasoning", "target_frontier_id", "exploration_complete"],
    "additionalProperties": False,
}

_SYS_COMMON = """\
You are the navigation planner for a wheeled robot that is mapping an indoor grocery store by
driving around it. You decide where it moves next. You are not describing the scene; you are
steering a real body through real geometry, and a bad decision wastes a step.

WORLD CONVENTIONS (these are exact - trust them over your intuition about the images):
- Coordinates are world meters on the horizontal plane: X and Z. Y (height) is irrelevant to you.
- Yaw is in degrees. Yaw 0 faces +Z. Yaw 90 faces +X. Yaw 180 faces -Z. Yaw 270 faces -X.
  A point at bearing b and distance d from the robot is at (x + d*sin(b), z + d*cos(b)).
- On the TOP-DOWN MAP image: +X is to the right, +Z is UP. White = known empty floor, black =
  known obstacle (shelves, walls), grey = not yet mapped. The red arrow is the robot, pointing
  the way it currently faces. Numbered blue markers are the frontiers listed below.
- A "frontier" is the boundary between mapped empty floor and unmapped space. Driving to one is
  what reveals new map. When no frontiers remain, the store is fully mapped.

YOUR OUTPUT CONTRACT (this is the same contract the deterministic planner satisfies):
Return the next WAYPOINT: a single world (x, z) point that
  1. makes progress toward the frontier you chose, and
  2. is reachable from the robot's CURRENT position by travelling in a STRAIGHT LINE that does
     not pass through any black/occupied cell.
Requirement 2 is the hard one and it is the whole job. The robot moves in a straight line toward
whatever you return. If that straight line crosses a shelf, the robot stops against it and the
step is wasted - it will NOT route around the obstacle for you. If the direct line to your chosen
frontier is blocked, do not return the frontier itself: return an intermediate waypoint that opens
up the corner first (down the aisle, past the shelf end), and you will get another turn once the
robot is there.

Keep the waypoint within about 3 meters of the robot unless the straight line is plainly clear.
Short, certain hops beat ambitious ones through walls.\
"""

_SYS_FULL = _SYS_COMMON + """

Return JSON only, matching the schema.\
"""

_SYS_GOAL = _SYS_COMMON + """

For this task you choose ONLY the frontier to head for - the path to it is computed for you, so
you do not need to emit a waypoint. Pick the frontier that best advances the mapping.

Return JSON only, matching the schema.\
"""


def _post_chat(base, payload, api_key, timeout, retries, backoff=2.0):
    """POST with retries, returning the parsed body, or None once retries are spent.

    Deliberately NOT annotate_probe.post_chat, which sys.exit()s on HTTPError. That is correct for
    a one-shot probe and ruinous here: an exploration run is 300 sequential steps against a shared
    remote server, and killing the process on one transient 5xx would lose the whole map. Returning
    None instead lets the planner record the failure as data (vlm_http_failures) and coast.
    """
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # vLLM puts the real complaint (bad schema, unsupported field) in the body, not the
            # status line, so surface it - a 400 here is our bug and will repeat on every retry.
            last = f"HTTP {e.code}: {e.read().decode()[:400]}"
            if 400 <= e.code < 500:
                print(f"[vlm] {last} (client error - not retrying)", flush=True)
                return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            wait = backoff * (2 ** attempt)
            print(f"[vlm] {last} - retry {attempt + 1}/{retries} in {wait:.0f}s", flush=True)
            time.sleep(wait)
    print(f"[vlm] giving up after {retries} retries: {last}", flush=True)
    return None


def render_map_png(grid, cur_world_xz, yaw_deg, clusters, dpi=110, figsize=8.0,
                   waypoint=None, crop_m=None):
    """Render the occupancy grid top-down as PNG bytes, annotated for the model.

    Uses the Figure/FigureCanvasAgg API rather than pyplot ON PURPOSE. matplotlib locks in its
    backend on the first `import matplotlib.pyplot`, and explore.py's save_snapshot() chooses that
    backend for the whole run (Agg, or an interactive one under --show-map). Going through pyplot
    here would fight that choice and could silently break --show-map; this path renders off-screen
    without touching global backend state either way.

    RESOLUTION TRAP (measured, see CLAUDE.md): the fridge annotation failure was ultimately
    effective resolution after the vision encoder's downscale - not the prompt, not the model. The
    same trap applies here. A 60m grid at 0.1m is 600x600 cells; once the encoder downsamples it,
    a 0.3m aisle gap can vanish into a smear, and the model would then be failing at PERCEPTION,
    not at spatial reasoning - which would confound exactly the thing this measures. Two
    mitigations: frontier coordinates are also given as exact text (so no target ever has to be
    visually localised), and the ASCII map carries the same occupancy at a resolution the encoder
    cannot touch. --map-crop-m is available if a zoomed local view is wanted, but it is off by
    default because cropping removes the global view and would re-introduce the very confound
    ("no global view") this module exists to eliminate.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Same free/occupied/unknown convention as explore.py's save_snapshot, so this image and the
    # saved grid_*.png read the same way for a human comparing them side by side.
    display = np.full(grid.log_odds.shape, 0.5)
    display[grid.log_odds < grid.FREE_THRESHOLD] = 1.0
    display[grid.log_odds > grid.OCCUPIED_THRESHOLD] = 0.0

    fig = Figure(figsize=(figsize, figsize))
    FigureCanvasAgg(fig)
    ax = fig.subplots()

    extent = (grid.origin, grid.origin + grid.n * grid.res,
              grid.origin, grid.origin + grid.n * grid.res)
    # .T + origin="lower" + world extent: X horizontal, Z vertical, ticks already in meters, so
    # the model reads world coordinates straight off the axes instead of converting cell indices.
    ax.imshow(display.T, origin="lower", cmap="gray", vmin=0, vmax=1, extent=extent)

    ax.plot(cur_world_xz[0], cur_world_xz[1], "o", color="red", markersize=9, zorder=5)
    ax.annotate("", xytext=(cur_world_xz[0], cur_world_xz[1]),
                xy=(cur_world_xz[0] + math.sin(math.radians(yaw_deg)) * 1.6,
                    cur_world_xz[1] + math.cos(math.radians(yaw_deg)) * 1.6),
                arrowprops=dict(color="red", width=2.2, headwidth=9), zorder=5)

    for c in clusters:
        wx, wz = grid.to_world(*c.centroid_cell)
        ax.plot(wx, wz, "s", color="deepskyblue", markersize=7,
                markeredgecolor="black", markeredgewidth=0.5, zorder=6)
        ax.annotate(str(c.vlm_id), (wx, wz), color="blue", fontsize=11, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points", zorder=7)

    if waypoint is not None:
        ax.plot([cur_world_xz[0], waypoint[0]], [cur_world_xz[1], waypoint[1]],
                "--", color="orange", linewidth=1.5, zorder=4)

    if crop_m:
        ax.set_xlim(cur_world_xz[0] - crop_m, cur_world_xz[0] + crop_m)
        ax.set_ylim(cur_world_xz[1] - crop_m, cur_world_xz[1] + crop_m)

    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Z (m)")
    ax.set_title("Top-down map: white=free, black=obstacle, grey=unmapped, red=robot")
    ax.grid(True, alpha=0.3, linestyle=":")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def render_ascii_map(grid, cur_world_xz, clusters, res_m=0.5, radius_m=None):
    """The same occupancy grid as a coarse text map.

    This is the strongest single concession to the VLM in the whole comparison, and it is here on
    purpose. The obvious rebuttal to a map-image result is "it just couldn't SEE the map" (a
    perception failure, per the resolution trap in render_map_png). A text grid removes that
    rebuttal: it passes through the tokenizer untouched, at a resolution no vision encoder can
    degrade. If the model still routes through a '#', it was told in plain text that '#' was there.

    Coarsening is conservative-for-safety: a coarse cell is '#' if ANY fine cell in it is occupied.
    That over-reports obstacles slightly (a 0.5m cell clipping one shelf corner reads solid), which
    can make a genuinely passable narrow gap look closed. It is the honest direction to err - the
    alternative under-reports walls and would invent passages that do not exist - but it is a real
    limitation of THIS view, not of the model, and must be reported as such: --ascii-map-res is the
    knob, and the model always also has the full-resolution map image and the exact text
    coordinates, neither of which is coarsened.
    """
    factor = max(1, int(round(res_m / grid.res)))
    # The TRUE world width of one character, which is res_m only when res_m is an exact multiple
    # of grid.res. Deriving it from grid.res * factor instead of reusing res_m is not pedantry:
    # labelling these axes with the requested res_m instead of the realised one put every
    # coordinate in this map out by a factor of `factor` (5x at the defaults), so the text map
    # contradicted both the stated pose and the map image's axes. The model was then being scored
    # on reconciling three mutually inconsistent coordinate frames - measuring this bug, not its
    # spatial reasoning. Caught only by printing the map next to a known pose; never label an axis
    # with a requested resolution.
    coarse_m = grid.res * factor
    n_coarse = grid.n // factor
    usable = n_coarse * factor
    lo = grid.log_odds[:usable, :usable]
    blocks = lo.reshape(n_coarse, factor, n_coarse, factor)

    occupied = (blocks > grid.OCCUPIED_THRESHOLD).any(axis=(1, 3))
    free = (blocks < grid.FREE_THRESHOLD).any(axis=(1, 3))

    chars = np.full((n_coarse, n_coarse), "?", dtype="<U1")
    chars[free] = "."
    chars[occupied] = "#"  # after free: any obstacle in the block wins (see docstring)

    def to_coarse(wx, wz):
        return (int((wx - grid.origin) / grid.res) // factor,
                int((wz - grid.origin) / grid.res) // factor)

    ax_, az_ = to_coarse(*cur_world_xz)
    if 0 <= ax_ < n_coarse and 0 <= az_ < n_coarse:
        chars[ax_, az_] = "A"
    for c in clusters:
        wx, wz = grid.to_world(*c.centroid_cell)
        cx, cz = to_coarse(wx, wz)
        if 0 <= cx < n_coarse and 0 <= cz < n_coarse and c.vlm_id < 10:
            chars[cx, cz] = str(c.vlm_id)

    # Trim to the mapped region (plus the robot), so the model reads a store-sized map instead of
    # a screen of '?' - the 60m grid is mostly untouched padding around a much smaller store.
    touched = np.argwhere(chars != "?")
    if len(touched) == 0:
        return "(nothing mapped yet)"
    x0, z0 = touched.min(axis=0)
    x1, z1 = touched.max(axis=0)
    if radius_m:
        r = int(radius_m / coarse_m)
        x0, x1 = max(x0, ax_ - r), min(x1, ax_ + r)
        z0, z1 = max(z0, az_ - r), min(z1, az_ + r)

    lines = []
    for cz in range(z1, z0 - 1, -1):  # high Z first: printed top-down, matching the map image
        wz = grid.origin + (cz + 0.5) * coarse_m
        row = "".join(chars[cx, cz] for cx in range(x0, x1 + 1))
        lines.append(f"z={wz:+6.1f} |{row}|")
    wx0 = grid.origin + (x0 + 0.5) * coarse_m
    wx1 = grid.origin + (x1 + 0.5) * coarse_m
    lines.append(f"         +{'-' * (x1 - x0 + 1)}+")
    lines.append(f"          x={wx0:+.1f} (left edge) .. x={wx1:+.1f} (right edge), "
                 f"{coarse_m:g}m per character")
    return "\n".join(lines)


def _describe_clusters(grid, clusters, cur_world_xz, yaw_deg):
    rows = []
    for c in clusters:
        wx, wz = grid.to_world(*c.centroid_cell)
        dx, dz = wx - cur_world_xz[0], wz - cur_world_xz[1]
        dist = math.hypot(dx, dz)
        bearing = normalize_deg(math.degrees(math.atan2(dx, dz)) - yaw_deg)
        side = "ahead" if abs(bearing) < 45 else (
            "right" if 45 <= bearing < 135 else ("behind" if abs(bearing) >= 135 else "left"))
        rows.append(f"  id={c.vlm_id}  at (x={wx:+.2f}, z={wz:+.2f})  size={c.size} cells  "
                    f"distance={dist:.2f}m  bearing={bearing:+.0f}deg ({side})")
    return "\n".join(rows)


class _VLMClientMixin:
    """Shared Qwen plumbing + the honest bookkeeping both VLM planner roles need."""

    def _init_vlm(self, *, uri, base_url, model, api_key, timeout, retries, mode,
                  think, max_tokens, temperature, ascii_map_res, map_crop_m,
                  include_clearance, capture_dir, debug):
        self.uri = uri
        self.base = resolve_base_url(base_url)
        self.model = model
        # Resolved at init, never trusted as a literal: the server 401s without the real
        # bearer (measured 2026-07-19), and the key may rotate - resolve_api_key reads
        # $UCL_API, then sari_env_old's conda state.
        self.api_key = resolve_api_key(api_key)
        self.timeout = timeout
        self.retries = retries
        self.mode = mode                      # "both" | "map" | "ego"
        self.think = think
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ascii_map_res = ascii_map_res    # 0 disables the ASCII map
        self.map_crop_m = map_crop_m
        self.include_clearance = include_clearance
        self.capture_dir = capture_dir
        self.vlm_debug = debug

        self.decisions = []      # one record per VLM call - the run's raw evidence
        self.blocked_history = []
        self.stats = {
            "vlm_calls": 0, "vlm_http_failures": 0, "vlm_unparseable": 0,
            "vlm_latency_s_total": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "premature_done": 0, "bad_frontier_id": 0, "waypoint_proposals": 0,
            "waypoint_out_of_bounds": 0, "waypoint_into_obstacle": 0,
            "waypoint_into_inflation": 0, "waypoint_crosses_obstacle": 0,
            "waypoint_crosses_inflation": 0, "waypoint_absurd_distance": 0,
            "waypoint_ok": 0, "coasted_on_stale_waypoint": 0, "held_no_waypoint": 0,
            # vlm-advised arm only; stay 0 elsewhere. Read TOGETHER, never alone: agree-rate
            # without the collision flags cannot distinguish "navigates well" from "retypes A*".
            "advice_offered": 0, "advice_agree": 0, "advice_deviate": 0,
        }

    def _pose(self):
        """Read the live pose. update() is handed (x, z) but not yaw, and yaw is needed to render
        the robot's heading and to make sense of the camera frame. A zero-delta TransformAgent is
        the established way to query pose without moving (explore.py:281 opens its run with exactly
        this call), so this stays a pure read and explore.py's loop needs no modification - which
        is what keeps the two arms sharing one identical loop.
        """
        state = TransformAgent((0, 0, 0), (0, 0, 0), uri=self.uri)
        return state["translation"], state["rotation"]

    def _camera_png(self, step_tag):
        result = RequestScreenshot(save_image=False, uri=self.uri)
        png = result["image"]
        if self.capture_dir:
            os.makedirs(self.capture_dir, exist_ok=True)
            with open(os.path.join(self.capture_dir, f"{step_tag}_ego.png"), "wb") as f:
                f.write(png)
        return png

    def _ask(self, system, schema, user_blocks, label):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_blocks},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # Qwen3.x reasons unless told not to; measured to burn the whole budget and loop
            # without reaching an answer (annotate_probe.py). The schema does the structuring.
            "chat_template_kwargs": {"enable_thinking": self.think},
            # MUST be response_format. `guided_json` is silently ignored by this vLLM build, and
            # silently is the trap: the prompt alone yields conforming JSON, so an ignored schema
            # looks identical to a working one. Only a negative control ("banana") exposed it.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": label, "schema": schema, "strict": True},
            },
        }
        t0 = time.time()
        resp = _post_chat(self.base, payload, self.api_key, self.timeout, self.retries)
        dt = time.time() - t0
        self.stats["vlm_calls"] += 1
        self.stats["vlm_latency_s_total"] += dt
        if resp is None:
            self.stats["vlm_http_failures"] += 1
            return None, dt
        usage = resp.get("usage", {})
        self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        self.stats["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        text = resp["choices"][0]["message"].get("content")
        if not text:
            # content=None means it never reached an answer (e.g. thinking ate max_tokens) - a
            # real outcome, not a glitch to route around.
            self.stats["vlm_unparseable"] += 1
            return None, dt
        try:
            return json.loads(text), dt
        except json.JSONDecodeError:
            self.stats["vlm_unparseable"] += 1
            if self.vlm_debug:
                print(f"[vlm] unparseable: {text[:300]}", flush=True)
            return None, dt

    def _build_blocks(self, grid, cur_world_xz, yaw_deg, clusters, step_tag, task_text):
        parts = []
        text = [
            task_text,
            "",
            f"ROBOT POSE: x={cur_world_xz[0]:+.2f}, z={cur_world_xz[1]:+.2f}, yaw={yaw_deg:.1f}deg",
            f"MAP: {grid.n}x{grid.n} cells at {grid.res}m, world extent "
            f"{grid.origin:+.1f}..{grid.origin + grid.n * grid.res:+.1f} on both X and Z.",
            "",
            f"FRONTIERS still unexplored ({len(clusters)}):",
            _describe_clusters(grid, clusters, cur_world_xz, yaw_deg),
        ]
        if self.blocked_history:
            text += [
                "",
                "YOUR PREVIOUS PROPOSALS THAT THE ROBOT COULD NOT EXECUTE (it stopped against "
                "something solid). Do not propose these again, and treat the space between the "
                "robot and them as untrustworthy:",
            ]
            for b in self.blocked_history[-6:]:
                text.append(f"  step {b['step']}: aimed at (x={b['x']:+.2f}, z={b['z']:+.2f}) "
                            f"- blocked")
        if self.ascii_map_res:
            text += [
                "",
                # Deliberately does not restate the cell size: the map's own footer prints the
                # REALISED resolution (grid.res * factor), and a second number stated here could
                # disagree with it - which is exactly the coordinate-frame inconsistency that
                # already cost one invalidated measurement. One source of truth per fact.
                "TEXT MAP ('#'=obstacle, '.'=free floor, '?'=unmapped, 'A'=robot, digits=frontier "
                "ids). Same map as the image; the cell size and axes are printed with it:",
                "",
                render_ascii_map(grid, cur_world_xz, clusters, res_m=self.ascii_map_res),
            ]
        parts.append({"type": "text", "text": "\n".join(text)})

        if self.mode in ("both", "map"):
            png = render_map_png(grid, cur_world_xz, yaw_deg, clusters, crop_m=self.map_crop_m)
            if self.capture_dir:
                os.makedirs(self.capture_dir, exist_ok=True)
                with open(os.path.join(self.capture_dir, f"{step_tag}_map.png"), "wb") as f:
                    f.write(png)
            parts.append({"type": "text", "text": "TOP-DOWN MAP (same map, as an image):"})
            parts.append(image_content_block(self.model, "image/png", _b64(png)))
        if self.mode in ("both", "ego"):
            parts.append({"type": "text", "text": "FIRST-PERSON CAMERA (what the robot sees now):"})
            parts.append(image_content_block(self.model, "image/png", _b64(self._camera_png(step_tag))))
        return parts

    def write_report(self, output_dir, tag="final"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"vlm_decisions_{tag}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for d in self.decisions:
                f.write(json.dumps(d) + "\n")
        return path


def _b64(raw):
    import base64
    return base64.b64encode(raw).decode()


class VLMFrontierPlanner(_VLMClientMixin):
    """role=full: the VLM picks the frontier AND emits the waypoint. No A* anywhere.

    This is the arm that actually tests NavReasonPlan.md's claim. It is a drop-in for
    FrontierPlanner - same update()/notify_blocked() surface - so explore.py's loop runs it
    unmodified and every non-planner variable is held fixed by construction rather than by
    discipline.

    Structure intentionally mirrors FrontierPlanner: same frontier detection, same DONE
    conditions, same no-movement circuit breaker. What is deleted is exactly astar() and
    simplify_path() - the spatial reasoning - and nothing else.
    """

    def __init__(self, grid, *, uri, base_url=None, model=DEFAULT_MODEL, api_key="none",
                 timeout=180.0, retries=2, mode="both", think=False, max_tokens=1024,
                 temperature=0.0, ascii_map_res=0.5, map_crop_m=None, include_clearance=False,
                 capture_dir=None, min_cluster_size=4, connectivity=8, body_radius=0.3,
                 goal_arrival_radius=0.4, replan_every=1, max_replans_without_moving=10,
                 debug=False):
        self.grid = grid
        self.min_cluster_size = min_cluster_size
        self.connectivity = connectivity
        self.body_radius = body_radius
        self.goal_arrival_radius = goal_arrival_radius
        self.replan_every = max(1, replan_every)
        self.max_replans_without_moving = max_replans_without_moving

        self.state = PlannerState.NEED_GOAL
        self._step = 0
        self._last_waypoint = None
        self._last_goal_world = None
        self._steps_since_ask = 0
        self._no_progress_pick_cell = None
        self._no_progress_pick_count = 0

        self._init_vlm(uri=uri, base_url=base_url, model=model, api_key=api_key, timeout=timeout,
                       retries=retries, mode=mode, think=think, max_tokens=max_tokens,
                       temperature=temperature, ascii_map_res=ascii_map_res,
                       map_crop_m=map_crop_m, include_clearance=include_clearance,
                       capture_dir=capture_dir, debug=debug)

    ADVICE_AGREE_RADIUS_M = 0.5  # within one body-diameter of the advice counts as "followed"

    def _advise(self, cur_world_xz, clusters):
        """Advisory hook - None in this arm. VLMAdvisedPlanner overrides. Kept on the base so
        update() stays one shared implementation instead of a diverging copy per arm."""
        return None

    def _validate(self, wx, wz, cur_world_xz, occupied_mask, raw_occupied_mask):
        """Classify a proposed waypoint. Records what is wrong with it; never fixes it.

        Fixing would destroy the measurement - a silently-repaired waypoint is indistinguishable
        from a good one in the results. The LiDAR clearance cap in explore.py's loop is what keeps
        the body safe, so a bad waypoint is paid for in a wasted step, not a collision.

        TWO STANDARDS, REPORTED SEPARATELY - collapsing them would overclaim
        --------------------------------------------------------------------
        `_inflate_occupied` marks every cell within body_radius of an obstacle, which is the
        standard A* itself plans against, so it is the fair like-for-like comparison. But a cell
        inside that margin is NOT necessarily a cell the map calls solid, and a metric named as
        though it were would not survive scrutiny:

          *_inflation  - the agent's body cannot occupy this / the swept body clips an obstacle.
                         The apples-to-apples number: exactly what A* is forbidden to emit.
          *_obstacle   - the cell (or the line) is literally recorded as occupied in the grid.
                         Strictly stronger, and un-arguable: this is a wall on the map.

        A measured subtlety that makes `into_obstacle` rare and easy to misread: LiDAR rays stop
        at a shelf's FACE, so a shelf's interior is never observed and stays `unknown` (log_odds
        0.0), not occupied. A waypoint placed deep inside a shelf therefore does NOT trip
        into_obstacle - it trips into_inflation (near the face) while sitting in nominally
        "unknown" space. The line from the agent to it still crosses the face, so
        crosses_obstacle is the flag that catches it, and crosses_obstacle is accordingly the
        headline number for the writeup: the straight route passes through cells the map records
        as solid.
        """
        flags = []
        self.stats["waypoint_proposals"] += 1
        cell = self.grid.cell(wx, wz)
        if not self.grid.in_bounds(*cell):
            flags.append("out_of_bounds")
            self.stats["waypoint_out_of_bounds"] += 1
            return flags
        if math.hypot(wx - cur_world_xz[0], wz - cur_world_xz[1]) > MAX_SANE_WAYPOINT_M:
            flags.append("absurd_distance")
            self.stats["waypoint_absurd_distance"] += 1
        if raw_occupied_mask[cell[0], cell[1]]:
            flags.append("into_obstacle")
            self.stats["waypoint_into_obstacle"] += 1
        elif occupied_mask[cell[0], cell[1]]:
            flags.append("into_inflation")
            self.stats["waypoint_into_inflation"] += 1

        cur_cell = self.grid.cell(*cur_world_xz)
        # _line_of_sight is the exact predicate A*'s simplify_path uses to guarantee its waypoints
        # are straight-line reachable, so reusing it compares the arms on an identical definition
        # rather than a reimplementation that could quietly differ.
        if not _line_of_sight(self.grid, cur_cell, cell, occupied_mask=raw_occupied_mask):
            flags.append("crosses_obstacle")
            self.stats["waypoint_crosses_obstacle"] += 1
        elif not _line_of_sight(self.grid, cur_cell, cell, occupied_mask=occupied_mask):
            flags.append("crosses_inflation")
            self.stats["waypoint_crosses_inflation"] += 1

        if not flags:
            self.stats["waypoint_ok"] += 1
        return flags

    def update(self, cur_world_xz, cur_cell):
        self._step += 1

        # Frontier detection and DONE conditions are byte-identical to FrontierPlanner's, so the
        # arms stop on the same criterion and their coverage is comparable.
        frontier_cells = self.grid.frontiers()
        if len(frontier_cells) == 0:
            self.state = PlannerState.DONE
            return NavCommand(kind="done")
        clusters = cluster_frontiers(frontier_cells, connectivity=self.connectivity,
                                     min_cluster_size=self.min_cluster_size)
        if not clusters:
            self.state = PlannerState.DONE
            return NavCommand(kind="done")

        # Same circuit breaker as FrontierPlanner: if we keep deciding from the exact same cell
        # with no movement, the run is over rather than looping to max_steps.
        if cur_cell == self._no_progress_pick_cell:
            self._no_progress_pick_count += 1
        else:
            self._no_progress_pick_cell = cur_cell
            self._no_progress_pick_count = 0
        if self._no_progress_pick_count >= self.max_replans_without_moving:
            print(f"[vlm] circuit breaker: {self.max_replans_without_moving} decisions from cell "
                  f"{cur_cell} with no movement -> done", flush=True)
            self.state = PlannerState.DONE
            return NavCommand(kind="done")

        # Coast toward an un-reached waypoint when --vlm-replan-every > 1. Default 1 re-asks every
        # step (most responsive, most expensive) - capability is not traded for cost here.
        self._steps_since_ask += 1
        if (self._last_waypoint is not None and self._steps_since_ask < self.replan_every
                and math.hypot(cur_world_xz[0] - self._last_waypoint[0],
                               cur_world_xz[1] - self._last_waypoint[1]) > self.goal_arrival_radius):
            return NavCommand(kind="goto_waypoint", target_world_xz=self._last_waypoint,
                              goal_world_xz=self._last_goal_world, replanned=False)
        self._steps_since_ask = 0

        for i, c in enumerate(clusters):
            c.vlm_id = i
        (px, _py, pz), rot = self._pose()
        yaw = rot[1]
        step_tag = f"step{self._step:04d}"

        # vlm-advised hook: base arm has no advisor (returns None), so this line is inert
        # everywhere except VLMAdvisedPlanner - the arms stay one-variable-apart.
        advice = self._advise((px, pz), clusters)
        task_text = "Decide the robot's next waypoint."
        if advice is not None:
            a_fid, (ax, az) = advice
            self.stats["advice_offered"] += 1
            task_text = (
                f"A CLASSICAL PLANNER (A*) computed a SUGGESTION from the same map you see: "
                f"head for frontier {a_fid}; its next straight-line-reachable waypoint is "
                f"({ax:+.2f}, {az:+.2f}). The suggestion is ADVISORY - you own the decision. "
                f"Follow it, or override it if your first-person camera or your reading of the "
                f"map indicates a better or safer move (the planner cannot see the camera).\n"
                f"Decide the robot's next waypoint.")

        blocks = self._build_blocks(
            self.grid, (px, pz), yaw, clusters, step_tag, task_text)
        decision, dt = self._ask(_SYS_FULL, DECISION_SCHEMA, blocks, "nav_decision")

        record = {"step": self._step, "pos": [px, pz], "yaw": yaw, "latency_s": round(dt, 2),
                  "n_frontiers": len(clusters)}

        if decision is None:
            # Never fall back to A* - that would contaminate the arm under measurement.
            if self._last_waypoint is not None:
                self.stats["coasted_on_stale_waypoint"] += 1
                record["outcome"] = "vlm_failed_coasted"
                self.decisions.append(record)
                return NavCommand(kind="goto_waypoint", target_world_xz=self._last_waypoint,
                                  goal_world_xz=self._last_goal_world, replanned=False)
            self.stats["held_no_waypoint"] += 1
            record["outcome"] = "vlm_failed_held"
            self.decisions.append(record)
            # Aim at our own position: the loop caps step_len ~0, prints "holding", and calls
            # notify_blocked - i.e. the failure is registered through the normal path.
            return NavCommand(kind="goto_waypoint", target_world_xz=(px, pz),
                              goal_world_xz=(px, pz), replanned=True)

        record["reasoning"] = decision.get("reasoning", "")
        record["raw"] = {k: decision.get(k) for k in
                         ("target_frontier_id", "waypoint_x", "waypoint_z", "exploration_complete")}

        if decision.get("exploration_complete"):
            # Advisory only, and logged - honouring it would let the arms terminate on different
            # criteria and make their coverage incomparable. Frontiers demonstrably remain here.
            self.stats["premature_done"] += 1
            record["premature_done"] = True

        fid = decision.get("target_frontier_id")
        goal_cluster = next((c for c in clusters if c.vlm_id == fid), None)
        if goal_cluster is None:
            self.stats["bad_frontier_id"] += 1
            record["bad_frontier_id"] = fid
            goal_cluster = clusters[0]
        goal_world = self.grid.to_world(*goal_cluster.centroid_cell)

        wx, wz = float(decision["waypoint_x"]), float(decision["waypoint_z"])
        occupied_mask = _inflate_occupied(self.grid, self.body_radius)
        raw_occupied_mask = self.grid.log_odds > self.grid.OCCUPIED_THRESHOLD
        flags = self._validate(wx, wz, (px, pz), occupied_mask, raw_occupied_mask)
        if advice is not None:
            # Attribution data for the writeup: how far did the model land from the advice,
            # and did it follow? Deviation is NOT an error - a deviation with a clean flag
            # set is the camera-override result this arm exists to detect.
            adist = math.hypot(wx - advice[1][0], wz - advice[1][1])
            agree = adist <= self.ADVICE_AGREE_RADIUS_M
            record["advice"] = {"frontier_id": advice[0],
                                "waypoint": [advice[1][0], advice[1][1]],
                                "dist_m": round(adist, 2), "agree": agree}
            self.stats["advice_agree" if agree else "advice_deviate"] += 1
        record["waypoint"] = [wx, wz]
        record["goal_world"] = list(goal_world)
        record["flags"] = flags
        record["outcome"] = "ok" if not flags else ",".join(flags)
        self.decisions.append(record)

        if self.vlm_debug or flags:
            print(f"[vlm] step={self._step} -> frontier {fid} waypoint=({wx:+.2f},{wz:+.2f}) "
                  f"{'FLAGS=' + ','.join(flags) if flags else 'clean'} ({dt:.1f}s) "
                  f"| {record['reasoning'][:110]}", flush=True)

        if "out_of_bounds" in flags:
            # Unusable as a target at all (grid.cell would index outside the array downstream).
            # Counted above; hold rather than crash the run.
            return NavCommand(kind="goto_waypoint", target_world_xz=(px, pz),
                              goal_world_xz=goal_world, replanned=True)

        self._last_waypoint = (wx, wz)
        self._last_goal_world = goal_world
        return NavCommand(kind="goto_waypoint", target_world_xz=(wx, wz),
                          goal_world_xz=goal_world, replanned=True)

    def notify_blocked(self, blocked_cell):
        """The loop calls this when swept_clearance_ahead capped the step near zero - i.e. the
        proposal was not physically executable. Fed back into the next prompt (see _build_blocks):
        NavReasonPlan.md:350 already specifies collision memory for the navigation prompt, so
        withholding it would understate what their own design intends the VLM to have.
        """
        self.blocked_history.append({"step": self._step,
                                     "x": float(blocked_cell[0]), "z": float(blocked_cell[1])})
        self._last_waypoint = None   # do not coast toward something that just failed
        self._steps_since_ask = self.replan_every  # force a fresh ask next step


class VLMAdvisedPlanner(VLMFrontierPlanner):
    """--planner vlm-advised: the VLM still owns every decision; A* whispers.

    The fourth arm on the authority spectrum, between `vlm` (no help) and `vlm-goal` (A*
    executes). Each ask, A*'s OWN best move from the current PARTIAL map - nearest reachable
    frontier by path cost, first simplified waypoint - is computed and injected into the
    prompt as an explicitly ADVISORY line. The VLM may follow or override; nothing is
    validated differently, nothing is corrected, and the executor still never falls back
    to A*.

    Why this arm exists (see phase4.1 doc, "vlm-advised"):
      * It is the MOST generous baseline constructible: map + coordinates + THE ANSWER. If
        the model still proposes waypoints that cross recorded walls, no examiner can say
        the setup starved it. If it executes faithfully, that measures the ceiling.
      * During EXPLORATION the advice is computed on an incomplete grid, so it can be
        genuinely wrong in ways the first-person camera can catch. A deviation with a clean
        flag set is the camera-override result - the one regime where VLM-in-the-loop could
        legitimately beat pure A*, and the only honest reason to keep a VLM in the loop.

    Attribution rule, load-bearing: read `advice_agree`/`advice_deviate` TOGETHER with the
    waypoint flags. Agree-rate ~1.0 with clean flags = A* with a VLM latency tax. Deviations
    with clean flags = camera overrides worth reading individually. Deviations WITH flags =
    the thesis claim, now with the answer having been handed over.

    On the FROZEN map this arm is pointless by construction - A* is near-perfect there, so
    the VLM can only match it (tax) or deviate (worse). It belongs to the exploration
    harness only; keep it out of the Phase 4.2 task-navigation A/B.
    """

    def _advise(self, cur_world_xz, clusters):
        """A*'s best move from the current partial map, or None when no frontier is reachable
        (advice failing to exist is itself informative - it means A* is stuck too, and the
        prompt then simply carries no advisory line rather than a fabricated one)."""
        occupied = _inflate_occupied(self.grid, self.body_radius)
        cur_cell = self.grid.cell(*cur_world_xz)
        best = None  # (path_len, vlm_id, first_waypoint_world)
        for c in clusters:
            path = astar(self.grid, cur_cell, tuple(c.centroid_cell),
                         connectivity=self.connectivity, occupied_mask=occupied)
            if path is None:
                continue
            if best is None or len(path) < best[0]:
                wps = simplify_path(path, self.grid, occupied_mask=occupied)
                # STRICTLY wps[1] - the only waypoint with guaranteed line-of-sight from the
                # CURRENT cell. A first draft skipped waypoints inside goal_arrival_radius
                # (capture_walk's executor habit) and MEASURED 2026-07-19, first live run:
                # 3 of 28 advised asks got flagged crosses_inflation with dist=0.0 - the VLM
                # had copied the advice EXACTLY and the ADVICE was the violation, because
                # cur -> wps[2] carries no line-of-sight guarantee. An advisor whose advice
                # fails the harness's own validator contaminates the attribution analysis.
                target = (self.grid.to_world(*wps[1]) if len(wps) > 1
                          else self.grid.to_world(*c.centroid_cell))
                best = (len(path), c.vlm_id, target)
        if best is None:
            return None
        return best[1], best[2]


class VLMGoalPlanner(FrontierPlanner, _VLMClientMixin):
    """role=goal: the VLM chooses WHICH frontier; A* still computes HOW to reach it.

    The decomposition matters more than it looks. role=full answers "is the VLM worse?"; this
    answers "worse at WHAT?" - and those are different theses. If goal choice is fine and only
    pathing collapses, the result is not the vague "VLMs can't navigate" but the precise, and much
    more defensible, "VLMs choose targets competently and fail at metric path planning" - which is
    also a direct empirical vindication of this repo's actual architecture (the graph owns spatial
    truth; the VLM judges what is in front of it), rather than a blanket dismissal that overclaims.

    Subclasses FrontierPlanner and replaces ONLY _pick_and_plan's ranking. The parent's
    _plan_to_cluster/_try_reach/_observation_vantages and its entire FOLLOWING state machine are
    inherited untouched, so the A* half of this arm is provably the same code explore.py runs.
    """

    def __init__(self, grid, *, uri, base_url=None, model=DEFAULT_MODEL, api_key="none",
                 timeout=180.0, retries=2, mode="both", think=False, max_tokens=1024,
                 temperature=0.0, ascii_map_res=0.5, map_crop_m=None, include_clearance=False,
                 capture_dir=None, debug=False, **planner_kwargs):
        FrontierPlanner.__init__(self, grid, **planner_kwargs)
        self._step = 0
        self._init_vlm(uri=uri, base_url=base_url, model=model, api_key=api_key, timeout=timeout,
                       retries=retries, mode=mode, think=think, max_tokens=max_tokens,
                       temperature=temperature, ascii_map_res=ascii_map_res,
                       map_crop_m=map_crop_m, include_clearance=include_clearance,
                       capture_dir=capture_dir, debug=debug)

    def _pick_and_plan(self, cur_cell):
        self._step += 1

        # Circuit breaker + frontier detection + DONE conditions: same as the parent's.
        if cur_cell == self._no_progress_pick_cell:
            self._no_progress_pick_count += 1
        else:
            self._no_progress_pick_cell = cur_cell
            self._no_progress_pick_count = 0
        if self._no_progress_pick_count >= self.max_replans_without_moving:
            self.state = PlannerState.DONE
            return NavCommand(kind="done")

        frontier_cells = self.grid.frontiers()
        if len(frontier_cells) == 0:
            self.state = PlannerState.DONE
            return NavCommand(kind="done")
        clusters = cluster_frontiers(frontier_cells, connectivity=self.connectivity,
                                     min_cluster_size=self.min_cluster_size)
        if not clusters:
            self.state = PlannerState.DONE
            return NavCommand(kind="done")

        for i, c in enumerate(clusters):
            c.vlm_id = i
        (px, _py, pz), rot = self._pose()
        step_tag = f"pick{self._step:04d}"
        blocks = self._build_blocks(self.grid, (px, pz), rot[1], clusters, step_tag,
                                    "Choose which frontier the robot should head for next.")
        decision, dt = self._ask(_SYS_GOAL, CLUSTER_CHOICE_SCHEMA, blocks, "frontier_choice")

        record = {"step": self._step, "pos": [px, pz], "yaw": rot[1], "latency_s": round(dt, 2),
                  "n_frontiers": len(clusters)}
        if decision is None:
            # No A* fallback for the CHOICE (that is the thing being measured), but the parent's
            # own ordering is the only other way to name a cluster; record it as a failed choice.
            record["outcome"] = "vlm_failed_used_first_cluster"
            self.decisions.append(record)
            ordered = clusters
        else:
            record["reasoning"] = decision.get("reasoning", "")
            if decision.get("exploration_complete"):
                self.stats["premature_done"] += 1
                record["premature_done"] = True
            fid = decision.get("target_frontier_id")
            chosen = next((c for c in clusters if c.vlm_id == fid), None)
            if chosen is None:
                self.stats["bad_frontier_id"] += 1
                record["bad_frontier_id"] = fid
                ordered = clusters
            else:
                record["outcome"] = "ok"
                record["chosen_frontier"] = fid
                record["goal_world"] = list(self.grid.to_world(*chosen.centroid_cell))
                # VLM's pick first, then the rest as fallbacks so an unreachable pick degrades
                # into "try the others" exactly as the parent would, instead of ending the run.
                ordered = [chosen] + [c for c in clusters if c is not chosen]
            self.decisions.append(record)

        for consider_blocklisted in (False, True):
            for cluster in ordered:
                if not consider_blocklisted and cluster.centroid_cell in self._blocklist:
                    continue
                path_world, _reached = self._plan_to_cluster(cur_cell, cluster)
                if path_world is not None:
                    self.goal_cluster = cluster
                    self.path_world = path_world
                    self.waypoint_idx = 1 if len(path_world) > 1 else 0
                    self.state = PlannerState.FOLLOWING
                    self._best_dist_to_goal = math.inf
                    self._steps_since_progress = 0
                    self._stuck_counter = 0
                    return NavCommand(kind="goto_waypoint",
                                      target_world_xz=self.path_world[self.waypoint_idx],
                                      goal_world_xz=self.grid.to_world(*cluster.centroid_cell),
                                      replanned=True)
                if not consider_blocklisted:
                    self._blocklist[cluster.centroid_cell] = self.goal_blocklist_steps

        self.state = PlannerState.DONE
        return NavCommand(kind="done")
