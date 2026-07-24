from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import os
import base64
import json
import time
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import re
from PIL import Image
from loguru import logger
import ast

# Repo-root api.env (overhaul/agent_core/ -> repo root is three parents up), resolved from
# __file__ so it loads regardless of CWD or checkout location.
load_dotenv(Path(__file__).resolve().parent.parent.parent / 'api.env')

from agent_core.sys_inst import (
    SYS_INST_ASSOCIATIVE_SEMANTIC,
    SYS_INST_ASSOCIATIVE_EPISODIC,
    SYS_INST_VLM_LEAN
)
from agent_core.memory import BASE_SEMANTIC_MEMORY
from toolset.actions_str import (
    NAVIGATION_ACTIONS,
    PERCEPTION_ACTIONS,
    MANIPULATION_ACTIONS,
)

def _extract_json(pattern, text: str) -> str:
    """Extract JSON string using regex, falling back to raw text if no code block found."""
    match = re.search(pattern, text)
    if match:
        return match[1]
    # Model returned raw dict/JSON without a code block wrapper
    return text.strip()


def _reach_move_steps(last_reach) -> Optional[int]:
    """If `last_reach` is a measured MOVE verdict, return its move_forward step count, else None.

    The verdict string format is fixed by manipulation.plan_reach + subtask_agents._last_reach_line
    ("MOVE - move_forward N (~0.X m): ..."). Only a MOVE verdict means "you are on the right shelf,
    just too far" - the one case _metric_approach handles instead of the graph candidate-hopper."""
    if not isinstance(last_reach, str) or not last_reach.startswith("MOVE"):
        return None
    m = re.search(r"move_forward\s+(\d+)", last_reach)
    return int(m.group(1)) if m else None


@dataclass
class OpenRouterConfig:
    model_id: str = 'google/gemini-2.5-flash-preview-05-20'
    temperature: float = 0.5
    mode: Literal['base', 'lean'] = 'base'
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: Optional[int] = None
    extra_body: Optional[dict] = None


def _ucl_creds():
    """UCL vLLM host + bearer key. Repo-root api.env (loaded at import above) is the canonical
    source; the conda-meta/state fallback survives for the legacy path where sari_env_old's
    python.exe is invoked DIRECTLY (no `conda activate`), which does not run the env-var hooks.

    The bearer is read as UCL_API_KEY first (api.env's spelling, matching its *_API_KEY
    convention) then UCL_API (legacy env / conda state) - both are accepted so neither source
    silently returns None."""
    host = os.getenv("UCL_BASE_URL")
    key = os.getenv("UCL_API_KEY") or os.getenv("UCL_API")
    if not (host and key):
        try:
            with open(r"C:/Sari/sari_env_old/conda-meta/state", encoding="utf-8") as f:
                sv = json.load(f).get("env_vars", {})
            host = host or sv.get("UCL_BASE_URL")
            key = key or sv.get("UCL_API_KEY") or sv.get("UCL_API")
        except OSError:
            pass
    return host, key


def ucl_qwen_config(temperature=0.5, mode='lean'):
    """Agent-runtime default since 2026-07-19 (user directive): the UCL vLLM server, replacing
    OpenRouter (retired for agent calls when its credits ran out - 402). The ANNOTATOR is
    unaffected and stays pinned to `claude -p` sonnet/medium - see CLAUDE.md."""
    host, key = _ucl_creds()
    if not host:
        raise RuntimeError("UCL_BASE_URL not found (looked in repo-root api.env, then "
                           "sari_env_old conda state)")
    # enable_thinking=False is LOAD-BEARING, not a tweak. MEASURED 2026-07-19: with the
    # default chat template this server thinks before answering - 245 completion tokens /
    # 6.2s for a trivial one-JSON ask vs 7 tokens / 0.3s with thinking off (35x), and the
    # full agent prompts ballooned to ~10 MINUTES per step. Same trap the annotation probe
    # measured ("spent its whole budget thinking, looped"). max_tokens caps runaway output;
    # the agent's JSON replies run ~500-700 tokens.
    return OpenRouterConfig(model_id='Qwen/Qwen3.6-27B', temperature=temperature, mode=mode,
                            api_key=key, base_url=f"http://{host}:8000/v1",
                            max_tokens=1536,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def _encode_image(image: Image.Image) -> dict:
    buf = BytesIO()
    image.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _build_content(*parts) -> list:
    """Build OpenRouter content list from mixed strings and PIL Images."""
    content = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, Image.Image):
            content.append(_encode_image(part))
        elif isinstance(part, str):
            content.append({"type": "text", "text": part})
    return content


class BaseAgent(ABC):
    _MAX_RETRIES = 5
    _RETRY_DELAYS = [5, 10, 20, 40, 60]  # seconds between attempts

    @abstractmethod
    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        self.config = config or OpenRouterConfig()

    @property
    def extractable_json_structured_output(self):
        return re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

    def _api_call_with_retry(self, client: OpenAI, messages: list) -> str:
        """Call the OpenRouter API with exponential-ish backoff on transient failures."""
        last_err = None
        for attempt in range(self._MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=self.config.model_id,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    extra_body=self.config.extra_body,
                )
                return resp.choices[0].message.content
            except (json.JSONDecodeError, Exception) as e:
                last_err = e
                delay = self._RETRY_DELAYS[min(attempt, len(self._RETRY_DELAYS) - 1)]
                logger.warning(
                    f"[API] Attempt {attempt + 1}/{self._MAX_RETRIES} failed "
                    f"({type(e).__name__}: {e}). Retrying in {delay}s..."
                )
                time.sleep(delay)
        raise RuntimeError(
            f"[API] All {self._MAX_RETRIES} attempts failed. Last error: {last_err}"
        )


class SemanticEpisodicAssociativeLearner(BaseAgent):
    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    def generate_content(self, system_instruction: str, image: Optional[Image.Image], text: str) -> str:
        content = _build_content(image, text) if image else [{"type": "text", "text": text}]
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content},
        ]
        return self._api_call_with_retry(self.client, messages)


class VLMAgent(BaseAgent):
    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        self.history: List[Dict[str, Any]] = []
        self.episodic_memory: str = ""
        self.base_semantic_memory: str = ""
        logger.info(f"VLMAgent initialized with model: {self.config.model_id}")

    def reset_history(self):
        self.history = []

    def send_message(self, content: list) -> str:
        self.history.append({"role": "user", "content": content})
        messages = [
            {"role": "system", "content": SYS_INST_VLM_LEAN},
            *self.history,
        ]
        reply = self._api_call_with_retry(self.client, messages)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def get_history_text(self, n: int = 8) -> str:
        result = ""
        for message in self.history[-n:]:
            role = message["role"]
            content = message["content"]
            if isinstance(content, list):
                text = " ".join(p["text"] for p in content if p.get("type") == "text")
            else:
                text = content
            result += f"{role.upper()}: {text}\n"
        return result.strip()


# ---- graph-advised navigator (the slamtest vlm-advised idea, ported to checkpoint hops) --------
# A DEDICATED per-hop navigator VLM - its own stateless call per hop, never the main reasoner,
# never sharing its history - picks the next checkpoint while the graph's shortest-path next hop
# rides along as an explicitly ADVISORY line (vlm_planner.VLMAdvisedPlanner's contract, discrete
# hops instead of frontier waypoints). See EmbodiedAgent._advised_goto for the attribution rules.
ADVISOR_SYS = (
    "You are the NAVIGATOR module of a store robot, walking between checkpoints of a known "
    "store graph one hop at a time. Each turn you get the robot's current camera view, the "
    "checkpoint it stands at, the destination checkpoint, the adjacent checkpoints it can move "
    "to, and the route planner's suggested next hop. Choose next_checkpoint from the ADJACENT "
    "list ONLY. The planner's suggestion is ADVICE, not an order - override it only when what "
    "you SEE justifies it (e.g. the sought product is already visible on a nearer shelf). If "
    "the current view already shows the goods you are being sent to, set stop_here true "
    "instead of moving on. Always give a one-sentence reason."
)
ADVISOR_SCHEMA = {
    "type": "object",
    "properties": {
        "next_checkpoint": {"type": "integer",
                            "description": "checkpoint id to move to; MUST be an adjacent id"},
        "stop_here": {"type": "boolean",
                      "description": "true = the target goods are visible from HERE; do not move"},
        "reason": {"type": "string"},
    },
    "required": ["next_checkpoint", "stop_here", "reason"],
}


class EmbodiedAgent:
    def __init__(self, vlm_config: Optional[OpenRouterConfig] = None,
                 associative_config: Optional[OpenRouterConfig] = None,
                 mode: Literal['base', 'lean'] = 'base',
                 nav_mode: Literal['vlm', 'graph', 'graph-advised'] = 'vlm',
                 resolver_backend: Literal['qwen', 'claude-cli'] = 'qwen',
                 advisor_backend: Literal['qwen', 'claude-cli'] = 'qwen') -> None:

        self.vlm_agent = VLMAgent(vlm_config)
        self.mode = mode
        # Phase 4.2 A/B switch. 'vlm': navigation mode hands the VLM its old action set
        # (the control arm). 'graph': navigation mode dispatches to resolver+goto_checkpoint
        # and the VLM NEVER sees a navigation action - it wakes up in front of shelves.
        # The swap lives HERE, in the mode router, not in the VLM's action list, so the VLM
        # cannot mix strategies and contaminate the arms (phase4.2 plan, "the one rule").
        # 'graph-advised': target selection identical to 'graph', but the DRIVE runs one graph
        # hop at a time through a dedicated navigator VLM that gets the shortest-path next hop
        # as advice (_advised_goto) - the vlm-advised authority arm, at checkpoint granularity.
        self.nav_mode = nav_mode
        # Which backend resolves the target -> candidate checkpoints in the graph arm.
        # DEFAULT 'qwen' since 2026-07-20 (user directive): a variance eval found qwen at
        # parity-to-better vs claude (overall 0.848 vs 0.815), and running it on qwen makes the
        # whole runtime self-hosted AND removes the graph arm's Claude-shaped planner advantage,
        # so the phase-4.2 A/B isolates navigation rather than planner model. 'claude-cli' stays
        # available for comparison.
        self.resolver_backend = resolver_backend
        # Which backend the graph-advised arm's PER-HOP navigator uses. Default qwen for the
        # same reasons as the resolver (self-hosted, no Claude-shaped advantage in an A/B);
        # independent of resolver_backend so the two roles can be mixed deliberately.
        self.advisor_backend = advisor_backend
        self._advised_llm_calls = 0     # lifetime advisor-call counter; harnesses read deltas
        self._advised_stats = []        # per-hop records {hop, cur, pick, advice, agreed, ...}
        self._advised_shot_idx = 0      # monotonically-named advisor screenshots
        self._graph_nav = None          # lazy: needs the sim up
        self._nav_candidates = []       # resolver output for the current task, in visit order
        self._nav_visited = set()
        self._nav_task = None
        self._nav_seeded = None         # 6.3 #1: plan-time candidates, if the orchestrator pre-resolved
        self._hands_active = None       # None = unknown; set on first _set_hands call
        self._hand_pose = None          # None = unknown; 'rest' when the router has parked the hand (6.1)
        self._mem_leg = None            # semantic-memory leg tag; the orchestrator sets it per leg (see _semantic_tag)

        if mode == 'lean':
            self.associative_learner = SemanticEpisodicAssociativeLearner(associative_config)
            self.set_semantic_memory()

    def set_semantic_memory(self) -> None:
        self.vlm_agent.base_semantic_memory = BASE_SEMANTIC_MEMORY
        logger.info("Base semantic memory set for VLMAgent.")

    def set_episodic_memory(self, episodic_memory: str) -> None:
        self.vlm_agent.episodic_memory = episodic_memory
        logger.info(f"Episodic memory updated: {episodic_memory}")

    def _semantic_tag(self, timestep: int) -> str:
        """Provenance label prefixed to each appended semantic-memory entry. When the orchestrator has
        set a current leg (`_mem_leg`, per-leg in subtask_agents.run_leg), the label carries it -
        `@ leg 2 step 5` - so a multi-leg task's accumulated `base_semantic_memory` no longer keys
        several legs' worth of entries under colliding `@ timestep 1`/`2`/... labels (each leg restarts
        `timestep` at 1, so the raw blob otherwise holds duplicate keys describing different places, and
        the semantic learner that re-reads the whole blob every step can't tell one leg from another).
        A single-episode caller (eval_pickup / env_simulation) leaves `_mem_leg` None and keeps the
        original `@ timestep {N}` format unchanged."""
        if self._mem_leg is not None:
            return f"@ leg {self._mem_leg} step {timestep}"
        return f"@ timestep {timestep}"

    # _compute_depth_hint was REMOVED with depth.py (Phase 4.2): the monocular hand-steps hint
    # is gone from BOTH arms identically; the manipulation phase owns picking a replacement
    # (LiDAR forward range or hoveredObject proximity - see phase4.2 plan, REMOVED #3).

    def _set_hands(self, active: bool):
        """Low-level hand enable/disable, state-tracked so the websocket call fires only on
        transitions. Phase 6.1: the WITHIN-TASK path no longer disables hands (disabling dropped
        carried items); this survives for the BETWEEN-TASK hard reset only - return_to_start stows
        (False), the next task re-activates via _set_hand_pose. Any toggle INVALIDATES the pose
        tracker: after Unity re-enables a hand its pose is unknown, so the next _set_hand_pose must
        re-drive. Both A/B arms share this identically."""
        if self._hands_active == active:
            return
        from sim.env import SetHandsActive
        SetHandsActive(active)
        self._hands_active = active
        self._hand_pose = None

    def _set_hand_pose(self, pose: str):
        """Ensure BOTH hands are ACTIVE and parked at the named pose ('rest') - the right hand at the
        left pose's x-mirror (manipulation.pose_for_hand, dual-hand 2026-07-23), so an item carried in
        EITHER hand rides safely. Phase 6.1 replaces _set_hands(active) on the nav/perception path:
        hands are no longer disabled, so a carried item is never dropped by a mode change - it rides at
        REST. Transition-only: the websocket drive fires only when the tracked pose changes (no
        per-step spam). The GRAB pose is NEVER set here - it is tool-internal
        (manipulation.extend_arm_until_grabbed and 6.2's scan/place tools)."""
        self._set_hands(True)                 # always active during a task; no-op if already active
        if self._hand_pose == pose:
            return
        from manip.manipulation import set_hand_pose
        for side in ("left", "right"):
            arrived, reported, resid = set_hand_pose(pose, hand=side)
            if not arrived:
                logger.warning(f"[hand-pose] {side} '{pose}' did not converge (resid={resid:.3f} m, "
                               f"reported={tuple(round(v, 3) for v in reported)}) - frame/clamp issue?")
        self._hand_pose = pose

    def _invalidate_hand_pose(self):
        """Manipulation mode may move the hand (the grab tool sets GRAB then REST; a manual hand poke
        moves it directly). Keep the hand ACTIVE but mark its pose UNKNOWN, so the next nav/perception
        step re-asserts REST even if a poke left it displaced. We deliberately do NOT force REST here -
        manipulation wants the hand free to reach/poke."""
        self._set_hands(True)
        self._hand_pose = None

    # ---- Phase 4.2 graph-navigation dispatcher -------------------------------------------

    def _graph_nav_session(self):
        """Lazy StoreMap+NavSession+resolver backend - constructed on first navigation
        dispatch because NavSession needs the sim live."""
        if self._graph_nav is None:
            from nav.store_map import StoreMap, NavSession
            sm = StoreMap()
            nav = NavSession(sm, stow_hands=False)  # hands managed per-leg, not per-session
            self._graph_nav = (sm, nav)
        return self._graph_nav

    def seed_nav_candidates(self, candidates, target_name=None):
        """Phase 6.3 #1: pre-seed the graph navigator with PLAN-TIME resolved candidates so
        _graph_navigate does NOT re-run the resolver at runtime - the orchestrator already resolved
        this leg's target once, at plan time (subtask_agents.plan_legs). Pass a candidate list to seed,
        or None/[] to clear (the next leg re-resolves fresh). Resets `_nav_task` so the next
        _graph_navigate re-initialises for this leg and picks up the seed. No-op in the vlm arm (that
        arm never calls _graph_navigate)."""
        self._nav_seeded = list(candidates) if candidates else None
        self._nav_seeded_name = target_name
        self._nav_task = None

    def _graph_navigate(self, main_task: str, nav_goal: str = None):
        """Execute one navigation-mode entry deterministically. Returns an arrival note for
        the VLM's next (perception) step, plus a FRESH screenshot - the one captured before
        the drive shows the wrong place.

        Resolver runs ONCE per task (cached); each navigation entry drives to the next
        unvisited candidate (goto_product's retry loop realised as mode-machine behaviour).
        No verifier LLM here - arm B is goto+face+ordinary perception, per the 4.2 plan.

        `nav_goal` is the NARROW per-hop intent handed to the advised navigator's GOAL line
        (the clean per-leg goal, not the full task blob); resolution + the cache key stay on
        the full `main_task`. Defaults to main_task when the caller ships no narrower goal."""
        from nav import locate_task
        from sim.env import RequestScreenshot
        from explore import step_agent

        sm, nav = self._graph_nav_session()

        if self._nav_task != main_task:
            if self._nav_seeded:
                # 6.3 #1: the orchestrator already resolved this leg at plan time - reuse those
                # candidates instead of paying for the resolver again (plan and execution can't then
                # disagree about where the target lives). Ordering is still runtime code's job below.
                self._nav_candidates = [c for c in self._nav_seeded if c in sm.by_id]
                self._nav_resolution = {"candidates": self._nav_candidates,
                                        "target_name": getattr(self, "_nav_seeded_name", None),
                                        "seeded": True}
                logger.info(f"[graph-nav] using {len(self._nav_candidates)} PLAN-SEEDED candidate(s): "
                            f"{self._nav_candidates}")
            else:
                # Resolver backend is selectable; default qwen (see __init__). Both return
                # (result_dict, envelope) with the same (system, prompt, schema, images) call shape.
                if self.resolver_backend == "claude-cli":
                    _resolve_call = lambda s, p, sc, im=(): locate_task.claude_json(s, p, sc, im)
                else:
                    _resolve_call = lambda s, p, sc, im=(): locate_task.qwen_json(s, p, sc, im)
                resolution, _ = locate_task.resolve(_resolve_call, sm, main_task)
                self._nav_resolution = resolution
                cands = resolution.get("candidates") or []
                self._nav_candidates = [c for c in cands if c in sm.by_id]
                logger.info(f"[graph-nav] resolved {resolution.get('target_name')!r} "
                            f"tier={resolution.get('tier')} candidates={self._nav_candidates}")
            self._nav_task = main_task
            self._nav_visited = set()
            if not self._nav_candidates:
                return ("## NAVIGATOR: could not resolve the target to any known location. "
                        "Proceed by exploring visually.\n", None)

        remaining = [c for c in self._nav_candidates if c not in self._nav_visited]
        if not remaining:
            # Candidates exhausted: allow revisits rather than stranding the agent, but say so.
            logger.warning("[graph-nav] all candidates visited; restarting candidate list")
            self._nav_visited = set()
            remaining = list(self._nav_candidates)

        # Pose desync landmine (phase4.2 plan): VLM/manipulation actions moved the agent
        # behind NavSession's back - resync before planning.
        nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
        x, z = nav.pos[0], nav.pos[2]
        target = min(remaining, key=lambda c: sm.hops(sm.nearest_checkpoint((x, z)), c) or 99)
        self._nav_visited.add(target)

        self._set_hand_pose("rest")   # 6.1: hands stay ACTIVE at REST through the drive (carry-safe)
        if self.nav_mode == "graph-advised":
            ok, end_cp = self._advised_goto(sm, nav, target, nav_goal or main_task)
        else:
            ok, end_cp = nav.goto(target), target

        info = sm.checkpoint(end_cp)
        fresh = RequestScreenshot(save_image=False, uri=nav.args.uri)["image"]
        if not ok:
            note = (f"## NAVIGATOR: could not reach checkpoint {target} (path blocked). "
                    f"You are at ({nav.pos[0]:.2f}, {nav.pos[2]:.2f}). Assess visually.\n")
        else:
            holds = ", ".join(info["holds"]) if info["holds"] else "unknown goods"
            stopped = ("" if end_cp == target else
                       f" (the navigator stopped short of checkpoint {target} because the goods "
                       f"looked visible from here)")
            note = (f"## ARRIVED VIA NAVIGATOR: checkpoint {end_cp}{stopped}, facing a shelf "
                    f"holding {holds}. {info['summary'] or ''} If the target is not visible "
                    f"here, choose *navigation* mode again and you will be taken to the next "
                    f"candidate location.\n")
        return note, fresh

    def _advised_goto(self, sm, nav, target, nav_goal):
        """graph-advised arm: drive to `target` ONE GRAPH HOP AT A TIME, each hop chosen by a
        dedicated navigator VLM (its own stateless call - never the main reasoner) with the
        graph's shortest-path next hop injected as an explicitly ADVISORY line. The port of
        vlm_planner.VLMAdvisedPlanner to the frozen checkpoint graph: the VLM owns every hop,
        A* whispers, and agree-rate is recorded per hop so obedience and navigation stay
        distinguishable (read _advised_stats TOGETHER with outcomes - agree ~1.0 = graph arm
        with a per-hop VLM latency+token tax; deviations/stop_here = the interesting rows).

        Where the slamtest arm was exploration-only (advice on a PARTIAL grid can be wrong;
        the camera can catch it), here the map is frozen and the route is near-perfect, so
        the honest upside is different: the camera can justify STOPPING EARLY (the sought
        product spotted before the destination checkpoint) - the one move the deterministic
        graph arm structurally cannot make. stop_here is that affordance.

        Guardrails, in repo discipline (degrade to the graph arm, never strand the task):
          * an invalid pick (non-adjacent/unknown id) is logged and REPLACED by the advice
            hop - one bad answer costs attribution for that hop, not the leg;
          * hop budget = 2x the shortest path + 2; on exhaustion, or a refused hop (executor
            said no - adjacency is a graph edge, reachability is the executor's call), fall
            back to the deterministic nav.goto(target);
          * intermediate hops keep face_shelf=False (no rotate tax mid-route); the final
            arrival - target reached or stop_here granted - faces the shelf like the graph arm.

        Returns (ok, end_cp): end_cp is where the agent actually stands (== target unless
        stop_here fired), so _graph_navigate's arrival note describes the real view."""
        from nav import locate_task
        from explore import step_agent

        if self.advisor_backend == "claude-cli":
            _ask = lambda s, p, sc, im=(): locate_task.claude_json(s, p, sc, im)
        else:
            _ask = lambda s, p, sc, im=(): locate_task.qwen_json(s, p, sc, im)

        shots_dir = os.path.join(sm.output_dir, "advised_nav")
        os.makedirs(shots_dir, exist_ok=True)
        tgt_info = sm.checkpoint(target)
        tgt_holds = ", ".join(tgt_info["holds"]) or "unannotated"

        nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
        cur = sm.nearest_checkpoint((nav.pos[0], nav.pos[2]))
        budget = 2 * (sm.hops(cur, target) or 1) + 2
        for hop in range(1, budget + 1):
            if cur == target:
                return True, cur
            path = sm.hop_path(cur, target)
            advice = path[1] if path and len(path) > 1 else None
            neigh = sm.checkpoint(cur)["neighbors"]
            if not neigh:
                break   # isolated node: nothing to pick from; deterministic fallback below
            lines = []
            for n in neigh:
                ni = sm.checkpoint(n)
                d = sm.hops(n, target)
                lines.append(f"cp{n}: {'?' if d is None else d} hop(s) from destination"
                             + (f" | holds {', '.join(ni['holds'])}" if ni["holds"] else "")
                             + (f" | {ni['summary']}" if ni["summary"] else ""))
            shot = nav.screenshot(os.path.join(shots_dir,
                                               f"hop_{self._advised_shot_idx:05d}.png"))
            self._advised_shot_idx += 1
            prompt = (
                f"## GOAL\nTask: {nav_goal}\n"
                f"Destination: checkpoint {target} (holds {tgt_holds})\n\n"
                f"## WHERE YOU ARE\nCheckpoint {cur}"
                + (f": {sm.checkpoint(cur)['summary']}" if sm.checkpoint(cur)["summary"] else "")
                + "\n\n## ADJACENT CHECKPOINTS (next_checkpoint MUST be one of these)\n"
                + "\n".join(lines)
                + (f"\n\n## PLANNER ADVICE\nShortest route next hop: cp{advice}"
                   if advice is not None else "")
            )
            try:
                result, _env = _ask(ADVISOR_SYS, prompt, ADVISOR_SCHEMA,
                                    (("current view", shot),))
            except Exception as e:  # noqa: BLE001 - one dead call degrades, never strands
                logger.warning(f"[advised-nav] advisor call failed ({type(e).__name__}: {e}); "
                               f"taking the advice hop")
                result = {}
            self._advised_llm_calls += 1
            pick = result.get("next_checkpoint") if isinstance(result, dict) else None
            stop = bool(result.get("stop_here")) if isinstance(result, dict) else False
            reason = (result.get("reason") or "")[:200] if isinstance(result, dict) else ""
            invalid = pick not in neigh and not stop
            if invalid:
                pick = advice if advice is not None else neigh[0]
            rec = {"hop": hop, "cur": cur, "target": target, "pick": pick, "advice": advice,
                   "agreed": (not stop and pick == advice), "invalid": invalid,
                   "stop_here": stop, "reason": reason}
            self._advised_stats.append(rec)
            logger.info(f"[advised-nav] hop {hop}/{budget} at cp{cur} -> "
                        f"{'STOP' if stop else f'cp{pick}'} (advice cp{advice}, "
                        f"{'agreed' if rec['agreed'] else 'INVALID' if invalid else 'deviated'})"
                        f"{': ' + reason if reason else ''}")
            if stop:
                # The camera-override result: the navigator judges the goods visible HERE.
                # Face the shelf like a normal arrival (goto to the current node is a
                # zero-length drive + the face/level epilogue).
                nav.goto(cur)
                return True, cur
            if not nav.goto(pick, face_shelf=(pick == target)):
                logger.warning(f"[advised-nav] executor refused cp{cur} -> cp{pick}; "
                               f"falling back to deterministic drive")
                break
            nav.pos, nav.rot, _ = step_agent((0, 0, 0), (0, 0, 0), nav.args.uri)
            cur = sm.nearest_checkpoint((nav.pos[0], nav.pos[2]))
        if cur == target:
            return True, cur
        logger.warning(f"[advised-nav] budget/refusal at cp{cur} (target cp{target}); "
                       f"degrading to the graph arm's deterministic goto")
        return nav.goto(target), target

    def _navigate_to_counter(self):
        """Deterministically drive to the checkout counter (the cp54 landmark) - the place subtask's
        'go to the counter' primitive, the navigation mirror of perception.center_to_counter. NO
        resolver LLM: the counter is a known singleton (store_map.go_to_counter looks cp54 up direct).
        Returns (arrival_note, fresh_png) exactly like _graph_navigate, so the actor's next
        (perception/place) step sees the post-drive frame, not the pre-drive one.

        Reuses the cached graph-nav session (stow_hands=False) and the same 6.1 carry-safe pattern as
        _graph_navigate: assert the STATE-TRACKED REST pose here, then let go_to_counter resync + drive
        (it does not touch the hands, so the tracker stays valid). NOTE: this is the primitive; making
        it LLM-selectable from the mode machine / typed subtasks is 6.3 dispatch wiring (A/B'd there),
        not done here."""
        from nav.store_map import go_to_counter
        from sim.env import RequestScreenshot

        _sm, nav = self._graph_nav_session()
        self._set_hand_pose("rest")   # 6.1: hands stay ACTIVE at REST through the drive (carry-safe)
        res = go_to_counter(nav)
        fresh = RequestScreenshot(save_image=False, uri=nav.args.uri)["image"]
        if not res.get("arrived"):
            note = (f"## NAVIGATOR: could not reach the checkout counter "
                    f"(checkpoint {res.get('checkpoint')}; {res.get('reason', 'path blocked')}). "
                    f"You are at ({res.get('x', 0.0):.2f}, {res.get('z', 0.0):.2f}). Assess visually.\n")
        else:
            note = (f"## ARRIVED VIA NAVIGATOR: the checkout counter (checkpoint {res['checkpoint']}). "
                    f"Centre the counter surface (center_to_counter), then place the held item.\n")
        return note, fresh

    def _checkout_held_item(self, hand="auto"):
        """Phase 6.3 dispatch: run the deterministic checkout MACRO on the held item(s) - drive to the
        counter, align on the scan pad, scan, and bag - as ONE call the VLM triggers with the
        `checkout_held_item` action (or the `_left`/`_right` variants, which pin `hand`). The VLM never
        sequences the align/scan/place steps (CLAUDE.md doctrine: geometry is deterministic; the VLM
        judges only what is in front of it); its only meaningful move on a checkout leg is to emit this.

        DUAL-HAND (2026-07-23): the default 'auto' dispatches store_map.checkout_held_items, which
        checks out EVERY held item in one fused pass - when carrying two, it sweep-scans both (verifying
        each off the receipt) BEFORE bagging either, off a single drive+align (scan-scan-bag-bag), and
        degrades to the single-hand checkout_held_item when only one hand holds. `_left`/`_right` still
        pin a single hand (checkout_held_item) for one-at-a-time control. Returns the macro's
        {success, scanned, placed, aligned, steps, reason, ...} dict (top-level scanned/placed are ANDed
        across held hands for 'auto'), which run_leg surfaces as `last_checkout` for the checkout
        completion predicate to grant/refuse the STOP.

        Reuses the cached carry-safe nav session (stow_hands=False) and the 6.1 pattern: assert the
        state-tracked REST pose first so the held item rides the drive; the macro owns its own
        GRAB/REST around each hand action (it does not stow), so the tracker stays valid."""
        from nav.store_map import checkout_held_item, checkout_held_items
        _sm, nav = self._graph_nav_session()
        self._set_hand_pose("rest")   # 6.1: hands stay ACTIVE at REST so the carried item survives
        res = checkout_held_items(nav) if hand == "auto" else checkout_held_item(nav, hand=hand)
        print(f"[checkout] success={res.get('success')} scanned={res.get('scanned')} "
              f"placed={res.get('placed')} aligned={res.get('aligned')} - {res.get('reason')}")
        return res

    def _metric_approach(self, move_steps: int):
        """Execute the MEASURED forward nudge from a `MOVE` reach verdict IN PLACE, without hopping
        graph candidates. Returns (note, fresh_png) mirroring _graph_navigate.

        OPTION-1 FIX (2026-07-22): under graph nav, mode==navigation otherwise routes to
        _graph_navigate, which A*-drives to the next UNVISITED candidate checkpoint - abandoning the
        very shelf the target sits on. But when the previous step's grab produced a measured
        `last_reach == "MOVE - move_forward N ..."`, the agent is already ON the right shelf and only
        measurably too far: it needs to creep N units forward along its CURRENT heading, not travel
        to another node. Before this fix that verdict + the graph navigator collided into an infinite
        candidate-hop loop (pickup run 0722_142259: cp32 -> 45 -> 52 -> 18, never closing the last
        ~0.3 m). `navigation` mode was overloaded - coarse checkpoint travel vs. this fine metric
        approach - and the metric move had no execution channel in graph mode. This is that channel.

        The move is deterministic geometry (matching CLAUDE.md's "geometry is deterministic; the VLM
        judges only what is in front of it"): plan_reach already measured the exact step count. We move
        it, then resume the VLM in perception with a re-center-and-retry note, because a body move
        shifts the centred target off-centre (the same finding behind the grab-recovery edit)."""
        from sim.env import move_forward, RequestScreenshot

        self._set_hand_pose("rest")   # 6.1: hands stay ACTIVE at REST through the nudge (carry-safe)
        move_forward(move_steps)   # body-relative along the current heading; env clamps to <=10
        fresh = RequestScreenshot(save_image=False)["image"]
        note = (f"## MOVED {move_steps} STEP(S) (~{move_steps * 0.1:.1f} m) FORWARD to close the "
                f"measured reach gap - you are still facing the same shelf. RE-CENTER on the target "
                f"with center_object_on_screen (the move shifted it off-centre), then retry the grab "
                f"in *manipulation*.\n")
        return note, fresh

    def _call_associative(self, system_instruction: str, image: Optional[Image.Image], text: str) -> str:
        content = _build_content(image, "## CURRENT OBSERVATION\n", text)
        resp = self.associative_learner.client.chat.completions.create(
            model=self.associative_learner.config.model_id,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content},
            ],
            temperature=self.associative_learner.config.temperature,
            max_tokens=self.associative_learner.config.max_tokens,
            extra_body=self.associative_learner.config.extra_body,
        )
        return resp.choices[0].message.content

    def _call_episodic(self, history_text: str) -> str:
        resp = self.associative_learner.client.chat.completions.create(
            model=self.associative_learner.config.model_id,
            messages=[
                {"role": "system", "content": SYS_INST_ASSOCIATIVE_EPISODIC},
                {"role": "user", "content": history_text},
            ],
            temperature=self.associative_learner.config.temperature,
            max_tokens=self.associative_learner.config.max_tokens,
            extra_body=self.associative_learner.config.extra_body,
        )
        return resp.choices[0].message.content

    def execute_lean(self, request, timestep):
        main_task = request['task']
        # The advised navigator's per-hop GOAL line wants the NARROW target intent, not the full task
        # blob. The orchestrator ships a clean per-leg goal here (request['nav_goal'] = leg_text) so the
        # navigator VLM isn't reading previous-leg findings / future goals when it only has to pick the
        # next hop or judge the product visible HERE - that cross-leg noise was measured as this arm's
        # extra deviation vs eval_pickup, whose task string is already one clean sentence and so
        # defaults straight through here. The resolver + nav cache key below still use the FULL
        # main_task; only the advisor prompt narrows. PROMPT CHANGE - A/B pending (attribute-tier +
        # named-product task), read the advised_hops JSONL: agreed should rise WITHOUT losing the
        # correct stop_here rows.
        nav_goal = request.get('nav_goal') or main_task
        raw_state = request.get('state')
        # Measured reach verdict from the previous step's grab (AGENT_STATE_DOC p). A "MOVE" verdict
        # steers the graph-nav branch below into a metric forward nudge instead of a candidate hop.
        reach_move_steps = _reach_move_steps(
            raw_state.get('last_reach') if isinstance(raw_state, dict) else None)
        state = str(request['state'])
        screenshot = str(request['image']).encode('utf-8')
        screenshot = base64.b64decode(screenshot)
        imagebytes = BytesIO(screenshot)
        screenshot = Image.open(imagebytes).convert('RGB')

        new_semantic_memory = ""
        recall = ""

        if timestep == 1:
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## SEMANTIC MEMORY: {self.vlm_agent.base_semantic_memory}\n"
                        f"## STATE: {state}\n")

            semantic_response_text = self._call_associative(
                SYS_INST_ASSOCIATIVE_SEMANTIC, screenshot, user_msg
            )
            print(f"SEMANTIC LEARNER RESPONSE: {semantic_response_text}")

            semantic_response = ast.literal_eval(_extract_json(
                self.associative_learner.extractable_json_structured_output,
                semantic_response_text
            ))
            new_semantic_memory = semantic_response['new_semantic_memory']
            recall = semantic_response['recall']
            agent_mode = semantic_response['mode']
            # MODE/ACTION HORIZON edit (2026-07-23): the single next step the learner committed to, fed
            # to the actor so it acts on THAT step and does not skip ahead to the grab (sys_inst rule 4).
            # Soft .get: an older learner reply without the field just omits the line below.
            next_action = semantic_response.get('next_action')

            # Location gate (2026-07-24): run_leg sets force_navigate when the graph says the agent is
            # not yet at the target's checkpoint(s). The graph owns spatial truth, so OVERRIDE a router
            # that hallucinated the target onto whatever is in front of it (e.g. a just-checked-out item
            # on the counter) and chose perception/manipulation to grab in place. Force *navigation*, and
            # drop any stale MOVE verdict so the nav branch drives a candidate HOP to the shelf instead
            # of a metric creep that inches against the wrong fixture forever.
            if request.get('force_navigate'):
                if agent_mode in ("perception", "manipulation"):
                    agent_mode = "navigation"
                if agent_mode == "navigation":
                    reach_move_steps = None

            nav_note = ""
            if agent_mode == "navigation" and self.nav_mode in ("graph", "graph-advised"):
                # Arm B: navigation executes deterministically; the VLM resumes in perception
                # at the new location, with a note and a FRESH frame (the screenshot captured
                # before the drive shows the wrong place).
                if reach_move_steps is not None:
                    # A measured MOVE verdict means "on the right shelf, just too far": creep the
                    # measured distance forward in place instead of hopping to another candidate
                    # checkpoint (which stranded the agent in a loop - see _metric_approach).
                    nav_note, _fresh_png = self._metric_approach(reach_move_steps)
                else:
                    nav_note, _fresh_png = self._graph_navigate(main_task, nav_goal)
                if _fresh_png is not None:
                    screenshot = Image.open(BytesIO(_fresh_png)).convert('RGB')
                agent_mode = "perception"

            # Phase 6.1: hands stay ACTIVE at REST for nav/perception so a carried item survives the
            # trip. In manipulation mode leave the hand free (the grab/place tool sets GRAB then
            # restores REST itself) but mark the pose UNKNOWN, so the next nav/perception step
            # re-asserts REST even if a manual poke displaced it. Both A/B arms share this.
            if agent_mode == "manipulation":
                self._invalidate_hand_pose()
            else:
                self._set_hand_pose("rest")

            if agent_mode == "perception":
                available_actions = f"{PERCEPTION_ACTIONS}\n\n"
            elif agent_mode == "navigation":
                available_actions = f"{NAVIGATION_ACTIONS}\n\n"
            elif agent_mode == "manipulation":
                available_actions = f"{MANIPULATION_ACTIONS}\n\n"
            elif agent_mode == "STOP":
                return {
                    'halt': True,
                    'text': "STOP action received, terminating execution...",
                    'agent_mode': agent_mode
                }

            self.vlm_agent.base_semantic_memory += f"{self._semantic_tag(timestep)}: {new_semantic_memory}\n"

            # Suppress the intended-action line once the graph dispatcher has already driven (nav_note
            # set): the learner's next_action ("move to the shelf") is then stale - the actor should
            # just perceive the fresh post-drive frame.
            next_action_line = (f"## THIS STEP'S INTENDED ACTION: {next_action}\n"
                                if next_action and not nav_note else "")
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"{next_action_line}"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}"
                        f"{nav_note}")

            vlm_content = _build_content(screenshot, "## CURRENT OBSERVATION\n" + user_msg)
            response_text = self.vlm_agent.send_message(vlm_content)
            print(f"VLMAgent RESPONSE: {response_text}")

            response_json = ast.literal_eval(_extract_json(
                self.vlm_agent.extractable_json_structured_output,
                response_text
            ))

            episodic_response_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
            episodic_response = ast.literal_eval(_extract_json(
                self.associative_learner.extractable_json_structured_output,
                episodic_response_text
            ))

            episodic_memory = (f"@ timestep {timestep}:\n"
                               f"## DENSE SUMMARY: {episodic_response['dense_summary']}\n"
                               f"## WHAT WORKED: {episodic_response['what_worked']}\n"
                               f"## WHAT TO AVOID: {episodic_response['what_to_avoid']}\n")
            self.set_episodic_memory(episodic_memory)

        else:
            user_msg = (f"## MAIN TASK: {main_task}\n"
                        f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## SEMANTIC MEMORY: {self.vlm_agent.base_semantic_memory}\n"
                        f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
                        f"## STATE: {state}\n")

            semantic_response_text = self._call_associative(
                SYS_INST_ASSOCIATIVE_SEMANTIC, screenshot, user_msg
            )
            semantic_response = ast.literal_eval(_extract_json(
                self.associative_learner.extractable_json_structured_output,
                semantic_response_text
            ))
            new_semantic_memory = semantic_response['new_semantic_memory']
            recall = semantic_response['recall']
            agent_mode = semantic_response['mode']
            # MODE/ACTION HORIZON edit (2026-07-23): single next step, fed to the actor (sys_inst rule 4).
            next_action = semantic_response.get('next_action')

            self.vlm_agent.base_semantic_memory += f"{self._semantic_tag(timestep)}: {new_semantic_memory}\n"
            print(f"SEMANTIC LEARNER RESPONSE: {semantic_response}")

            # Location gate (2026-07-24): see the timestep==1 branch above. run_leg's force_navigate
            # lets the graph override a router that put the target where the agent already stands and
            # chose perception/manipulation; force *navigation* and drop the stale MOVE verdict so the
            # nav branch does a candidate HOP to the shelf, not a metric creep in place.
            if request.get('force_navigate'):
                if agent_mode in ("perception", "manipulation"):
                    agent_mode = "navigation"
                if agent_mode == "navigation":
                    reach_move_steps = None

            nav_note = ""
            if agent_mode == "navigation" and self.nav_mode in ("graph", "graph-advised"):
                # Arm B: navigation executes deterministically; the VLM resumes in perception
                # at the new location, with a note and a FRESH frame (the screenshot captured
                # before the drive shows the wrong place).
                if reach_move_steps is not None:
                    # A measured MOVE verdict means "on the right shelf, just too far": creep the
                    # measured distance forward in place instead of hopping to another candidate
                    # checkpoint (which stranded the agent in a loop - see _metric_approach).
                    nav_note, _fresh_png = self._metric_approach(reach_move_steps)
                else:
                    nav_note, _fresh_png = self._graph_navigate(main_task, nav_goal)
                if _fresh_png is not None:
                    screenshot = Image.open(BytesIO(_fresh_png)).convert('RGB')
                agent_mode = "perception"

            # Phase 6.1: hands stay ACTIVE at REST for nav/perception so a carried item survives the
            # trip. In manipulation mode leave the hand free (the grab/place tool sets GRAB then
            # restores REST itself) but mark the pose UNKNOWN, so the next nav/perception step
            # re-asserts REST even if a manual poke displaced it. Both A/B arms share this.
            if agent_mode == "manipulation":
                self._invalidate_hand_pose()
            else:
                self._set_hand_pose("rest")

            if agent_mode == "perception":
                available_actions = f"{PERCEPTION_ACTIONS}\n\n"
            elif agent_mode == "navigation":
                available_actions = f"{NAVIGATION_ACTIONS}\n\n"
            elif agent_mode == "manipulation":
                available_actions = f"{MANIPULATION_ACTIONS}\n\n"
            elif agent_mode == "STOP":
                return {
                    'halt': True,
                    'text': "STOP action received, terminating execution..."
                }

            next_action_line = (f"## THIS STEP'S INTENDED ACTION: {next_action}\n"
                                if next_action and not nav_note else "")
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"{next_action_line}"
                        f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}"
                        f"{nav_note}")

            vlm_content = _build_content(screenshot, "## CURRENT OBSERVATION\n" + user_msg)
            response_text = self.vlm_agent.send_message(vlm_content)

            response_json = ast.literal_eval(_extract_json(
                self.vlm_agent.extractable_json_structured_output,
                response_text
            ))

            episodic_response_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
            episodic_response = ast.literal_eval(_extract_json(
                self.associative_learner.extractable_json_structured_output,
                episodic_response_text
            ))

            episodic_memory = (f"@ timestep {timestep}:\n"
                               f"## DENSE SUMMARY: {episodic_response['dense_summary']}\n"
                               f"## WHAT WORKED: {episodic_response['what_worked']}\n"
                               f"## WHAT TO AVOID: {episodic_response['what_to_avoid']}\n")
            self.set_episodic_memory(episodic_memory)

        with open('semantic_memory.txt', 'w') as f:
            f.write(self.vlm_agent.base_semantic_memory)
        with open('episodic_memory.txt', 'w') as f:
            f.write(self.vlm_agent.episodic_memory)

        return {
            'halt': False,
            'nav_note': nav_note,  # non-empty iff the graph dispatcher drove this step
            'text': response_text,
            'agent_mode': agent_mode,
            'semantic': semantic_response_text,   # mode router: mode / recall / new_semantic_memory
            'episodic': episodic_response_text,    # reflection: dense_summary / what_worked / what_to_avoid
        }
