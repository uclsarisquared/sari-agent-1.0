from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import copy
import os
import base64
import json
import tempfile
import time
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
import re
from PIL import Image
from loguru import logger
import ast

# Repo-root config.env (agent/agent_core/ -> repo root is three parents up), resolved from
# __file__ so it loads regardless of CWD or checkout location.
load_dotenv(Path(__file__).resolve().parent.parent.parent / 'config.env')

from agent_core.sys_inst import (
    SYS_INST_ASSOCIATIVE_SEMANTIC,
    SYS_INST_ASSOCIATIVE_EPISODIC,
    SYS_INST_VLM_LEAN
)
from agent_core.memory import base_semantic_memory
from agent_core.models import agent_model
from agent_core.context import SemanticLog
from agent_core.context_policy import ContextPolicy, validate_context_policy
# Per-role token attribution. The meter counts every SDK call whether or not it is tagged; the
# `role` blocks below only decide WHICH reasoner each call is billed to, so an ablation can read
# off what the component it removed was actually costing. Import-cheap and sim-free.
from agent_core import token_meter
from toolset.actions_str import (
    NAVIGATION_ACTIONS,
    PERCEPTION_ACTIONS,
    MANIPULATION_ACTIONS,
    INSPECTION_ACTIONS,
)

def _extract_json(pattern, text: str) -> str:
    """Extract JSON string using regex, falling back to raw text if no code block found."""
    match = re.search(pattern, text)
    if match:
        return match[1]
    # Model returned raw dict/JSON without a code block wrapper
    return text.strip()


# Safe default the semantic learner degrades to when its JSON reply is truncated or malformed.
# MEASURED 2026-07-24 (orchestrator run, leg 4 step 1): the learner hand-traced a BFS through the
# whole connectivity map into `recall` ("21 -> 20 -> 19 ... 54 connects to 32"), overflowed the
# 1536-token cap mid-string, and ast.literal_eval raised "unterminated string literal" - killing the
# step (recovered only on the next). Navigation is the correct recovery: the overflow happens ONLY
# while the learner is route-planning, and the deterministic graph navigator (_graph_navigate, A*
# over the store map) - not this discarded reply - actually computes the path.
_SEMANTIC_FALLBACK = {
    'new_semantic_memory': '',
    'recall': '',
    'mode': 'navigation',
    'next_action': None,
    'reported_answer': '',
}


def _parse_semantic_response(pattern, text: str) -> dict:
    """ast.literal_eval the learner's JSON, degrading to _SEMANTIC_FALLBACK on a truncated/malformed
    reply instead of raising (an unguarded raise aborts the whole step). The returned dict always
    carries new_semantic_memory / recall / mode / next_action / reported_answer, so the callers'
    direct indexing is safe even when the model omitted a field."""
    raw = _extract_json(pattern, text)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict) and 'mode' in parsed:
            return {**_SEMANTIC_FALLBACK, **parsed}
        logger.warning("[learner] reply parsed but missing 'mode'; using navigation fallback")
    except (SyntaxError, ValueError, TypeError) as e:
        logger.warning(f"[learner] unparseable reply ({type(e).__name__}: {e}); "
                       "using navigation fallback")
    return dict(_SEMANTIC_FALLBACK)


def _stop_response(semantic_response: dict, semantic_response_text: str) -> dict:
    """Build the one STOP payload contract shared by both execute_lean branches."""
    answer = semantic_response.get("reported_answer") if isinstance(semantic_response, dict) else ""
    return {
        "halt": True,
        "text": "STOP action received, terminating execution...",
        "agent_mode": "STOP",
        "reported_answer": answer if isinstance(answer, str) else "",
        "semantic": semantic_response_text,
    }


def _resolve_agent_mode(agent_mode: str, force_navigate: bool = False,
                        force_manipulate: bool = False, inspect_mode: str = None) -> str:
    """Apply orchestrator mode overrides without changing the learner's classifier prompt.

    STOP is always preserved. Inspect mode is a hard leg-scope route (held -> manipulation,
    unheld -> perception) and therefore cannot enter navigation; outside inspection, the graph
    location gate wins defensively if both legacy overrides are supplied.
    """
    if agent_mode == "STOP":
        return agent_mode
    if inspect_mode == "held":
        return "manipulation"
    if inspect_mode == "visual":
        return "perception"
    if force_navigate:
        return ("navigation"
                if agent_mode in ("perception", "manipulation")
                else agent_mode)
    if force_manipulate and agent_mode == "perception":
        return "manipulation"
    return agent_mode


def _available_actions(agent_mode: str, held_item_inspection: bool = False) -> str:
    """Return the actor vocabulary for the effective mode."""
    if agent_mode == "perception":
        return f"{PERCEPTION_ACTIONS}\n\n"
    if agent_mode == "navigation":
        return f"{NAVIGATION_ACTIONS}\n\n"
    if agent_mode == "manipulation":
        actions = INSPECTION_ACTIONS if held_item_inspection else MANIPULATION_ACTIONS
        return f"{actions}\n\n"
    raise ValueError(f"unsupported agent mode: {agent_mode!r}")


# The episodic reflector emits the same single-quoted Python-literal dict as the learner, so it has
# the same failure mode: an apostrophe in a value ("Kellogg's", "it's") breaks ast.literal_eval.
# When that happens the step keeps going with an empty reflection rather than crashing.
_EPISODIC_FALLBACK = {'dense_summary': '', 'what_worked': '', 'what_to_avoid': ''}


def _safe_ast_dict(pattern, text: str, fallback: dict, tag: str = "parse") -> dict:
    """ast.literal_eval an extracted ```json block, degrading to `fallback` on a malformed/truncated
    reply instead of raising. The actor prompt asks for a single-quoted Python-literal dict, so any
    apostrophe in a value (product names like "Kellogg's", prose like "it's") can break the literal
    parse - and an unguarded raise here aborts the WHOLE step (measured 2026-07-24, the "Get a
    cereal" run: every actor/episodic reply that named a cereal brand crashed the step). Returns
    fallback merged with whatever fields did parse, so callers' direct indexing stays safe."""
    raw = _extract_json(pattern, text)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return {**fallback, **parsed}
        logger.warning(f"[{tag}] reply parsed but was not a dict; using fallback")
    except (SyntaxError, ValueError, TypeError) as e:
        logger.warning(f"[{tag}] unparseable reply ({type(e).__name__}: {e}); using fallback")
    return dict(fallback)


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


def _endpoint_creds():
    """OpenAI-compatible endpoint host + bearer key. Repo-root config.env (loaded at import above)
    is the canonical source; the conda-meta/state fallback survives for the legacy path where
    sari_env_old's python.exe is invoked DIRECTLY (no `conda activate`), which does not run the
    env-var hooks.

    The host comes back BARE - no scheme, no trailing slash - because every caller builds
    f"http://{host}:8000/v1" from it. MEASURED 2026-07-26: config.env held the scheme
    ("http://202.92.159.240"), so that f-string produced "http://http://202.92.159.240:8000/v1",
    httpx tried to resolve the hostname `http`, and the SDK's retries turned a DNS failure into a
    ~35s APIConnectionError that Sari Bench could only record as a bare agent_error. The
    scheme-tolerant readers (annotate_probe.resolve_base_url, locate_task) already stripped it;
    this path did not. Normalising here rather than in config.env fixes all three agent-runtime call
    sites (agent_vlm_config, orchestrator._llm_client, perception.CLIENT) at once."""
    host = os.getenv("OPENAI_API_URL")
    key = os.getenv("OPENAI_API_KEY")
    if not (host and key):
        try:
            conda_state = os.getenv("SARI_CONDA_STATE", r"C:/Sari/sari_env_old/conda-meta/state")
            with open(conda_state, encoding="utf-8") as f:
                sv = json.load(f).get("env_vars", {})
            host = host or sv.get("OPENAI_API_URL")
            key = key or sv.get("OPENAI_API_KEY")
        except OSError:
            pass
    if host:
        host = host.strip().split("//", 1)[-1].strip("/").split("/", 1)[0]
        # A port in the var would collide with the :8000 the callers append. Explicit error
        # beats another silent malformed-URL round trip.
        if ":" in host:
            raise RuntimeError(f"OPENAI_API_URL must be a bare host without a port, got {host!r} "
                               "(the code appends :8000/v1 itself)")
    return host, key


def agent_vlm_config(temperature=0.5, mode='lean'):
    """Agent-runtime default since 2026-07-19 (user directive): the OpenAI API compatible endpoint
    from config.env, replacing OpenRouter (retired for agent calls when its credits ran out - 402).
    The model id comes from $SARI_MODEL (agent_core.models). The ANNOTATOR is unaffected and stays
    pinned to `claude -p` sonnet/medium - see CLAUDE.md."""
    host, key = _endpoint_creds()
    if not host:
        raise RuntimeError("OPENAI_API_URL not found (looked in repo-root config.env, then "
                           "sari_env_old conda state)")
    # enable_thinking=False is LOAD-BEARING, not a tweak. MEASURED 2026-07-19: with the
    # default chat template this endpoint thinks before answering - 245 completion tokens /
    # 6.2s for a trivial one-JSON ask vs 7 tokens / 0.3s with thinking off (35x), and the
    # full agent prompts ballooned to ~10 MINUTES per step. Same trap the annotation probe
    # measured ("spent its whole budget thinking, looped"). max_tokens caps runaway output;
    # the agent's JSON replies run ~500-700 tokens. Ignored by servers that don't support it.
    return OpenRouterConfig(model_id=agent_model(), temperature=temperature, mode=mode,
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


API_MAX_ATTEMPTS = 10
API_RETRY_DELAYS = (1, 2, 4, 8, 15, 30, 30, 30, 30)


def _is_transient_api_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, RateLimitError, TimeoutError, ConnectionError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


def call_with_api_retries(operation):
    """Retry transient OpenAI failures, logging every attempt, and re-raise the final error after ten attempts."""
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            result = operation()
            if attempt > 0:
                logger.info(f"[api-retry] attempt {attempt + 1}/{API_MAX_ATTEMPTS} succeeded")
            return result
        except Exception as error:
            remaining = API_MAX_ATTEMPTS - (attempt + 1)
            if not _is_transient_api_error(error) or attempt + 1 == API_MAX_ATTEMPTS:
                logger.error(
                    f"[api-retry] attempt {attempt + 1}/{API_MAX_ATTEMPTS} failed "
                    f"({type(error).__name__}: {error}); giving up, {remaining} tries left"
                )
                raise
            delay = API_RETRY_DELAYS[attempt]
            logger.warning(
                f"[api-retry] attempt {attempt + 1}/{API_MAX_ATTEMPTS} failed "
                f"({type(error).__name__}: {error}); retrying in {delay}s, {remaining} tries left"
            )
            time.sleep(delay)


class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        self.config = config or OpenRouterConfig()

    @property
    def extractable_json_structured_output(self):
        return re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

    def _api_call_with_retry(self, client: OpenAI, messages: list) -> str:
        """Make up to ten quiet attempts, then preserve and raise the final API failure."""
        def request():
            resp = client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extra_body=self.config.extra_body,
            )
            return resp.choices[0].message.content

        return call_with_api_retries(request)


class SemanticEpisodicAssociativeLearner(BaseAgent):
    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            max_retries=0,
        )

    def generate_content(self, system_instruction: str, image: Optional[Image.Image], text: str) -> str:
        content = _build_content(image, text) if image else [{"type": "text", "text": text}]
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content},
        ]
        return self._api_call_with_retry(self.client, messages)


class VLMAgent(BaseAgent):
    def __init__(
        self,
        config: Optional[OpenRouterConfig] = None,
        context_policy: ContextPolicy = ContextPolicy(),
    ) -> None:
        super().__init__(config)
        self.context_policy = validate_context_policy(context_policy)
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            max_retries=0,
        )
        self.history: List[Dict[str, Any]] = []
        self.episodic_memory: str = ""
        self.semantic_log = SemanticLog("", self.context_policy)
        logger.info(f"VLMAgent initialized with model: {self.config.model_id}")

    def reset_history(self):
        self.history = []

    def send_message(self, content: list) -> str:
        self.history.append({"role": "user", "content": content})
        messages = [
            {"role": "system", "content": SYS_INST_VLM_LEAN},
            *self._outbound_history(),
        ]
        # Around the retry helper, not inside it, so the retries of a flaky actor call are billed to
        # the actor too - they are what the server was really charged for.
        with token_meter.role(token_meter.ROLE_ACTOR):
            reply = self._api_call_with_retry(self.client, messages)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _outbound_history(self) -> list[dict[str, Any]]:
        """Return an API-only history view with old user images removed for A6.

        The retained history remains byte-for-byte untouched because the episodic learner and
        diagnostics consume it independently of the actor's outbound context.
        """
        keep = self.context_policy.actor_image_history
        if keep is None:
            return self.history

        user_indices = [
            index for index, message in enumerate(self.history)
            if message.get("role") == "user"
        ]
        newest = set(user_indices[-keep:])
        outbound: list[dict[str, Any]] = []
        for index, message in enumerate(self.history):
            content = message.get("content")
            if (
                index not in newest
                and message.get("role") == "user"
                and isinstance(content, list)
            ):
                filtered = [
                    copy.deepcopy(part)
                    for part in content
                    if not (isinstance(part, dict) and part.get("type") == "image_url")
                ]
                outbound.append({**message, "content": filtered})
            else:
                outbound.append(message)
        return outbound

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
                 advisor_backend: Literal['qwen', 'claude-cli'] = 'qwen',
                 map_output_dir: Optional[str] = None,
                 run_dir: Optional[str] = None,
                 context_policy: ContextPolicy = ContextPolicy()) -> None:

        self.context_policy = validate_context_policy(context_policy)
        self.vlm_agent = VLMAgent(vlm_config, context_policy=self.context_policy)
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
        # Which slamtest output dir the graph arm loads its map (topology/annotations/grid) from.
        # None -> StoreMap's DEFAULT_OUTPUT_DIR (slamtest/output). Threaded from the entry points'
        # --output-dir so a run can be pointed at an alternate map without touching the default.
        self._map_output_dir = map_output_dir
        # Runtime/debug artifacts belong to the attempt, never to the shared frozen map. The
        # environment fallback lets programmatic callers join the orchestrator's context without
        # another argument; no context preserves the legacy standalone filenames.
        active_run_dir = run_dir or os.environ.get("SARI_RUN_DIR")
        self._run_dir = os.path.abspath(active_run_dir) if active_run_dir else None
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
        # Rendered from the map dir THIS agent navigates (see agent_core/memory), so a run pointed
        # at an alternate map can't be given prose describing the default one.
        self.vlm_agent.semantic_log = SemanticLog(
            base_semantic_memory(self._map_output_dir), self.context_policy
        )
        logger.info("Base semantic memory set for VLMAgent.")

    def set_episodic_memory(self, episodic_memory: str) -> None:
        self.vlm_agent.episodic_memory = episodic_memory
        logger.info(f"Episodic memory updated: {episodic_memory}")

    def _run_artifact(self, name: str) -> str:
        return os.path.join(self._run_dir, name) if self._run_dir else name

    @staticmethod
    def _write_text_atomic(path: str, content: str) -> None:
        """Publish a complete snapshot so live readers never observe a half-written memory file."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                                         dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

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

    def _restore_hands_after_inspection(self):
        """Restore canonical transforms and clear any closed-but-empty ("ghost") grippers.

        Unity tracks the grip toggle separately from the held-item attachment.  If an inspection
        move loses the item, the old state channel can therefore leave a hand closed and report it
        as occupied forever.  That prevents every later grab.  Current simulator builds expose both
        signals; after ResetHands, open only a gripper that is closed *without* an attached item.
        Legitimate carried items remain untouched.
        """
        self._set_hands(True)
        self._hand_pose = None
        from sim.env import (
            ResetHands,
            ToggleLeftGrip,
            ToggleRightGrip,
            TransformHands,
        )

        state = ResetHands()
        recovered_ghost_grips = []
        toggles = {"left": ToggleLeftGrip, "right": ToggleRightGrip}
        for side in ("left", "right"):
            holding_key = f"{side}HoldingItem"
            closed_key = f"{side}GripClosedState"
            # Older simulator replies do not expose attachment state.  Do not guess there: opening
            # a genuinely held item would be worse than retaining the legacy behavior.
            if holding_key not in state or closed_key not in state:
                continue
            if state[closed_key] and not state[holding_key]:
                toggles[side]()
                recovered_ghost_grips.append(side)

        if recovered_ghost_grips:
            zero = (0, 0, 0)
            state = TransformHands(zero, zero, zero, zero)
            still_ghosted = [
                side for side in recovered_ghost_grips
                if state.get(f"{side}GripClosedState")
                and not state.get(f"{side}HoldingItem")
            ]
            if still_ghosted:
                raise RuntimeError(
                    "could not open closed-but-empty inspection hand(s): "
                    + ", ".join(still_ghosted)
                )

        self._hand_pose = "rest"
        return {
            "restored": True,
            "recovered_ghost_grips": recovered_ghost_grips,
            "hands": {
                side: {
                    "translation": state.get(f"{side}Translation"),
                    "rotation": state.get(f"{side}Rotation"),
                    "gripped": state.get(f"{side}GrippedState"),
                    "holding_item": state.get(f"{side}HoldingItem"),
                    "grip_closed": state.get(f"{side}GripClosedState"),
                }
                for side in ("left", "right")
            },
        }

    # ---- Phase 4.2 graph-navigation dispatcher -------------------------------------------

    def _graph_nav_session(self):
        """Lazy StoreMap+NavSession+resolver backend - constructed on first navigation
        dispatch because NavSession needs the sim live."""
        if self._graph_nav is None:
            from nav.store_map import StoreMap, NavSession
            from sim.env import default_uri
            sm = (StoreMap(output_dir=self._map_output_dir) if self._map_output_dir
                  else StoreMap())
            # Without an explicit uri, NavSession falls back to capture_walk's parser default
            # (ws://localhost:8080/commands) instead of this attempt's leased sandbox port - every
            # graph-nav command then dials a port nothing is listening on and times out on the
            # handshake, stranding the agent at its starting checkpoint. default_uri() reads
            # SARI_WS_URI, which the runner sets per-lease (see sari_bench/runner.py).
            nav = NavSession(sm, uri=default_uri(), stow_hands=False)
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
                with token_meter.role(token_meter.ROLE_RESOLVER):
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

        shots_dir = (os.path.join(self._run_dir, "advised_nav") if self._run_dir
                     else os.path.join(sm.output_dir, "advised_nav"))
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
                with token_meter.role(token_meter.ROLE_ADVISOR):
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
        with token_meter.role(token_meter.ROLE_SEMANTIC):
            return self.associative_learner._api_call_with_retry(
                self.associative_learner.client,
                [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content},
                ],
            )

    def _call_episodic(self, history_text: str) -> str:
        # Split from the semantic pass even though both run on the same learner and the same client:
        # they are separately removable, and telling them apart is the point of the roles.
        with token_meter.role(token_meter.ROLE_EPISODIC):
            return self.associative_learner._api_call_with_retry(
                self.associative_learner.client,
                [
                    {"role": "system", "content": SYS_INST_ASSOCIATIVE_EPISODIC},
                    {"role": "user", "content": history_text},
                ],
            )

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
                        f"## SEMANTIC MEMORY: {self.vlm_agent.semantic_log.render()}\n"
                        f"## STATE: {state}\n")

            semantic_response_text = self._call_associative(
                SYS_INST_ASSOCIATIVE_SEMANTIC, screenshot, user_msg
            )
            print(f"SEMANTIC LEARNER RESPONSE: {semantic_response_text}")

            semantic_response = _parse_semantic_response(
                self.associative_learner.extractable_json_structured_output,
                semantic_response_text
            )
            new_semantic_memory = semantic_response['new_semantic_memory']
            recall = semantic_response['recall']
            agent_mode = semantic_response['mode']
            # MODE/ACTION HORIZON edit (2026-07-23): the single next step the learner committed to, fed
            # to the actor so it acts on THAT step and does not skip ahead to the grab (sys_inst rule 4).
            # Soft .get: an older learner reply without the field just omits the line below.
            next_action = semantic_response.get('next_action')

            # Orchestrator overrides leave STOP/navigation alone. The graph location gate takes
            # precedence; otherwise an inspect leg with an already-held item may promote perception
            # to manipulation so the actor can choose a safe reorientation.
            force_navigate = bool(request.get('force_navigate'))
            force_manipulate = bool(request.get('force_manipulate'))
            inspect_mode = request.get("inspect_mode")
            agent_mode = _resolve_agent_mode(
                agent_mode, force_navigate, force_manipulate, inspect_mode=inspect_mode)
            # Held inspection STOP must preserve the exact presented/rotated item pose until the
            # orchestrator verifies this timestep's frozen screenshot. run_leg's inspect-only
            # finally restores BOTH hand translations and rotations after the verdict/leg exit.
            # All non-inspection STOP paths retain their existing pre-return REST behavior below.
            if agent_mode == "STOP" and inspect_mode == "held":
                return _stop_response(semantic_response, semantic_response_text)
            if force_navigate and agent_mode == "navigation":
                # Drop a stale MOVE verdict so forced navigation performs a candidate hop.
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

            if agent_mode == "STOP":
                return _stop_response(semantic_response, semantic_response_text)
            available_actions = _available_actions(
                agent_mode,
                held_item_inspection=(inspect_mode == "held" and agent_mode == "manipulation"),
            )

            self.vlm_agent.semantic_log.append(
                self._semantic_tag(timestep), new_semantic_memory
            )

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

            # The actor's action dict is parsed DOWNSTREAM (subtask_agents / eval_pickup) off the
            # returned 'text'. Parsing it here as well was dead - the result (response_json) was
            # never read - and its only live effect was to crash the entire step when a single-
            # quoted value carried an apostrophe ("Kellogg's Coco Pops"). Dropped 2026-07-24.
            episodic_response_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
            episodic_response = _safe_ast_dict(
                self.associative_learner.extractable_json_structured_output,
                episodic_response_text, _EPISODIC_FALLBACK, tag="episodic")

            episodic_memory = (f"@ timestep {timestep}:\n"
                               f"## DENSE SUMMARY: {episodic_response['dense_summary']}\n"
                               f"## WHAT WORKED: {episodic_response['what_worked']}\n"
                               f"## WHAT TO AVOID: {episodic_response['what_to_avoid']}\n")
            self.set_episodic_memory(episodic_memory)

        else:
            user_msg = (f"## MAIN TASK: {main_task}\n"
                        f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## SEMANTIC MEMORY: {self.vlm_agent.semantic_log.render()}\n"
                        f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
                        f"## STATE: {state}\n")

            semantic_response_text = self._call_associative(
                SYS_INST_ASSOCIATIVE_SEMANTIC, screenshot, user_msg
            )
            semantic_response = _parse_semantic_response(
                self.associative_learner.extractable_json_structured_output,
                semantic_response_text
            )
            new_semantic_memory = semantic_response['new_semantic_memory']
            recall = semantic_response['recall']
            agent_mode = semantic_response['mode']
            # MODE/ACTION HORIZON edit (2026-07-23): single next step, fed to the actor (sys_inst rule 4).
            next_action = semantic_response.get('next_action')

            self.vlm_agent.semantic_log.append(
                self._semantic_tag(timestep), new_semantic_memory
            )
            print(f"SEMANTIC LEARNER RESPONSE: {semantic_response}")

            # Same override resolution as timestep 1. Keeping this after the unchanged learner call
            # means the mode-classifier prompt and response contract are untouched.
            force_navigate = bool(request.get('force_navigate'))
            force_manipulate = bool(request.get('force_manipulate'))
            inspect_mode = request.get("inspect_mode")
            agent_mode = _resolve_agent_mode(
                agent_mode, force_navigate, force_manipulate, inspect_mode=inspect_mode)
            # See timestep 1: do not erase the evidence pose before the STOP guard consumes the
            # frame. Inspect-leg cleanup owns the eventual canonical translation/rotation restore.
            if agent_mode == "STOP" and inspect_mode == "held":
                return _stop_response(semantic_response, semantic_response_text)
            if force_navigate and agent_mode == "navigation":
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

            if agent_mode == "STOP":
                return _stop_response(semantic_response, semantic_response_text)
            available_actions = _available_actions(
                agent_mode,
                held_item_inspection=(inspect_mode == "held" and agent_mode == "manipulation"),
            )

            next_action_line = (f"## THIS STEP'S INTENDED ACTION: {next_action}\n"
                                if next_action and not nav_note else "")
            episodic_line = (
                f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
                if self.context_policy.episodic_in_actor else ""
            )
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"{next_action_line}"
                        f"{episodic_line}"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}"
                        f"{nav_note}")

            vlm_content = _build_content(screenshot, "## CURRENT OBSERVATION\n" + user_msg)
            response_text = self.vlm_agent.send_message(vlm_content)

            # The actor's action dict is parsed DOWNSTREAM (subtask_agents / eval_pickup) off the
            # returned 'text'. Parsing it here as well was dead - the result (response_json) was
            # never read - and its only live effect was to crash the entire step when a single-
            # quoted value carried an apostrophe ("Kellogg's Coco Pops"). Dropped 2026-07-24.
            episodic_response_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
            episodic_response = _safe_ast_dict(
                self.associative_learner.extractable_json_structured_output,
                episodic_response_text, _EPISODIC_FALLBACK, tag="episodic")

            episodic_memory = (f"@ timestep {timestep}:\n"
                               f"## DENSE SUMMARY: {episodic_response['dense_summary']}\n"
                               f"## WHAT WORKED: {episodic_response['what_worked']}\n"
                               f"## WHAT TO AVOID: {episodic_response['what_to_avoid']}\n")
            self.set_episodic_memory(episodic_memory)

        self._write_text_atomic(
            self._run_artifact("semantic_memory.txt"), self.vlm_agent.semantic_log.render())
        self._write_text_atomic(
            self._run_artifact("episodic_memory.txt"), self.vlm_agent.episodic_memory)

        return {
            'halt': False,
            'nav_note': nav_note,  # non-empty iff the graph dispatcher drove this step
            'text': response_text,
            'agent_mode': agent_mode,
            'semantic': semantic_response_text,   # mode router: mode / recall / new_semantic_memory
            'episodic': episodic_response_text,    # reflection: dense_summary / what_worked / what_to_avoid
        }
