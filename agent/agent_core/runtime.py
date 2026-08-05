"""Composition root and single-step pipeline for the embodied agent."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import os
from typing import Literal, Optional

from loguru import logger
from PIL import Image

from agent_core import token_meter
from agent_core.actors import AssociativeLearner, VLMAgent
from agent_core.context_policy import ContextPolicy, validate_context_policy
from agent_core.contracts import (
    AgentMode,
    EpisodicReflection,
    SemanticDecision,
    available_actions,
    parse_episodic_reflection,
    parse_semantic_decision,
    reach_move_steps,
    resolve_agent_mode,
    stop_response,
)
from agent_core.hands import HandController
from agent_core.llm import LLMConfig, build_content
from agent_core.memory_runtime import MemoryRuntime
from agent_core.navigation import GraphNavigator
from agent_core.sys_inst import SYS_INST_ASSOCIATIVE_EPISODIC, SYS_INST_ASSOCIATIVE_SEMANTIC


@dataclass(frozen=True)
class StepRequest:
    task: str
    nav_goal: str
    raw_state: object
    state_text: str
    screenshot: Image.Image
    force_navigate: bool
    force_manipulate: bool
    inspect_mode: Optional[str]
    timestep: int

    @classmethod
    def from_mapping(cls, request: dict, timestep: int) -> "StepRequest":
        task = request["task"]
        image_bytes = base64.b64decode(str(request["image"]).encode("utf-8"))
        screenshot = Image.open(BytesIO(image_bytes)).convert("RGB")
        raw_state = request.get("state")
        return cls(
            task=task,
            nav_goal=request.get("nav_goal") or task,
            raw_state=raw_state,
            state_text=str(request["state"]),
            screenshot=screenshot,
            force_navigate=bool(request.get("force_navigate")),
            force_manipulate=bool(request.get("force_manipulate")),
            inspect_mode=request.get("inspect_mode"),
            timestep=timestep,
        )

    @property
    def first_step(self) -> bool:
        return self.timestep == 1

    @property
    def measured_move_steps(self) -> Optional[int]:
        last_reach = (
            self.raw_state.get("last_reach") if isinstance(self.raw_state, dict) else None
        )
        return reach_move_steps(last_reach)


class EmbodiedAgent:
    """Coordinate actor, learner, memory, hands, and navigation services."""

    def __init__(
        self,
        vlm_config: Optional[LLMConfig] = None,
        associative_config: Optional[LLMConfig] = None,
        mode: Literal["base", "lean"] = "base",
        nav_mode: Literal["vlm", "graph", "graph-advised"] = "vlm",
        resolver_backend: Literal["endpoint", "claude-cli"] = "endpoint",
        advisor_backend: Literal["endpoint", "claude-cli"] = "endpoint",
        map_output_dir: Optional[str] = None,
        run_dir: Optional[str] = None,
        context_policy: ContextPolicy = ContextPolicy(),
    ) -> None:
        self.context_policy = validate_context_policy(context_policy)
        self.vlm_agent = VLMAgent(vlm_config, context_policy=self.context_policy)
        self.mode = mode
        self.nav_mode = nav_mode
        self.resolver_backend = resolver_backend
        self.advisor_backend = advisor_backend
        self._map_output_dir = map_output_dir
        active_run_dir = run_dir or os.environ.get("SARI_RUN_DIR")
        self._run_dir = os.path.abspath(active_run_dir) if active_run_dir else None
        self._mem_leg = None

        self._hands = HandController()
        self._navigation = GraphNavigator(
            self._hands,
            nav_mode=nav_mode,
            resolver_backend=resolver_backend,
            advisor_backend=advisor_backend,
            map_output_dir=map_output_dir,
            run_dir=self._run_dir,
        )
        self._memory = MemoryRuntime(
            self.vlm_agent,
            context_policy=self.context_policy,
            map_output_dir=map_output_dir,
            run_dir=self._run_dir,
        )
        self._runtime_initialized = True

        if mode == "lean":
            self.associative_learner = AssociativeLearner(associative_config)
            self.set_semantic_memory()

    # ------------------------------------------------------------------ services

    def _hand_service(self) -> HandController:
        service = self.__dict__.get("_hands")
        if service is None:
            service = HandController()
            self.__dict__["_hands"] = service
        return service

    def _navigation_service(self) -> GraphNavigator:
        service = self.__dict__.get("_navigation")
        if service is None:
            service = GraphNavigator(
                self._hand_service(),
                nav_mode=self.__dict__.get("nav_mode", "vlm"),
                resolver_backend=self.__dict__.get("resolver_backend", "endpoint"),
                advisor_backend=self.__dict__.get("advisor_backend", "endpoint"),
                map_output_dir=self.__dict__.get("_map_output_dir"),
                run_dir=self.__dict__.get("_run_dir"),
            )
            self.__dict__["_navigation"] = service
        service.hands = self._hand_service()
        service.nav_mode = self.__dict__.get("nav_mode", service.nav_mode)
        service.resolver_backend = self.__dict__.get(
            "resolver_backend", service.resolver_backend
        )
        service.advisor_backend = self.__dict__.get("advisor_backend", service.advisor_backend)
        service.map_output_dir = self.__dict__.get("_map_output_dir")
        service.run_dir = self.__dict__.get("_run_dir")
        return service

    def _memory_service(self) -> MemoryRuntime:
        service = self.__dict__.get("_memory")
        if service is None:
            service = MemoryRuntime(
                self.__dict__.get("vlm_agent"),
                context_policy=self.__dict__.get("context_policy", ContextPolicy()),
                map_output_dir=self.__dict__.get("_map_output_dir"),
                run_dir=self.__dict__.get("_run_dir"),
                leg=self.__dict__.get("_mem_leg"),
            )
            self.__dict__["_memory"] = service
        service.vlm_agent = self.__dict__.get("vlm_agent")
        service.context_policy = self.__dict__.get("context_policy", service.context_policy)
        service.map_output_dir = self.__dict__.get("_map_output_dir")
        service.run_dir = self.__dict__.get("_run_dir")
        service.leg = self.__dict__.get("_mem_leg")
        return service

    # --------------------------------------------------------- compatibility API

    @property
    def _hands_active(self):
        return self._hand_service().active

    @_hands_active.setter
    def _hands_active(self, value) -> None:
        self._hand_service().active = value

    @property
    def _hand_pose(self):
        return self._hand_service().pose

    @_hand_pose.setter
    def _hand_pose(self, value) -> None:
        self._hand_service().pose = value

    @property
    def _graph_nav(self):
        return self._navigation_service().graph_nav

    @_graph_nav.setter
    def _graph_nav(self, value) -> None:
        self._navigation_service().graph_nav = value

    @property
    def _advised_llm_calls(self):
        return self._navigation_service().advised_llm_calls

    @_advised_llm_calls.setter
    def _advised_llm_calls(self, value) -> None:
        self._navigation_service().advised_llm_calls = value

    @property
    def _advised_stats(self):
        return self._navigation_service().advised_stats

    @_advised_stats.setter
    def _advised_stats(self, value) -> None:
        self._navigation_service().advised_stats = value

    @property
    def _advised_shot_idx(self):
        return self._navigation_service().advised_shot_idx

    @_advised_shot_idx.setter
    def _advised_shot_idx(self, value) -> None:
        self._navigation_service().advised_shot_idx = value

    @property
    def _nav_candidates(self):
        return self._navigation_service().candidates

    @_nav_candidates.setter
    def _nav_candidates(self, value) -> None:
        self._navigation_service().candidates = value

    @property
    def _nav_visited(self):
        return self._navigation_service().visited

    @_nav_visited.setter
    def _nav_visited(self, value) -> None:
        self._navigation_service().visited = value

    @property
    def _nav_task(self):
        return self._navigation_service().task

    @_nav_task.setter
    def _nav_task(self, value) -> None:
        self._navigation_service().task = value

    @property
    def _nav_seeded(self):
        return self._navigation_service().seeded

    @_nav_seeded.setter
    def _nav_seeded(self, value) -> None:
        self._navigation_service().seeded = value

    @property
    def _nav_seeded_name(self):
        return self._navigation_service().seeded_name

    @_nav_seeded_name.setter
    def _nav_seeded_name(self, value) -> None:
        self._navigation_service().seeded_name = value

    @property
    def _nav_resolution(self):
        return self._navigation_service().resolution

    @_nav_resolution.setter
    def _nav_resolution(self, value) -> None:
        self._navigation_service().resolution = value

    def set_semantic_memory(self) -> None:
        self._memory_service().reset_semantic()

    def set_episodic_memory(self, episodic_memory: str) -> None:
        self._memory_service().set_episodic(episodic_memory)

    def _run_artifact(self, name: str) -> str:
        return self._memory_service().artifact_path(name)

    @staticmethod
    def _write_text_atomic(path: str, content: str) -> None:
        MemoryRuntime.write_text_atomic(path, content)

    def _semantic_tag(self, timestep: int) -> str:
        return self._memory_service().semantic_tag(timestep)

    def _set_hands(self, active: bool) -> None:
        self._hand_service().set_active(active)

    def _set_hand_pose(self, pose: str) -> None:
        self._hand_service().set_pose(pose)

    def _invalidate_hand_pose(self) -> None:
        self._hand_service().invalidate_pose()

    def _restore_hands_after_inspection(self) -> dict:
        return self._hand_service().restore_after_inspection()

    def _graph_nav_session(self):
        return self._navigation_service().session()

    def seed_nav_candidates(self, candidates, target_name=None) -> None:
        self._navigation_service().seed_candidates(candidates, target_name)

    def begin_leg(self, candidates, target_name, leg_index: int) -> int:
        """Reset leg-local conversation/navigation state and return a semantic-log mark."""
        self.vlm_agent.reset_history()
        self.seed_nav_candidates(candidates, target_name)
        self._mem_leg = leg_index
        self._memory_service().leg = leg_index
        return self.vlm_agent.semantic_log.mark()

    def _graph_navigate(self, main_task: str, nav_goal: Optional[str] = None):
        result = self._navigation_service().navigate(main_task, nav_goal)
        return result.note, result.image_bytes

    def _advised_goto(self, store_map, nav, target, nav_goal):
        return self._navigation_service().advised_goto(store_map, nav, target, nav_goal)

    def _navigate_to_counter(self):
        result = self._navigation_service().navigate_to_counter()
        return result.note, result.image_bytes

    def _checkout_held_item(self, hand: str = "auto") -> dict:
        return self.checkout_held_item(hand)

    def checkout_held_item(self, hand: str = "auto") -> dict:
        return self._navigation_service().checkout_held_item(hand)

    def restore_hands_after_inspection(self) -> dict:
        return self._restore_hands_after_inspection()

    def close(self) -> None:
        """Release the lazily-created simulator navigation session, if any."""
        navigation = self.__dict__.get("_navigation")
        if navigation is not None and navigation.graph_nav:
            navigation.graph_nav[1].close()

    def _metric_approach(self, move_steps: int):
        result = self._navigation_service().metric_approach(move_steps)
        return result.note, result.image_bytes

    # --------------------------------------------------------------- LLM passes

    def _call_associative(
        self, system_instruction: str, image: Optional[Image.Image], text: str
    ) -> str:
        content = build_content(image, "## CURRENT OBSERVATION\n", text)
        with token_meter.role(token_meter.ROLE_SEMANTIC):
            return self.associative_learner._api_call_with_retry(
                self.associative_learner.client,
                [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content},
                ],
            )

    def _call_episodic(self, history_text: str) -> str:
        with token_meter.role(token_meter.ROLE_EPISODIC):
            return self.associative_learner._api_call_with_retry(
                self.associative_learner.client,
                [
                    {"role": "system", "content": SYS_INST_ASSOCIATIVE_EPISODIC},
                    {"role": "user", "content": history_text},
                ],
            )

    def _semantic_prompt(self, step: StepRequest) -> str:
        if step.first_step:
            return (
                f"## CURRENT TIMESTEP: {step.timestep}\n"
                f"## MAIN TASK: {step.task}\n"
                f"## SEMANTIC MEMORY: {self.vlm_agent.semantic_log.render()}\n"
                f"## STATE: {step.state_text}\n"
            )
        return (
            f"## MAIN TASK: {step.task}\n"
            f"## CURRENT TIMESTEP: {step.timestep}\n"
            f"## SEMANTIC MEMORY: {self.vlm_agent.semantic_log.render()}\n"
            f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
            f"## STATE: {step.state_text}\n"
        )

    def _actor_prompt(
        self,
        step: StepRequest,
        decision: SemanticDecision,
        mode: str,
        actions: str,
        nav_note: str,
    ) -> str:
        next_action_line = (
            f"## THIS STEP'S INTENDED ACTION: {decision.next_action}\n"
            if decision.next_action and not nav_note
            else ""
        )
        if step.first_step:
            return (
                f"## CURRENT TIMESTEP: {step.timestep}\n"
                f"## MAIN TASK: {step.task}\n"
                f"## RECALL FROM SEMANTIC MEMORY: {decision.recall}\n"
                f"{next_action_line}"
                f"## STATE: {step.state_text}\n"
                f"## AGENT MODE: {mode}\n"
                f"## AVAILABLE ACTIONS:\n{actions}"
                f"{nav_note}"
            )
        episodic_line = (
            f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
            if self.context_policy.episodic_in_actor
            else ""
        )
        return (
            f"## CURRENT TIMESTEP: {step.timestep}\n"
            f"## RECALL FROM SEMANTIC MEMORY: {decision.recall}\n"
            f"{next_action_line}"
            f"{episodic_line}"
            f"## STATE: {step.state_text}\n"
            f"## AGENT MODE: {mode}\n"
            f"## AVAILABLE ACTIONS:\n{actions}"
            f"{nav_note}"
        )

    @staticmethod
    def _format_episodic(timestep: int, reflection: EpisodicReflection) -> str:
        return (
            f"@ timestep {timestep}:\n"
            f"## DENSE SUMMARY: {reflection.dense_summary}\n"
            f"## WHAT WORKED: {reflection.what_worked}\n"
            f"## WHAT TO AVOID: {reflection.what_to_avoid}\n"
        )

    def _stop_and_persist(self, decision: SemanticDecision, semantic_text: str) -> dict:
        # STOP is a real final observation. Persisting it avoids the old timestep-dependent
        # behavior where later steps mutated memory in-process but no STOP path wrote artifacts.
        # object.__new__-constructed unit-test doubles have no artifact context. A real
        # runtime is marked by __init__; tests that deliberately supply a run directory
        # still exercise persistence without leaking files into the process CWD.
        if self.__dict__.get("_runtime_initialized") or self.__dict__.get("_run_dir"):
            self._memory_service().persist()
        return stop_response(decision, semantic_text)

    # ----------------------------------------------------------- single pipeline

    def execute_lean(self, request: dict, timestep: int) -> dict:
        step = StepRequest.from_mapping(request, timestep)
        semantic_text = self._call_associative(
            SYS_INST_ASSOCIATIVE_SEMANTIC, step.screenshot, self._semantic_prompt(step)
        )
        decision = parse_semantic_decision(
            self.associative_learner.extractable_json_structured_output, semantic_text
        )
        logger.info(f"[semantic-learner] {decision.as_dict()}")

        mode = resolve_agent_mode(
            decision.mode,
            step.force_navigate,
            step.force_manipulate,
            inspect_mode=step.inspect_mode,
        )
        self.vlm_agent.semantic_log.append(
            self._semantic_tag(timestep), decision.new_semantic_memory
        )

        # Held inspection evidence must remain posed until the guard consumes the frozen frame.
        if mode == AgentMode.STOP.value and step.inspect_mode == "held":
            return self._stop_and_persist(decision, semantic_text)

        move_steps = step.measured_move_steps
        if step.force_navigate and mode == AgentMode.NAVIGATION.value:
            move_steps = None

        nav_note = ""
        screenshot = step.screenshot
        if mode == AgentMode.NAVIGATION.value and self.nav_mode in ("graph", "graph-advised"):
            nav_note, fresh_png = (
                self._metric_approach(move_steps)
                if move_steps is not None
                else self._graph_navigate(step.task, step.nav_goal)
            )
            if fresh_png is not None:
                screenshot = Image.open(BytesIO(fresh_png)).convert("RGB")
            mode = AgentMode.PERCEPTION.value

        if mode == AgentMode.MANIPULATION.value:
            self._invalidate_hand_pose()
        else:
            self._set_hand_pose("rest")

        if mode == AgentMode.STOP.value:
            return self._stop_and_persist(decision, semantic_text)

        actions = available_actions(
            mode,
            held_item_inspection=(
                step.inspect_mode == "held" and mode == AgentMode.MANIPULATION.value
            ),
        )
        actor_prompt = self._actor_prompt(step, decision, mode, actions, nav_note)
        response_text = self.vlm_agent.send_message(
            build_content(screenshot, "## CURRENT OBSERVATION\n" + actor_prompt)
        )
        logger.info(f"[actor] {response_text}")

        episodic_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
        reflection = parse_episodic_reflection(
            self.associative_learner.extractable_json_structured_output, episodic_text
        )
        self.set_episodic_memory(self._format_episodic(timestep, reflection))
        self._memory_service().persist()

        return {
            "halt": False,
            "nav_note": nav_note,
            "text": response_text,
            "agent_mode": mode,
            "semantic": semantic_text,
            "episodic": episodic_text,
        }
