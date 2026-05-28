from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import re
from PIL import Image
from loguru import logger
import ast
import math

load_dotenv('C:\\Sari\\sari-agent-1.0\\api.env')

from sys_inst import (
    SYS_INST_ASSOCIATIVE_SEMANTIC,
    SYS_INST_ASSOCIATIVE_EPISODIC,
    SYS_INST_VLM_LEAN
)
from memory import BASE_SEMANTIC_MEMORY
from actions_str import (
    NAVIGATION_ACTIONS,
    PERCEPTION_ACTIONS,
    MANIPULATION_ACTIONS,
)

from depth import estimate_depth


@dataclass
class OpenRouterConfig:
    model_id: str = 'google/gemini-2.5-flash-preview-05-20'
    temperature: float = 0.5
    mode: Literal['base', 'lean'] = 'base'
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))


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
    @abstractmethod
    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        self.config = config or OpenRouterConfig()

    @property
    def extractable_json_structured_output(self):
        return re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

class NavigationPlanError(Exception):
    pass


class HumanNavigationReasoner:
    """
    Prompts the operator for waypoints via stdin.
    Used to validate the waypoint-following architecture before committing to an LLM reasoner.
    """
    ARRIVAL_RADIUS = 0.5  # world units

    def plan(self, current_x, current_z, current_yaw, target_x, target_z,
             target_shelf="", known_obstacles="", navigation_memory="") -> list[dict]:
        print("\n" + "=" * 60)
        print("[NAVIGATION REASONER] Input required.")
        print(f"  Current position : x={current_x:.2f}, z={current_z:.2f}, yaw={current_yaw:.1f}°")
        print(f"  Target position  : x={target_x:.2f}, z={target_z:.2f}  ({target_shelf})")
        if known_obstacles:
            print(f"  Known obstacles  :\n{known_obstacles}")
        if navigation_memory:
            print(f"  Past navigation records:\n{navigation_memory}")
        print()
        print("Enter 2-5 waypoints as a JSON list.")
        print("Each rationale should be an ACTION HINT — facing direction + steps, e.g.:")
        print('  [{"x": 2.0, "z": 4.5, "rationale": "Pan right to yaw≈0, move_forward ~8 steps"},')
        print('   {"x": 2.0, "z": 7.6, "rationale": "Continue move_forward ~31 steps"},')
        print('   {"x": 3.01, "z": 7.65, "rationale": "Pan right to yaw≈270, move_forward ~10 steps"}]')
        print("Then press Enter twice.")
        print("=" * 60)

        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)

        raw = " ".join(lines).strip()
        try:
            waypoints = json.loads(raw)
            if not isinstance(waypoints, list):
                raise ValueError("Expected a JSON array.")
            return waypoints
        except Exception as e:
            raise NavigationPlanError(f"Could not parse your waypoint input: {e}\nRaw: {raw[:200]}")

    @staticmethod
    def distance_2d(ax, az, bx, bz) -> float:
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)

class SemanticEpisodicAssociativeLearner(BaseAgent):
    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config.api_key,
        )

    def generate_content(self, system_instruction: str, image: Optional[Image.Image], text: str) -> str:
        content = _build_content(image, text) if image else [{"type": "text", "text": text}]
        resp = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content},
            ],
            temperature=self.config.temperature,
        )
        return resp.choices[0].message.content


class VLMAgent(BaseAgent):
    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
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
        resp = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": SYS_INST_VLM_LEAN},
                *self.history,
            ],
            temperature=self.config.temperature,
        )
        reply = resp.choices[0].message.content
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


class EmbodiedAgent:
    def __init__(self, vlm_config: Optional[OpenRouterConfig] = None,
                 associative_config: Optional[OpenRouterConfig] = None,
                 mode: Literal['base', 'lean'] = 'base') -> None:

        self.vlm_agent = VLMAgent(vlm_config)
        self.mode = mode

        if mode == 'lean':
            self.associative_learner = SemanticEpisodicAssociativeLearner(associative_config)
            self.set_semantic_memory()
            self.navigation_reasoner = HumanNavigationReasoner()
            self.current_waypoints: list[dict] = []
            self.current_waypoint_idx: int = 0
            self.current_nav_target: Optional[tuple[float, float]] = None
            self.navigation_memory: str = ""          # separate from semantic memory; read by nav reasoner and VLM (nav mode only)

    def set_semantic_memory(self) -> None:
        self.vlm_agent.base_semantic_memory = BASE_SEMANTIC_MEMORY
        logger.info("Base semantic memory set for VLMAgent.")

    def set_episodic_memory(self, entry: str, max_entries: int = 5) -> None:
        existing = [e for e in self.vlm_agent.episodic_memory.strip().split("\n\n") if e]
        existing.append(entry)
        self.vlm_agent.episodic_memory = "\n\n".join(existing[-max_entries:])
        logger.info(f"Episodic memory updated ({len(existing[-max_entries:])} entries).")

    def _check_waypoint_advance(self, ax: float, az: float):
        if not self.current_waypoints or self.current_waypoint_idx >= len(self.current_waypoints):
            return
        wp = self.current_waypoints[self.current_waypoint_idx]
        if self.navigation_reasoner.distance_2d(ax, az, wp['x'], wp['z']) <= self.navigation_reasoner.ARRIVAL_RADIUS:
            logger.info(f"Waypoint {self.current_waypoint_idx + 1}/{len(self.current_waypoints)} reached.")
            self.current_waypoint_idx += 1

    def _get_current_waypoint_text(self) -> str:
        if not self.current_waypoints or self.current_waypoint_idx >= len(self.current_waypoints):
            return ""
        wp = self.current_waypoints[self.current_waypoint_idx]
        n, total = self.current_waypoint_idx + 1, len(self.current_waypoints)
        return (f"## NAVIGATION WAYPOINT ({n}/{total}): "
                f"Move to (x={wp['x']}, z={wp['z']}). "
                f"Action hint: {wp.get('rationale', '')}\n")

    @staticmethod
    def _extract_nav_target(recall: str) -> Optional[tuple[float, float]]:
        m = re.search(r'translation\s*:\s*\(\s*([0-9.]+)\s*,\s*[0-9.]+\s*,\s*([0-9.]+)\s*\)', recall)
        return (float(m.group(1)), float(m.group(2))) if m else None

    @staticmethod
    def _extract_target_shelf(recall: str) -> str:
        m = re.search(r'Shelf\s+\d+', recall)
        return m.group(0) if m else "unknown shelf"

    def _maybe_plan_navigation(self, agent_mode: str, recall: str,
                                ax: float, az: float, yaw: float) -> str:
        if agent_mode != "navigation":
            self.current_waypoints = []
            self.current_waypoint_idx = 0
            self.current_nav_target = None
            return ""

        self._check_waypoint_advance(ax, az)
        new_target = self._extract_nav_target(recall)
        need_replan = (not self.current_waypoints) or (
            new_target is not None and new_target != self.current_nav_target
        )

        if not need_replan:
            return self._get_current_waypoint_text()

        if new_target is None:
            logger.warning("Navigation mode but no target coords in recall. Skipping planner.")
            return self._get_current_waypoint_text()

        tx, tz = new_target
        if self.navigation_reasoner.distance_2d(ax, az, tx, tz) <= self.navigation_reasoner.ARRIVAL_RADIUS:
            self.current_waypoints = [{'x': tx, 'z': tz, 'rationale': 'Already at target.'}]
            self.current_waypoint_idx = 0
            self.current_nav_target = new_target
            return self._get_current_waypoint_text()

        try:
            shelf = self._extract_target_shelf(recall)
            waypoints = self.navigation_reasoner.plan(
                ax, az, yaw, tx, tz,
                target_shelf=shelf,
                navigation_memory=self.navigation_memory,   # pass past records to reasoner
            )
            self.current_waypoints = waypoints
            self.current_waypoint_idx = 0
            self.current_nav_target = new_target
            summary = "; ".join(f"({w['x']},{w['z']})" for w in waypoints)
            # append to navigation_memory — NOT semantic memory
            self.navigation_memory += f"[PATH] start=({ax:.1f},{az:.1f}) → {shelf} → waypoints={summary} → outcome=PENDING\n"
            logger.info(f"Navigation plan: {summary}")
        except NavigationPlanError as e:
            logger.error(f"Navigation planner failed: {e}. Proceeding without waypoints.")

        return self._get_current_waypoint_text()

    def _record_collision(self, ax: float, az: float, timestep: int):
        self.navigation_memory += f"[COLLISION] position=({ax:.1f},{az:.1f}) → timestep={timestep}\n"
        logger.warning(f"Collision recorded at ({ax:.1f},{az:.1f}) timestep={timestep}")

    def _compute_depth_hint(self, main_task: str, depth_array) -> str:
        try:
            from perception import detect_object_via_moondream, estimate_steps_from_depth
            item = detect_object_via_moondream(main_task)
            if item is None:
                return ""
            steps = estimate_steps_from_depth(item["box"], depth_array)
            return f"## ESTIMATED HAND STEPS TO TARGET: {steps}\n"
        except Exception as e:
            logger.warning(f"Could not compute depth hint for manipulation: {e}")
            return ""

    def _call_associative(self, system_instruction: str, image: Optional[Image.Image], depth_map: Optional[Image.Image], text: str) -> str:
        content = _build_content(image, "## CURRENT OBSERVATION\n", depth_map, "## CURRENT DEPTH MAP\n", text)
        resp = self.associative_learner.client.chat.completions.create(
            model=self.associative_learner.config.model_id,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content},
            ],
            temperature=self.associative_learner.config.temperature,
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
        )
        return resp.choices[0].message.content

    def execute_lean(self, request, timestep):
        main_task = request['task']
        _raw_state = request['state']
        if isinstance(_raw_state, dict):
            _t = _raw_state.get('translation', (0, 0, 0))
            _r = _raw_state.get('rotation', (0, 0, 0))
            _colliding = _raw_state.get('isColliding', False)
        else:
            _t = _r = (0, 0, 0)
            _colliding = False
        agent_x   = float(_t[0])
        agent_z   = float(_t[2])
        agent_yaw = float(_r[1]) if len(_r) > 1 else 0.0
        state = str(_raw_state)
        if _colliding:
            self._record_collision(agent_x, agent_z, timestep)
        screenshot = str(request['image']).encode('utf-8')
        screenshot = base64.b64decode(screenshot)
        imagebytes = BytesIO(screenshot)
        screenshot = Image.open(imagebytes).convert('RGB')

        depth_image, depth_array = estimate_depth(imagebytes)

        new_semantic_memory = ""
        recall = ""

        if timestep == 1:
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## SEMANTIC MEMORY: {self.vlm_agent.base_semantic_memory}\n"
                        f"## STATE: {state}\n")

            semantic_response_text = self._call_associative(
                SYS_INST_ASSOCIATIVE_SEMANTIC, screenshot, depth_image, user_msg
            )
            print(f"SEMANTIC LEARNER RESPONSE: {semantic_response_text}")

            semantic_response = re.search(
                self.associative_learner.extractable_json_structured_output,
                semantic_response_text
            )[1]
            semantic_response = ast.literal_eval(semantic_response)
            new_semantic_memory = semantic_response['new_semantic_memory']
            recall = semantic_response['recall']
            agent_mode = semantic_response['mode']

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

            self.vlm_agent.base_semantic_memory += f"@ timestep {timestep}: {new_semantic_memory}\n"

            depth_hint = self._compute_depth_hint(main_task, depth_array) if agent_mode == "manipulation" else ""
            waypoint_text = self._maybe_plan_navigation(
                agent_mode=agent_mode, recall=recall,
                ax=agent_x, az=agent_z, yaw=agent_yaw,
            )
            nav_memory_block = (f"## NAVIGATION MEMORY:\n{self.navigation_memory}\n"
                                if agent_mode == "navigation" and self.navigation_memory else "")
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}"
                        f"{depth_hint}"
                        f"{nav_memory_block}"
                        f"{waypoint_text}")

            vlm_content = _build_content(screenshot, "## CURRENT OBSERVATION\n", depth_image, "## CURRENT DEPTH MAP\n" + user_msg)
            response_text = self.vlm_agent.send_message(vlm_content)
            print(f"VLMAgent RESPONSE: {response_text}")

            response_json = re.search(
                self.vlm_agent.extractable_json_structured_output,
                response_text
            )[1]
            response_json = ast.literal_eval(response_json)

            episodic_response_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
            episodic_response = re.search(
                self.associative_learner.extractable_json_structured_output,
                episodic_response_text
            )[1]
            episodic_response = ast.literal_eval(episodic_response)

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
                SYS_INST_ASSOCIATIVE_SEMANTIC, screenshot, depth_image, user_msg
            )
            semantic_response = re.search(
                self.associative_learner.extractable_json_structured_output,
                semantic_response_text
            )[1]
            semantic_response = ast.literal_eval(semantic_response)
            new_semantic_memory = semantic_response['new_semantic_memory']
            recall = semantic_response['recall']
            agent_mode = semantic_response['mode']

            self.vlm_agent.base_semantic_memory += f"@ timestep {timestep}: {new_semantic_memory}\n"
            print(f"SEMANTIC LEARNER RESPONSE: {semantic_response}")

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

            depth_hint = self._compute_depth_hint(main_task, depth_array) if agent_mode == "manipulation" else ""
            waypoint_text = self._maybe_plan_navigation(
                agent_mode=agent_mode, recall=recall,
                ax=agent_x, az=agent_z, yaw=agent_yaw,
            )
            nav_memory_block = (f"## NAVIGATION MEMORY:\n{self.navigation_memory}\n"
                                if agent_mode == "navigation" and self.navigation_memory else "")
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}"
                        f"{depth_hint}"
                        f"{nav_memory_block}"
                        f"{waypoint_text}")

            vlm_content = _build_content(screenshot, "## CURRENT OBSERVATION\n", depth_image, "## CURRENT DEPTH MAP\n" + user_msg)
            response_text = self.vlm_agent.send_message(vlm_content)

            response_json = re.search(
                self.vlm_agent.extractable_json_structured_output,
                response_text
            )[1]
            response_json = ast.literal_eval(response_json)

            episodic_response_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
            episodic_response = re.search(
                self.associative_learner.extractable_json_structured_output,
                episodic_response_text
            )[1]
            episodic_response = ast.literal_eval(episodic_response)

            episodic_memory = (f"@ timestep {timestep}:\n"
                               f"## DENSE SUMMARY: {episodic_response['dense_summary']}\n"
                               f"## WHAT WORKED: {episodic_response['what_worked']}\n"
                               f"## WHAT TO AVOID: {episodic_response['what_to_avoid']}\n")
            self.set_episodic_memory(episodic_memory)

        with open('semantic_memory.txt', 'w', encoding='utf-8') as f:
            f.write(self.vlm_agent.base_semantic_memory)
        with open('episodic_memory.txt', 'w', encoding='utf-8') as f:
            f.write(self.vlm_agent.episodic_memory)

        return {
            'halt': False,
            'text': response_text,
            'agent_mode': agent_mode,
        }
