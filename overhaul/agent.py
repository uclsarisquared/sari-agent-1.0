from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import os
import base64
import json
import math
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import re
from PIL import Image
from loguru import logger
import ast

load_dotenv('C:\\Sari\\sari-agent-1.0\\api.env')

from sys_inst import (
    SYS_INST_ASSOCIATIVE_SEMANTIC,
    SYS_INST_ASSOCIATIVE_EPISODIC,
    SYS_INST_VLM_LEAN
)
from memory import BASE_SEMANTIC_MEMORY
from actions_str import (
    NAVIGATION_ACTIONS,
    SCAN_ACTIONS,
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


class NavigationPlanError(Exception):
    pass


class HumanNavigationReasoner:
    ARRIVAL_RADIUS = 0.5

    def plan(self, current_x: float, current_z: float, current_yaw: float,
             target_x: float, target_z: float, target_yaw: int = 0,
             target_shelf: str = "", navigation_memory: str = "") -> list:
        print("\n" + "=" * 60)
        print("[NAVIGATION REASONER] Waypoints required.")
        print(f"  Current : x={current_x:.2f}, z={current_z:.2f}, yaw={current_yaw:.1f}°")
        print(f"  Target  : x={target_x:.2f}, z={target_z:.2f}, yaw={target_yaw}°  ({target_shelf})")
        if navigation_memory:
            print(f"  Nav memory:\n{navigation_memory}")
        print()
        print("Enter 2-5 waypoints as a JSON array. Rationale must be an action hint")
        print("(facing direction + step count), e.g.:")
        print('  [{"x": 1.84, "z": 0.78, "rationale": "move_left ~10 steps to left_corridor"},')
        print('   {"x": 1.84, "z": 6.73, "rationale": "move_forward ~60 steps north"},')
        print('   {"x": 1.14, "z": 6.73, "rationale": "move_left ~7 steps, then turn_left_90 to yaw=270"}]')
        print("Press Enter twice when done.")
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
            if not isinstance(waypoints, list) or not waypoints:
                raise ValueError("Expected a non-empty JSON array.")
            return waypoints
        except Exception as e:
            raise NavigationPlanError(f"Could not parse waypoint input: {e}\nRaw: {raw[:200]}")

    @staticmethod
    def distance_2d(ax: float, az: float, bx: float, bz: float) -> float:
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)


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
                 mode: Literal['base', 'lean'] = 'base',
                 map_path: Optional[str] = None) -> None:

        self.vlm_agent = VLMAgent(vlm_config)
        self.mode = mode

        if mode == 'lean':
            self.associative_learner = SemanticEpisodicAssociativeLearner(associative_config)
            self.set_semantic_memory()

            _map_path = Path(map_path) if map_path else Path(__file__).parent / "maps" / "layout_2.json"
            with open(_map_path) as f:
                self._store_map = json.load(f)
            self._shelf_lookup: Dict[str, Any] = {s['id']: s for s in self._store_map['shelves']}
            logger.info(f"Store map loaded: {self._store_map['layout_id']} ({len(self._shelf_lookup)} shelves)")

            self.navigation_reasoner = HumanNavigationReasoner()
            self.current_waypoints: list = []
            self.current_waypoint_idx: int = 0
            self.current_nav_target: Optional[tuple] = None
            self.navigation_memory: str = ""
            self._position_gate_cleared: bool = False

    def set_semantic_memory(self) -> None:
        self.vlm_agent.base_semantic_memory = BASE_SEMANTIC_MEMORY
        logger.info("Base semantic memory set for VLMAgent.")

    def reset_nav_state(self) -> None:
        self.current_waypoints = []
        self.current_waypoint_idx = 0
        self.current_nav_target = None
        self._position_gate_cleared = False
        self.navigation_memory = ""

    def set_episodic_memory(self, entry: str, max_entries: int = 5) -> None:
        existing = [e for e in self.vlm_agent.episodic_memory.strip().split("\n\n") if e]
        existing.append(entry)
        self.vlm_agent.episodic_memory = "\n\n".join(existing[-max_entries:])
        logger.info(f"Episodic memory updated ({len(existing[-max_entries:])} entries).")

    def _lookup_shelf(self, shelf_id: str) -> Optional[Dict]:
        return self._shelf_lookup.get(shelf_id.upper())

    def _extract_target_shelf_id(self, recall: str) -> Optional[str]:
        # Match map-ID style: S11, S12_13
        m = re.search(r'\bS(\d+(?:_\d+)?)\b', recall, re.IGNORECASE)
        if m:
            return m.group(0).upper()
        # Match natural-language style: "Shelf 11", "Shelf 12-13", "Shelf 12_13"
        m = re.search(r'\bShelf\s+(\d+(?:[_\-]\d+)?)\b', recall, re.IGNORECASE)
        if m:
            nums = re.sub(r'[-\s]', '_', m.group(1))
            return f"S{nums}"
        return None

    def _check_waypoint_advance(self, ax: float, az: float) -> None:
        if not self.current_waypoints or self.current_waypoint_idx >= len(self.current_waypoints):
            return
        wp = self.current_waypoints[self.current_waypoint_idx]
        if HumanNavigationReasoner.distance_2d(ax, az, wp['x'], wp['z']) <= HumanNavigationReasoner.ARRIVAL_RADIUS:
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

    def _maybe_plan_navigation(self, agent_mode: str, recall: str,
                                ax: float, az: float, yaw: float) -> str:
        if agent_mode != "navigation":
            self.current_waypoints = []
            self.current_waypoint_idx = 0
            self.current_nav_target = None
            return ""

        self._check_waypoint_advance(ax, az)

        shelf_id = self._extract_target_shelf_id(recall)
        shelf = self._lookup_shelf(shelf_id) if shelf_id else None
        new_target = (shelf['approach']['x'], shelf['approach']['z']) if shelf else None

        need_replan = (not self.current_waypoints) or (
            new_target is not None and new_target != self.current_nav_target
        )

        if not need_replan:
            return self._get_current_waypoint_text()

        if new_target is None:
            logger.warning("Navigation mode but no shelf ID found in recall. Skipping planner.")
            return self._get_current_waypoint_text()

        tx, tz = new_target
        if HumanNavigationReasoner.distance_2d(ax, az, tx, tz) <= HumanNavigationReasoner.ARRIVAL_RADIUS:
            self.current_waypoints = [{'x': tx, 'z': tz, 'rationale': 'Already at target.'}]
            self.current_waypoint_idx = 0
            self.current_nav_target = new_target
            return self._get_current_waypoint_text()

        try:
            target_yaw = shelf['approach']['yaw'] if shelf else 0
            waypoints = self.navigation_reasoner.plan(
                ax, az, yaw, tx, tz,
                target_yaw=target_yaw,
                target_shelf=shelf_id or "unknown",
                navigation_memory=self.navigation_memory,
            )
            self.current_waypoints = waypoints
            self.current_waypoint_idx = 0
            self.current_nav_target = new_target
            self._position_gate_cleared = False
            summary = "; ".join(f"({w['x']},{w['z']})" for w in waypoints)
            self.navigation_memory += f"[PATH] start=({ax:.1f},{az:.1f}) → {shelf_id} → waypoints={summary} → outcome=PENDING\n"
            logger.info(f"Navigation plan: {summary}")
        except NavigationPlanError as e:
            logger.error(f"Navigation planner failed: {e}. Proceeding without waypoints.")

        return self._get_current_waypoint_text()

    def _record_collision(self, ax: float, az: float, timestep: int) -> None:
        self.navigation_memory += f"[COLLISION] position=({ax:.1f},{az:.1f}) @ timestep={timestep}\n"
        logger.warning(f"Collision recorded at ({ax:.1f},{az:.1f}) timestep={timestep}")

    def _face_span_text(self, shelf_id: Optional[str]) -> str:
        if not shelf_id:
            return ""
        shelf = self._lookup_shelf(shelf_id)
        if not shelf:
            return ""
        fs = shelf['face_span']
        if 'x_min' in fs:
            return f"## SHELF SCAN BOUNDS: strafe range x∈[{fs['x_min']}, {fs['x_max']}]\n"
        return f"## SHELF SCAN BOUNDS: strafe range z∈[{fs['z_min']}, {fs['z_max']}]\n"

    def end_episode(self, trajectory_log: list, outcome: str = "unknown") -> None:
        if not trajectory_log:
            return
        aisle_defs = "\n".join(
            f"- {a['id']}: x∈[{a['bounds']['x_min']},{a['bounds']['x_max']}] "
            f"z∈[{a['bounds']['z_min']},{a['bounds']['z_max']}]"
            for a in self._store_map.get('aisles', [])
        )
        traj_text = "\n".join(
            f"t={e['timestep']} pos=({e['x']:.2f},{e['z']:.2f}) yaw={e['yaw']:.0f} "
            f"mode={e['mode']} actions={e['actions']} colliding={e['isColliding']}"
            for e in trajectory_log
        )
        history_text = self.vlm_agent.get_history_text(n=12)
        episode_input = (
            f"## OUTCOME: {outcome}\n"
            f"## TRAJECTORY LOG:\n{traj_text}\n\n"
            f"## AISLE CORRIDORS:\n{aisle_defs}\n\n"
            f"## VLM REASONING EXCERPTS (last 12 turns):\n{history_text}\n"
        )
        episodic_response_text = self._call_episodic(episode_input)
        episodic_raw = self._extract_json(
            self.associative_learner.extractable_json_structured_output, episodic_response_text
        )
        if episodic_raw is None:
            logger.warning(f"No JSON in episodic response: {episodic_response_text}")
            return
        episodic_response = ast.literal_eval(episodic_raw)
        entry = (f"## DENSE SUMMARY: {episodic_response['dense_summary']}\n"
                 f"## WHAT WORKED: {episodic_response['what_worked']}\n"
                 f"## WHAT TO AVOID: {episodic_response['what_to_avoid']}\n")
        self.set_episodic_memory(entry)
        with open('episodic_memory.txt', 'w', encoding='utf-8') as f:
            f.write(self.vlm_agent.episodic_memory)
        logger.info("Episode reflection complete.")

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

    @staticmethod
    def _extract_json(pattern, text):
        m = re.search(pattern, text) if text else None
        return m.group(1) if m else None

    def execute_lean(self, request, timestep):
        main_task = request['task']
        _bare_task = request.get('bare_task', main_task)
        _raw_state = request['state']
        screenshot_bytes = base64.b64decode(str(request['image']).encode('utf-8'))
        imagebytes = BytesIO(screenshot_bytes)
        screenshot = Image.open(imagebytes).convert('RGB')
        # depth_image, depth_array = estimate_depth(imagebytes)  # temporarily disabled
        depth_image, depth_array = None, None

        # Extract numeric state for nav infrastructure
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

        # Pre-extract target shelf from task for proximity hint to semantic learner
        _task_shelf_id = self._extract_target_shelf_id(_bare_task)
        _shelf = None
        _approach_hint = ""
        if _task_shelf_id:
            _shelf = self._lookup_shelf(_task_shelf_id)
            if _shelf:
                _ap = _shelf['approach']
                _dist = HumanNavigationReasoner.distance_2d(agent_x, agent_z, _ap['x'], _ap['z'])
                _approach_hint = (
                    f"## TARGET SHELF: {_task_shelf_id} | "
                    f"Approach position: x={_ap['x']}, z={_ap['z']}, yaw={_ap['yaw']}° | "
                    f"Distance from approach: {_dist:.2f}m\n"
                )

        # Semantic reasoner
        semantic_input = (
            f"## CURRENT TIMESTEP: {timestep}\n"
            f"## MAIN TASK: {main_task}\n"
            f"{_approach_hint}"
            f"## SEMANTIC MEMORY: {self.vlm_agent.base_semantic_memory}\n"
            f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
            f"## STATE: {state}\n"
        )
        semantic_response_text = self._call_associative(
            SYS_INST_ASSOCIATIVE_SEMANTIC, screenshot, depth_image, semantic_input
        )
        print(f"SEMANTIC LEARNER RESPONSE: {semantic_response_text}")

        semantic_raw = self._extract_json(
            self.associative_learner.extractable_json_structured_output, semantic_response_text
        )
        if semantic_raw is None:
            print(f"[ERROR] No JSON in semantic response: {semantic_response_text}")
            return {'halt': True, 'text': semantic_response_text or '', 'agent_mode': 'STOP'}

        semantic_response = ast.literal_eval(semantic_raw)
        new_semantic_memory = semantic_response['new_semantic_memory']
        recall = semantic_response['recall']
        agent_mode = semantic_response['mode']
        self.vlm_agent.base_semantic_memory += f"@ timestep {timestep}: {new_semantic_memory}\n"
        print(f"SEMANTIC LEARNER: mode={agent_mode} | recall={recall[:80]}...")

        # Position gate: override mode until the agent is correctly positioned at the target shelf
        if self.current_waypoints and self.current_waypoint_idx < len(self.current_waypoints):
            # Gate 1: waypoints still in progress
            agent_mode = "navigation"
            logger.info(f"Gate 1: waypoint {self.current_waypoint_idx + 1}/{len(self.current_waypoints)} pending. Forcing navigation.")
        elif not self._position_gate_cleared and _shelf and agent_mode not in ("manipulation", "STOP"):
            # Gate 2: waypoints exhausted but position/yaw not yet verified
            _ap = _shelf['approach']
            _gate_dist = HumanNavigationReasoner.distance_2d(agent_x, agent_z, _ap['x'], _ap['z'])
            _agent_yaw_cardinal = int(round(agent_yaw / 90) * 90) % 360
            if _gate_dist > 1.0 or _agent_yaw_cardinal != _ap['yaw']:
                self.current_waypoints = []
                agent_mode = "navigation"
                logger.info(f"Gate 2: position not ready (dist={_gate_dist:.2f}m, yaw={_agent_yaw_cardinal}° vs {_ap['yaw']}°). Forcing navigation.")
            else:
                self._position_gate_cleared = True
                logger.info(f"Gate 2: cleared (dist={_gate_dist:.2f}m, yaw={_agent_yaw_cardinal}°). Releasing mode control.")

        # Mode routing
        if agent_mode == "perception":
            available_actions = f"{PERCEPTION_ACTIONS}\n\n"
        elif agent_mode == "navigation":
            available_actions = f"{NAVIGATION_ACTIONS}\n\n"
        elif agent_mode == "scan":
            available_actions = f"{SCAN_ACTIONS}\n\n"
        elif agent_mode == "manipulation":
            available_actions = f"{MANIPULATION_ACTIONS}\n\n"
        elif agent_mode == "STOP":
            return {'halt': True, 'text': "STOP action received, terminating execution...", 'agent_mode': agent_mode}
        else:
            logger.warning(f"Unknown mode '{agent_mode}', defaulting to navigation.")
            available_actions = f"{NAVIGATION_ACTIONS}\n\n"
            agent_mode = "navigation"

        # Navigation planning
        waypoint_text = self._maybe_plan_navigation(agent_mode, recall, agent_x, agent_z, agent_yaw)
        if agent_mode == "navigation" and not waypoint_text:
            waypoint_text = "## NAVIGATION NOTE: No waypoint available yet — target shelf not identified. Use get_heading or snap_to_cardinal to confirm orientation. Do not move.\n"

        # Mode-specific context blocks
        depth_hint = self._compute_depth_hint(main_task, depth_array) if agent_mode == "manipulation" else ""
        shelf_id = self._extract_target_shelf_id(recall)
        scan_bounds = self._face_span_text(shelf_id) if agent_mode == "scan" else ""
        nav_memory_block = (
            f"## NAVIGATION MEMORY:\n{self.navigation_memory}\n"
            if agent_mode == "navigation" and self.navigation_memory else ""
        )

        # VLM action planner
        user_msg = (
            f"## CURRENT TIMESTEP: {timestep}\n"
            f"## MAIN TASK: {main_task}\n"
            f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
            f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
            f"## STATE: {state}\n"
            f"## AGENT MODE: {agent_mode}\n"
            f"## AVAILABLE ACTIONS:\n{available_actions}"
            f"{depth_hint}"
            f"{scan_bounds}"
            f"{nav_memory_block}"
            f"{waypoint_text}"
        )
        vlm_content = _build_content(screenshot, "## CURRENT OBSERVATION\n", depth_image, "## CURRENT DEPTH MAP\n" + user_msg)
        response_text = self.vlm_agent.send_message(vlm_content)
        print(f"VLMAgent RESPONSE: {response_text}")

        response_raw = self._extract_json(
            self.vlm_agent.extractable_json_structured_output, response_text
        )
        if response_raw is None:
            print(f"[ERROR] No JSON in VLM response: {response_text}")
            return {'halt': True, 'text': response_text or '', 'agent_mode': agent_mode}
        ast.literal_eval(response_raw)  # validate parse only; caller re-parses from response_text

        with open('semantic_memory.txt', 'w', encoding='utf-8') as f:
            f.write(self.vlm_agent.base_semantic_memory)

        return {'halt': False, 'text': response_text, 'agent_mode': agent_mode}
