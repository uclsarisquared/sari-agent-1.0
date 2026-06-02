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

    def set_semantic_memory(self) -> None:
        self.vlm_agent.base_semantic_memory = BASE_SEMANTIC_MEMORY
        logger.info("Base semantic memory set for VLMAgent.")

    def set_episodic_memory(self, entry: str, max_entries: int = 5) -> None:
        existing = [e for e in self.vlm_agent.episodic_memory.strip().split("\n\n") if e]
        existing.append(entry)
        self.vlm_agent.episodic_memory = "\n\n".join(existing[-max_entries:])
        logger.info(f"Episodic memory updated ({len(existing[-max_entries:])} entries).")

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
        state = str(request['state'])
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
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}"
                        f"{depth_hint}")

            vlm_content = _build_content(screenshot, "## CURRENT OBSERVATION\n", depth_image, "## CURRENT DEPTH MAP\n" + user_msg)
            response_text = self.vlm_agent.send_message(vlm_content)
            print(f"VLMAgent RESPONSE: {response_text}")

            response_raw = self._extract_json(
                self.vlm_agent.extractable_json_structured_output, response_text
            )
            if response_raw is None:
                print(f"[ERROR] No JSON in VLM response: {response_text}")
                return {'halt': True, 'text': response_text or '', 'agent_mode': agent_mode}
            response_json = ast.literal_eval(response_raw)

            episodic_response_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
            episodic_raw = self._extract_json(
                self.associative_learner.extractable_json_structured_output, episodic_response_text
            )
            if episodic_raw is None:
                print(f"[ERROR] No JSON in episodic response: {episodic_response_text}")
                return {'halt': True, 'text': response_text or '', 'agent_mode': agent_mode}
            episodic_response = ast.literal_eval(episodic_raw)

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
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}"
                        f"{depth_hint}")

            vlm_content = _build_content(screenshot, "## CURRENT OBSERVATION\n", depth_image, "## CURRENT DEPTH MAP\n" + user_msg)
            response_text = self.vlm_agent.send_message(vlm_content)

            response_raw = self._extract_json(
                self.vlm_agent.extractable_json_structured_output, response_text
            )
            if response_raw is None:
                print(f"[ERROR] No JSON in VLM response: {response_text}")
                return {'halt': True, 'text': response_text or '', 'agent_mode': agent_mode}
            response_json = ast.literal_eval(response_raw)

            episodic_response_text = self._call_episodic(self.vlm_agent.get_history_text(n=8))
            episodic_raw = self._extract_json(
                self.associative_learner.extractable_json_structured_output, episodic_response_text
            )
            if episodic_raw is None:
                print(f"[ERROR] No JSON in episodic response: {episodic_response_text}")
                return {'halt': True, 'text': response_text or '', 'agent_mode': agent_mode}
            episodic_response = ast.literal_eval(episodic_raw)

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
