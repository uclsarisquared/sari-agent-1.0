import os
import re
import ast
import base64
from io import BytesIO
from dataclasses import dataclass, field
from typing import Literal, List, Dict, Any

from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
from loguru import logger

from reference import SYSTEM_INSTRUCTIONS_REF, BASE_SEMANTIC_MEMORY

load_dotenv('api.env')


@dataclass
class OpenRouterCfg:
    model: str = "google/gemini-3.1-pro-preview"
    temperature: float = 0.5
    mode: Literal["inf_base", "inf_super"] = "inf_base"


class OpenRouterAgent:
    def __init__(self, cfg: OpenRouterCfg) -> None:
        self.cfg = cfg
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY'),
        )
        self.metrics = {'total_tokens_used': 0}
        self.main_history: List[Dict[str, Any]] = []

        if self.cfg.mode == "inf_super":
            self.base_semantic_memory = BASE_SEMANTIC_MEMORY
            self.base_semantic_memory += "\n**Added Semantic Memory:**\n"
            self.episodic_memory = ""

    @property
    def extractable_json(self):
        return re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

    def _image_part(self, image: Image.Image) -> Dict:
        buf = BytesIO()
        image.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

    def _build_content(self, image: Image.Image, text: str) -> Any:
        if image and text:
            return [self._image_part(image), {"type": "text", "text": text}]
        if image:
            return [self._image_part(image)]
        return text

    def _call(self, system: str, history: List[Dict], image: Image.Image = None, text: str = "") -> str:
        messages = (
            [{"role": "system", "content": system}]
            + history
            + [{"role": "user", "content": self._build_content(image, text)}]
        )
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
        )
        self.metrics['total_tokens_used'] += resp.usage.total_tokens
        return resp.choices[0].message.content

    def _decode_screenshot(self, request) -> Image.Image:
        raw = base64.b64decode(str(request['image']).encode('utf-8'))
        return Image.open(BytesIO(raw)).convert("RGB")

    def _parse_json(self, text: str) -> Dict:
        match = re.search(self.extractable_json, text)
        if not match:
            raise ValueError(f"No JSON block found in response:\n{text}")
        return ast.literal_eval(match.group(1))

    def _reset_memory(self) -> None:
        self.main_history = []
        self.metrics = {'total_tokens_used': 0}
        if self.cfg.mode == "inf_super":
            self.base_semantic_memory = BASE_SEMANTIC_MEMORY + "\n**Added Semantic Memory:**\n"
            self.episodic_memory = ""

    def generate(self, request, timestep):
        if timestep == 1:
            self._reset_memory()
        if self.cfg.mode == "inf_base":
            return self._generate_inf_base(request, timestep)
        elif self.cfg.mode == "inf_super":
            return self._generate_inf_super(request, timestep)
        raise ValueError(f"Unknown mode: {self.cfg.mode}")

    def _generate_inf_base(self, request, timestep):
        screenshot = self._decode_screenshot(request)
        state = str(request['state'])
        actions_history = str(request['actions'])

        text = f"## CURRENT TIMESTEP: {timestep}\n"
        if timestep == 1:
            text += f"## MAIN TASK: {request['task']}\n"
        text += f"## STATE: {state}\n## ACTIONS HISTORY: {actions_history}\n"

        response_text = self._call(
            SYSTEM_INSTRUCTIONS_REF["inf_base"],
            self.main_history,
            screenshot,
            f"## CURRENT OBSERVATION:\n{text}",
        )

        self.main_history += [
            {"role": "user",      "content": self._build_content(screenshot, f"## CURRENT OBSERVATION:\n{text}")},
            {"role": "assistant", "content": response_text},
        ]

        logger.info(f"[RESPONSE]: {response_text}")
        return {'text': response_text, 'history': self.main_history[-2:]}

    def _generate_inf_super(self, request, timestep):
        screenshot = self._decode_screenshot(request)
        state = str(request['state'])

        # semantic learner — stateless (no shared history)
        sem_text = f"## CURRENT TIMESTEP: {timestep}\n"
        if timestep == 1:
            sem_text += f"## MAIN TASK: {request['task']}\n"
        sem_text += f"## SEMANTIC MEMORY: {self.base_semantic_memory}\n{state}"

        sem_response = self._call(
            SYSTEM_INSTRUCTIONS_REF["inf_super"]["semantic"],
            [],
            screenshot,
            f"## CURRENT OBSERVATION:\n{sem_text}",
        )
        sem_json = self._parse_json(sem_response)
        new_semantic_memory = sem_json['new_semantic_memory']
        recall = sem_json['recall']
        self.base_semantic_memory += f"@ timestep {timestep}: {new_semantic_memory}\n"

        # main agent
        main_text = f"## CURRENT TIMESTEP: {timestep}\n"
        if timestep == 1:
            main_text += f"## MAIN TASK: {request['task']}\n"
        main_text += f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
        if timestep > 1:
            main_text += f"## EPISODIC MEMORY: {self.episodic_memory}\n"
        main_text += state

        response_text = self._call(
            SYSTEM_INSTRUCTIONS_REF["inf_super"]["general"],
            self.main_history,
            screenshot,
            f"## CURRENT OBSERVATION:\n{main_text}",
        )

        self.main_history += [
            {"role": "user",      "content": self._build_content(screenshot, f"## CURRENT OBSERVATION:\n{main_text}")},
            {"role": "assistant", "content": response_text},
        ]

        # check for STOP before running episodic learner
        action_json = self._parse_json(response_text)
        if "STOP" in action_json.get('actions', []):
            logger.info("STOP action detected.")
            self._log_memory(new_semantic_memory, recall)
            return response_text

        # episodic learner — stateless, text-only
        last_exchange = ""
        for msg in self.main_history[-2:]:
            content = msg['content']
            if isinstance(content, list):
                content = " ".join(p['text'] for p in content if p.get('type') == 'text')
            last_exchange += f"{msg['role']}: {content}\n"

        epi_response = self._call(
            SYSTEM_INSTRUCTIONS_REF["inf_super"]["episodic"],
            [],
            None,
            last_exchange.strip(),
        )
        epi_json = self._parse_json(epi_response)
        self.episodic_memory = (
            f"@ timestep {timestep}\n"
            f"## DENSE SUMMARY: {epi_json['dense_summary']}\n"
            f"## SURPRISE: {epi_json['surprise']}\n"
            f"## WHAT WORKED: {epi_json['what_worked']}\n"
            f"## WHAT TO AVOID: {epi_json['what_to_avoid']}\n"
        )

        self._log_memory(new_semantic_memory, recall)

        with open('base_semantic_memory.txt', 'w') as f:
            f.write(self.base_semantic_memory)

        return {'text': response_text, 'history': self.main_history[-2:]}

    def _log_memory(self, new_semantic_memory: str, recall: str):
        logger.info(">" * 50)
        logger.info(f"New Semantic Memory: {new_semantic_memory}")
        logger.info("*" * 50)
        logger.info(f"Recall: {recall}")
        logger.info("*" * 50)
        logger.info(f"Episodic Memory: {self.episodic_memory}")
        logger.info("<" * 50)
