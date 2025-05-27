from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import re
import base64
from io import BytesIO
from PIL import Image
from loguru import logger
import ast

load_dotenv('../api.env')

from sys_inst import (
    SYS_INST_ASSOCIATIVE_SEMANTIC,
    SYS_INST_ASSOCIATIVE_EPISODIC,
    SYS_INST_VLM_LEAN
)
from memory import BASE_SEMANTIC_MEMORY
from actions_str import (
    NAVIGATION_ACTIONS,
    PERCEPTION_ACTIONS,
)


@dataclass
class GeminiConfig:
    model_id: Literal['gemini-2.5-flash-preview-05-20', 'gemini-2.5-pro-preview-05-06'] = 'gemini-2.5-flash-preview-05-20'
    max_thinking_tokens: int = 1024
    temperature: float = 0.5
    mode: Literal['base', 'lean'] = 'base'
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))

class BaseAgent(ABC):
    """Base class for all agents."""
    @abstractmethod
    def __init__(self, config: Optional[Union[Dict[str, Any], GeminiConfig]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def generate_config(self):
        """Configuration of text generation."""
        pass

    @property
    def extractable_json_structured_output(self):
        return re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)


class SemanticEpisodicAssociativeLearner(BaseAgent):
    """A semantic-episodic associative learner that utilizes semantic and episodic memory to learn and recall information."""
    def __init__(self, config: Optional[Union[Dict[str, Any], GeminiConfig]] = None) -> None:
        super().__init__(config)

        self.client = genai.Client(api_key=self.config.api_key)

    def generate_config(self):
        if self.config.mode == 'lean':
            semantic_gen_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.config.max_thinking_tokens),
                system_instruction=SYS_INST_ASSOCIATIVE_SEMANTIC,
                temperature=self.config.temperature,
            )
            episodic_gen_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                system_instruction=SYS_INST_ASSOCIATIVE_EPISODIC,
                temperature=self.config.temperature,
            )
            return semantic_gen_config, episodic_gen_config
        elif self.config.mode == 'base':
            raise ValueError("Base mode does not support SemanticEpisodicAssociativeLearner.")

class VLMAgent(BaseAgent):
    """
    An agent that is embedded in the environment and can interact with it.
    It uses semantic and episodic memory to learn and recall information.
    """
    def __init__(self, config: Optional[Union[Dict[str, Any], GeminiConfig]] = None) -> None:
        super().__init__(config)

        self.client = genai.Client(api_key=self.config.api_key)
        self.init_chat_session()

    def init_chat_session(self):
        self.chat = self.client.chats.create(
            model=self.config.model_id,
            config=self.generate_config(),
        )
        logger.info(f"Chat session initialized with model: {self.config.model_id}")

    def generate_config(self):
        if self.config.mode == 'lean':
            main_gen_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.config.max_thinking_tokens),
                system_instruction=SYS_INST_VLM_LEAN,
                temperature=self.config.temperature,
            )
        elif self.config.mode == 'base':
            raise ValueError("Base mode does not support VLMAgent.")
        return main_gen_config

class EmbodiedAgent:
    """An agent that is embodied in the environment and can interact with it."""
    def __init__(self, vlm_config: Optional[Union[Dict[str, Any], GeminiConfig]] = None,
                 associative_config: Optional[Union[Dict[str, Any], GeminiConfig]] = None,
                 mode: Literal['base', 'lean']='base') -> None:

        self.vlm_agent = VLMAgent(vlm_config)
        self.mode = mode

        if mode == 'lean':
            self.associative_learner = SemanticEpisodicAssociativeLearner(associative_config)
            self.semantic_gen_config = self.associative_learner.generate_config()[0]
            self.episodic_gen_config = self.associative_learner.generate_config()[1]
            self.set_semantic_memory()

    def set_semantic_memory(self) -> None:
        self.vlm_agent.base_semantic_memory = BASE_SEMANTIC_MEMORY
        logger.info("Base semantic memory set for VLMAgent.")

    def set_episodic_memory(self, episodic_memory: str) -> None:
        """Set the episodic memory for the agent."""
        self.vlm_agent.episodic_memory = episodic_memory
        logger.info(f"Episodic memory updated: {episodic_memory}")

    def execute_lean(self, request, timestep):
        main_task = request['task']
        state = str(request['state'])
        screenshot = str(request['image']).encode('utf-8')
        screenshot = base64.b64decode(screenshot)
        imagebytes = BytesIO(screenshot)
        screenshot = Image.open(imagebytes).convert('RGB')

        curr_obs_header = "## CURRENT OBSERVATION\n"

        new_semantic_memory = ""
        recall = ""

        if timestep == 1:
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## SEMANTIC MEMORY: {self.vlm_agent.base_semantic_memory}\n"
                        f"## STATE: {state}\n")
            
            contents = [curr_obs_header, screenshot, user_msg]

            semantic_response = self.associative_learner.client.models.generate_content(
                model=self.associative_learner.config.model_id,
                contents=contents,
                config=self.semantic_gen_config
            )

            print(f"SEMANTIC LEARNER RESPONSE: {semantic_response.text}")

            semantic_response = re.search(
                self.associative_learner.extractable_json_structured_output,
                semantic_response.text
            )[1]
            semantic_response = ast.literal_eval(semantic_response)
            new_semantic_memory = semantic_response['new_semantic_memory']
            recall = semantic_response['recall']
            agent_mode = semantic_response['mode']

            if agent_mode == "perception":
                available_actions = f"{PERCEPTION_ACTIONS}\n\n"
            elif agent_mode == "navigation":
                available_actions = f"{NAVIGATION_ACTIONS}\n\n"
            elif agent_mode == "STOP":
                return {
                    'halt': True,
                    'text': "STOP action received, terminating execution...",
                    'agent_mode': agent_mode
                }

            self.vlm_agent.base_semantic_memory += f"@ timestep {timestep}: {new_semantic_memory}\n"

            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}")

            contents = [curr_obs_header, screenshot, user_msg]

            response = self.vlm_agent.chat.send_message(
                message=contents,
                config=self.vlm_agent.generate_config()
            )
            
            print(f"VLMAgent RESPONSE: {response.text}")

            response_json = re.search(
                self.vlm_agent.extractable_json_structured_output,
                response.text
            )[1]
            response_json = ast.literal_eval(response_json)
            action = response_json['actions']

            history = ""
            for message in self.vlm_agent.chat.get_history()[-2:]:
                role = message.role
                content = message.parts[0].text
                history += f"{role.upper()}: {content}\n"
            history = history.strip()

            episodic_response = self.associative_learner.client.models.generate_content(
                model=self.associative_learner.config.model_id,
                config=self.episodic_gen_config,
                contents=[history]
            )

            episodic_response = re.search(
                self.associative_learner.extractable_json_structured_output,
                episodic_response.text
            )[1]
            episodic_response = ast.literal_eval(episodic_response)

            dense_summary = episodic_response['dense_summary']
            what_worked = episodic_response['what_worked']
            what_to_avoid = episodic_response['what_to_avoid']

            episodic_memory = (f"@ timestep {timestep}:\n"
                               f"## DENSE SUMMARY: {dense_summary}\n"
                               f"## WHAT WORKED: {what_worked}\n"
                               f"## WHAT TO AVOID: {what_to_avoid}\n")

            self.set_episodic_memory(episodic_memory)
        else:
            user_msg = (f"## MAIN TASK: {main_task}\n"
                        f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## SEMANTIC MEMORY: {self.vlm_agent.base_semantic_memory}\n"
                        f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
                        f"## STATE: {state}\n")
            contents = [curr_obs_header, screenshot, user_msg]

            semantic_response = self.associative_learner.client.models.generate_content(
                model=self.associative_learner.config.model_id,
                contents=contents,
                config=self.semantic_gen_config
            )

            semantic_response = re.search(
                self.associative_learner.extractable_json_structured_output,
                semantic_response.text
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
            elif agent_mode in ["STOP", "manipulation"]:
                return {
                    'halt': True,
                    'text': "STOP action received, terminating execution..."
                }

            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"## EXISTING EPISODIC MEMORY: {self.vlm_agent.episodic_memory}\n"
                        f"## STATE: {state}\n"
                        f"## AGENT MODE: {agent_mode}\n"
                        f"## AVAILABLE ACTIONS:\n{available_actions}")

            contents = [curr_obs_header, screenshot, user_msg]
            response = self.vlm_agent.chat.send_message(
                message=contents,
                config=self.vlm_agent.generate_config()
            )

            response_json = re.search(
                self.vlm_agent.extractable_json_structured_output,
                response.text
            )[1]
            response_json = ast.literal_eval(response_json)
            action = response_json['actions']

            history = ""
            for message in self.vlm_agent.chat.get_history()[-2:]:
                role = message.role
                content = message.parts[0].text
                history += f"{role.upper()}: {content}\n"
            history = history.strip()

            episodic_response = self.associative_learner.client.models.generate_content(
                model=self.associative_learner.config.model_id,
                config=self.episodic_gen_config,
                contents=[history]
            )

            episodic_response = re.search(
                self.associative_learner.extractable_json_structured_output,
                episodic_response.text
            )[1]
            episodic_response = ast.literal_eval(episodic_response)

            dense_summary = episodic_response['dense_summary']
            what_worked = episodic_response['what_worked']
            what_to_avoid = episodic_response['what_to_avoid']

            episodic_memory = (f"@ timestep {timestep}:\n"
                               f"## DENSE SUMMARY: {dense_summary}\n"
                               f"## WHAT WORKED: {what_worked}\n"
                               f"## WHAT TO AVOID: {what_to_avoid}\n")
            self.set_episodic_memory(episodic_memory)

        # Write base semantic memory to file
        with open('semantic_memory.txt', 'w') as f:
            f.write(self.vlm_agent.base_semantic_memory)
        # Write episodic memory to file
        with open('episodic_memory.txt', 'w') as f:
            f.write(self.vlm_agent.episodic_memory)

        agent_response = {
            'halt': False,
            'text': response.text,
            'agent_mode': agent_mode,
        }
        return agent_response