from google import genai
from google.genai import types
import os
from PIL import Image
from io import BytesIO
import base64
import re
import ast
from dotenv import load_dotenv
from dataclasses import dataclass
import sys
from typing import Literal
from loguru import logger

from reference import SYSTEM_INSTRUCTIONS_REF, BASE_SEMANTIC_MEMORY

load_dotenv('config.env')

MODE = sys.argv[1] if len(sys.argv) > 1 else "tank"

@dataclass
class GeminiProCfg:
    max_thinking_tokens: int = 3072
    temperature: float = 0.5
    mode: Literal["inf_base", "inf_super", "swift"] = "inf_base"

class GeminiPro:
    def __init__(self, cfg: GeminiProCfg) -> None:
        self.model_id = "gemini-2.5-pro-preview-05-06"
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.cfg = cfg

        self.metrics = {'thinking_tokens_used': 0, 'total_tokens_used': 0}

        if self.cfg.mode in ["inf_base", "inf_super"]:
            self.init_chat_session()

        if self.cfg.mode == "inf_super":
            self.base_semantic_memory = BASE_SEMANTIC_MEMORY
            self.base_semantic_memory += f"\n**Added Semantic Memory:**\n"
            self.episodic_memory = ""

    def init_chat_session(self):
        self.chat = self.client.chats.create(
            model=self.model_id,
            config=self.generate_config,
        )
        logger.info(f"Chat session initialized...")

    @property
    def generate_config(self):
        if self.cfg.mode == "inf_base":
            return types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.cfg.max_thinking_tokens),
                system_instruction=SYSTEM_INSTRUCTIONS_REF[self.cfg.mode],
                temperature=self.cfg.temperature,
            )
        elif self.cfg.mode == "inf_super":
            return types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.cfg.max_thinking_tokens),
                system_instruction=SYSTEM_INSTRUCTIONS_REF[self.cfg.mode]['general'],
                temperature=self.cfg.temperature,
            )
        elif self.cfg.mode == "swift":
            return ...

    @property
    def extractable_json_structured_output(self):
        return re.compile(r'```\s*json\s*([\s\S]*?)\s*```', re.DOTALL)

    def generate(self, request, timestep):
        if self.cfg.mode == "inf_base":
            return self._generate_inf_base(request, timestep)
        elif self.cfg.mode == "inf_super":
            return self._generate_inf_super(request, timestep)
        elif self.cfg.mode == "swift":
            return self._generate_swift(request, timestep)
        else:
            raise ValueError("Invalid mode. Choose 'inf_base', 'inf_super', or 'swift'.")

    def _generate_inf_base(self, request, timestep):
        if timestep == 1:
            main_task = request['task']

        state = str(request['state'])
        screenshot = str(request['image']).encode('utf-8')
        actions_history = str(request['actions'])
        screenshot = base64.b64decode(screenshot)
        imagebytes = BytesIO(screenshot)
        screenshot = Image.open(imagebytes).convert("RGB")

        curr_obs_header = "## CURRENT OBSERVATION:\n"
        if timestep == 1:
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## STATE: {state}"
                        f"## ACTIONS HISTORY: {actions_history}\n"
                        )
        else:
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## STATE: {state}"
                        f"## ACTIONS HISTORY: {actions_history}\n"
                        )
        contents = [curr_obs_header, screenshot, user_msg]

        response = self.chat.send_message(message=contents)

        print(f"[RESPONSE]: {response.text}")
        print("-" * 50)
        
        payload = {
            'text': response.text,
            'history': self.chat.get_history()[-2:] 
        }
        return payload

    def _generate_inf_super(self, request, timestep):
        if timestep == 1:
            main_task = request['task']

        state = str(request['state'])
        screenshot = str(request['image']).encode('utf-8')
        screenshot = base64.b64decode(screenshot)
        imagebytes = BytesIO(screenshot)
        screenshot = Image.open(imagebytes).convert("RGB")

        curr_obs_header = "## CURRENT OBSERVATION:\n"
        
        new_semantic_memory = ""
        recall = ""
        
        if timestep == 1:
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## SEMANTIC MEMORY: {self.base_semantic_memory}\n"
                        f"{state}")
            
            contents = [curr_obs_header, screenshot, user_msg]
            
            # semantic memory synthesis
            semantic_learner_response = self.chat.send_message(
                message=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS_REF[self.cfg.mode]['semantic'],
                    temperature=self.cfg.temperature,
                )
            )
            
            semantic_learner_response_json = re.search(self.extractable_json_structured_output, semantic_learner_response.text)[1]
            semantic_learner_response_json = ast.literal_eval(semantic_learner_response_json)
            # TODO: Add logger here for both new_semantic_memory and recall
            new_semantic_memory = semantic_learner_response_json['new_semantic_memory']
            recall = semantic_learner_response_json['recall']
            self.base_semantic_memory += f"@ timestep {timestep}: {new_semantic_memory}\n"

            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## MAIN TASK: {main_task}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"{state}")
    
            contents = [curr_obs_header, screenshot, user_msg]

            response = self.chat.send_message(
                message=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS_REF[self.cfg.mode]['general'],
                    temperature=self.cfg.temperature,
                )
            )

            
            # check if 'STOP' in response.text
            response_json = re.search(self.extractable_json_structured_output, response.text)[1]
            response_json = ast.literal_eval(response_json)
            action = response_json['actions']

            for act in action:
                if act == "STOP":
                    print("STOP action detected. Exiting...")
                    
                    print(">" * 50)
                    print(f"New Semantic Memory: {new_semantic_memory}")
                    print('*' * 50)
                    print(f"Recall: {recall}")
                    print('*' * 50)
                    print(f"Episodic Memory: {self.episodic_memory}")
                    print('<' * 50)

                    return response.text

            # get conversation history
            history = ""
            for message in self.chat.get_history()[-2:]:    # get the last two messages
                role = message.role
                content = message.parts[0].text
                history += f"{role}: {content}\n"
            history = history.strip()

            # episodic memory synthesis
            episodic_learner_response = self.chat.send_message(
                message=[history],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),    # no thinking budget
                    system_instruction=SYSTEM_INSTRUCTIONS_REF[self.cfg.mode]['episodic'],
                    temperature=self.cfg.temperature,
                )
            )

            episodic_learner_response_json = re.search(self.extractable_json_structured_output, episodic_learner_response.text)[1]
            episodic_learner_response_json = ast.literal_eval(episodic_learner_response_json)
            
            dense_summary = episodic_learner_response_json['dense_summary']
            surprise = episodic_learner_response_json['surprise']
            what_worked = episodic_learner_response_json['what_worked']
            what_to_avoid = episodic_learner_response_json['what_to_avoid']
            
            episodic_memory = (f"@ timestep {timestep}\n"
                               f"## DENSE SUMMARY: {dense_summary}\n"
                               f"## SURPRISE: {surprise}\n"
                               f"## WHAT WORKED: {what_worked}\n"
                               f"## WHAT TO AVOID: {what_to_avoid}\n")

            self.episodic_memory = episodic_memory
        else:
            # use updated semantic memory
            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## SEMANTIC MEMORY: {self.base_semantic_memory}\n"
                        f"{state}")
            contents = [curr_obs_header, screenshot, user_msg]

            semantic_learner_response = self.chat.send_message(
                message=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS_REF[self.cfg.mode]['semantic'],
                    temperature=self.cfg.temperature,
                )
            )
            
            # semantic memory synthesis
            semantic_learner_response_json = re.search(self.extractable_json_structured_output, semantic_learner_response.text)[1]
            semantic_learner_response_json = ast.literal_eval(semantic_learner_response_json)
            new_semantic_memory = semantic_learner_response_json['new_semantic_memory']
            recall = semantic_learner_response_json['recall']
            self.base_semantic_memory += f"@ timestep {timestep}: {new_semantic_memory}\n"

            user_msg = (f"## CURRENT TIMESTEP: {timestep}\n"
                        f"## RECALL FROM SEMANTIC MEMORY: {recall}\n"
                        f"## EPISODIC MEMORY: {self.episodic_memory}\n"
                        f"{state}")
            contents = [curr_obs_header, screenshot, user_msg]

            response = self.chat.send_message(
                message=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS_REF[self.cfg.mode]['general'],
                    temperature=self.cfg.temperature,
                )
            )

            # check if 'STOP' in response.text
            response_json = re.search(self.extractable_json_structured_output, response.text)[1]
            response_json = ast.literal_eval(response_json)
            action = response_json['actions']
            for act in action:
                if act == "STOP":
                    print("STOP action detected. Exiting...")
                    
                    print(">" * 50)
                    print(f"New Semantic Memory: {new_semantic_memory}")
                    print('*' * 50)
                    print(f"Recall: {recall}")
                    print('*' * 50)
                    print(f"Episodic Memory: {self.episodic_memory}")
                    print('<' * 50)
                    
                    return response.text

            # get conversation history
            history = ""
            for message in self.chat.get_history()[-2:]:
                role = message.role
                content = message.parts[0].text
                history += f"{role}: {content}\n"
            history = history.strip()

            # episodic memory synthesis
            episodic_learner_response = self.chat.send_message(
                message=[history],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),    # no thinking budget
                    system_instruction=SYSTEM_INSTRUCTIONS_REF[self.cfg.mode]['episodic'],
                    temperature=self.cfg.temperature,
                )
            )

            episodic_learner_response_json = re.search(self.extractable_json_structured_output, episodic_learner_response.text)[1]
            episodic_learner_response_json = ast.literal_eval(episodic_learner_response_json)

            dense_summary = episodic_learner_response_json['dense_summary']
            surprise = episodic_learner_response_json['surprise']
            what_worked = episodic_learner_response_json['what_worked']
            what_to_avoid = episodic_learner_response_json['what_to_avoid']

            episodic_memory = (f"@ timestep {timestep}\n"
                               f"## DENSE SUMMARY: {dense_summary}\n"
                               f"## SURPRISE: {surprise}\n"
                               f"## WHAT WORKED: {what_worked}\n"
                               f"## WHAT TO AVOID: {what_to_avoid}\n")
            
            self.episodic_memory = episodic_memory
        
        print('>' * 50)
        print(f"New Semantic Memory: {new_semantic_memory}")
        print('*' * 50)
        print(f"Recall: {recall}")
        print('*' * 50)
        print(f"Episodic Memory: {self.episodic_memory}")
        print('<' * 50)

        ## write base semantic memory to file
        with open('base_semantic_memory.txt', 'w') as f:
            f.write(self.base_semantic_memory)

        payload = {
            'text': response.text,
            'history': self.chat.get_history()[-2:]
        }
        return payload