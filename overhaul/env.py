from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv('api.env')

from reference import BASE_SEMANTIC_MEMORY

@dataclass
class GeminiConfig:
    model_id: Literal['gemini-2.5-flash-preview-04-17', 'gemini-2.5-pro-preview-05-06'] = 'gemini-2.5-flash-preview-04-17'
    max_thinking_tokens: int = 3072
    temperature: float = 0.5
    mode: Literal['base', 'super'] = 'base'
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

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Main execution method for the agent."""
        pass


class SemanticEpisodicAssociativeLearner(BaseAgent):
    """
    A semantic-episodic associative learner that utilizes semantic and episodic memory
    to learn and recall information. It will suggest candidate actions based on the current context
    to be passed to the VLM. 
    """
    def __init__(self, config: Optional[Union[Dict[str, Any], GeminiConfig]] = None) -> None:
        super().__init__(config)

        self.client = genai.Client(api_key=self.config.api_key)

    def execute(self, *args, **kwargs) -> Any:
        """Main execution method for the agent."""
        pass

    def generate_config(self):
        if self.config.mode == 'base':
            return types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=self.config.max_thinking_tokens),
                system_instruction=...,
                temperature=self.config.temperature,
            )