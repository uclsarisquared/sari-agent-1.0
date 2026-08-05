"""OpenAI-compatible endpoint configuration and shared chat-call utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field
from io import BytesIO
import json
import os
from pathlib import Path
import time
from typing import Literal, Optional
from urllib.parse import urlsplit

from dotenv import load_dotenv
from loguru import logger
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from PIL import Image

from agent_core.contracts import JSON_BLOCK_PATTERN
from agent_core.models import agent_model


load_dotenv(Path(__file__).resolve().parent.parent.parent / "config.env")


@dataclass
class LLMConfig:
    model_id: str = "google/gemini-2.5-flash-preview-05-20"
    temperature: float = 0.5
    mode: Literal["base", "lean"] = "base"
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: Optional[int] = None
    extra_body: Optional[dict] = None


# Compatibility name retained while callers migrate to the transport-neutral spelling.
OpenRouterConfig = LLMConfig


def normalize_endpoint_root(raw: str) -> str:
    """Normalize and validate an endpoint root whose port is configuration-owned."""
    endpoint = raw.strip().rstrip("/")
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    parsed = urlsplit(endpoint)
    if not parsed.hostname:
        raise RuntimeError(f"OPENAI_API_URL has no host: {endpoint!r}")
    try:
        port = parsed.port
    except ValueError as error:
        raise RuntimeError(f"OPENAI_API_URL has an invalid port: {endpoint!r}") from error
    if port is None:
        raise RuntimeError(
            "OPENAI_API_URL must include the endpoint port "
            "(for example http://host:8000)"
        )
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise RuntimeError(
            "OPENAI_API_URL must contain only scheme, host, and port; the code appends /v1"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def endpoint_creds() -> tuple[Optional[str], Optional[str]]:
    """Return the normalized endpoint root (including its port) and bearer key.

    ``OPENAI_API_URL`` owns transport location: scheme, host, and port. Callers
    own the OpenAI API prefix and append ``/v1`` themselves.
    """
    endpoint = os.getenv("OPENAI_API_URL")
    key = os.getenv("OPENAI_API_KEY")
    if not (endpoint and key):
        try:
            conda_state = os.getenv("SARI_CONDA_STATE", r"C:/Sari/sari_env_old/conda-meta/state")
            with open(conda_state, encoding="utf-8") as handle:
                values = json.load(handle).get("env_vars", {})
            endpoint = endpoint or values.get("OPENAI_API_URL")
            key = key or values.get("OPENAI_API_KEY")
        except OSError:
            pass
    if endpoint:
        endpoint = normalize_endpoint_root(endpoint)
    return endpoint, key


def agent_vlm_config(temperature: float = 0.5, mode: str = "lean") -> LLMConfig:
    endpoint, key = endpoint_creds()
    if not endpoint:
        raise RuntimeError(
            "OPENAI_API_URL not found (looked in repo-root config.env, then "
            "sari_env_old conda state)"
        )
    return LLMConfig(
        model_id=agent_model(),
        temperature=temperature,
        mode=mode,
        api_key=key,
        base_url=f"{endpoint}/v1",
        max_tokens=1536,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def encode_image(image: Image.Image) -> dict:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def build_content(*parts) -> list:
    content = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, Image.Image):
            content.append(encode_image(part))
        elif isinstance(part, str):
            content.append({"type": "text", "text": part})
    return content


API_MAX_ATTEMPTS = 10
API_RETRY_DELAYS = (1, 2, 4, 8, 15, 30, 30, 30, 30)


def is_transient_api_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, RateLimitError, TimeoutError, ConnectionError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


def call_with_api_retries(operation):
    """Retry transient endpoint failures and re-raise the final original error."""
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            result = operation()
            if attempt:
                logger.info(f"[api-retry] attempt {attempt + 1}/{API_MAX_ATTEMPTS} succeeded")
            return result
        except Exception as error:
            remaining = API_MAX_ATTEMPTS - (attempt + 1)
            if not is_transient_api_error(error) or attempt + 1 == API_MAX_ATTEMPTS:
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
    raise AssertionError("retry loop exhausted without returning or raising")


class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

    @property
    def extractable_json_structured_output(self):
        return JSON_BLOCK_PATTERN

    def _api_call_with_retry(self, client: OpenAI, messages: list) -> str:
        def request():
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extra_body=self.config.extra_body,
            )
            return response.choices[0].message.content

        return call_with_api_retries(request)
