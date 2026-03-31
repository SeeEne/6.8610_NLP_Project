"""Unified LLM client using OpenRouter's OpenAI-compatible API.

Provides a single interface to call GPT, Claude, Gemini, DeepSeek, Qwen,
and any other model available on OpenRouter. Supports temperature sampling
for pass@k evaluation and structured output parsing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Model registry — loaded from config/models.yaml
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _load_model_registry() -> dict[str, str]:
    path = _CONFIG_DIR / "models.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


MODEL_REGISTRY: dict[str, str] = _load_model_registry()


@dataclass
class ModelConfig:
    """Configuration for a single LLM call."""

    model: str  # alias from MODEL_REGISTRY or a full OpenRouter model ID
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    n: int = 1  # number of completions (for pass@k sampling)

    @property
    def model_id(self) -> str:
        return MODEL_REGISTRY.get(self.model, self.model)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    model: str
    choices: list[str]  # one per n
    usage: dict  # prompt_tokens, completion_tokens, total_tokens
    latency_s: float
    raw: dict = field(default_factory=dict, repr=False)


class LLMClient:
    """Unified client for calling LLMs through OpenRouter.

    Usage:
        client = LLMClient()  # reads OPENROUTER_API_KEY from env
        resp = client.call(
            ModelConfig(model="gpt-4o", temperature=0.8, n=5),
            system="You are a Python code generator.",
            prompt="Write a function that ...",
        )
        for code in resp.choices:
            print(code)
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY in your .env file "
                "or pass api_key= to LLMClient()."
            )
        self._client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self._api_key,
        )

    def call(
        self,
        config: ModelConfig,
        prompt: str,
        system: str = "",
    ) -> LLMResponse:
        """Make a synchronous LLM call.

        Args:
            config: Model and sampling parameters.
            prompt: The user message.
            system: Optional system message.

        Returns:
            LLMResponse with one choice per requested completion.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        response = self._client.chat.completions.create(
            model=config.model_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            n=config.n,
        )
        latency = time.time() - t0

        raw = response.model_dump()
        choices = [c.message.content or "" for c in response.choices]
        usage = raw.get("usage", {}) or {}

        return LLMResponse(
            model=config.model_id,
            choices=choices,
            usage=usage,
            latency_s=round(latency, 3),
            raw=raw,
        )

    def call_batch(
        self,
        config: ModelConfig,
        prompts: list[str],
        system: str = "",
    ) -> list[LLMResponse]:
        """Call the same model on multiple prompts sequentially.

        For high-throughput use, consider wrapping with asyncio or threading.
        """
        return [self.call(config, prompt, system) for prompt in prompts]

    @staticmethod
    def available_models() -> dict[str, str]:
        """Return the alias -> model_id mapping."""
        return dict(MODEL_REGISTRY)
