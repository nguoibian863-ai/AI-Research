import json
import logging
import re
from typing import Any, Dict, Optional, Type
import httpx
from pydantic import BaseModel, ValidationError
from backend.llm.backend import LLMBackend, LLMResponse
from backend.core.errors import ModelInferenceError

logger = logging.getLogger(__name__)

# Reasoning models (SmolLM3, Qwen3, DeepSeek-R1) may emit a <think> block before the answer.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_CLOSE = "</think>"


def strip_think_block(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    # The chat template may prefill the opening <think> tag, so only "</think>" appears in the output.
    if _THINK_CLOSE in text:
        text = text.rsplit(_THINK_CLOSE, 1)[1]
    return text.strip()


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M",
        timeout: float = 120.0,
        temperature: float = 0.1,
        context_window: int = 4096,
        max_output_tokens: int = 1024,
        system_prefix: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        # Prepended to every system prompt, e.g. "/no_think" to disable SmolLM3 reasoning mode.
        self.system_prefix = system_prefix

    def _build_system(self, system_prompt: Optional[str]) -> Optional[str]:
        parts = [p for p in (self.system_prefix, system_prompt) if p]
        return "\n".join(parts) if parts else None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("num_predict", self.max_output_tokens),
                "num_ctx": kwargs.get("num_ctx", self.context_window)
            }
        }
        system = self._build_system(system_prompt)
        if system:
            payload["system"] = system

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()

            content = strip_think_block(data.get("response", ""))
            prompt_eval_count = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)

            return LLMResponse(
                content=content,
                prompt_tokens=prompt_eval_count,
                completion_tokens=eval_count,
                total_tokens=prompt_eval_count + eval_count,
                model_name=self.model
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise ModelInferenceError(f"Ollama generation error: {e}") from e

    def structured_generate(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        schema_dict = schema.model_json_schema()

        base_system = self._build_system(system_prompt)
        enriched_system = (
            (base_system + "\n" if base_system else "") +
            "You MUST respond ONLY with a valid JSON object matching this schema. "
            "Do not include markdown codeblocks or preamble.\n"
            f"JSON Schema: {json.dumps(schema_dict)}"
        )

        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "system": enriched_system,
            "format": schema_dict,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("num_predict", self.max_output_tokens),
                "num_ctx": kwargs.get("num_ctx", self.context_window)
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()

            raw_content = strip_think_block(data.get("response", ""))
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

            parsed_instance = schema.model_validate_json(raw_content)

            prompt_eval_count = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)

            return LLMResponse(
                content=raw_content,
                parsed=parsed_instance,
                prompt_tokens=prompt_eval_count,
                completion_tokens=eval_count,
                total_tokens=prompt_eval_count + eval_count,
                model_name=self.model
            )
        except ValidationError as ve:
            logger.error(f"Ollama structured output schema mismatch: {ve}. Content was: {raw_content}")
            raise ModelInferenceError(f"Schema validation error: {ve}") from ve
        except Exception as e:
            logger.error(f"Ollama structured generation failed: {e}")
            raise ModelInferenceError(f"Ollama call failed: {e}") from e
