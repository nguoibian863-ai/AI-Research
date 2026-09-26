import json
import logging
from typing import Any, Dict, Optional, Type
import httpx
from pydantic import BaseModel, ValidationError
from backend.llm.backend import LLMBackend, LLMResponse
from backend.core.errors import ModelInferenceError

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        timeout: float = 120.0,
        temperature: float = 0.1
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("num_predict", 1024)
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()

            content = data.get("response", "")
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
        
        # Guide system prompt with explicit JSON formatting
        enriched_system = (
            (system_prompt + "\n" if system_prompt else "") +
            "You MUST respond ONLY with a valid JSON object matching this schema. "
            "Do not include markdown codeblocks or preamble.\n"
            f"JSON Schema: {json.dumps(schema_dict)}"
        )

        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "system": enriched_system,
            "format": schema_dict,  # Ollama 0.3+ native JSON schema constraint
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("num_predict", 1024)
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()

            raw_content = data.get("response", "").strip()
            # Clean markdown code block wraps if model outputs them
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
