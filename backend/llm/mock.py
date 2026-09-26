import json
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
from backend.llm.backend import LLMBackend, LLMResponse


class MockLLMBackend(LLMBackend):
    """Predictable mock backend for unit testing and offline development."""

    def __init__(self, canned_responses: Optional[Dict[str, Any]] = None):
        self.canned_responses = canned_responses or {}
        self.call_history = []

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        self.call_history.append({"prompt": prompt, "system_prompt": system_prompt, "kwargs": kwargs})
        content = self.canned_responses.get("generate", "Mock response citing [E1] from LLM")
        return LLMResponse(
            content=content,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=len(prompt.split()) + len(content.split()),
            model_name="mock-3b"
        )

    def structured_generate(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        self.call_history.append({
            "prompt": prompt,
            "schema": schema.__name__,
            "system_prompt": system_prompt,
            "kwargs": kwargs
        })

        # Check if canned response matches schema name
        schema_name = schema.__name__
        if schema_name in self.canned_responses:
            data = self.canned_responses[schema_name]
        else:
            # Construct a dummy instance with schema defaults or basic mock data
            data = {}
            for field_name, field_info in schema.model_fields.items():
                if field_info.default is not None and not str(field_info.default).startswith("PydanticUndefined"):
                    data[field_name] = field_info.default
                elif field_info.annotation == str:
                    data[field_name] = f"Mock {field_name}"
                elif field_info.annotation == int:
                    data[field_name] = 1
                elif field_info.annotation == float:
                    data[field_name] = 0.95
                elif field_info.annotation == list or getattr(field_info.annotation, "__origin__", None) == list:
                    data[field_name] = []
                elif field_info.annotation == dict or getattr(field_info.annotation, "__origin__", None) == dict:
                    data[field_name] = {}
                else:
                    data[field_name] = None

        instance = schema.model_validate(data)
        content_json = instance.model_dump_json()

        return LLMResponse(
            content=content_json,
            parsed=instance,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content_json.split()),
            total_tokens=len(prompt.split()) + len(content_json.split()),
            model_name="mock-3b"
        )
