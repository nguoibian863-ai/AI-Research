from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    parsed: Optional[Any] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""


class LLMBackend(ABC):
    """Abstract interface for all LLM backends (Ollama, llama.cpp, Mock, API)."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        """Standard raw text generation."""
        pass

    @abstractmethod
    def structured_generate(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Constrained structured generation strictly returning instances conforming to schema.
        Model 3B/small models require grammar or JSON schema constraints.
        """
        pass
