import json
import httpx
import pytest
import backend.llm.ollama as ollama_module
from backend.llm.ollama import OllamaBackend, strip_think_block
from backend.llm.schemas import ResearchPlanSchema
from backend.core.errors import ModelInferenceError

PLAN_JSON = json.dumps({
    "goal": "Compare A and B",
    "tasks": [{"task_id": "t1", "description": "Find A metric", "expected_evidence": "mAP"}],
    "key_hypotheses": []
})


@pytest.fixture
def fake_ollama(monkeypatch):
    """Routes OllamaBackend's httpx.Client to an in-process MockTransport and records request payloads."""
    captured = {"requests": [], "response": ""}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["requests"].append(json.loads(request.content))
        return httpx.Response(200, json={
            "response": captured["response"],
            "prompt_eval_count": 10,
            "eval_count": 5
        })

    real_client = httpx.Client
    monkeypatch.setattr(
        ollama_module.httpx, "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    return captured


def test_strip_think_block():
    assert strip_think_block("<think>\nreasoning...\n</think>\n{\"a\": 1}") == "{\"a\": 1}"
    assert strip_think_block("plain answer") == "plain answer"


def test_system_prefix_sent_and_think_block_stripped(fake_ollama):
    backend = OllamaBackend(model="hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M", system_prefix="/no_think")
    fake_ollama["response"] = f"<think>\n\n</think>\n{PLAN_JSON}"

    res = backend.structured_generate("Plan this", schema=ResearchPlanSchema)

    payload = fake_ollama["requests"][0]
    assert payload["model"] == "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"
    assert payload["system"].startswith("/no_think\n")
    assert payload["format"] == ResearchPlanSchema.model_json_schema()
    assert res.parsed.tasks[0].task_id == "t1"
    assert res.total_tokens == 15


def test_generate_without_prefix_omits_system(fake_ollama):
    backend = OllamaBackend()
    fake_ollama["response"] = "<think>hmm</think>Final answer"

    res = backend.generate("Question?")

    assert "system" not in fake_ollama["requests"][0]
    assert res.content == "Final answer"


def test_invalid_json_raises_model_inference_error(fake_ollama):
    backend = OllamaBackend(system_prefix="/no_think")
    fake_ollama["response"] = "{\"goal\": \"missing tasks\"}"

    with pytest.raises(ModelInferenceError):
        backend.structured_generate("Plan this", schema=ResearchPlanSchema)
