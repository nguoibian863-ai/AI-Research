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


def test_strip_think_block_with_prefilled_opening_tag():
    # Observed with SmolLM3 GGUF on Ollama: reasoning text followed by a bare closing tag.
    raw = "Okay, the user just wrote a greeting...\nAlright, that should cover it.\n</think>\n\nHello! How can I assist you today?"
    assert strip_think_block(raw) == "Hello! How can I assist you today?"


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


def test_structured_generate_retries_and_repairs_on_validation_error(monkeypatch):
    """
    P1 Test: JSON retry/repair loop.
    Verifies that structured_generate catches schema validation errors on attempt 1,
    sends error feedback to the LLM in attempt 2, and successfully parses the repaired response.
    Verifies calls_made=2 and total_tokens accumulates across both calls.
    """
    requests_received = []
    responses = [
        "{\"goal\": \"missing tasks\"}",  # Attempt 1: schema error (missing tasks)
        PLAN_JSON                        # Attempt 2: valid JSON matching schema
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests_received.append(json.loads(request.content))
        resp_content = responses.pop(0)
        return httpx.Response(200, json={
            "response": resp_content,
            "prompt_eval_count": 12,
            "eval_count": 8
        })

    real_client = httpx.Client
    monkeypatch.setattr(
        ollama_module.httpx, "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )

    backend = OllamaBackend(system_prefix="/no_think")
    res = backend.structured_generate("Plan this task", schema=ResearchPlanSchema)

    assert len(requests_received) == 2
    assert res.calls_made == 2
    assert res.parsed.tasks[0].task_id == "t1"
    assert res.total_tokens == 40  # (12 + 8) * 2

    # Verify attempt 2 prompt contained validation error feedback
    second_request_prompt = requests_received[1]["prompt"]
    assert "Your previous output failed schema validation" in second_request_prompt
    assert "missing" in second_request_prompt.lower()

