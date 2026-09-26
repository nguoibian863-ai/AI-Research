"""
Ollama Local Benchmark Script
Measures token generation speed (tokens/sec), latency, and verifies JSON schema constrained decoding.
"""

import time
import httpx
import json

OLLAMA_URL = "http://127.0.0.1:11434"
TEST_PROMPT = (
    "Provide a 3-point technical comparison between PointPillars and CenterPoint 3D detectors "
    "for autonomous driving, focusing on NDS, voxelization, and latency."
)


def run_benchmark(model_name: str = "smollm3:3b"):
    print(f"\n=======================================================")
    print(f"Benchmarking Local Ollama Model: {model_name}")
    print(f"=======================================================")

    # 1. Check server
    try:
        res = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        models = [m["name"] for m in res.json().get("models", [])]
        print(f"Ollama server is active. Installed models: {', '.join(models)}")
        if model_name not in models and not any(m.startswith(model_name) for m in models):
            print(f"[WARNING] Model '{model_name}' not found locally. Available models are listed above.")
    except Exception as e:
        print(f"[ERROR] Could not connect to Ollama server at {OLLAMA_URL}: {e}")
        return

    # 2. Benchmark Raw Generation
    print(f"\n[Test 1] Raw Generation Speed Benchmark...")
    payload = {
        "model": model_name,
        "prompt": TEST_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
            "num_ctx": 4096
        }
    }

    start_time = time.time()
    try:
        res = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120.0)
        elapsed = time.time() - start_time
        res.raise_for_status()
        data = res.json()

        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0

        print(f"  - Total Elapsed Time : {elapsed:.2f}s")
        print(f"  - Prompt Eval Tokens : {prompt_eval_count}")
        print(f"  - Generated Tokens   : {eval_count}")
        print(f"  - Generation Speed   : {tokens_per_sec:.2f} tokens/second")
    except Exception as e:
        print(f"  - Benchmark failed: {e}")

    # 3. Benchmark Constrained JSON Output
    print(f"\n[Test 2] JSON Schema Constrained Decoding Benchmark...")
    schema = {
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
            "latency_ms": {"type": "number"},
            "accuracy_nds": {"type": "number"}
        },
        "required": ["model_name", "latency_ms", "accuracy_nds"]
    }
    json_payload = {
        "model": model_name,
        "prompt": "Output the estimated metrics for CenterPoint on nuScenes.",
        "format": schema,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 128,
            "num_ctx": 4096
        }
    }

    start_time = time.time()
    try:
        res = httpx.post(f"{OLLAMA_URL}/api/generate", json=json_payload, timeout=60.0)
        elapsed = time.time() - start_time
        res.raise_for_status()
        data = res.json()
        raw_json = data.get("response", "").strip()
        parsed = json.loads(raw_json)
        print(f"  - Structured JSON Output Validated: {parsed}")
        print(f"  - Structured Latency: {elapsed:.2f}s")
    except Exception as e:
        print(f"  - JSON constrained test failed: {e}")


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
    run_benchmark(model)
