"""
Ollama Local Benchmark Script
Measures token generation speed (tokens/sec), latency, VRAM allocation, and verifies JSON schema constrained decoding.
"""

import re
import time
import subprocess
import httpx
import json

OLLAMA_URL = "http://127.0.0.1:11434"
TEST_PROMPT = (
    "Provide a 3-point technical comparison between PointPillars and CenterPoint 3D detectors "
    "for autonomous driving, focusing on NDS, voxelization, and latency."
)


def get_vram_usage():
    """Queries nvidia-smi for current GPU memory usage if available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        ).strip()
        used, total, util = output.split(",")
        return f"{used.strip()} MB / {total.strip()} MB (GPU Util: {util.strip()}%)"
    except Exception:
        return "N/A (nvidia-smi not accessible or running on CPU)"


def get_ollama_ps():
    """Queries Ollama /api/ps endpoint to inspect loaded model VRAM footprint."""
    try:
        res = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=5.0)
        data = res.json().get("models", [])
        if data:
            for m in data:
                size_vram_mb = m.get("size_vram", 0) / (1024 * 1024)
                size_total_mb = m.get("size", 0) / (1024 * 1024)
                return f"{m.get('name')}: VRAM {size_vram_mb:.1f} MB / Total {size_total_mb:.1f} MB"
        return "No model actively loaded in Ollama VRAM"
    except Exception:
        return "Ollama ps not available"


DEFAULT_MODEL = "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"


def no_think_system(model_name: str):
    """SmolLM3 disables its reasoning mode via a '/no_think' system-prompt flag."""
    return "/no_think" if "smollm3" in model_name.lower() else None


def strip_think_block(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # The chat template may prefill "<think>", leaving only the closing tag in the output.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def run_benchmark(model_name: str = DEFAULT_MODEL):
    print(f"\n=======================================================")
    print(f"Benchmarking Local Ollama Model: {model_name} (Profile: Laptop 4GB)")
    print(f"=======================================================")

    # 1. Check server and models
    try:
        res = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        models = [m["name"] for m in res.json().get("models", [])]
        print(f"Ollama server is active. Installed models: {', '.join(models)}")
        if model_name not in models and not any(m.startswith(model_name) for m in models):
            print(f"[WARNING] Model '{model_name}' not found locally. Using installed model if available.")
    except Exception as e:
        print(f"[ERROR] Could not connect to Ollama server at {OLLAMA_URL}: {e}")
        return

    print(f"Initial GPU Status: {get_vram_usage()}")
    system = no_think_system(model_name)

    # 2. Benchmark Raw Generation
    print(f"\n[Test 1] Raw Generation Speed Benchmark...")
    payload = {
        "model": model_name,
        "prompt": TEST_PROMPT,
        "stream": False,
        **({"system": system} if system else {}),
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
            "num_ctx": 2048
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
        print(f"  - Post-gen Ollama VRAM: {get_ollama_ps()}")
        print(f"  - Post-gen GPU Memory : {get_vram_usage()}")
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
        **({"system": system} if system else {}),
        "options": {
            "temperature": 0.1,
            "num_predict": 128,
            "num_ctx": 2048
        }
    }

    start_time = time.time()
    try:
        res = httpx.post(f"{OLLAMA_URL}/api/generate", json=json_payload, timeout=60.0)
        elapsed = time.time() - start_time
        res.raise_for_status()
        data = res.json()
        raw_json = strip_think_block(data.get("response", ""))
        parsed = json.loads(raw_json)
        print(f"  - Structured JSON Output Validated: {parsed}")
        print(f"  - Structured Latency: {elapsed:.2f}s")
    except Exception as e:
        print(f"  - JSON constrained test failed: {e}")


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    run_benchmark(model)
