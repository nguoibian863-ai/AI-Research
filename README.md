# Local Deep Research Agent

> **Local-first, single-user, single-PC research tool with grounded evidence and provenance verification.**

## Overview

Local Deep Research Agent is an autonomous research system designed to run on personal hardware (e.g. RTX 3050 Laptop / 4GB VRAM + 32GB RAM). It performs structured deep research on complex technical topics, parses web pages and academic papers, extracts immutable atomic evidence, and generates fully cited reports where **every factual claim is verifiable down to the exact quote, page, and source URL**.

### Key Architectural Pillars
- **LLM as Reasoning Engine:** Swappable backend (Ollama / llama.cpp / mock) with native JSON Schema constrained decoding.
- **Python State Machine:** Strict deterministic state transitions across 14 research phases and hard limit enforcement (anti-infinite loop).
- **SQLite Single Source of Truth:** Immutable raw evidence layer (`raw_evidences`) and partitioned trajectory logging (`data/trajectories/{raw, candidate, gold, eval}`) for Phase 2 fine-tuning.
- **Verification Stack:** 3-layer verification (Source Integrity, Rule-based Numeric Integrity, Semantic Claim Entailment).

---

## Project Structure

```text
AI_Research/
├── backend/
│   ├── api/             # FastAPI REST endpoints
│   ├── core/            # State Machine, limits, budget, errors, engine
│   ├── db/              # SQLite database manager & repositories
│   ├── llm/             # LLMBackend abstraction, schemas, Ollama & Mock backends
│   ├── tools/           # Web search (DuckDuckGo) & fetcher (Trafilatura)
│   ├── sources/         # Source management & deduplication
│   ├── parsing/         # PDF & HTML chunking
│   ├── retrieval/       # BM25 + FAISS hybrid retrieval
│   ├── evidence/        # Atomic evidence extraction & provenance
│   ├── verification/    # Integrity & numeric verification
│   └── writer/          # Grounded report writer & citations
├── config/
│   └── settings.yaml    # System settings & hard limits
├── data/
│   ├── research.db      # SQLite local database
│   └── trajectories/    # Dataset partitioned: raw, candidate, gold, eval
├── scripts/             # Startup PowerShell scripts
└── tests/               # Pytest automated test suite
```

---

## Quick Start

### 1. Requirements
- Python 3.11+
- [Ollama](https://ollama.com/) (optional for local LLM inference, e.g. `qwen3:8b`, `deepseek-r1:8b`)

### 2. Setup Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run Tests
```powershell
.\.venv\Scripts\pytest -v
```

### 4. Start Backend API
```powershell
.\scripts\start_backend.ps1
```
Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.
