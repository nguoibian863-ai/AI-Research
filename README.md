# Local Deep Research Agent

> **Local-first, single-user, single-PC research tool with grounded evidence and provenance verification.**

## Overview

Local Deep Research Agent is an autonomous research system designed to run on personal hardware (e.g. RTX 3050 Laptop / 4GB VRAM + 32GB RAM). It performs structured deep research on complex technical topics, parses web pages and academic papers, extracts immutable atomic evidence, and generates fully cited reports where **every factual claim is verifiable down to the exact quote, page, and source URL**.

---

## Implementation Status by Milestone

### ✅ Week 1: Core Engine & Deterministic Flow (Completed & Closed)
- **Deterministic Python State Machine:** 14 states (`INIT` → `PLAN` → `SEARCH` → `FETCH` → `CLEAN` → `RETRIEVE` → `EXTRACT` → `EVALUATE` → `VERIFY` → `WRITE` → `DONE`/`PARTIAL`/`FAILED`) strictly controlled by Python logic to prevent infinite LLM loops.
- **SQLite Single Source of Truth:** Schema with WAL mode and foreign keys. Includes immutable raw evidence (`raw_evidences`) and partitioned trajectory storage (`data/trajectories/{raw, candidate, gold, eval}`) for Phase 2 fine-tuning.
- **Full State Restoration (`load_state`):** Guarantees that across decoupled HTTP requests (`/session`, `/plan`, `/search`, `/fetch`), the engine seamlessly reconstructs plan, visited queries, sources, and open questions without state drift.
- **Execution Budget Tracker Enforcement:** Active enforcement of hard limits (`max_search_calls`, `max_fetch_calls`, `max_llm_calls`, `max_runtime_seconds`).
- **Swappable LLM Backend:** Abstract `LLMBackend` interface with `MockLLMBackend` and `OllamaBackend` featuring native JSON Schema constrained decoding, configured with `smollm3:3b` default profile (with `num_ctx: 4096`) optimized for 4GB VRAM.
- **Tools & Deduplication:** Web search tool with query categorization, URL deduplication, and max sources limits. Web fetch tool with Trafilatura HTML cleaning and local disk caching.
- **End-to-End Runner:** `ResearchEngine.run_week1()` and `POST /api/research/run` implementing Question → Plan → Search → Fetch → Grounded Initial Answer.
- **Automated Testing Suite:** 12/12 unit and multi-request integration tests passing in `pytest`.

### ⏳ Week 2: Parsing & Hybrid Retrieval (In Progress)
- [ ] PyMuPDF page-aware and section-aware PDF text extraction.
- [ ] Contextual section-based chunking with token overlap.
- [ ] BM25 keyword index for exact metrics and model names.
- [ ] CPU-optimized FAISS vector search for semantic matching.
- [ ] Hybrid search merge and reranking.

### 📅 Weeks 3–5: Evidence, Verification, UI & Benchmark (Roadmap)
- [ ] Atomic evidence extraction (1 Fact = 1 Evidence) and Provenance.
- [ ] Gap Evaluator and iterative search loop termination.
- [ ] 3-Layer Verification: Source Integrity, Rule-based Numeric Integrity, Semantic Claim Entailment.
- [ ] Grounded Report Writer with claim-to-source click navigation.
- [ ] Next.js Frontend with Click-to-Verify citations.
- [ ] Benchmark 100 research tasks, metrics logging, and clean gold trajectory generation.

---

## Project Structure

```text
AI_Research/
├── backend/
│   ├── api/             # FastAPI REST endpoints (/session, /plan, /search, /fetch, /answer, /run)
│   ├── core/            # State Machine, transitions, limits, budget tracker, engine
│   ├── db/              # SQLite database manager & repository layer
│   ├── llm/             # LLMBackend abstraction, schemas, Ollama & Mock backends
│   ├── tools/           # Web search (DuckDuckGo) & fetcher (Trafilatura + cache)
│   ├── sources/         # Source management & deduplication
│   ├── parsing/         # PDF & HTML chunking
│   ├── retrieval/       # BM25 + FAISS hybrid retrieval
│   ├── evidence/        # Atomic evidence extraction & provenance
│   ├── verification/    # Integrity & numeric verification
│   └── writer/          # Grounded report writer & citations
├── config/
│   └── settings.yaml    # System settings, profiles & hard limits
├── data/
│   ├── research.db      # SQLite local database (Source of Truth)
│   └── trajectories/    # Dataset partitioned: raw, candidate, gold, eval
├── scripts/             # Startup PowerShell scripts
└── tests/               # Pytest automated test suite
```

---

## Quick Start

### 1. Requirements
- Python 3.11+
- [Ollama](https://ollama.com/) (optional, default model: `smollm3:3b` or `qwen3:8b`)

### 2. Setup Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run Automated Tests
```powershell
.\.venv\Scripts\pytest -v
```

### 4. Start Backend API
```powershell
.\scripts\start_backend.ps1
```
Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.

### 5. Execute an End-to-End Research Run
```powershell
curl -X POST http://127.0.0.1:8000/api/research/run -H "Content-Type: application/json" -d '{"goal": "Compare PointPillars and CenterPoint 3D object detection accuracy"}'
```
