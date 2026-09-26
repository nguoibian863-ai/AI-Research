# Local Deep Research Agent

> **Local-first, single-user, single-PC research tool with grounded evidence and provenance verification.**

## Overview

Local Deep Research Agent is an autonomous research system designed to run on personal hardware (e.g. RTX 3050 Laptop / 4GB VRAM + 32GB RAM). It performs structured deep research on complex technical topics, parses web pages and academic papers, extracts immutable atomic evidence, and generates fully cited reports where **every factual claim is verifiable down to the exact quote, page, and source URL**.

---

## Implementation Status by Milestone

### ✅ Week 1: Core Engine & Hardened Foundation (Verified & Passed)
- **Deterministic Python State Machine:** 14 states (`INIT` → `PLAN` → `SEARCH` → `FETCH` → `CLEAN` → `RETRIEVE` → `EXTRACT` → `EVALUATE` → `VERIFY` → `WRITE` → `DONE`/`PARTIAL`/`FAILED`) strictly controlled by Python logic.
- **SQLite Single Source of Truth:** Schema with WAL mode, foreign keys, and persistent budget counters (`search_calls`, `fetch_calls`, `llm_calls`, `tokens_consumed`).
- **P0 Evidence Integrity Guard:** Dummy raw evidence writing removed from fetch; `raw_evidences` table is kept pristine for verified verbatim claim citations only.
- **P0 Trajectory Governance:** Initial synthesis trajectories strictly placed in `raw` partition with `verified=False` and `quality_score=0.0` (zero Fake Confidence before Phase 4 verifier).
- **P0 Full State Restoration (`load_state`):** State, plan, open questions, queries, and sources are restored from SQLite across independent HTTP requests.
- **P0 Active Budget & Step Enforcement:** `ExecutionBudgetTracker` actively enforces `max_search_calls`, `max_fetch_calls`, `max_llm_calls`, `max_runtime_seconds`. `run_week1` executes through `EVALUATE` and `evaluate_next_step()`, properly tracking `state.step`.
- **P1 Robust Error Handling & HTTP Statuses:** Engine transitions to `FAILED` or `PARTIAL` on exceptions, recording `error_message`. FastAPI maps `StateTransitionError` → 409 Conflict, `BudgetExceededError` → 429, `ModelInferenceError` → 502, `KeyError` → 404.
- **P1 Observability & Local Logging:** Dual output to console and `logs/research.log` tagged with `[sess_xxx][PHASE]`, configured in the FastAPI lifespan (no import-time side effects).
- **P2 Benchmark & Isolated Test Suite:** 22/22 automated tests passing in ~0.6s with zero external network dependencies and temporary DB fixtures; added `scripts/benchmark_ollama.py`.

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
