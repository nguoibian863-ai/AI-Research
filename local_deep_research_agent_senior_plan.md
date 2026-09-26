# Local Deep Research Agent — Senior AI Systems Engineer Implementation Plan

## 0. Executive Summary

### Mục tiêu

Xây dựng một **Local Deep Research Agent** chạy chủ yếu trên máy cá nhân, có khả năng:

- nhận một câu hỏi nghiên cứu;
- lập kế hoạch nghiên cứu có cấu trúc;
- tìm thông tin trên Internet;
- đọc webpage, PDF, paper, GitHub;
- trích xuất evidence;
- lưu provenance đầy đủ;
- kiểm tra số liệu và citation;
- phát hiện thiếu thông tin / conflict;
- viết báo cáo grounded;
- cho phép click từ claim về đúng evidence và source;
- lưu trajectory để phục vụ evaluation và fine-tuning ở Phase 2.

### Scope hệ thống

Hệ thống là:

> **Local-first, single-user, single-PC research tool.**

Không phải cloud platform.

Không phải SaaS.

Không phải distributed crawler.

Không phải multi-agent platform ở MVP.

Internet chỉ đóng vai trò **external information source**.

### Kiến trúc định hướng

```text
User
 ↓
Planner
 ↓
Structured Research Plan
 ↓
Query Generator
 ↓
Search Tools
 ↓
Source Manager
 ↓
Web/PDF Cleaner
 ↓
Hybrid Retriever
 ↓
Raw Evidence Layer
 ↓
Atomic Evidence Extractor
 ↓
Evidence Store
 ↓
Gap Evaluator
 ├── thiếu → Search tiếp
 └── đủ
      ↓
Source Integrity Verifier
      ↓
Evidence Integrity Verifier
      ↓
Semantic Claim Verifier
      ↓
Grounded Writer
      ↓
Claim ↔ Evidence ↔ Source
```

### Nguyên tắc thiết kế

```text
LLM
= reasoning / planning / extraction / synthesis

Python
= state machine / hard limits / validation / transitions

SQLite
= source of truth

BM25 + Vector Index
= retrieval

Tools
= Internet access

Verifier
= integrity + anti-hallucination layer

Local filesystem
= PDFs / cache / reports / trajectories / indexes
```

---

# 1. Project Goals

## 1.1 Primary Goal

Tạo một research system mà:

> Mọi factual claim quan trọng đều có thể truy ngược về evidence gốc.

### Ví dụ

```text
Report Claim
"Model X achieved 71.2 NDS on nuScenes."
      ↓
claim_id
      ↓
evidence_id
      ↓
raw_quote
      ↓
PDF page 7
      ↓
source URL
```

---

## 1.2 Secondary Goals

Hệ thống phải:

- chạy được với model local nhỏ;
- không phụ thuộc cloud database;
- không phụ thuộc framework orchestration nặng;
- có thể thay model backend mà không sửa core research engine;
- có benchmark rõ ràng;
- thu được trajectory sạch;
- sẵn sàng fine-tune bằng LoRA/QLoRA sau MVP.

---

# 2. Non-Goals

Trong MVP không làm:

- Multi-agent architecture;
- LangGraph;
- Kubernetes;
- Redis;
- PostgreSQL;
- Pinecone;
- Cloud deployment;
- Multi-user;
- Authentication phức tạp;
- RBAC;
- distributed crawling;
- autonomous self-training;
- knowledge graph lớn;
- complex long-term memory;
- auto fine-tuning;
- RL;
- 20k training dataset;
- advanced 3D agent visualization.

Các phần này chỉ xem xét khi MVP chứng minh được core value.

---

# 3. Hardware Constraints

## Máy hiện tại

```text
GPU: RTX 3050 Laptop
VRAM: 4GB
RAM: 32GB
OS: Windows
```

## Nguyên tắc resource allocation

### GPU

Chỉ ưu tiên cho:

- local LLM inference;
- optional CUDA acceleration.

### CPU + RAM

Chạy:

- scraping;
- HTML cleaning;
- PDF parsing;
- embeddings;
- SQLite;
- BM25;
- FAISS;
- backend;
- UI;
- preprocessing.

---

# 4. Model Strategy

## 4.1 MVP

Model local nhỏ, quantized.

Candidates:

```text
SmolLM3-3B Q4
Qwen small coding/reasoning model
```

Runtime:

```text
llama.cpp
```

hoặc:

```text
Ollama
```

Chỉ chọn một runtime chính thức sau benchmark ban đầu.

---

## 4.2 Không hard-code model

Interface:

```python
class LLMBackend:
    def generate(self, prompt, **kwargs):
        ...

    def structured_generate(self, prompt, schema, **kwargs):
        ...
```

Sau này có thể thay:

```text
3B
→ 7B
→ 14B
→ 32B
→ API model
```

mà không thay core engine.

---

# 5. Core Architecture

```text
                    USER
                      │
                      ▼
              Research Planner
                      │
                      ▼
              Structured Plan
                      │
                      ▼
               Query Generator
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
        Web          arXiv        GitHub
          │           │            │
          └───────────┼────────────┘
                      ▼
               Source Manager
                      │
                      ▼
               Parser / Cleaner
                      │
                      ▼
             Hybrid Retrieval
            BM25 + Vector Search
                      │
                      ▼
                Raw Evidence
                      │
                      ▼
            Atomic Fact Extractor
                      │
                      ▼
               Evidence Store
                      │
                      ▼
                Gap Evaluator
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
      Need More               Sufficient
           │                     │
           └──── SEARCH          ▼
                         Integrity Verification
                                  │
                                  ▼
                         Semantic Verification
                                  │
                                  ▼
                              Claim Store
                                  │
                                  ▼
                          Grounded Writer
                                  │
                                  ▼
                    Claim ↔ Evidence ↔ Source
```

---

# 6. Local Project Architecture

```text
ResearchAI/
│
├── backend/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── research.py
│   │   ├── sessions.py
│   │   ├── sources.py
│   │   └── reports.py
│   │
│   ├── core/
│   │   ├── engine.py
│   │   ├── state.py
│   │   ├── transitions.py
│   │   ├── limits.py
│   │   ├── errors.py
│   │   └── budget.py
│   │
│   ├── llm/
│   │   ├── backend.py
│   │   ├── llama_cpp.py
│   │   ├── ollama.py
│   │   ├── schemas.py
│   │   └── constrained_output.py
│   │
│   ├── tools/
│   │   ├── web_search.py
│   │   ├── web_fetch.py
│   │   ├── arxiv.py
│   │   ├── github.py
│   │   └── pdf_fetch.py
│   │
│   ├── sources/
│   │   ├── manager.py
│   │   ├── deduplicator.py
│   │   ├── metadata.py
│   │   └── ranking.py
│   │
│   ├── parsing/
│   │   ├── html_cleaner.py
│   │   ├── pdf_parser.py
│   │   ├── chunker.py
│   │   └── table_parser.py
│   │
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── embeddings.py
│   │   ├── faiss_index.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── evidence/
│   │   ├── raw.py
│   │   ├── extractor.py
│   │   ├── repository.py
│   │   ├── models.py
│   │   └── provenance.py
│   │
│   ├── verification/
│   │   ├── source_integrity.py
│   │   ├── evidence_integrity.py
│   │   ├── numeric.py
│   │   ├── semantic.py
│   │   ├── contradiction.py
│   │   └── metrics.py
│   │
│   ├── writer/
│   │   ├── report.py
│   │   └── citation.py
│   │
│   └── db/
│       ├── database.py
│       ├── repositories.py
│       └── migrations/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── types/
│
├── data/
│   ├── research.db
│   ├── pdf/
│   ├── web_cache/
│   ├── indexes/
│   ├── evidence/
│   ├── reports/
│   └── trajectories/
│       ├── raw/
│       ├── candidate/
│       ├── gold/
│       └── eval/
│
├── evaluation/
│   ├── benchmark.json
│   ├── cases/
│   ├── metrics.py
│   └── run_eval.py
│
├── training/
│   ├── prepare_dataset.py
│   ├── train_lora.py
│   └── evaluate_adapter.py
│
├── scripts/
│   ├── start_backend.ps1
│   ├── start_frontend.ps1
│   ├── start_llm.ps1
│   └── rebuild_index.py
│
├── tests/
│
├── logs/
│
└── config/
    └── settings.yaml
```

---

# 7. State Machine

## 7.1 States

```text
INIT
PLAN
SEARCH
FETCH
CLEAN
RETRIEVE
EXTRACT
EVALUATE
VERIFY
WRITE
DONE
PARTIAL
FAILED
CANCELLED
```

---

## 7.2 Transition principle

Python quyết định transition.

LLM không được tự quyết vòng lặp.

```python
if state.step >= limits.max_research_steps:
    transition("VERIFY")

elif state.open_questions:
    transition("SEARCH")

else:
    transition("VERIFY")
```

---

# 8. Research State

```python
ResearchState = {
    "session_id": "...",
    "goal": "...",
    "plan": {...},

    "visited_queries": [],
    "source_ids": [],
    "evidence_ids": [],
    "claim_ids": [],

    "open_questions": [],

    "step": 0,
    "phase": "INIT",

    "status": "RUNNING"
}
```

Không lưu full HTML/PDF text trong state.

---

# 9. Hard Limits

Initial config:

```python
MAX_RESEARCH_STEPS = 5

MAX_SEARCH_CALLS = 12
MAX_FETCH_CALLS = 20

MAX_SEARCH_RESULTS = 8
MAX_SOURCES_PER_RUN = 15

MAX_TOTAL_CHUNKS = 300
MAX_CHUNK_TOKENS = 600

MAX_EVIDENCE_ITEMS = 30

RETRIEVAL_TOP_K = 5

MAX_LLM_CALLS = 30

MAX_RUNTIME_SECONDS = 600

LLM_CONTEXT = 4096
MAX_OUTPUT_TOKENS = 1024
```

Sau benchmark:

```text
4096
→ 6144
→ 8192
```

nếu phần cứng ổn định.

---

# 10. Token Budget By Phase

Không dùng full context cho tất cả bước.

```python
TOKEN_BUDGET = {
    "planner": 1800,
    "query_generator": 1000,
    "extractor": 3000,
    "gap_evaluator": 2000,
    "verifier": 2000,
    "writer": 6000
}
```

---

# 11. Structured Output

Model 3B bắt buộc dùng constrained output ở các bước quan trọng.

Các schema cần ép:

```text
ResearchPlan
SearchQuery
GapEvaluation
AtomicEvidence
SemanticVerification
```

Ưu tiên:

- JSON Schema;
- GBNF;
- grammar constrained decoding.

Không chỉ dựa vào prompt.

---

# 12. Search Strategy

Không chỉ dùng một loại query.

## 12.1 Discovery Query

Tìm nguồn chính.

```text
3D object detection nuScenes 2026
```

## 12.2 Evidence Query

Tìm metric cụ thể.

```text
"Model X" NDS nuScenes
```

## 12.3 Verification Query

Xác minh claim.

```text
"Model X" 71.2 NDS
```

## 12.4 Contradiction Query

Tìm bằng chứng ngược.

```text
"Model X" reproduction benchmark
```

---

## 12.5 Search Providers

| Provider | Vai trò | Ghi chú |
|---|---|---|
| DuckDuckGo (`ddgs`) | Web search mặc định | Kết quả lẫn lộn, có rate limit |
| OpenAlex API | Nguồn học thuật: DOI, năm, venue, số trích dẫn | Miễn phí, không cần key |
| arXiv API | Paper gốc, arXiv id, version | Dùng cho canonical key (mục 15) |
| SearXNG (tự host, tuỳ chọn) | Metasearch local-first | Giảm phụ thuộc DuckDuckGo |

Metadata từ OpenAlex/arXiv (DOI, arXiv id, năm, venue) được dùng trực tiếp cho `source_score` (mục 14.1) và dedup (mục 15), không cần LLM.

---

# 13. Source Manager

Mỗi source phải có:

```json
{
  "source_id": "src_001",
  "url": "...",
  "title": "...",
  "authors": [],
  "published_at": "...",
  "retrieved_at": "...",
  "source_type": "paper",
  "domain": "...",
  "canonical_key": "..."
}
```

---

# 14. Source Ranking

Không dùng một ranking đơn giản.

Mỗi source đánh giá theo:

```text
authority
primary_source
recency
directness
independence
reproducibility
```

## 14.1 Source Credibility Score

Tính hoàn toàn bằng Python (không gọi LLM), deterministic, test được.

```text
source_score (0–100) =
    authority       30%   domain/venue: arxiv, doi, .edu, docs chính thức, repo gốc
  + primary_source  30%   paper/repo gốc > blog tổng hợp > diễn đàn/Q&A
  + directness      20%   core entities của goal xuất hiện trong title/snippet
  + recency         10%   published_at so với thời điểm research
  + independence    10%   không trùng canonical_key với source khác
```

Không chấm chỉ bằng whitelist domain cố định: một blog tóm tắt trên domain uy tín vẫn phải thấp hơn paper gốc.

## 14.2 Quy tắc sử dụng

```text
score < 40
→ không fetch (tiết kiệm fetch budget)

evidence kế thừa source_score
→ writer ưu tiên evidence từ source điểm cao

2 evidence mâu thuẫn
→ trình bày cả hai kèm source_score (xem mục 25), không tự chọn
```

Lưu `source_score` và từng thành phần vào bảng `sources` để giải thích được vì sao một source bị loại.

> Tham khảo: module credibility scoring của `tarun7r/deep-research-agent` (MIT). Chỉ lấy ý tưởng chấm điểm; không dùng kiến trúc LangGraph/ReAct của repo đó (xem P3).

---

# 15. Source Deduplication

Một nội dung có thể xuất hiện ở:

```text
arXiv
project page
GitHub
blog
```

Không được xem là bốn nguồn độc lập nếu cùng dựa trên một paper.

Cần:

```text
canonical paper id
DOI
arXiv id
normalized title
author matching
```

---

# 16. Parsing Pipeline

## Web

```text
HTML
→ trafilatura
→ clean text
→ metadata
```

## PDF

```text
PDF
→ PyMuPDF
→ page-aware text
→ section detection
→ chunking
```

---

## Nâng cấp tuỳ chọn

| Tool | Dùng cho | Ràng buộc |
|---|---|---|
| `pymupdf4llm` | PDF → Markdown giữ bảng, giảm quote rời rạc kiểu ô bảng | Cùng hệ PyMuPDF, nhẹ |
| GROBID (Docker, CPU) | Tách section, references, metadata paper chính xác | Thêm một service Java |

Không dùng `marker` hoặc `docling` bản đầy đủ trên máy 4GB (quá nặng VRAM).

---

# 17. Chunking Strategy

Không chỉ cắt cứng theo token.

Ưu tiên:

```text
section
→ heading
→ paragraph
→ token limit
```

Config ban đầu:

```python
MAX_CHUNK_TOKENS = 600
CHUNK_OVERLAP = 80
```

---

# 18. Hybrid Retrieval

Không dùng FAISS-only.

```text
Query
  │
  ├── BM25
  │
  └── Vector Search
         │
         ▼
       Merge
         │
         ▼
      Rerank
         │
         ▼
       TOP-K
```

BM25 tốt với:

- model names;
- metrics;
- exact values;
- years;
- versions.

Vector search tốt với:

- semantic questions;
- conceptual matching.

---

## 18.1 Rerank

```text
Merge (RRF)
→ cross-encoder reranker nhỏ chạy CPU (FastEmbed, họ bge-reranker)
→ rule-based boosts chỉ dùng làm tie-breaker, đã scale theo 1/(k+1)
→ TOP-K
```

Reranker lỗi hoặc chưa tải được model → fallback về thứ tự RRF, ghi log rõ ràng.

---

# 19. SQLite Is Source of Truth

SQLite lưu:

```text
sessions
queries
sources
documents
chunks
raw_evidences
evidences
claims
verifications
reports
trajectories
```

FAISS chỉ giữ vector index.

Nếu FAISS hỏng:

```text
rebuild from SQLite
```

---

## 19.1 Cross-Session Research Memory (P2, sau MVP)

Hiện mỗi session bắt đầu từ con số 0. SQLite đã có đủ dữ liệu để làm memory, nhưng chỉ được nhớ những gì **đã kiểm chứng và có provenance**.

| Loại memory | Nội dung | Dùng khi |
|---|---|---|
| Evidence Memory | evidence VERIFIED + raw quote + source từ các session trước | Trước SEARCH: tra theo core entities của goal |
| Source Memory | `source_score`, domain bị chặn, URL fetch lỗi, canonical key | Trước FETCH: bỏ qua source đã biết là kém/lỗi |
| User Preferences | ngôn ngữ report, citation style, lĩnh vực ưu tiên | Khi WRITE |

Luồng:

```text
PLAN
→ Memory Lookup (evidence theo entity, còn hạn)
→ coverage đủ? → dùng lại, chỉ SEARCH phần còn thiếu
→ coverage thiếu → SEARCH như bình thường
```

Quy tắc bắt buộc:

- chỉ nhớ **evidence đã VERIFIED kèm đầy đủ provenance**; không bao giờ nhớ answer/summary do LLM viết;
- evidence dùng lại phải có `retrieved_at`; quá hạn (mặc định 90 ngày, số liệu benchmark có thể 30 ngày) → coi như thiếu, search lại;
- report phải đánh dấu evidence nào lấy từ memory, evidence nào vừa thu thập;
- memory là cache của SQLite, không phải source of truth mới; xoá memory không làm mất dữ liệu gốc;
- mỗi lần dùng lại vẫn đi qua verifier như evidence mới (quote vẫn phải khớp raw text đã lưu).

Acceptance:

- câu hỏi lặp lại dùng ít search/fetch call hơn;
- không có evidence quá hạn trong report;
- report ghi rõ nguồn gốc memory/new.

---

# 20. Raw Evidence Layer

Đây là P0.

Không lưu chỉ normalized fact.

Phải lưu raw text bất biến.

```json
{
  "raw_evidence_id": "raw_001",

  "source_id": "src_010",

  "raw_quote":
    "Model A obtains 71.2 NDS while Model B obtains 72.4.",

  "page": 7,

  "section": "Experiments",

  "char_start": 10241,
  "char_end": 10300
}
```

---

# 21. Atomic Evidence

LLM tạo interpretation từ raw evidence.

```json
{
  "evidence_id": "ev_001",

  "raw_evidence_id": "raw_001",

  "subject": "Model A",

  "predicate": "achieved",

  "object": {
    "metric": "NDS",
    "value": 71.2
  },

  "confidence": 0.94
}
```

Raw evidence không được sửa.

---

# 22. Evidence Rule

```text
1 evidence = 1 atomic fact
```

Không:

```text
Model X uses A, achieves B, is faster than C and beats D.
```

Phải tách thành nhiều evidence.

---

# 23. Verification Architecture

Ba tầng.

## Layer 1 — Source Integrity

Kiểm tra:

```text
URL
title
author
date
paper identity
duplicate
```

---

## Layer 2 — Evidence Integrity

Kiểm:

```text
raw_quote tồn tại?
page đúng?
section đúng?
number đúng?
table cell đúng?
```

---

## Layer 3 — Claim Entailment

LLM chỉ nhận:

```text
Claim
+
Evidence
```

và trả:

```text
SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED
CONTRADICTED
```

Thứ tự ưu tiên checker:

```text
1. NLI / fact-check model nhỏ chạy CPU (họ MiniCheck hoặc cross-encoder NLI DeBERTa-small)
2. LLM 3B chỉ là fallback, không phải checker chính
```

Checker chuyên dụng ổn định hơn dùng chính model sinh report để tự chấm.

---

# 24. Numeric Verification

Rule-based Python xử lý trước.

Ví dụ:

```text
claim = 71.2 NDS
source = 68.4 NDS
```

Result:

```text
NUMERIC_MISMATCH
```

Không cần hỏi LLM.

Áp dụng cho cả câu trong report có `[E#]`:

```text
mọi con số trong câu
→ phải xuất hiện trong exact_quote của ít nhất một evidence được trích dẫn trong câu đó
→ nếu không: claim.status = NUMERIC_MISMATCH, session = PARTIAL
```

Một câu trích dẫn nhiều evidence → 1 claim với nhiều `evidence_ids`, không tách thành nhiều claim trùng nội dung.

Claim chỉ có citation mà chưa qua entailment → status `CITED`, không gán `verified = true`.

---

# 25. Contradiction Detection

Nếu:

```text
source A = 71.2
source B = 69.8
```

Không chọn số ngẫu nhiên.

Tạo:

```text
CONFLICT
```

Writer phải trình bày conflict hoặc source context.

---

# 26. Claim Schema

```json
{
  "claim_id": "cl_001",

  "text": "Model X achieved 71.2 NDS.",

  "evidence_ids": [
    "ev_001"
  ],

  "verification": {
    "citation_entailment": true,
    "numeric_match": true,
    "source_valid": true
  }
}
```

---

# 27. Writer Boundary

Writer chỉ nhận:

```text
goal
verified claims
verified evidences
source metadata
```

Không nhận raw Internet result.

Không được tự tạo metric mới.

## 27.1 Section-by-Section Writing

Với context 4096, không viết cả report trong một lần gọi.

```text
report outline (từ plan)
→ với mỗi section:
     chọn evidence liên quan section đó (retrieval trên evidence store)
     writer viết section với [E#] citations
     verifier kiểm tra section trước khi ghép
→ ghép sections
→ Evidence & Provenance Table
```

Mỗi lần gọi writer chỉ nhận evidence của section đó, giữ prompt trong `TOKEN_BUDGET["writer"]`.

Section không có evidence → ghi rõ "không đủ evidence", không để LLM tự lấp.

---

# 28. Answerability

Trước khi viết:

```text
coverage sufficient?
```

Nếu không đủ:

```text
status = PARTIAL
```

Không được viết chắc chắn khi evidence thiếu.

---

# 29. UI

## Tech Stack

```text
Frontend:
Next.js
TypeScript
Tailwind CSS
shadcn/ui

Backend:
FastAPI
Python
```

---

## MVP UI

Chỉ cần:

```text
Research Input
Progress
Sources (kèm source_score)
Evidence
Rejected Evidence (lý do bị loại: NUMERIC_MISMATCH, SUBJECT_MISMATCH, ...)
Report
Click-to-Verify
```

Progress cập nhật theo phase thật của state machine (không giả lập).

Không làm animation phức tạp.

---

## UI Layout

```text
┌─────────────────────────────────────────────┐
│ Research Query                              │
│ [........................................]  │
│               Start Research                │
├──────────────┬──────────────────────────────┤
│ Progress     │ Current Action               │
│              │                              │
│ ✓ Plan       │ Query #3                     │
│ ✓ Search     │                              │
│ ✓ Fetch      │ Sources: 8                   │
│ → Extract    │ Evidence: 21                 │
│ ○ Verify     │                              │
│ ○ Write      │                              │
├──────────────┴──────────────────────────────┤
│ Report                                      │
│                                             │
│ Model X achieved 71.2 NDS [1]               │
│                                             │
└─────────────────────────────────────────────┘
```

Click `[1]`:

```text
source
→ page
→ section
→ raw quote
→ highlight
```

---

# 30. Cache Strategy

Web/PDF tải về phải cache local.

```text
Internet
 ↓
Local Cache
 ↓
Parser
 ↓
SQLite
```

Không tải lại cùng source nếu hash/version không đổi.

Chỉ cache ở mức source (web/PDF), không cache toàn bộ report theo topic: cùng câu hỏi phải được research lại để không trả về kết luận cũ.

---

# 31. Offline Degradation

Local-first không có nghĩa offline-only.

Khi mất mạng:

```text
Local PDF
Local web cache
Existing evidence
Existing reports
```

vẫn dùng được.

Search mới sẽ fail gracefully.

---

# 32. Error Handling

Không chỉ DONE/FAILED.

Phải có:

```text
DONE
PARTIAL
FAILED
CANCELLED
```

Ví dụ:

```text
web timeout
→ retry
→ fail
→ PARTIAL
```

nếu vẫn còn evidence đủ để tạo report giới hạn.

---

# 33. Observability

Log local:

```text
session_id
phase
step
query
tool
latency
VRAM
RAM
LLM calls
search calls
source count
evidence count
errors
```

Không cần cloud tracing.

---

# 34. Evaluation Framework

## Benchmark Size

MVP:

```text
100 research tasks
```

Chia thành:

```text
10 Simple factual
10 Multi-source factual
10 Numeric benchmark
10 Conflicting sources
10 Missing information
10 Recent information
10 PDF-heavy
10 Table-heavy
10 Long-form synthesis
10 Adversarial/noisy sources
```

---

# 35. Evaluation Metrics

## Citation Coverage

```text
claims with citation
/
factual claims
```

Target:

```text
>95%
```

---

## Citation Entailment

Citation thực sự support claim.

Target:

```text
>90%
```

---

## Numeric Accuracy

Target:

```text
>95%
```

Critical metrics nên hướng đến:

```text
100%
```

---

## Unsupported Claim Rate

Target:

```text
<10%
```

---

## Source Validity

```text
valid source
/
all cited sources
```

Target:

```text
>95%
```

---

## Evidence Recall

```text
important facts found
/
important facts expected
```

Đây là metric bắt buộc để tránh hệ thống "đúng nhưng thiếu".

---

# 36. Runtime Metrics

Log:

```text
VRAM peak
RAM peak
research latency
LLM calls
search calls
fetch calls
sources/read
evidence count
tokens
```

---

# 37. Five-Week Roadmap

---

## Week 1 — Core Engine

### Goal

```text
Question
→ Plan
→ Search
→ Fetch
→ Simple Answer
```

### Tasks

- project scaffold;
- FastAPI;
- LLM backend interface;
- llama.cpp/Ollama benchmark;
- ResearchState;
- state machine;
- hard limits;
- structured plan schema;
- web search tool;
- fetch tool;
- local logging.

### Acceptance Criteria

- một research session chạy end-to-end;
- không infinite loop;
- Planner output valid JSON;
- max step được enforce;
- local LLM backend swap được.

---

# Week 2 — Parsing + Retrieval

### Tasks

- trafilatura;
- PyMuPDF;
- source metadata;
- section-aware chunking;
- BM25;
- embedding model;
- FAISS;
- hybrid retrieval;
- web/PDF cache.

### Acceptance Criteria

- web article parse đúng;
- PDF page-aware;
- exact metric query BM25 tìm được;
- semantic query vector tìm được;
- hybrid retrieval chạy được.

---

# Week 3 — Evidence Engine

### Tasks

- SQLite schema;
- raw evidence;
- atomic evidence;
- provenance;
- source dedupe;
- source credibility scoring (mục 14.1);
- OpenAlex / arXiv search provider (mục 12.5);
- Evidence Store;
- Gap Evaluator;
- open questions;
- search continuation logic.

### Acceptance Criteria

- claim lineage tồn tại;
- raw quote immutable;
- evidence có page/section/source;
- agent biết khi nào còn thiếu evidence;
- source duplicate được phát hiện;
- source có `source_score`, source < 40 không bị fetch.

---

# Week 4 — Verification + UI

### Tasks

- source integrity verifier;
- evidence integrity verifier;
- numeric verifier (kể cả câu có `[E#]`, mục 24);
- semantic verifier bằng NLI model nhỏ chạy CPU (mục 23);
- cross-encoder reranker (mục 18.1);
- contradiction handling;
- claim store;
- grounded writer theo từng section (mục 27.1);
- Next.js UI;
- progress view;
- evidence view (kèm rejected evidence);
- click-to-source.

### Acceptance Criteria

- wrong numeric claim bị reject;
- unsupported claim bị reject;
- click claim truy về raw source;
- conflict được đánh dấu;
- PARTIAL state hiển thị đúng;
- mỗi lần gọi writer nằm trong `TOKEN_BUDGET["writer"]`.

---

# Week 5 — Evaluation + Trajectory

### Tasks

- benchmark 100 tasks;
- run evaluation;
- log resource metrics;
- trajectory schema;
- trajectory logging cho `query_generation` và `evidence_extraction` kèm nhãn verifier (mục 43.1);
- export JSONL theo partition;
- trajectory filtering;
- failure analysis;
- bug fixing;
- architecture stabilization;
- (tuỳ chọn) export report Markdown/HTML + citation style (APA/IEEE).

### Acceptance Criteria

- benchmark chạy reproducible;
- metrics được xuất;
- failure được phân nhóm;
- clean trajectories được lưu;
- MVP scope đóng.

---

# 38. Priority

## P0

- State Machine
- Raw Evidence
- Evidence Store
- Provenance
- Claim ↔ Evidence mapping
- Integrity Verification
- Numeric Verification
- Hard Limits
- LLM backend abstraction

## P1

- Web Search
- PDF
- Hybrid Retrieval
- Source Dedup
- Source Credibility Scoring
- Academic Search Providers (OpenAlex / arXiv)
- Cross-encoder Reranker
- NLI Entailment Verifier
- Gap Evaluation
- Contradiction Detection
- Section-by-Section Writer

## P2

- UI
- GitHub search
- local cache optimization
- report formatting (Markdown/HTML export, citation styles)
- PDF nâng cao (`pymupdf4llm`, GROBID)
- AI development context (Phụ lục A.3)
- Cross-Session Research Memory (mục 19.1)

## P3

- Fine-tuning
- Multi-agent
- LangGraph
- advanced memory
- advanced visualization

---

# 39. Risks

## P0 — Evidence Corruption

### Problem

Extractor hiểu sai raw source.

### Mitigation

```text
raw immutable evidence
+
offset/page
+
evidence integrity verification
```

---

## P0 — Fake Confidence

### Problem

Evidence thiếu nhưng writer vẫn kết luận chắc chắn.

### Mitigation

```text
coverage
+
answerability
+
PARTIAL state
```

---

## P1 — Agent Drift

### Mitigation

```text
goal summary
visited queries
open questions
hard step budget
```

---

## P1 — Source Duplication

### Mitigation

```text
DOI
arXiv ID
canonical URL
normalized title
author matching
```

---

## P1 — PDF Table Parsing

### Mitigation

Phase 1:

- text extraction;
- basic table detection.

Phase 2:

- dedicated table parser.

---

## P1 — 3B Model Reasoning Limits

### Mitigation

- constrained decoding;
- task decomposition;
- short prompts;
- rule-first verification;
- backend swappable.

---

# 40. Definition of Done — MVP

MVP chỉ hoàn thành khi:

- [ ] chạy local;
- [ ] local LLM hoạt động;
- [ ] search Internet được;
- [ ] đọc web được;
- [ ] đọc PDF được;
- [ ] hybrid retrieval hoạt động;
- [ ] raw evidence được lưu;
- [ ] atomic evidence được tạo;
- [ ] provenance đầy đủ;
- [ ] mỗi factual claim có evidence;
- [ ] numeric verification hoạt động;
- [ ] semantic verification hoạt động;
- [ ] contradiction được xử lý;
- [ ] click claim về source;
- [ ] hard limits hoạt động;
- [ ] loop không vô hạn;
- [ ] PARTIAL/FAILED được xử lý;
- [ ] benchmark 100 tasks chạy được;
- [ ] trajectory được lưu;
- [ ] có failure report;
- [ ] không phụ thuộc cloud database.

---

# 41. Phase 2 — Research Model

Chỉ bắt đầu sau khi MVP ổn.

Pipeline:

```text
Research Runs
 ↓
Verifier
 ↓
Trajectory Filtering
 ↓
Human Review
 ↓
Gold Dataset
 ↓
QLoRA
 ↓
A/B Evaluation
```

---

# 42. Trajectory Reusability & Dataset Governance Strategy

## 42.1 Dataset — Tài sản cốt lõi dài hạn (Long-Term Asset)

Dữ liệu train hoàn toàn có thể dùng lại và **bắt buộc phải thiết kế ngay từ đầu để tái sử dụng**:

> Model có thể thay đổi, harness có thể cải tiến, nhưng **Gold Dataset + Provenance đầy đủ** mới là tài sản lâu dài và giá trị nhất của toàn bộ project.

Một clean trajectory chuẩn thu được từ chuỗi workflow end-to-end:

```text
Question
 ↓
Research Plan
 ↓
Search Query
 ↓
Tool Calls
 ↓
Sources
 ↓
Evidence
 ↓
Verification
 ↓
Final Answer
```

---

## 42.2 Các mục đích tái sử dụng chính

Clean trajectory dataset có thể phục vụ nhiều bài toán kỹ thuật khác nhau:

1. **Fine-tune model khác:** Hôm nay train SmolLM3-3B, sau này chuyển sang Qwen 7B, 14B hoặc các model thế hệ mới vẫn tái sử dụng được trọn vẹn dataset.
2. **Train lại cùng model (Iterative Improvement):** Bổ sung thêm các trajectory mới sau mỗi đợt research rồi retrain adapter.
3. **Train từng kỹ năng chuyên biệt (Modular Skill Training):**
   - Query decomposition & planning.
   - Tool selection & query formulation.
   - Atomic evidence extraction & grounding.
   - Citation & entailment behavior.
4. **Evaluation Benchmark (Hold-out Test Set):** Giữ riêng một phần dữ liệu chất lượng cao làm test set chuẩn mực, tuyệt đối không đưa vào train.
5. **Knowledge Distillation:** Sử dụng model mạnh (DeepSeek-R1, Qwen-14B/32B hoặc Frontier API) tạo trajectory mẫu mực, sau đó distill hành vi chuẩn cho model local 3B.

---

## 42.3 Nguyên tắc phân tầng dữ liệu (Dataset Partitioning)

> **Nguyên tắc sống còn:** Không bao giờ dùng lại nguyên xi mọi raw log.

Toàn bộ log trajectory phải được phân tách thành 4 tầng nghiêm ngặt:

```text
data/trajectories/
├── raw/
│   └── Toàn bộ log của agent sinh ra trong quá trình chạy (chưa qua kiểm định)
│
├── candidate/
│   └── Log đã vượt qua bộ Verifier tự động (Source, Evidence, Entailment hợp lệ)
│
├── gold/
│   └── Dữ liệu sạch, chất lượng cao nhất sau lọc kỹ và Human Review, dùng để train
│
└── eval/
    └── Dữ liệu giữ riêng biệt tuyệt đối làm test set / benchmark, không đưa vào train
```

---

## 42.4 Metadata Governance cho từng Trajectory Sample

Mỗi sample trong dataset bắt buộc phải đính kèm metadata có cấu trúc để phục vụ slicing và filtering dữ liệu:

```json
{
  "task_type": "evidence_extraction",
  "model_source": "smollm3-3b",
  "verified": true,
  "quality_score": 0.96,
  "created_at": "2026-09-26T10:50:00Z",
  "source_ids": ["src_001", "src_004"],
  "language": "en",
  "has_valid_citations": true
}
```

Nhờ metadata này, hệ thống dễ dàng lọc các lát cắt dữ liệu mục tiêu:
- Chỉ lấy sample có `quality_score > 0.95`.
- Chỉ lấy task thuộc nhóm `tool_selection` hoặc `evidence_extraction`.
- Chỉ lấy sample tiếng Anh hoặc tiếng Việt.
- Chỉ lấy sample có citation hoàn toàn chính xác.

---

## 42.5 Phòng chống rò rỉ dữ liệu (Anti-Data-Leakage)

> **Cảnh báo P0 về Data Leakage:**
> Tuyệt đối không lấy cùng một task hoặc biến thể rất gần (near-duplicate/paraphrased) của task trong tập train đưa vào benchmark (`eval/` hoặc `benchmark.json`).

Nếu vi phạm nguyên tắc này, kết quả benchmark sẽ bị sai lệch nghiêm trọng: hệ thống tưởng model thông minh lên nhưng thực chất là model đã "học vẹt" trước đề bài.

---

# 43. Fine-Tuning Strategy

Không train knowledge.

Train behavior.

## Dataset A

```text
Query Decomposition
```

## Dataset B

```text
Tool Selection
```

## Dataset C

```text
Atomic Evidence Extraction
```

## Dataset D

```text
Gap Detection
```

Không gộp tất cả ngay từ đầu.

Bắt đầu với Dataset C (Atomic Evidence Extraction): đây là điểm yếu rõ nhất của model 3B khi chạy thật.

## 43.1 Thu thập dữ liệu ngay trong MVP (không cần GPU)

```text
query_generation
→ input: goal + plan
→ output: queries
→ nhãn: số source liên quan mang về

evidence_extraction
→ input: chunk
→ output: facts
→ nhãn: verifier (VERIFIED / NUMERIC_MISMATCH / SUBJECT_MISMATCH / ...)
```

Fact VERIFIED → mẫu SFT. Cặp (VERIFIED, bị loại) trên cùng chunk → mẫu DPO.

Metadata theo mục 42.4. Xuất JSONL theo partition vào `data/trajectories/`.

## 43.2 Nguồn dữ liệu

| Nguồn | Vai trò |
|---|---|
| Trajectory của chính hệ thống | Nguồn chính, đúng schema và format JSON |
| Chưng cất từ model lớn local (`qwen3:14b`, `deepseek-r1:8b`) | Teacher tạo mẫu, verifier lọc lại |
| distilabel | Pipeline sinh dữ liệu tổng hợp |
| Argilla hoặc Label Studio | Human review candidate → gold |

Dùng API cloud làm teacher → kiểm tra điều khoản sử dụng output để train.

---

# 44. Phase 2 Dataset Targets

Initial:

```text
500–2,000 clean samples
```

Sau đó:

```text
5k
10k
20k
```

nếu thực sự có dữ liệu chất lượng.

## 44.1 Dataset công khai tham khảo

| Kỹ năng | Dataset |
|---|---|
| A — Query Decomposition | HotpotQA, MuSiQue, 2WikiMultihopQA, StrategyQA |
| B — Tool Selection | Dataset function-calling (họ xLAM/APIGen) + trajectory của hệ thống |
| C — Atomic Evidence Extraction | QASPER, SciFact, FEVER |
| D — Gap Detection / Citation | ALCE, ExpertQA |

Quy tắc:

- chuyển về đúng schema của hệ thống (vd. `ExtractedEvidencesSchema`);
- dữ liệu của chính hệ thống chiếm ≥ 50% mỗi dataset;
- kiểm tra license từng dataset (nhiều bộ chỉ cho phép phi thương mại);
- dedup và kiểm tra leakage với `eval/` (mục 42.5).

---

# 45. Phase 2 Training

```text
Base:
SmolLM3-3B

Method:
QLoRA

Quantization:
4-bit

Train:
LoRA adapter only
```

Không full fine-tune trên phần cứng hiện tại.

## 45.1 Framework

| Tool | Khi nào dùng |
|---|---|
| Unsloth | Mặc định cho QLoRA ít VRAM (kiểm tra hỗ trợ SmolLM3 trước) |
| TRL + PEFT + bitsandbytes | Khi Unsloth không hỗ trợ model; `SFTTrainer`, `DPOTrainer` |
| LLaMA-Factory | Cần giao diện / cấu hình YAML |
| Axolotl | Train bằng config, chạy trên Linux/WSL |

## 45.2 Ràng buộc 4GB VRAM

```text
sequence length: 1024–2048
batch size: 1 + gradient accumulation
gradient checkpointing: bật
```

Chạy thử một lượt ngắn để đo VRAM trước khi train thật. Không đủ → thuê GPU cloud theo giờ chỉ cho bước train, inference vẫn local.

## 45.3 Định dạng & triển khai

- train đúng chat template của SmolLM3 ở chế độ `/no_think` như lúc chạy thật;
- LoRA adapter → GGUF (llama.cpp) → `ADAPTER` trong Ollama Modelfile;
- LLM backend không đổi, chỉ đổi model name trong config.

## 45.4 Cấu trúc thư mục

```text
training/
├── export_trajectories.py   # SQLite → JSONL theo partition/task_type
├── promote.py               # raw → candidate (verifier) → gold (human review)
├── dedup_leakage.py         # chống trùng train/eval
├── datasets/                # build Dataset A–D
├── train_qlora.py
├── configs/smollm3_qlora.yaml
└── eval_ab.py               # base vs LoRA (mục 46)
```

---

# 46. A/B Evaluation

```text
Base model
vs
Research LoRA
```

Metrics:

- query quality;
- search efficiency;
- tool selection;
- evidence extraction;
- citation accuracy;
- completion rate.

Kiểm tra thêm kỹ năng chung không bị suy giảm (EleutherAI `lm-evaluation-harness`).

Nếu adapter không cải thiện đáng kể:

```text
không deploy
```

---

# 47. Senior Engineering Recommendation

Nếu đây là project của team, kiến trúc được **approve để bắt đầu implementation** với điều kiện:

1. Raw Evidence Layer là bắt buộc.
2. Python giữ quyền transition.
3. SQLite là source of truth.
4. Retrieval dùng hybrid, không vector-only.
5. Verifier có Source Integrity + Evidence Integrity + Semantic Entailment.
6. LLM backend phải swappable.
7. Fine-tuning không nằm trong MVP.
8. Multi-agent không nằm trong MVP.
9. UI chỉ làm phần phục vụ verification.
10. Benchmark phải có trước khi tuyên bố model/harness cải thiện.

---

# 48. Final Architecture Principle

```text
Model nhỏ không phải vấn đề lớn nhất.

Harness yếu mới là vấn đề lớn.
```

Project này nên được xây theo triết lý:

```text
LLM
không phải
Source of Truth

LLM
là
Reasoning Component

Source of Truth
=
Evidence + Provenance
```

Core value cuối cùng:

> **Mọi kết luận quan trọng đều truy ngược được về evidence thực tế mà hệ thống đã đọc.**

Đó là tiêu chuẩn kỹ thuật để phân biệt project này với một RAG/Web Search wrapper thông thường.

---

# Phụ lục A — Tooling, References & AI Development Context

## A.1 Repo tham khảo

Chỉ học ý tưởng, không dùng làm nền. Kiểm tra license trước khi sao chép bất kỳ đoạn code nào và ghi rõ nguồn.

| Repo | Học gì | Không dùng |
|---|---|---|
| `tarun7r/deep-research-agent` (MIT) | Credibility scoring (mục 14.1), viết report theo section, UI tiến độ | LangGraph/ReAct, citation chỉ ở cấp URL |
| `stanford-oval/storm` | Outline nhiều góc nhìn, viết section có citation (mục 27.1) | — |
| `langchain-ai/local-deep-researcher` | Vòng lặp tóm tắt → tìm gap → search tiếp với Ollama (Gap Evaluator) | Để LLM tự quyết vòng lặp |
| `assafelovic/gpt-researcher` | Chọn/tổng hợp nhiều nguồn, định dạng report | — |
| `huggingface/smol-course` | SFT/DPO/eval cho model nhỏ (Phase 2) | — |
| Search-R1 và tương tự | Thiết kế reward, format trajectory cho search | RL training (không chạy được trên 4GB) |
| EleutherAI `lm-evaluation-harness` | Kiểm tra suy giảm kỹ năng sau fine-tune | — |

## A.2 Không đưa vào project

- LangGraph, CrewAI hoặc agent ReAct tự do (vi phạm "Python quyết định transition");
- `marker`, `docling` bản đầy đủ (quá nặng cho 4GB);
- Ragas/DeepEval với LLM judge cần model mạnh hoặc API cloud;
- cache toàn bộ report theo topic (mục 30).

## A.3 AI Development Context

Cấu trúc để AI coding assistant (Claude Code) làm việc nhất quán giữa các phiên:

```text
CLAUDE.md                     # nguyên tắc bất biến, lệnh test, ràng buộc phần cứng
.claude/
├── settings.json             # hook SessionStart: cài dependency + chạy pytest
└── skills/
    ├── review-project/       # quy trình review: fetch → diff → pytest → probe → đối chiếu acceptance
    ├── run-e2e/              # checklist chạy thật với Ollama
    └── week-acceptance/      # kiểm tra tiêu chí nghiệm thu từng tuần
docs/
├── context/                  # kiến trúc hiện tại, quyết định thiết kế (ADR)
├── references/               # ghi chú repo/tool bên ngoài: link, license, học gì, không dùng gì
└── reviews/                  # báo cáo review từng tuần
third_party/                  # (gitignore) clone repo tham khảo để đọc, không commit
```

`CLAUDE.md` tối thiểu phải ghi:

- Python quyết định transition, LLM không tự quyết vòng lặp;
- không gán source khi không chắc chắn, reject thay vì đoán;
- mọi số liệu phải xuất hiện nguyên văn trong quote;
- mọi fix phải kèm test hồi quy;
- báo cáo tiến độ chỉ dựa trên kết quả đã kiểm chứng (test + chạy thật).

Skill có sẵn nên dùng: `/code-review` cho mỗi commit quan trọng, `/security-review` trước khi mở API ra ngoài `127.0.0.1` (SSRF qua fetch URL, đọc file local qua PDF tool).
