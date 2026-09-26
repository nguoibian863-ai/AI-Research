# Kiến trúc hiện tại

Mô tả những gì **đã được implement** (không phải kế hoạch). Kế hoạch đầy đủ: `local_deep_research_agent_senior_plan.md`.

## Luồng end-to-end (`ResearchEngine.run_week1`)

```text
Question
→ PLAN      LLM → ResearchPlanSchema (JSON schema constrained, retry 1 lần khi sai)
→ loop (tối đa max_research_steps):
    SEARCH    open_questions (nếu có) hoặc LLM sinh 2–4 query ngắn
              → lọc domain chặn → lọc liên quan (core entities, whole-word)
              → dedup canonical key → source_score (0–100, bỏ nếu < 40) → lưu sources
    FETCH     HTML → HTMLCleaner (trafilatura) | PDF → PDFFetchTool (PyMuPDF, page-aware)
              → documents → SectionAwareChunker → chunks (≤ max_total_chunks)
    CLEAN     chunk các document chưa có chunk
    RETRIEVE  BM25 + FAISS (FastEmbed) → RRF → ScoreReranker (boost đã scale)
              + retrieve riêng từng core entity
    EVALUATE  coverage theo chunk → thực thể thiếu thành open_questions → quay lại SEARCH
→ EXTRACT   từng chunk (≤5): LLM đề xuất facts → find_quote_in_text → expand câu
            → verify_atomic_fact (độ dài, subject, số liệu, so sánh) → raw_evidences + evidences
→ WRITE     prompt chỉ gồm evidence [E#] → report → Evidence & Provenance Table
            → claims từ câu trong report (SUPPORTED / UNSUPPORTED)
→ DONE | PARTIAL (kèm error_message giải thích)
```

## Trạng thái & kiểm soát

- 14 phase (`state.py`), transition hợp lệ trong `transitions.py`; terminal: DONE, PARTIAL, FAILED, CANCELLED.
- `ExecutionBudgetTracker`: search/fetch/LLM calls, runtime; counters lưu trong bảng `sessions`.
- `_handle_phase_error`: lỗi runtime → PARTIAL (có source) hoặc FAILED; bỏ qua `StateTransitionError` và state đã terminal.
- HTTP mapping: 404 SessionNotFound, 409 StateTransition, 429 Budget, 502 ModelInference.

## Dữ liệu (SQLite, `data/research.db`)

```text
sessions → queries, sources → documents → chunks
sessions → raw_evidences (source_id, chunk_id, page, section, char_start/end)
         → evidences (subject, predicate, object_json)
         → claims (câu report, evidence_ids, verification, status)
         → reports, trajectories (partition raw/candidate/gold/eval)
```

Offset bất biến: `document.text[chunk.char_start:chunk.char_end] == chunk.text`; `raw_evidence.char_start/end` tuyệt đối theo document.

## Chưa có (theo plan)

- OpenAlex/arXiv provider (12.5)
- Cross-encoder reranker (18.1), NLI entailment verifier (23)
- Writer theo section (27.1), UI đầy đủ (29.1) — đã có Session Viewer chỉ đọc tại `/ui` (29.0)
- Trajectory cho `query_generation` / `evidence_extraction` (43.1), `training/` (45.4)
- Research memory liên session (19.1)
