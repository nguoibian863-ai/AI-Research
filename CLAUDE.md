# CLAUDE.md — Local Deep Research Agent

Context gốc cho AI coding assistant. Đọc file này trước khi làm bất cứ việc gì trong repo.

- Plan tổng thể: `local_deep_research_agent_senior_plan.md` (roadmap mục 37, priority mục 38, Phụ lục A)
- Kiến trúc hiện tại: `docs/context/architecture.md`
- Quyết định thiết kế đã chốt: `docs/context/decisions.md`
- Trạng thái & vấn đề còn mở: `docs/context/status.md` (cập nhật sau mỗi vòng review)
- Báo cáo từng tuần: `docs/week*_review.md`

## Dự án

Research agent local-first, single-user, chạy trên laptop GPU 4GB VRAM + 32GB RAM.
Mục tiêu cốt lõi: **mọi kết luận quan trọng đều truy ngược được về quote nguyên văn + page/section + source đã đọc.**

## Nguyên tắc bất biến (không được vi phạm)

1. **Python quyết định transition.** LLM không tự quyết vòng lặp, không tự gọi tool tự do (không LangGraph/ReAct/CrewAI).
2. **SQLite là source of truth.** FAISS/BM25 chỉ là index, rebuild được từ SQLite.
3. **Không bao giờ đoán source.** Không resolve được `source_id` → reject evidence, không fallback sang source khác.
4. **Raw evidence bất biến.** `raw_evidences` chỉ ghi quote nguyên văn đã kiểm chứng, không ghi dữ liệu giả/placeholder.
5. **Số liệu phải có trong quote.** Mọi con số trong `value`/`statement`/câu report có `[E#]` phải xuất hiện trong quote được trích dẫn.
6. **Không fake confidence.** Không hard-code `verified=True`, `quality_score` cao, hay `DONE` khi chưa kiểm chứng. Thiếu evidence → `PARTIAL` kèm lý do.
7. **Trạng thái terminal bất biến.** Không gán `state.phase` trực tiếp; mọi chuyển trạng thái đi qua `StateMachine.transition`. `StateTransitionError` là lỗi client, không làm hỏng session.
8. **LLM backend swappable.** Code nghiệp vụ chỉ dùng `LLMBackend`, không gọi Ollama trực tiếp.

## Quy ước làm việc

- **Làm việc và push trực tiếp trên `main`** (đã được chủ repo đồng ý).
- **Mọi fix phải kèm test hồi quy** tái hiện lỗi; test phải fail trên code cũ.
- Test không được gọi mạng, không cần GPU, không ghi vào `data/research.db` thật: dùng fixture trong `tests/conftest.py` (`isolated_engine`, `client`, `LocalHashEmbeddingBackend`, `MockLLMBackend`).
- Test cần mạng/model thật → đánh dấu `@pytest.mark.slow`.
- Báo cáo tiến độ chỉ dựa trên kết quả đã kiểm chứng (pytest + chạy E2E thật). Không ghi "100%" khi chưa chạy thật.
- Commit message tiếng Việt, dạng `feat(weekN): ...`, `fix(weekN): ...`, `docs(...): ...`.
- Không copy nguyên code repo khác vào project; chỉ ghi chú ý tưởng, kiểm tra license (Phụ lục A của plan).

## Lệnh thường dùng

```bash
pip install -r requirements.txt
python -m pytest -q                      # toàn bộ test (~3s, offline)
python -m pytest -q -m "not slow"        # bỏ test cần mạng/model
uvicorn backend.main:app --host 127.0.0.1 --port 8000   # hoặc scripts/start_backend.ps1 (Windows)
python scripts/benchmark_ollama.py       # đo tokens/s, VRAM, JSON constrained decoding
python scripts/show_session.py           # liệt kê session; thêm <session_id> hoặc --latest để xem chi tiết + xuất data/reports/<id>.md
```

E2E thật (cần Ollama chạy local):

```bash
curl -X POST http://127.0.0.1:8000/api/research/run -H "Content-Type: application/json" \
  -d '{"goal": "Compare YOLOv8 and RT-DETR accuracy and latency on COCO"}'
```

## Môi trường & model

- Máy chủ repo: Windows, project tại `D:\AI\AI_Research`, dùng `.venv` + PowerShell.
- LLM: Ollama `hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M` (~2.2GB VRAM, ~68 tok/s), `num_ctx` 4096 — **không tăng context** vì VRAM chỉ còn ~0.4GB.
- SmolLM3 tắt reasoning bằng `system_prefix: "/no_think"` trong **system prompt** (đặt trong user prompt không có tác dụng). Output có thể chỉ chứa `</think>` → đã có `strip_think_block`.
- Embedding: FastEmbed `BAAI/bge-small-en-v1.5` (CPU). Lần chạy đầu cần mạng để tải model; lỗi tải → server báo lỗi ngay lúc khởi động (không fallback lặng lẽ).
- Config: `config/settings.yaml` (limits, llm, retrieval).

## Bản đồ code

```text
backend/core/       engine.py (orchestrator run_week1), state.py, transitions.py, budget.py, coverage.py
backend/llm/        backend.py (interface), ollama.py, mock.py, schemas.py
backend/tools/      web_search.py (ddgs), web_fetch.py, pdf_fetch.py
backend/parsing/    html_cleaner.py, pdf_parser.py, chunker.py
backend/retrieval/  bm25.py, embeddings.py, faiss_index.py, hybrid.py (RRF), reranker.py
backend/evidence/   extractor.py (quote invariance + verify_atomic_fact), verifier.py (citation table)
backend/sources/    dedup.py (canonical key: arXiv/DOI/title/URL)
backend/db/         database.py (schema + migration), repositories.py
backend/api/        research.py (FastAPI routes)
```
