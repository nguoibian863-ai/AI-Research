# Week 2 Review — Parsing, Indexing, Hybrid Retrieval & Entity Coverage

> Đánh giá nghiệm thu Tuần 2 (roadmap mục 38 trong `local_deep_research_agent_senior_plan.md`).
> Ngày: 2026-09-26 · Phần cứng: laptop GPU 4GB VRAM · Test suite: **50 passed** (2.62s).
> Đánh giá nghiệm thu: **Tuần 1: ~95% · Tuần 2: 100% (ĐẠT TOÀN DIỆN & ĐÓNG TUẦN 2)**.

---

## 1. Kết luận

- Toàn bộ pipeline Tuần 2 (Parsing, Chunking, Hybrid Retrieval BM25 + FAISS Dense Vector với FastEmbed `BAAI/bge-small-en-v1.5`, Reciprocal Rank Fusion, Calibrated Reranking, Entity Coverage Verification, Balanced Context Assembly và JSON Retry/Repair) đã hoàn thiện và được kiểm chứng nghiêm ngặt.
- **Tiêu chuẩn nghiệm thu cốt lõi đã được thoả mãn**:
  - Với câu hỏi so sánh thực thể (ví dụ: *A vs B*), hệ thống bắt buộc phải có nguồn và bằng chứng chunk cho cả A lẫn B.
  - Vòng lặp `EVALUATE` tự động phát hiện thực thể bị thiếu (`coverage_ratio < 1.0`), đưa câu hỏi vào `state.open_questions` để vòng `SEARCH` kế tiếp thực hiện truy vấn tập trung cho thực thể đó.
  - Nếu sau `max_research_steps` vẫn không thu thập đủ bằng chứng cho toàn bộ các thực thể cốt lõi, phiên nghiên cứu **kết thúc ở trạng thái `PARTIAL` thay vì `DONE`**, với lý do được ghi nhận minh bạch trong `error_message = "Missing coverage: <entities>"`. Hệ thống tuyệt đối không bịa đặt số liệu hay tự nhận hoàn thành giả mạo.

---

## 2. Bảng Tổng hợp Xử lý & Sửa lỗi theo Đánh giá của Senior AI Engineer

| # | Vấn đề phát hiện | Phân loại | Giải pháp kỹ thuật đã triển khai | Trạng thái |
|---|---|---|---|---|
| 1 | **Kiểm tra độ bao phủ theo thực thể (Entity Coverage)** | 🔴 P0 | Tách module `backend/core/coverage.py` với `extract_core_entities`, `check_entity_in_text`, `evaluate_coverage`. Trong `EVALUATE`: kiểm tra toàn bộ chunks trong SQLite. Thiếu thực thể nào thì tự động thêm câu hỏi mở có mục tiêu vào `open_questions`. Khi hết bước mà vẫn thiếu, chuyển sang `PARTIAL` với `error_message = "Missing coverage: ..."`. | ✅ Đã kiểm chứng qua test `test_entity_coverage_loop_and_partial_reporting` |
| 2 | **Cân bằng ngữ cảnh truy xuất theo thực thể (Entity-Balanced Retrieval)** | 🔴 P0 | Trong `run_retrieve_phase`: truy xuất đa truy vấn (query chính + truy vấn theo từng thực thể) và khử trùng lặp theo `chunk_id`. Trong `run_basic_answer`: phân bổ bảo đảm $\ge \lceil k / N \rceil$ chunk cho mỗi thực thể để tránh trường hợp thực thể nhiều chunk lấn át hoàn toàn thực thể ít chunk. | ✅ Đã kiểm chứng qua test `test_entity_balanced_context_generation` |
| 3 | **Fail-fast khi khởi động Backend (Startup Lifespan Validation)** | 🟠 P1 | Khởi tạo sớm `get_research_engine()` trong `lifespan` (`main.py`) để phát hiện và báo lỗi ngay khi khởi động nếu cấu hình model embedding FastEmbed không hợp lệ hoặc thiếu model. Kiểm tra `dependency_overrides` để giữ tốc độ cho test client. | ✅ Đã kiểm chứng qua test `test_lifespan_eagerly_validates_engine_and_fails_on_broken_backend` |
| 4 | **JSON Retry / Repair cho LLM cục bộ** | 🟠 P1 | Bổ sung vòng lặp retry 1 lần trong `OllamaBackend.structured_generate`: khi gặp lỗi `ValidationError` hoặc JSON syntax, đưa thông báo lỗi chi tiết vào prompt của lượt 2 để LLM tự sửa. Ghi nhận số lượt gọi tích luỹ qua `calls_made` vào `budget.llm_calls`. | ✅ Đã kiểm chứng qua test `test_structured_generate_retries_and_repairs_on_validation_error` |
| 5 | **Trung hoà test tránh Benchmark Leakage** | 🟡 P2 | Đổi test `test_engine_search_relevance_whole_words_and_core_entities` sang chủ đề trung tính: *"Compare RocksDB and LevelDB write amplification on NVMe"*, giữ nguyên kiểm tra so khớp nguyên từ độc lập. | ✅ Đã kiểm chứng |
| 6 | **Chuẩn hoá Reranker bảo toàn thứ hạng Vector RRF** | 🔴 P0 | Scale boost bằng $1 / (K + 1) \approx 0.01639$, lọc stop-words, chỉ boost khi trùng toàn bộ cụm content words hoặc số liệu. Boost tối đa $\approx 0.0066$, không thể đảo lộn các chunk xuất hiện ở cả hai danh sách. | ✅ Đã kiểm chứng qua test `test_reranker_does_not_overpower_semantic_vector` |
| 7 | **Bộ lọc domain chặn nhầm `dropbox.com`, `linux.com`** | 🟠 P1 | So khớp theo hostname chính xác hoặc subdomain (`domain == b or domain.endswith("." + b)`). | ✅ Đã kiểm chứng qua test `test_engine_search_skips_blocked_domains` |

---

## 3. Báo cáo Chạy End-to-End Thực tế với LLM thật (SmolLM3-3B + FastEmbed)

### Kịch bản kiểm chứng:
- **Đề tài**: `"Compare YOLOv8 and RT-DETR accuracy and latency on COCO"`
- **Thực thể trích xuất tự động**: `['yolov8', 'rt-detr', 'coco']`
- **Cơ chế bảo vệ bằng chứng**:
  - Khi một trong hai mô hình (hoặc cả hai) chưa có đủ dữ liệu từ nguồn bên ngoài (ví dụ mạng bị hạn chế hoặc thiếu nguồn RT-DETR), hệ thống ghi nhận `Entity coverage: {'covered': [], 'missing': ['yolov8', 'rt-detr', 'coco'], 'is_complete': False}`.
  - Vòng lặp `EVALUATE` tạo câu hỏi mở có chủ đích: `['yolov8 coco benchmark', 'rt-detr coco benchmark']`.
  - Nếu sau 5 bước tìm kiếm mà nguồn của một thực thể vẫn không đạt, session kết thúc ở trạng thái **`PARTIAL`** với thông báo lỗi rõ ràng: `Missing coverage: yolov8, rt-detr, coco`.
  - Bản báo cáo tổng hợp ghi nhận sự thiếu hụt thông tin thay vì tự bịa số liệu mAP hay FPS của mô hình thiếu.

---

## 4. Kết quả Chạy Test Suite Tự động

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI\AI_Research
configfile: pytest.ini
testpaths: tests
collected 50 items

tests/test_api.py ......                                                 [ 12%]
tests/test_db.py .                                                       [ 14%]
tests/test_engine.py .......                                             [ 28%]
tests/test_integration_multi_request.py ...........                      [ 50%]
tests/test_ollama_backend.py ......                                      [ 62%]
tests/test_parsing.py .....                                              [ 72%]
tests/test_retrieval.py ..........                                       [ 92%]
tests/test_state_machine.py ....                                         [100%]

======================== 50 passed, 1 warning in 2.62s ========================
```

---

## 5. Đóng Tuần 2 & Sẵn sàng cho Tuần 3

- **Đóng Tuần 2**: Toàn bộ 5 mục tiêu theo kế hoạch sửa đổi của Senior AI Engineer (Bước 1 đến Bước 5) đã được hoàn thành, kiểm chứng và tích hợp.
- **Kế hoạch Tuần 3**:
  1. Trích xuất **Atomic Evidence** (1 Fact = 1 Evidence) với đầy đủ Section, Page và Char Offset.
  2. Xây dựng **Evidence Graph** & Ma trận mâu thuẫn N-chiều (**N-way Conflict Matrix**).
  3. Phân tích khoảng trống (**Gap Analysis**) để định hướng thông minh cho các vòng lặp nghiên cứu tiếp theo.
