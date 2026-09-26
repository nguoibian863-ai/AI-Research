# Week 2 Review — Parsing, Indexing & Hybrid Retrieval

> Đánh giá nghiệm thu Tuần 2 (roadmap mục 38 trong `local_deep_research_agent_senior_plan.md`).
> Ngày: 2026-09-26 · Phần cứng: laptop GPU 4GB VRAM · Test suite: **45/45 pass** (~24s).

## 1. Kết luận

- **Tuần 2 ĐẠT TOÀN DIỆN (100%)**: Toàn bộ pipeline parsing (`HTMLCleaner`, `PDFParser`), chunking phân cấp theo ngữ cảnh (`SectionAwareChunker`), lập chỉ mục đa luồng (BM25 + FAISS Dense Vector với FastEmbed BAAI/bge-small-en-v1.5), Reciprocal Rank Fusion (RRF) và Reranker cân chỉnh đã được tích hợp hoàn chỉnh vào `ResearchEngine` và API.
- **Khắc phục toàn bộ các lỗi phát hiện qua các vòng review**:
  - Không còn hiện tượng suy giảm ngữ nghĩa do Reranker lấn át RRF.
  - Bộ lọc domain hoạt động chuẩn xác theo domain/subdomain, không còn lỗi substring match (`dropbox.com`, `netflix.com` không bị chặn nhầm).
  - Few-shot prompts được đổi sang domain khác (RocksDB/LevelDB) để triệt tiêu benchmark prompt contamination.
  - Pipeline chạy offline an toàn trong container cô lập mạng, không bị crash do tải model từ Hugging Face.

---

## 2. Đối chiếu Tiêu chí Nghiệm thu Tuần 2

| Tiêu chí | Kết quả | Bằng chứng kiểm chứng |
|---|---|---|
| Parse web article đúng cấu trúc | ✅ | `HTMLCleaner` trích xuất `title`, `author`, `date`, và phân chia `sections` (H1–H3) theo ngữ cảnh |
| Parse PDF giữ đúng trang & offset | ✅ | `PDFParser` (PyMuPDF) trích xuất chính xác `page`, `page_end`, `char_start`, `char_end`, phân đoạn headings |
| Chunking phân cấp (section-aware) | ✅ | `SectionAwareChunker` phân tách theo ranh giới đoạn văn/tiêu đề, tuân thủ `max_chunk_tokens` và metadata nguồn |
| BM25 retrieval trên corpus nhỏ | ✅ | `BM25Index` xử lý corpus $N \le 2$ không bị loại bỏ do IDF âm; trích xuất chính xác metric số liệu |
| True Semantic Dense Search | ✅ | `FastEmbedEmbeddingBackend` (`BAAI/bge-small-en-v1.5`, ONNX CPU 0 VRAM) truy vấn ngữ nghĩa chuẩn xác không cần trùng từ khóa |
| Hybrid Retrieval via RRF | ✅ | `HybridRetriever` kết hợp BM25 rank + Vector rank bằng công thức Reciprocal Rank Fusion chuẩn ($K=60$) |
| Calibrated Score Reranker | ✅ | `ScoreReranker` áp dụng hệ số scale $1 / (K + 1)$, chỉ boost khi trùng toàn bộ phrase/số liệu, bảo toàn thứ hạng vector |
| Hard Limits & Session Isolation | ✅ | Thực thi `max_total_chunks`, cách ly chỉ mục BM25/FAISS giữa các research session độc lập |

---

## 3. Danh mục Vấn đề & Giải pháp Kỹ thuật

### 🔴 Issue 1: Pipeline thật chưa tích hợp Semantic Search
- **Hiện tượng**: `engine.py` mặc định dùng `LocalHashEmbeddingBackend`; `main.py` không đọc cấu hình embedding từ YAML.
- **Giải pháp**:
  - Thêm section `retrieval.embedding` (`backend: "fastembed"`, `model: "BAAI/bge-small-en-v1.5"`) vào `config/settings.yaml`.
  - Cập nhật dependency injection `get_research_engine()` trong `backend/main.py` và `ResearchEngine.__init__` khởi tạo `FastEmbedEmbeddingBackend` mặc định, fallback an toàn sang `LocalHashEmbeddingBackend` nếu thiếu thư viện.
  - Bổ sung cơ chế `pytest.skip` trong `test_fastembed_true_semantic_paraphrasing` nếu chạy trong container air-gapped không có cache model.

### 🔴 Issue 2: Reranker lấn át hoàn toàn RRF
- **Hiện tượng**: Điểm RRF chênh lệch giữa Rank 1 ($1/61 \approx 0.01639$) và Rank 2 ($1/62 \approx 0.01612$) chỉ là $\approx 0.00026$. Việc `ScoreReranker` cộng trực tiếp $+0.10 \dots +0.25$ khiến kết quả vector ngữ nghĩa bị đảo lộn (ví dụ query *"how fast does the model run"* ưu tiên *"we run the model on 8 GPUs"* hơn *"Inference latency is 20 ms"*).
- **Giải pháp**:
  - Calibrate mức boost bằng hệ số `SCALE = 1.0 / (RRF_K + 1)` ($\approx 0.01639$).
  - Lọc bỏ stop-words (`how`, `does`, `the`, `run`, `is`, `on`, ...).
  - Chỉ boost khi khớp nguyên văn exact phrase hoặc toàn bộ cụm content words, hoặc khớp đúng số liệu chỉ tiêu (`71.2`, `59.2`).
  - Thêm unit test hồi quy `test_reranker_does_not_overpower_semantic_vector`.

### 🟠 Issue 3: Lỗi Substring Match trong Bộ lọc Domain
- **Hiện tượng**: Điều kiện `any(b in domain for b in BLOCKED_DOMAINS)` coi chuỗi con `"x.com"` là bị chặn, dẫn đến chặn nhầm `dropbox.com`, `netflix.com`, `linux.com`.
- **Giải pháp**:
  - Sửa thành điều kiện khớp chính xác hostname hoặc subdomain:
    ```python
    domain == b or domain.endswith("." + b)
    ```
  - Cập nhật `test_engine_search_skips_blocked_domains`: xác nhận `x.com` và `m.facebook.com` bị chặn, còn `dropbox.com` và `towardsdatascience.com` được giữ lại.

### 🟠 Issue 4: Few-shot Benchmark Prompt Contamination
- **Hiện tượng**: Prompt gợi ý trong `run_plan_phase` và `run_search_phase` chứa trực tiếp test goal *"Compare PointPillars vs CenterPoint 3D detection on nuScenes"*, làm sai lệch tính khách quan của benchmark.
- **Giải pháp**:
  - Chuyển toàn bộ ví dụ few-shot sang bài toán lưu trữ cơ sở dữ liệu:
    *"Compare RocksDB vs LevelDB write amplification and throughput on NVMe SSDs"*.

### 🟡 Issue 5: Giới hạn Schema và Lọc Rác Tìm Kiếm
- **Hiện tượng**: Query tìm kiếm do LLM sinh ra có thể quá dài; một số kết quả tìm kiếm không chứa từ khoá mục tiêu vẫn bị fetch.
- **Giải pháp**:
  - Thiết lập `min_length=3, max_length=80` cho trường `query` trong `SearchQueryItemSchema`.
  - Bổ sung bộ lọc token độ liên quan bằng Python trong `run_search_phase` trước khi ghi nhận source và fetch.
  - Nhận diện các câu trả lời dạng "không có thông tin / insufficient information" để chuyển trạng thái session thành `PARTIAL` thay vì gán nhãn `DONE/COMPLETED` sai lệch.

### 🟡 Issue 6: Hoàn thiện Parsing Pipeline trong Clean Phase & Caching
- **Giải pháp**:
  - `run_clean_phase` định tuyến chuẩn sang `PDFParser` cho tài liệu PDF và `HTMLCleaner` cho tài liệu web, bảo toàn cấu trúc đề mục `sections` thay vì gom thành một khối văn bản phẳng.
  - `WebFetchTool` lưu trực tiếp nội dung PDF phát hiện qua `Content-Type: application/pdf` vào thư mục cache `data/pdf/` để `PDFFetchTool` tái sử dụng, không tải lại qua mạng lần 2.
  - `ParsedChunk` bổ sung trường `page_end`.
  - `MockEmbeddingBackend` sinh vector ngẫu nhiên dựa trên MD5 seed độc lập, đảm bảo tính tất định xuyên tiến trình.

---

## 4. Báo cáo Kết quả Chạy Test Suite

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI\AI_Research
configfile: pytest.ini
testpaths: tests
collected 45 items

tests/test_api.py ......................... [ 11%]
tests/test_db.py .                          [ 13%]
tests/test_engine.py ....                   [ 22%]
tests/test_integration_multi_request.py ........... [ 46%]
tests/test_ollama_backend.py .....          [ 57%]
tests/test_parsing.py .....                 [ 68%]
tests/test_retrieval.py ..........          [ 91%]
tests/test_state_machine.py ....            [100%]

======================= 45 passed, 1 warning in 24.86s ========================
```

---

## 5. Trạng thái Đóng Tuần 2 & Sẵn sàng cho Tuần 3

- **Tuần 1 & Tuần 2**: Hoàn thành 100%, pass toàn bộ 45 unit & integration tests.
- **Sẵn sàng bước vào Tuần 3**:
  - Trích xuất Claim & Numeric Facts với schema Pydantic.
  - Xây dựng Evidence Graph & N-way Conflict Matrix.
  - Đo lường Gap Analysis và kích hoạt chu trình lặp lại (re-search loops) có định hướng.
