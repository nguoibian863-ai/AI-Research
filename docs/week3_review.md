# Week 3 Review — Atomic Evidence Extraction, Quote Invariance & Grounded Citations (Vertical Slice)

> Đánh giá tiến độ Tuần 3: Hoàn thiện Vertical Slice theo phương châm Groundedness, Provenance & Claim Lineage.  
> Ngày: 2026-09-26 · Phần cứng: laptop GPU 4GB VRAM · Test suite: **72 passed** (5.80s).  
> Tiến độ: **Tuần 1: ~95% · Tuần 2: 95% · Tuần 3: 100% Hoàn tất toàn diện (Verified E2E với mô hình thật)**.

---

## 1. Kết luận & Đột phá Kỹ thuật

Vertical Slice Tuần 3 hoàn thiện chu trình nghiên cứu dựa trên bằng chứng nguyên tử với khả năng bảo vệ tính xác thực nghiêm ngặt:
$$\text{Question} \longrightarrow \text{Search} \longrightarrow \text{Fetch} \longrightarrow \text{Parse} \longrightarrow \text{Retrieve} \longrightarrow \text{Extract Atomic Evidence} \longrightarrow \text{Synthesize Report with [E#] & Real Claims}$$

Sau đợt rà soát thứ tư của Senior AI Engineer, toàn bộ các tồn đọng về kiểm chứng số liệu, gộp claim, nới lỏng ngữ cảnh đa lĩnh vực và chấm độ tin cậy nguồn theo Plan 14 đã được giải quyết dứt điểm:

1. **Kiểm tra số liệu của câu có trích dẫn (`Plan 24: Numeric Verification & NUMERIC_MISMATCH`)**:
   - Trước đây: Kịch bản *"CenterPoint reaches 95.0 NDS [E1], … PointPillars at 10.0 NDS [E2]"* ghi số liệu bịa nhưng vẫn được gắn `SUPPORTED / verified: True` và kết thúc `DONE` do hệ thống chỉ kiểm tra sự tồn tại của token `[E#]`.
   - Hiện tại: Triển khai kiểm chứng số liệu cấp độ câu. Với mọi câu có trích dẫn `[E#]`, mọi con số đo lường thực chất trong câu bắt buộc phải xuất hiện trong trích dẫn nguyên văn (`exact_quote` hoặc `raw_quote`) của ít nhất một evidence được dẫn chứng. Nếu không khớp:
     - Trạng thái claim: `status="NUMERIC_MISMATCH"`, `verified=False`, ghi nhận danh sách `mismatched_numbers`.
     - Phiên nghiên cứu: **Bắt buộc chuyển sang `PARTIAL`** với thông điệp: `"Report contains numeric claims mismatched with cited evidence"`.
   - Bỏ gán cứng `SUPPORTED`: Trước khi có bước kiểm tra entailment ở Tuần 4, các claim có trích dẫn hợp lệ chuyển sang trạng thái chuẩn mực là `status="CITED"`.
2. **Gộp Claim trùng lặp cho câu trích dẫn nhiều nguồn (`Claim Deduplication per Sentence`)**:
   - Trước đây: Một câu trích dẫn 2 evidence (ví dụ: `[E1] [E2]`) bị vòng lặp tạo thành 2 dòng claim riêng biệt có nội dung văn bản y hệt nhau.
   - Hiện tại: Mỗi câu trong báo cáo chỉ tạo đúng **1 claim duy nhất**, liên kết toàn bộ danh sách `evidence_ids=[evi_1, evi_2, ...]` và `citation_indices=[1, 2]`.
3. **Mở rộng nhận diện tập dữ liệu đa lĩnh vực & Quy tắc cú pháp tổng quát**:
   - Trước đây: Danh mục chỉ gồm các dataset Computer Vision, khiến các benchmark cơ sở dữ liệu (`TPC-C`), LLM (`MMLU`, `GSM8K`), hoặc CPU (`SPEC`) bị nhầm thành đối thủ so sánh.
   - Hiện tại: 
     - Bổ sung hàm cú pháp tổng quát `extract_context_entities`: Tự động trích xuất các thực thể xuất hiện sau giới từ (`on`, `in`, `using`, `across`, `under`, `with`) trong mục tiêu nghiên cứu (ví dụ: *"Compare Postgres and MySQL on TPC-C"* $\rightarrow$ `tpc-c` là ngữ cảnh).
     - Mở rộng `DATASET_BENCHMARK_TERMS` bao quát Database (`tpcc`, `tpch`, `ycsb`, `db_bench`), LLM (`mmlu`, `gsm8k`, `humaneval`, `arc`, `hellaswag`, `swe-bench`), CPU/Hardware (`spec`, `coremark`, `cinebench`, `geekbench`).
4. **Loại trừ năm xuất bản và số đếm nhỏ không có đơn vị (`extract_substantive_numbers`)**:
   - Trước đây: Mọi con số không trích dẫn đều gây phạt `PARTIAL`, kể cả năm xuất bản (`2019`) hay số lượng (`2 models`).
   - Hiện tại: Hàm `extract_substantive_numbers` loại trừ các năm lịch 4 chữ số (`1900–2099`) và các số nguyên nhỏ ($\le 5$) không đi kèm đơn vị đo lường khoa học (`%`, `ms`, `fps`, `mAP`, `NDS`, `ops/s`). Các số liệu khoa học thực chất (số thập phân, số có đơn vị đo, số đo lớn) được kiểm soát nghiêm ngặt 100%.
5. **Chấm độ tin cậy nguồn theo Plan Mục 14 (`Source Credibility Scoring`)**:
   - Triển khai module `backend/sources/credibility.py` đánh giá đa chiều theo 6 tiêu chí của Plan:
     1. `authority`: Uy tín học thuật/tổ chức của domain (`TIER_1_ACADEMIC`, `TIER_2_OFFICIAL`, `TIER_3_COMMUNITY`).
     2. `primary_source`: Nguồn gốc nguyên bản (bài báo, repo chính thức) vs blog tổng hợp.
     3. `recency`: Độ mới xuất bản tính theo tuổi thọ tài liệu.
     4. `directness`: Số liệu đo đạc thực nghiệm trực tiếp vs ý kiến định tính.
     5. `independence`: Xuất bản độc lập, phản biện ngang hàng hoặc mã nguồn mở.
     6. `reproducibility`: Có sẵn code, weights hoặc bộ dữ liệu công khai.
   - Điểm số tổng hợp `credibility_score` và chi tiết được lưu trữ trong bảng SQLite `sources` và trả về qua API.

---

## 2. Bảng Tổng hợp Xử lý & Sửa lỗi Toàn diện

| # | Vấn đề phát hiện | Phân loại | Giải pháp kỹ thuật đã triển khai | Trạng thái kiểm chứng |
|---|---|---|---|---|
| 1 | **Có trích dẫn [E#] nhưng ghi sai số liệu vẫn DONE** | 🔴 P0 | Triển khai đối soát số liệu Plan 24: mọi số trong câu phải có trong quote, nếu không $\rightarrow$ `NUMERIC_MISMATCH` và `PARTIAL`. | ✅ `test_cited_sentence_with_mismatched_numbers_triggers_numeric_mismatch_and_partial` |
| 2 | **Câu trích dẫn 2 evidence sinh 2 dòng claim trùng** | 🟡 P2 | Gom nhóm theo câu: lưu đúng 1 claim với `evidence_ids=[E1, E2]`. | ✅ `test_sentence_citing_multiple_evidences_deduplicates_to_single_claim` |
| 3 | **Gán cứng verified: True và status SUPPORTED** | 🟡 P2 | Đổi sang `status="CITED"`, `verified=True` chỉ khi số liệu khớp 100% với quote được trích. | ✅ Đã kiểm chứng trong cả mock và model thật |
| 4 | **Năm (2019) và số đếm nhỏ (2 models) gây false PARTIAL** | 🟠 P1 | `extract_substantive_numbers`: bỏ qua năm 1900-2099 và số 1-5 không có đơn vị đo. | ✅ `test_year_and_small_integer_counts_do_not_trigger_uncited_claims` |
| 5 | **Dataset ngoài Computer Vision bị nhầm thành đối thủ** | 🟠 P1 | Quy tắc cú pháp giới từ `extract_context_entities` + mở rộng `DATASET_BENCHMARK_TERMS` đa miền (DB, LLM, CPU). | ✅ `test_cross_domain_benchmarks_and_syntactic_context_extraction` |
| 6 | **Chấm độ tin cậy nguồn theo Plan Mục 14** | 🟡 P2 | Triển khai chấm điểm 6 chiều: `authority`, `primary_source`, `recency`, `directness`, `independence`, `reproducibility`. | ✅ `test_source_credibility_scoring_plan_14` |
| 7 | **Chặn mở rộng trích dẫn chứa dấu chấm thập phân** | 🔴 P0 | Thay thế `.count(".") > 1` bằng regex nhận diện kết thúc câu có loại trừ `\d\.\d`. | ✅ `test_expand_quote_to_sentence_ignores_decimal_numbers` |
| 8 | **Fallback gán nguồn sai (`sess_sources[0]`)** | 🔴 P0 | Loại bỏ fallback. Nếu không tra cứu được nguồn, hủy bỏ evidence ngay lập tức. | ✅ `test_extractor_rejects_evidence_when_source_id_unresolvable` |

---

## 3. Bằng chứng Chạy Thực tế Đa Lĩnh vực với Mô hình Cục bộ Thật (SmolLM3-3B)

Thực thi kiểm chứng E2E với mô hình `SmolLM3-3B-GGUF:Q4_K_M` qua Ollama trên 3 lĩnh vực chuyên sâu (`scratch/run_3_domains_e2e.py`):

### 3.1. Lĩnh vực 1: Thị giác Máy tính (Computer Vision)
- **Goal**: `Evaluate CenterPoint 3D detection on nuScenes`
- **Source**: `https://arxiv.org/abs/2006.11275` (arXiv Tier 1 Academic, Điểm uy tín: `0.94`)
- **Evidence Count**: 1 evidence nguyên tử được trích xuất thành công (`60.3 mAP and 67.3 NDS`).
- **Claim Lineage**:
  - `Claim 1`: *"CenterPoint achieves 60.3 mAP and 67.3 NDS on the nuScenes 3D benchmark..."* $\rightarrow$ `Status: UNSUPPORTED` (mô hình viết câu mở đầu nêu số đo nhưng quên đánh dấu `[E1]`).
  - `Claim 2`: *"These results, cited in [E1], indicate that CenterPoint outperforms the nuScenes benchmark with a detection accuracy of 60.3% (mAP) and a non-detection score of 67.3%..."* $\rightarrow$ `Status: CITED`, `Verified: True`, `Numeric Match: True`.
- **Kết luận phiên**: `PARTIAL` do câu mở đầu có số đo chưa kèm trích dẫn (hệ thống bắt lỗi trung thực, không bỏ lọt số liệu tự do).

### 3.2. Lĩnh vực 2: Cơ sở Dữ liệu & Lưu trữ (Database Systems)
- **Goal**: `Compare RocksDB and LevelDB on db_bench throughput`
- **Source**: `https://doi.org/10.1145/3318464.3389700` (ACM DOI, Điểm uy tín: `0.797`)
- **Evidence Count**: 2 evidence nguyên tử được trích xuất thành công:
  - `[E1]`: RocksDB đạt 420,000 ops/s so với LevelDB đạt 185,000 ops/s.
  - `[E2]`: RocksDB multi-threaded compaction làm giảm write stall.
- **Claim Lineage**:
  - `Claim 1`: *"RocksDB achieves 420,000 operations per second (ops/s) in random write throughput in the db_bench benchmark, as stated in [E1]."* $\rightarrow$ `Status: CITED`, `Verified: True`, `Evidence: ['evi_e65818b5']`.
  - `Claim 2`: *"**Note:** While [E2] highlights that RocksDB's multi-threaded compaction reduces write stalls..."* $\rightarrow$ `Status: CITED`, `Verified: True`, `Evidence: ['evi_7fa3f961']`.
  - Các gạch đầu dòng lặp lại số liệu không gắn citation $\rightarrow$ `Status: UNSUPPORTED`.

### 3.3. Lĩnh vực 3: Mô hình Ngôn ngữ Lớn (LLM / NLP)
- **Goal**: `Evaluate Llama 3 on MMLU benchmark`
- **Source**: `https://arxiv.org/abs/2407.21783` (arXiv Tier 1 Academic, Điểm uy tín: `0.97`)
- **Evidence Count**: 1 evidence nguyên tử trích xuất được (`Llama 3 8B achieves 68.4% and Llama 3 70B reaches 82.0% accuracy`).
- **Claim Lineage**:
  - `Claim 1`: *"[E1] Llama 3 8B achieves 68.4% on the standard MMLU 5-shot benchmark."* $\rightarrow$ `Status: CITED`, `Verified: True`, `Evidence: ['evi_163a4063']`.
  - `Claim 2`: *"[E2] Llama 3 70B achieves 82.0% on the standard MMLU 5-shot benchmark."* $\rightarrow$ `Status: UNSUPPORTED` (mô hình tự ý đánh dấu `[E2]` trong khi chỉ có 1 evidence `[E1]`). Hệ thống phát hiện index không tồn tại và hạ cấp thành `UNSUPPORTED`.

---

## 4. Kết quả Test Suite Tự động

Toàn bộ **72/72 tests** đều pass 100% (thời gian chạy: 5.80s):

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI\AI_Research
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1, asyncio-1.4.0
collected 72 items

tests\test_api.py ......                                                 [  8%]
tests\test_db.py .                                                       [  9%]
tests\test_engine.py .......                                             [ 19%]
tests\test_evidence.py ......................                            [ 50%]
tests\test_integration_multi_request.py ...........                      [ 65%]
tests\test_ollama_backend.py ......                                      [ 73%]
tests\test_parsing.py .....                                              [ 80%]
tests\test_retrieval.py ..........                                       [ 94%]
tests\test_state_machine.py ....                                         [100%]

======================== 72 passed, 1 warning in 5.80s ========================
```

---

## 5. Tổng kết Trạng thái Roadmap & Định hướng Tuần 4

| Giai đoạn | Trạng thái | Ghi chú hoàn thiện |
|---|---|---|
| **Tuần 1: Core Engine** | ✅ **~95%** | State machine, SQLite persistence, Trajectory logging, Budget limits. |
| **Tuần 2: Parsing & Retrieval** | ✅ **~95%** | PDF/HTML chunker, Hybrid BM25 + FAISS Vector, FastEmbed, Entity-balanced coverage. |
| **Tuần 3: Evidence Engine** | 🟢 **100%** | Trích xuất evidence nguyên tử, Quote invariance, Numeric verification (Plan 24), Claim lineage thật, Chấm uy tín nguồn (Plan 14). |
| **Tuần 4: Verification & UI** | ⏳ *Sắp tới* | Viết báo cáo theo section, NLI Entailment verification, Giao diện tương tác highlight citation. |
| **Tuần 5: Evaluation** | ⏳ *Kế hoạch* | Bộ benchmark 100 câu hỏi đánh giá groundedness & latency. |
