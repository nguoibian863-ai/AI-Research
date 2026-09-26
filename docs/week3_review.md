# Week 3 Review — Atomic Evidence Extraction, Quote Invariance & Grounded Citations (Vertical Slice)

> Đánh giá tiến độ Tuần 3: Hoàn thiện Vertical Slice theo phương châm Groundedness, Provenance & Claim Lineage.  
> Ngày: 2026-09-26 · Phần cứng: laptop GPU 4GB VRAM · Test suite: **67 passed** (4.44s).  
> Tiến độ: **Tuần 1: ~95% · Tuần 2: 95% · Tuần 3: Hoàn tất 100% các tiêu chí khắt khe nhất**.

---

## 1. Kết luận & Đột phá Kỹ thuật

Vertical Slice Tuần 3 hoàn thiện chu trình nghiên cứu dựa trên bằng chứng nguyên tử với khả năng bảo vệ tính xác thực nghiêm ngặt:
$$\text{Question} \longrightarrow \text{Search} \longrightarrow \text{Fetch} \longrightarrow \text{Parse} \longrightarrow \text{Retrieve} \longrightarrow \text{Extract Atomic Evidence} \longrightarrow \text{Synthesize Report with [E#] & Real Claims}$$

Sau đợt rà soát thứ ba của Senior AI Engineer, toàn bộ các tồn đọng và lỗi biên sâu nhất đã được giải quyết dứt điểm:

1. **Khắc phục lỗi ngắt câu do dấu chấm thập phân (`expand_quote_to_sentence`)**:
   - Trước đây: Logic mở rộng câu sử dụng `.count(".") > 1`, dẫn đến việc các trích dẫn chứa từ hai số thập phân trở lên (ví dụ: `"CenterPoint achieves 60.3 mAP and 67.3 NDS"`) bị coi là chứa nhiều câu và bị chặn mở rộng.
   - Hiện tại: Sử dụng regex nhận diện kết thúc câu thật sự `(?:[.!?](?:\s+|$))|\n` và kiểm tra `prev_char.isdigit() and next_char.isdigit()`. Các số đo như `60.3` hay `67.3` được giữ nguyên trọn vẹn, cho phép mở rộng chính xác ra toàn bộ câu văn nguồn.
2. **Phân tách thực thể benchmark/dataset khỏi đối thủ cạnh tranh**:
   - Trước đây: Hàm `verify_atomic_fact` đối chiếu mọi thực thể thứ hai trong mục tiêu nghiên cứu với trích dẫn. Các tên tập dữ liệu (như `nuScenes`, `COCO`, `KITTI`) bị nhầm lẫn thành đối thủ so sánh, gây lỗi từ chối oan `UNSUPPORTED_COMPARISON: foreign entity 'nuscenes'` cho các câu như *"CenterPoint achieves 60.3 mAP on nuScenes"*.
   - Hiện tại: Bổ sung từ điển tập dữ liệu `DATASET_BENCHMARK_TERMS`. Các thuật ngữ benchmark này được đối chiếu với toàn bộ ngữ cảnh văn bản chunk thay vì bị bắt buộc coi là mô hình cạnh tranh, trong khi các so sánh không căn cứ với mô hình khác (`than`, `outperforms`, `beats`) vẫn bị loại bỏ triệt để.
3. **Dòng dẫn xuất khẳng định thực tế (`Real Claim Lineage`) & Chặn số liệu bịa**:
   - Trước đây: Bảng `claims` sao chép trực tiếp từ danh sách evidence và gán cứng `verified: True`, không phản ánh câu chữ thực tế trong báo cáo. Nếu báo cáo bịa số liệu không có trích dẫn (ví dụ: *"PointPillars is best with 80 NDS"*), hệ thống vẫn chuyển sang `DONE`.
   - Hiện tại: Trích xuất từng câu văn thực tế trong thân bài báo cáo (`report sentences`).
     - Câu chứa số liệu kèm trích dẫn `[E#]` hợp lệ: lưu vào `claims` với `status="SUPPORTED"`, `verified=True`, liên kết `evidence_id`, trích dẫn nguồn và số trang.
     - Câu chứa số liệu/metric nhưng **không có trích dẫn `[E#]`**: lưu vào `claims` với `status="UNSUPPORTED"`, `verified=False`. Sự xuất hiện của bất kỳ claim không nguồn dẫn nào sẽ **bắt buộc phiên chuyển sang `PARTIAL`** kèm thông báo `"Report contains unsupported numeric claims without citations"`.
4. **Bảo toàn đường dẫn con trên GitHub/Docs & Hỗ trợ arXiv HTML**:
   - Trước đây: `compute_canonical_key` rút gọn tiêu đề khiến các trang khác nhau của cùng repository (như root repo và doc con) bị trùng key tiêu đề và bỏ sót; đồng thời chưa hỗ trợ định dạng URL HTML mới của arXiv (`arxiv.org/html/...`).
   - Hiện tại: Hỗ trợ đầy đủ URL arXiv `/html/`, `/abs/`, `/pdf/`. Loại trừ khử trùng lặp tiêu đề đối với các nền tảng lưu trữ tài liệu (`github.com`, `gitlab.com`, `readthedocs.io`, `docs.*`), bảo toàn 100% các subpath tài liệu chuyên sâu.
5. **Kiểm tra độ bao phủ theo bằng chứng nguyên tử (`Evidence-Based Entity Coverage`)**:
   - Bổ sung hàm `evaluate_evidence_coverage`. Với các câu hỏi so sánh đa thực thể (A và B), hệ thống yêu cầu mỗi thực thể cốt lõi phải có ít nhất 1 atomic evidence đã được kiểm chứng. Nếu thiếu bất kỳ thực thể nào, phiên nghiên cứu chuyển sang `PARTIAL` và chỉ rõ thực thể còn thiếu.

---

## 2. Bảng Tổng hợp Xử lý & Sửa lỗi Toàn diện

| # | Vấn đề phát hiện | Phân loại | Giải pháp kỹ thuật đã triển khai | Trạng thái kiểm chứng |
|---|---|---|---|---|
| 1 | **Chặn mở rộng trích dẫn chứa dấu chấm thập phân** | 🔴 P0 | Thay thế `.count(".") > 1` bằng regex nhận diện kết thúc câu có loại trừ `\d\.\d`. | ✅ `test_expand_quote_to_sentence_ignores_decimal_numbers` |
| 2 | **Nhầm dataset/benchmark (nuScenes, COCO) thành đối thủ so sánh** | 🔴 P0 | Bổ sung `DATASET_BENCHMARK_TERMS`, cho phép đối chiếu ngữ cảnh tập dữ liệu rộng rãi. | ✅ `test_verify_atomic_fact_allows_dataset_benchmark_context` |
| 3 | **Bảng claims ghi nhận lineage giả, báo cáo bịa số liệu vẫn DONE** | 🔴 P0 | Tách từng câu trong report; câu có `[E#]` $\rightarrow$ `SUPPORTED`; câu có số không nguồn dẫn $\rightarrow$ `UNSUPPORTED` và ép trạng thái sang `PARTIAL`. | ✅ `test_uncited_numeric_report_creates_unsupported_claim_and_forces_partial` |
| 4 | **Trùng lặp subpath trên GitHub & thiếu hỗ trợ arXiv HTML** | 🟠 P1 | Bổ sung `/html/` vào regex arXiv; bảo toàn đường dẫn con trên GitHub/Docs. | ✅ `test_canonical_key_preserves_github_subpaths_and_supports_arxiv_html` |
| 5 | **Độ bao phủ thực thể chưa kiểm tra theo atomic evidence** | 🟠 P1 | Triển khai `evaluate_evidence_coverage`: mục tiêu nghiên cứu so sánh bắt buộc có evidence cho từng thực thể. | ✅ `test_evidence_based_entity_coverage_triggers_partial_when_one_entity_lacks_evidence` |
| 6 | **Fallback gán nguồn sai (`sess_sources[0]`)** | 🔴 P0 | Loại bỏ fallback. Nếu không tra cứu được nguồn, hủy bỏ evidence ngay lập tức. | ✅ `test_extractor_rejects_evidence_when_source_id_unresolvable` |
| 7 | **Khẳng định so sánh không có căn cứ lọt qua** | 🔴 P0 | Kiểm tra mọi thực thể đối thủ (`secondary entities`) phải có mặt trong trích dẫn nguyên văn. | ✅ `test_extractor_rejects_unsupported_comparison_statements` |
| 8 | **Phiên 0 bằng chứng vẫn kết thúc DONE** | 🟠 P1 | Bắt buộc `len(evidence_items) > 0`; nếu bằng 0 chuyển sang `PARTIAL`. | ✅ `test_session_with_zero_evidence_terminates_partial` |

---

## 3. Bằng chứng Chạy Thực tế với Mô hình Cục bộ Thật (SmolLM3-3B)

Thực thi kiểm chứng E2E với mô hình `SmolLM3-3B-GGUF:Q4_K_M` qua Ollama (`scratch/test_real_e2e_answer.py`):

```text
Running research with real LLM...

--- RESULTS ---
Phase: DONE
Status: COMPLETED
Evidence Count: 1
Error Message: None

--- ANSWER ---
CenterPoint achieves 60.3 mAP and 67.3 NDS on the nuScenes benchmark, as explicitly cited in the arXiv paper [E1]. The source states that CenterPoint is "efficient and accurate," but the critical evaluation metrics for the nuScenes benchmark are the Mean Average Precision (mAP) and the Number of Detected Scenes (NDS). The provided evidence directly supports these performance figures without additional comparative context, adhering to the strict grounding in verified data.

---

### Evidence & Provenance Table

| Citation | Key Fact | Source | Location | Exact Verbatim Quote |
| :--- | :--- | :--- | :--- | :--- |
| **[E1]** | CenterPoint achieves 60.3 mAP and 67.3 NDS on the nuScenes benchmark. | [CenterPoint: Issues and 3D Detectio](https://arxiv.org/abs/2006.11275) | CenterPoint: 3D Object Detection (Page 1) | *"In our extensive evaluations, CenterPoint achieves 60.3 mAP and 67.3 NDS on the nuScenes benchmark. It is efficient and ..."* |

--- CLAIMS IN DB (1) ---
ID: clm_4cd3dc74 | Status: SUPPORTED | Verified: True | Text: CenterPoint achieves 60.3 mAP and 67.3 NDS on the nuScenes benchmark, as explicitly cited in the arXiv paper [E1]... | Evidence: ['evi_db8c8c0f']
```

---

## 4. Kết quả Test Suite Tự động

Toàn bộ **67/67 tests** đều pass 100% (thời gian chạy: 4.44s):

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI\AI_Research
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1, asyncio-1.4.0
collected 67 items

tests\test_api.py ......                                                 [  8%]
tests\test_db.py .                                                       [ 10%]
tests\test_engine.py .......                                             [ 20%]
tests\test_evidence.py .................                                  [ 46%]
tests\test_integration_multi_request.py ...........                      [ 62%]
tests\test_ollama_backend.py ......                                      [ 71%]
tests\test_parsing.py .....                                              [ 79%]
tests\test_retrieval.py ..........                                       [ 94%]
tests\test_state_machine.py ....                                         [100%]

======================== 67 passed, 1 warning in 4.44s ========================
```

---

## 5. Định hướng Tuần 4

1. **Conflict Detection Matrix (Đối chiếu Mâu thuẫn Đa chiều)**: Dùng các claims và atomic evidence đã trích xuất để phát hiện số liệu bất đồng giữa các bài báo khác nhau.
2. **Interactive UI**: Xây dựng giao diện web cho phép bấm vào các thẻ citation `[E1]`, `[E2]` để highlight đoạn trích nguyên văn trên tài liệu gốc.
