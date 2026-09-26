# Week 3 Review — Atomic Evidence Extraction, Quote Invariance & Grounded Citations (Vertical Slice)

> Đánh giá tiến độ Tuần 3: Hoàn thiện Vertical Slice theo phương châm Groundedness, Provenance & Claim Lineage.  
> Ngày: 2026-09-26 · Phần cứng: laptop GPU 4GB VRAM · Test suite: **62 passed** (3.23s).  
> Tiến độ: **Tuần 1: ~95% · Tuần 2: 95% · Tuần 3: Hoàn tất toàn bộ yêu cầu cốt lõi**.

---

## 1. Kết luận & Đột phá Kỹ thuật

Vertical Slice Tuần 3 hoàn thiện chu trình nghiên cứu dựa trên bằng chứng nguyên tử với khả năng bảo vệ tính xác thực nghiêm ngặt:
$$\text{Question} \longrightarrow \text{Search} \longrightarrow \text{Fetch} \longrightarrow \text{Parse} \longrightarrow \text{Retrieve} \longrightarrow \text{Extract Atomic Evidence} \longrightarrow \text{Synthesize Report with [E#] & Claims}$$

Sau đợt rà soát thứ hai của Senior AI Engineer, toàn bộ các tồn đọng và lỗi biên đã được giải quyết dứt điểm:

1. **Loại bỏ triệt để Fallback sai nguồn (`sess_sources[0]`)**:
   - Trước đây khi `source_id` bị thiếu trên chunk, mã nguồn tạm gán về `sess_sources[0]`. Điều này dẫn tới nguy cơ gán nhầm số liệu của bài báo B cho bài báo A.
   - Hiện tại: `EvidenceExtractor` thực hiện tra cứu 3 cấp (từ chunk metadata $\rightarrow$ tra cứu ngược từ SQLite `chunks` $\rightarrow$ `documents` $\rightarrow$ `sources`). Nếu không xác minh được nguồn, bằng chứng bị **hủy bỏ lập tức** (`REJECTED`) để bảo toàn tính toàn vẹn nguồn dẫn.
2. **Chặn đứng khẳng định so sánh không có căn cứ (`UNSUPPORTED_COMPARISON`)**:
   - Trước đây: Câu khẳng định không có số liệu sai như *"PointPillars is more accurate than CenterPoint"* đi kèm trích dẫn chỉ nói về PointPillars (*"PointPillars achieves 59.2 NDS"*) vẫn lọt qua vì không vi phạm `NUMERIC_MISMATCH`.
   - Hiện tại: Hàm `verify_atomic_fact` kiểm tra thực thể đối thủ (`secondary entities` trích xuất từ goal hoặc mệnh đề so sánh `than`, `outperforms`, `beats`). Nếu câu khẳng định nhắc đến thực thể B nhưng trích dẫn nguyên văn không chứa B, bằng chứng bị **từ chối ngay** với mã lỗi `UNSUPPORTED_COMPARISON`.
3. **Giải quyết vấn đề mô hình nhỏ (SmolLM3-3B) trích 0 bằng chứng**:
   - Khắc phục hiện tượng mô hình nhỏ chỉ trích ô bảng ngắn (như `'0.19 (m)'`) và để trống `statement=''`.
   - Bổ sung `min_length=10` cho `statement`, `min_length=2` cho `subject`, `min_length=15` cho `raw_quote` trong `AtomicFactItemSchema`.
   - Thêm ví dụ neutral few-shot và cơ chế mở rộng câu (`expand_quote_to_sentence`): nếu đoạn trích thô là mảnh ô bảng ngắn hoặc thiếu chủ ngữ, hàm sẽ mở rộng an toàn ra ranh giới câu trọn vẹn của đoạn văn nguồn, đảm bảo thỏa mãn cả độ dài quote và chủ ngữ mà vẫn giữ khớp nguyên văn 100%.
   - Kết quả chạy thật với `SmolLM3-3B`: Trích xuất thành công **2/2 và 3/3 bằng chứng hợp lệ** với đầy đủ statement, subject và exact quote.
4. **Quy tắc phiên 0 bằng chứng**:
   - Nếu một phiên kết thúc mà không trích xuất được bằng chứng nguyên tử hợp lệ nào, phiên **bắt buộc chuyển sang `PARTIAL`** với thông điệp rõ ràng: `"No verified atomic evidence extracted"`.
5. **Khử trùng lặp nguồn chuẩn hóa (`compute_canonical_key`)**:
   - Tự động nhận diện và hợp nhất các biến thể arXiv (`/abs/`, `/pdf/`, hậu tố version `v1`, `v2` $\rightarrow$ `arxiv:2006.11275`).
   - Chuẩn hóa DOI (`doi:10.xxxx/...`) và slug tiêu đề bài viết.
6. **Bảng `claims` & Dòng dẫn xuất (`Claim Lineage`)**:
   - Bổ sung endpoint `GET /api/research/session/{session_id}/claims` và trường `claims` trong response của `/api/research/session/{session_id}`.
   - Mỗi phát hiện kèm trích dẫn `[E#]` được ghi vào bảng SQLite `claims`, liên kết `evidence_id`, nội dung khẳng định và trạng thái `SUPPORTED`.

---

## 2. Bảng Tổng hợp Xử lý & Sửa lỗi theo Đánh giá

| # | Vấn đề phát hiện | Phân loại | Giải pháp kỹ thuật đã triển khai | Trạng thái kiểm chứng |
|---|---|---|---|---|
| 1 | **Fallback gán nguồn sai (`sess_sources[0]`)** | 🔴 P0 | Xóa bỏ fallback đầu tiên. Nếu tra cứu SQLite không tìm thấy nguồn gốc chunk, từ chối lưu bằng chứng. | ✅ `test_extractor_rejects_evidence_when_source_id_unresolvable` |
| 2 | **Khẳng định so sánh không có số vẫn lọt qua kiểm chứng** | 🔴 P0 | Trong `verify_atomic_fact`: kiểm tra mọi thực thể thứ hai (foreign/secondary entities) trong `statement` bắt buộc phải có mặt trong `quote_lower`. | ✅ `test_extractor_rejects_unsupported_comparison_statements` |
| 3 | **Model thật (SmolLM3-3B) trích 0 bằng chứng** | 🔴 P0 | Schema ràng buộc `min_length`, neutral few-shot prompt, và cơ chế `expand_quote_to_sentence` cứu các ô bảng ngắn. | ✅ Đã kiểm chứng thực tế: trích thành công 2/2 facts bằng mô hình thật |
| 4 | **Phiên 0 bằng chứng vẫn ghi nhận DONE** | 🟠 P1 | Ràng buộc `len(evidence_items) > 0` trong `run_basic_answer`. Nếu bằng 0, chuyển sang `PARTIAL` kèm thông báo lỗi trung thực. | ✅ `test_session_with_zero_evidence_terminates_partial` |
| 5 | **Trùng lặp nguồn qua nhiều URL khác nhau** | 🟠 P1 | Triển khai `compute_canonical_key`: chuẩn hóa arXiv, DOI và tiêu đề bài báo trong pha `SEARCH` và `FETCH`. | ✅ `test_compute_canonical_key_deduplication` |
| 6 | **Chưa có bảng claims ghi nhận dòng dẫn xuất** | 🟡 P2 | Lưu trữ từng khẳng định trích dẫn vào bảng `claims`, cung cấp API `GET /api/research/session/{id}/claims`. | ✅ `test_claims_table_populated_and_retrieved_via_api` |

---

## 3. Bằng chứng Chạy Thực tế với Mô hình Cục bộ Thật (SmolLM3-3B)

Thực thi kiểm chứng E2E với mô hình `SmolLM3-3B-GGUF:Q4_K_M` qua Ollama:

```text
2026-09-26 16:36:15,950 [INFO] backend.evidence.extractor: [sess_real_e2e_test][EXTRACT] LLM proposed 1 candidate facts for chunk chk_cp_01.
2026-09-26 16:36:17,764 [INFO] backend.evidence.extractor: [sess_real_e2e_test][EXTRACT] LLM proposed 1 candidate facts for chunk chk_pp_01.
2026-09-26 16:36:17,772 [INFO] backend.evidence.extractor: [sess_real_e2e_test][EXTRACT] Successfully verified and stored 2 atomic evidence items.
2026-09-26 16:36:17,775 [INFO] backend.core.transitions: [StateMachine] Session sess_real_e2e_test: EXTRACT -> WRITE (Synthesizing report)
2026-09-26 16:36:33,578 [INFO] backend.core.engine: [sess_real_e2e_test][ANSWER] Final report saved.
Extracted 2 evidence items.

================== SYNTHESIZED REPORT =================
...
### Evidence & Provenance Table

| Citation | Key Fact | Source | Location | Exact Verbatim Quote |
| :--- | :--- | :--- | :--- | :--- |
| **[E1]** | CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes 3D detection benchmark. | [CenterPoint: Center-based 3D Object](https://arxiv.org/abs/2006.11275) | Experiments (Page 6) | *"CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes 3D detection benchmark."* |
| **[E2]** | PointPillars achieves 59.2 NDS and 40.1 mAP on nuScenes with real-time latency. | [PointPillars: Fast Encoders for Obj](https://arxiv.org/abs/1812.05784) | Experiments (Page 5) | *"PointPillars achieves 59.2 NDS and 40.1 mAP on nuScenes with real-time latency."* |

================== CLAIMS LINEAGE (2) =================
- Claim: CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes 3D detection benchmark.
  Evidence IDs: ['evi_e1088f2e']
  Verification: {'verified': True, 'citation_index': 1, 'quote': 'CenterPoint achieves 60.3 mAP and 67.3 NDS on nuScenes 3D detection benchmark.'}
- Claim: PointPillars achieves 59.2 NDS and 40.1 mAP on nuScenes with real-time latency.
  Evidence IDs: ['evi_3c39fe0d']
  Verification: {'verified': True, 'citation_index': 2, 'quote': 'PointPillars achieves 59.2 NDS and 40.1 mAP on nuScenes with real-time latency.'}
```

---

## 4. Kết quả Test Suite Tự động

Toàn bộ **62/62 tests** đều pass 100% (thời gian chạy: 3.23s):

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI\AI_Research
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1, asyncio-1.4.0
collected 62 items

tests\test_api.py ......                                                 [  9%]
tests\test_db.py .                                                       [ 11%]
tests\test_engine.py .......                                             [ 22%]
tests\test_evidence.py ............                                      [ 41%]
tests\test_integration_multi_request.py ...........                      [ 59%]
tests\test_ollama_backend.py ......                                      [ 69%]
tests\test_parsing.py .....                                              [ 77%]
tests\test_retrieval.py ..........                                       [ 93%]
tests\test_state_machine.py ....                                         [100%]

======================== 62 passed, 1 warning in 3.23s ========================
```

---

## 5. Định hướng Tuần 4

1. **Conflict Detection Matrix (Đối chiếu Mâu thuẫn Đa chiều)**: Dùng các claims và atomic evidence đã trích xuất để phát hiện số liệu bất đồng giữa các bài báo khác nhau.
2. **Interactive UI**: Xây dựng giao diện web cho phép bấm vào các thẻ citation `[E1]`, `[E2]` để highlight đoạn trích nguyên văn trên tài liệu gốc.
