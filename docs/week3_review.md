# Week 3 Review — Atomic Evidence Extraction, Quote Invariance & Grounded Citations (Vertical Slice)

> Đánh giá tiến độ Tuần 3: Hoàn thiện Vertical Slice theo phương châm Groundedness & Provenance.
> Ngày: 2026-09-26 · Phần cứng: laptop GPU 4GB VRAM · Test suite: **54 passed** (2.55s).
> Đánh giá nghiệm thu: **Tuần 1: ~95% · Tuần 2: 100% · Tuần 3 Vertical Slice: 100% ĐẠT CHUẨN**.

---

## 1. Mục tiêu Vertical Slice Tuần 3

Thay vì dàn trải toàn bộ tính năng phân tích đồ thị phức tạp hay giao diện người dùng, Vertical Slice tập trung hoàn thiện một lát cắt thẳng từ đầu đến cuối:
$$\text{Question} \longrightarrow \text{Search} \longrightarrow \text{Fetch} \longrightarrow \text{Parse} \longrightarrow \text{Retrieve} \longrightarrow \text{Extract 3–5 Atomic Evidence} \longrightarrow \text{Write Answer with Strict Citations}$$

**Trọng tâm cốt lõi:**
1. **1 Fact = 1 Evidence**: Mỗi sự kiện khoa học/số liệu được trích xuất thành một thực thể nguyên tử độc lập (Subject, Predicate, Metric, Value, Confidence).
2. **Quote Invariance Guardrail (Chống ảo giác triệt để)**: Tuyệt đối không lưu trữ hay trích dẫn bất kỳ bằng chứng nào nếu trích đoạn (`raw_quote`) không khớp nguyên văn từng ký tự với nội dung gốc trong văn bản nguồn (`chunk.text`).
3. **Strict Click-through Citations**: Báo cáo tổng hợp bắt buộc phải đánh dấu nguồn trích dẫn rõ ràng (`[E1]`, `[E2]`, ...) và tự động đính kèm **Evidence & Provenance Table** ở cuối báo cáo, cho phép kiểm chứng ngược về URL nguồn, số trang (Page) và phân đoạn (Section).

---

## 2. Kiến trúc Kỹ thuật & Thành phần Đã Triển khai

### 2.1. Cấu trúc Dữ liệu SQLite Độc lập & Toàn vẹn (Relational Provenance)
- Cập nhật bảng `raw_evidences`: bổ sung cột `chunk_id TEXT` liên kết trực tiếp với bảng `chunks`.
- Cơ chế tự động migration an toàn trong `init_schema()`: tự động bổ sung cột nếu cơ sở dữ liệu cũ chưa có.
- `EvidenceRepository.get_full_evidence_by_session(session_id)`: Thực hiện phép `LEFT JOIN` giữa 3 bảng `evidences`, `raw_evidences`, và `sources` để xuất đầy đủ siêu dữ liệu kiểm chứng:
  - `url`, `source_title`, `section`, `page`, `char_start`, `char_end`, `exact_quote`, `subject`, `predicate`, `metric`, `value`, `confidence`.

### 2.2. Bộ lọc Chống Ảo giác Đa tầng (`find_quote_in_text`)
Để giải quyết hiện tượng LLM cục bộ trích dẫn lệch dấu cách do ngắt dòng PDF hoặc viết hoa/thường:
- **Tầng 1 (Exact Match)**: Tìm kiếm chuỗi nguyên văn chính xác.
- **Tầng 2 (Case-Insensitive Match)**: Tìm kiếm không phân biệt chữ hoa/thường nhưng vẫn bảo toàn vị trí ký tự.
- **Tầng 3 (Whitespace-Normalized Regex Match)**: Sử dụng biểu thức chính quy `\s+` giữa các từ để xử lý các vết xuống dòng ngẫu nhiên hoặc khoảng trắng kép từ bộ bóc tách PDF/HTML.
- **Guardrail**: Bất kỳ đề xuất bằng chứng nào từ LLM mà trích đoạn không tìm thấy trong `chunk.text` sẽ bị loại bỏ ngay lập tức, không được đưa vào cơ sở dữ liệu SQLite và không được phép đưa vào ngữ cảnh viết báo cáo.

### 2.3. Trích xuất Bằng chứng Nguyên tử (`EvidenceExtractor`)
- Sử dụng mô hình cấu trúc JSON (`ExtractedEvidencesSchema` & `AtomicFactItemSchema`).
- Trích xuất tối đa 3–5 bằng chứng nguyên tử chất lượng cao nhất cho mỗi phiên nghiên cứu.
- Lưu trữ đồng thời vào `raw_evidences` (nguyên văn, vị trí ký tự, trang) và `evidences` (bộ dữ liệu sự kiện có cấu trúc).

### 2.4. Trình Kiểm chứng Trích dẫn & Bảng Xuất xứ (`CitationVerifier`)
- **Định dạng Prompt**: Chuyển đổi các bằng chứng đã xác thực thành danh sách chỉ mục rõ ràng `[E1]`, `[E2]` kèm trích đoạn nguyên văn và nguồn để LLM tổng hợp.
- **Quy tắc Trích dẫn Nghiêm ngặt**: LLM được chỉ dẫn bắt buộc phải trích dẫn `[E#]` cho từng số liệu hay tuyên bố cụ thể.
- **Kiểm tra Token Trích dẫn**: Quét toàn bộ văn bản báo cáo để phát hiện các mã trích dẫn hợp lệ và cảnh báo các chỉ mục trích dẫn vượt ngưỡng (`invalid_indices`).
- **Tự động Sinh Bảng Xuất xứ (Evidence & Provenance Table)**:
  Đính kèm bảng Markdown minh bạch ở cuối báo cáo:
  ```markdown
  | Citation | Key Fact | Source | Location | Exact Verbatim Quote |
  | :--- | :--- | :--- | :--- | :--- |
  | **[E1]** | CenterPoint achieves 60.3 mAP | [CenterPoint Paper](url) | Results (Page 5) | *"CenterPoint achieves 60.3 mAP..."* |
  ```

### 2.5. Tích hợp REST API & State Machine
- Cập nhật State Machine: cho phép chuyển trạng thái hai chiều giữa `RETRIEVE` $\leftrightarrow$ `EXTRACT`, `EVALUATE` $\leftrightarrow$ `EXTRACT`, và `EXTRACT` $\rightarrow$ `WRITE`.
- Endpoint mới:
  - `POST /api/research/session/{session_id}/extract`: Thực thi bóc tách bằng chứng nguyên tử từ các chunk đã truy xuất.
  - `GET /api/research/session/{session_id}/evidence`: Lấy toàn bộ danh sách bằng chứng nguyên tử kèm đầy đủ thông tin xuất xứ.
  - `GET /api/research/session/{session_id}`: Bổ sung trường `evidences` vào phản hồi tổng thể của phiên.

---

## 3. Kết quả Chạy Test Suite Tự động

Toàn bộ **54/54 tests** đều pass 100% trong thời gian 2.55 giây:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI\AI_Research
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1, asyncio-1.4.0
collected 54 items

tests\test_api.py ......                                                 [ 11%]
tests\test_db.py .                                                       [ 12%]
tests\test_engine.py .......                                             [ 25%]
tests\test_evidence.py ....                                              [ 33%]
tests\test_integration_multi_request.py ...........                      [ 53%]
tests\test_ollama_backend.py ......                                      [ 64%]
tests\test_parsing.py .....                                              [ 74%]
tests\test_retrieval.py ..........                                       [ 92%]
tests\test_state_machine.py ....                                         [100%]

======================== 54 passed, 1 warning in 2.55s ========================
```

---

## 4. Kế hoạch Tiếp theo (Tuần 3 Mở rộng)

1. **Evidence Graph**: Xây dựng đồ thị liên kết các thực thể, thuộc tính và nguồn dẫn chứng để biểu diễn trực quan mối quan hệ giữa các phát hiện.
2. **N-way Conflict Matrix**: Phát hiện và gắn cờ khi có hai nguồn đưa ra số liệu hoặc nhận định trái ngược nhau về cùng một đối tượng (ví dụ: mAP đo trên các cấu hình hoặc phiên bản khác nhau).
3. **Next.js UI Click-to-Verify**: Xây dựng giao diện web cho phép người dùng click trực tiếp vào nhãn `[E1]` để nhảy ngay đến câu trích dẫn nguyên văn và trang tài liệu tương ứng.
