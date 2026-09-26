# Trạng thái hiện tại

> Cập nhật sau mỗi vòng review. Lần cập nhật: 2026-09-26, sau commit sửa context entity + credibility (xem `git log`).

## Tiến độ

| Tuần | Trạng thái | Ghi chú |
|---|---|---|
| 1 — Core Engine | ~95% | Đã chạy E2E thật với SmolLM3 |
| 2 — Parsing + Retrieval | ~95% | Đã chạy E2E thật (YOLOv8 vs RT-DETR) |
| 3 — Evidence Engine | ~92% | Numeric check, credibility score (lọc trước FETCH, truyền vào writer) đã có; **chưa chạy E2E thật sau các bản sửa** |
| 4 — Verification + UI | chưa bắt đầu | |
| 5 — Evaluation + Trajectory | chưa bắt đầu | |

Test: 84 passed + 1 skipped (test FastEmbed cần mạng).

Xem kết quả: `http://127.0.0.1:8000/ui` (Session Viewer, plan 29.0) hoặc `python scripts/show_session.py`.

## Đã giải quyết ở `4f3872c`

- Câu có `[E#]` sai số liệu → `NUMERIC_MISMATCH`, session `PARTIAL`.
- Một câu nhiều citation → 1 claim, nhiều `evidence_ids`; trạng thái `CITED` thay `SUPPORTED`.
- Năm (2019…) và số đếm nhỏ không đơn vị không còn gây `PARTIAL`.
- Danh sách benchmark mở rộng (CSDL, LLM, phần cứng) + nhận diện context entity theo giới từ.

## Đã giải quyết (review 4f3872c)

- Context entity không còn nuốt đối thủ so sánh: bỏ `with`, bỏ qua "in comparison with", operand trước mệnh đề ngữ cảnh luôn là thực thể bắt buộc.
- Credibility score thang 0–100 theo plan 14.1 (authority 30, primary_source 30, directness 20, recency 10, independence 10); so khớp domain theo host/subdomain; mạng xã hội < 40; repo code chỉ là primary khi owner/repo trùng thực thể của goal.
- Source < `min_source_score` (40) bị bỏ trước FETCH; evidence mang `source_score`; writer thấy evidence điểm cao trước và thấy điểm trong prompt.
- Claim `CITED` không còn `verified: True` (entailment `PENDING`); lý do PARTIAL không còn ghi "no valid citations" khi câu có citation nhưng sai số.

## Phát hiện từ lần chạy E2E thật đang diễn ra (`sess_a90c2f47fd8e`)

- Mỗi lần retrieve đều embed lại toàn bộ chunk (177 chunk ≈ 20–37s/lần trên CPU, ~2 phút/vòng RETRIEVE) → **đã sửa**: index theo session chỉ embed chunk mới.
- Retrieve/coverage theo thực thể chạy cả cho dataset (`coco`) → **đã sửa**: chỉ dùng thực thể được so sánh.

## Đã sửa sau E2E thật (chờ chạy lại để xác nhận)

1. Statement phải được quote hỗ trợ: ≥ 50% từ nội dung của statement có trong quote, nếu không → `STATEMENT_NOT_GROUNDED`.
2. Chọn chunk để trích xuất theo vòng giữa các thực thể được so sánh, tối đa 2 chunk/tài liệu; thực thể không có chunk ứng viên được retrieve riêng.
3. Coverage theo evidence tính theo `subject` (chấp nhận biến thể YOLOv8n, RT-DETR-R50), không tính "có nhắc tên".
4. Chuẩn hoá trích dẫn `**E1**`, `(E1)`, `E1`, `[E1, E2]` → `[E1]`; prompt writer bắt buộc citation cuối câu, câu có số phải có citation.
5. Prompt planner/query cấm lập kế hoạch làm thí nghiệm và tìm tham số cấu hình.
6. So khớp thực thể chấp nhận hậu tố biến thể (`text_mentions_entity`) khi chọn chunk.

## Vấn đề còn mở (ưu tiên từ trên xuống)

1. **Chạy lại 2 câu hỏi E2E** (YOLOv8/RT-DETR, PostgreSQL/MySQL) để xác nhận các bản sửa trên — điều kiện đóng Tuần 3.
2. Statement đọc sai quote nhưng trùng từ nhiều (vd "MySQL is the best open-source tool") vẫn qua → cần NLI (plan 23, Tuần 4).
3. Tỉ lệ quote bị loại "not in source chunk" cao (model diễn đạt lại thay vì trích nguyên văn) → dữ liệu cho Dataset C (plan 43).
4. Chưa ghi trajectory cho `query_generation` / `evidence_extraction` (plan 43.1).

## Lịch sử kết quả chạy thật

| Ngày | Goal | Kết quả |
|---|---|---|
| 2026-09-26 | Compare PointPillars and CenterPoint on nuScenes | DONE nhưng nguồn rác (Facebook, WhatsApp…), answer "no information" |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO | DONE, 8 nguồn liên quan nhưng **0 nguồn RT-DETR** |
| 2026-09-26 | (Tuần 3, `sess_ebc602b16c47`) | 0 evidence (quote là ô bảng, statement rỗng) |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO (`sess_a90c2f47fd8e`) | PARTIAL, 264s, 15 source, 177 chunk, **5 evidence nhưng đều từ paper RF-DETR**; lý do: câu tóm tắt nhắc lại "5 AP" không trích dẫn |
| 2026-09-26 | Compare PostgreSQL and MySQL on TPC-C (`sess_84c5aa35ff8f`) | PARTIAL, 212s, 5 evidence về tool BenchBase (không có số liệu hiệu năng); **chặn đúng 2 số bịa** (301,030 / 717,480 TPS); writer viết `**E1**` nên "no valid citations" |

