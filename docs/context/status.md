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

Test: 76 passed + 1 skipped (test FastEmbed cần mạng).

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

## Vấn đề còn mở (ưu tiên từ trên xuống) — từ E2E thật `sess_a90c2f47fd8e`, `sess_84c5aa35ff8f`

1. **Statement không được quote hỗ trợ vẫn được lưu** (chưa có entailment). Vd E1 "RF-DETR achieves state-of-the-art accuracy…" nhưng quote chỉ là "We evaluate RF-DETR on COCO for fair comparison…"; E1 run 2 "MySQL is the best open-source tool" đọc sai quote về BenchBase. Cần kiểm tra overlap statement↔quote ngay, NLI ở Tuần 4.
2. **Evidence thiếu đa dạng**: cả 5 evidence run 1 từ một paper (RF-DETR); extractor lấy 5 chunk đầu (đều từ retrieve chính), không chia theo thực thể/tài liệu.
3. **Coverage theo evidence chỉ kiểm "có nhắc tên"**: RF-DETR paper nhắc YOLOv8/RT-DETR nên coverage coi là đủ; cần tính theo `subject` của evidence.
4. **Writer không dùng đúng định dạng `[E#]`**: run 2 viết `**E1**` → "no valid citations"; run 1 dùng `[E1]` làm nhãn đầu đoạn.
5. **Planner vẫn lập kế hoạch làm thí nghiệm** ("Train YOLOv8 model on COCO…") thay vì tìm kết quả đã công bố; query run 2 lạc sang `max_connections`, `max_prepared_statements`.
6. Tỉ lệ quote bị loại "not in source chunk" cao (model diễn đạt lại thay vì trích nguyên văn).
7. Chưa ghi trajectory cho `query_generation` / `evidence_extraction` (plan 43.1).

## Lịch sử kết quả chạy thật

| Ngày | Goal | Kết quả |
|---|---|---|
| 2026-09-26 | Compare PointPillars and CenterPoint on nuScenes | DONE nhưng nguồn rác (Facebook, WhatsApp…), answer "no information" |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO | DONE, 8 nguồn liên quan nhưng **0 nguồn RT-DETR** |
| 2026-09-26 | (Tuần 3, `sess_ebc602b16c47`) | 0 evidence (quote là ô bảng, statement rỗng) |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO (`sess_a90c2f47fd8e`) | PARTIAL, 264s, 15 source, 177 chunk, **5 evidence nhưng đều từ paper RF-DETR**; lý do: câu tóm tắt nhắc lại "5 AP" không trích dẫn |
| 2026-09-26 | Compare PostgreSQL and MySQL on TPC-C (`sess_84c5aa35ff8f`) | PARTIAL, 212s, 5 evidence về tool BenchBase (không có số liệu hiệu năng); **chặn đúng 2 số bịa** (301,030 / 717,480 TPS); writer viết `**E1**` nên "no valid citations" |

