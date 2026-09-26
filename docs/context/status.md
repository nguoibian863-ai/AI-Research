# Trạng thái hiện tại

> Cập nhật sau mỗi vòng review. Lần cập nhật: 2026-09-26, commit `7a45c22` (code) / `57994ed` (plan).

## Tiến độ

| Tuần | Trạng thái | Ghi chú |
|---|---|---|
| 1 — Core Engine | ~95% | Đã chạy E2E thật với SmolLM3 |
| 2 — Parsing + Retrieval | ~95% | Đã chạy E2E thật (YOLOv8 vs RT-DETR) |
| 3 — Evidence Engine | ~85% | **Chưa chạy E2E thật sau các bản sửa** |
| 4 — Verification + UI | chưa bắt đầu | |
| 5 — Evaluation + Trajectory | chưa bắt đầu | |

Test: 66 passed + 1 skipped (test FastEmbed cần mạng).

## Vấn đề còn mở (ưu tiên từ trên xuống)

1. **Câu report có `[E#]` nhưng số liệu sai vẫn `SUPPORTED`, `verified: True`, session `DONE`.** Cần numeric check theo plan mục 24; trạng thái `CITED` thay cho `verified: True`.
2. Một câu trích nhiều evidence sinh nhiều dòng claim trùng nội dung → gộp thành 1 claim, nhiều `evidence_ids`.
3. `DATASET_BENCHMARK_TERMS` chỉ có dataset thị giác máy tính → cần quy tắc chung (thực thể sau "on/using/in") hoặc mở rộng.
4. Mọi số không trích dẫn (kể cả năm) → PARTIAL; nên bỏ qua năm và số nhỏ không đơn vị.
5. Chưa có lần chạy E2E thật sau khi sửa Tuần 3 (lần gần nhất `sess_ebc602b16c47`: 6 fact đề xuất, 0 được lưu).
6. Chưa có source credibility scoring (plan 14.1) — tiêu chí nghiệm thu Tuần 3.
7. Chưa ghi trajectory cho `query_generation` / `evidence_extraction` (plan 43.1).

## Lịch sử kết quả chạy thật

| Ngày | Goal | Kết quả |
|---|---|---|
| 2026-09-26 | Compare PointPillars and CenterPoint on nuScenes | DONE nhưng nguồn rác (Facebook, WhatsApp…), answer "no information" |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO | DONE, 8 nguồn liên quan nhưng **0 nguồn RT-DETR** |
| 2026-09-26 | (Tuần 3, `sess_ebc602b16c47`) | 0 evidence (quote là ô bảng, statement rỗng) |
