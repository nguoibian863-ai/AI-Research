# Trạng thái hiện tại

> Cập nhật sau mỗi vòng review. Lần cập nhật: 2026-09-26, commit `4f3872c`.

## Tiến độ

| Tuần | Trạng thái | Ghi chú |
|---|---|---|
| 1 — Core Engine | ~95% | Đã chạy E2E thật với SmolLM3 |
| 2 — Parsing + Retrieval | ~95% | Đã chạy E2E thật (YOLOv8 vs RT-DETR) |
| 3 — Evidence Engine | ~88% | Numeric check cho câu có `[E#]` đã có; credibility score đã tính nhưng **chưa được dùng**; **chưa chạy E2E thật sau các bản sửa** |
| 4 — Verification + UI | chưa bắt đầu | |
| 5 — Evaluation + Trajectory | chưa bắt đầu | |

Test: 71 passed + 1 skipped (test FastEmbed cần mạng).

## Đã giải quyết ở `4f3872c`

- Câu có `[E#]` sai số liệu → `NUMERIC_MISMATCH`, session `PARTIAL`.
- Một câu nhiều citation → 1 claim, nhiều `evidence_ids`; trạng thái `CITED` thay `SUPPORTED`.
- Năm (2019…) và số đếm nhỏ không đơn vị không còn gây `PARTIAL`.
- Danh sách benchmark mở rộng (CSDL, LLM, phần cứng) + nhận diện context entity theo giới từ.

## Vấn đề còn mở (ưu tiên từ trên xuống)

1. **`extract_context_entities` nuốt mất đối thủ so sánh.** Giới từ `with`/`in` quá rộng: "Compare Rust **with Go**" → `go` bị coi là context; "in comparison **with PointPillars**" → `pointpillars` bị loại khỏi coverage, còn `perform` lại thành thực thể bắt buộc.
2. **Credibility score chưa được dùng**: không lọc trước FETCH, không truyền vào evidence/writer (tiêu chí nghiệm thu Tuần 3: "source < 40 không bị fetch").
3. Thang điểm 0–1 trong khi plan dùng 0–100; điểm thấp nhất có thể đạt ~0.56 nên ngưỡng 0.40 không bao giờ kích hoạt; `facebook.com` (0.67) cao hơn blog Medium (0.58).
4. So khớp domain bằng chuỗi con: `github.com.evil.io` và `pacm.org` được xếp hạng như GitHub/ACM. Repo GitHub bất kỳ (0.90) cao hơn bài IEEE (0.89).
5. Lý do PARTIAL gây hiểu nhầm: câu có citation nhưng sai số → vẫn ghi thêm "Report contains no valid citations".
6. `CITED` vẫn kèm `verified: True` dù chưa qua entailment (plan mục 24).
7. Chưa có lần chạy E2E thật sau khi sửa Tuần 3 (lần gần nhất `sess_ebc602b16c47`: 0 evidence).
8. Chưa ghi trajectory cho `query_generation` / `evidence_extraction` (plan 43.1).

## Lịch sử kết quả chạy thật

| Ngày | Goal | Kết quả |
|---|---|---|
| 2026-09-26 | Compare PointPillars and CenterPoint on nuScenes | DONE nhưng nguồn rác (Facebook, WhatsApp…), answer "no information" |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO | DONE, 8 nguồn liên quan nhưng **0 nguồn RT-DETR** |
| 2026-09-26 | (Tuần 3, `sess_ebc602b16c47`) | 0 evidence (quote là ô bảng, statement rỗng) |
