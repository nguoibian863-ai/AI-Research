# Trạng thái hiện tại

> Cập nhật sau mỗi vòng review. Lần cập nhật: 2026-09-26, sau review `a05af31`.

## Tiến độ

| Tuần | Trạng thái | Ghi chú |
|---|---|---|
| 1 — Core Engine | ~95% | Đã chạy E2E thật với SmolLM3 |
| 2 — Parsing + Retrieval | ~95% | Đã chạy E2E thật (YOLOv8 vs RT-DETR) |
| 3 — Evidence Engine | ~95% | Đã chạy E2E thật xác nhận các bản sửa: quote nguyên văn, chuẩn hoá [E#], claim lineage CITED, không bịa số liệu, dừng trung thực |
| 4 — Verification + UI | 🟢 bắt đầu | Đã có Session Viewer (/ui). Còn: NLI, writer theo section, UI đầy đủ |
| 5 — Evaluation + Trajectory | chưa bắt đầu | |

Test: 95 passed + 1 skipped (test FastEmbed cần mạng).

Xem kết quả: `http://127.0.0.1:8000/ui` (Session Viewer, plan 29.0) hoặc `python scripts/show_session.py`.
Xuất trajectory: `python scripts/export_trajectories.py` (Plan 43.1).

## Đã giải quyết (Plan 12.5 & 43.1)

- **Search Providers học thuật đa nguồn (Plan 12.5)**:
  - Bổ sung `ArxivSearchProvider`: truy vấn Atom XML feed của arXiv, trích xuất paper gốc, DOI, arXiv ID, abstract snippet.
  - Bổ sung `OpenAlexSearchProvider`: truy vấn OpenAlex API, trích xuất venue, publication year, citation count, DOI, và tái cấu trúc abstract từ inverted index; xử lý 429 graceful.
  - `CompositeSearchProvider` / `WebSearchTool`: kết hợp tìm kiếm học thuật và web tổng hợp, deduplicate tự động bằng `canonical_key`.
  - Kết nối dữ liệu `source_type="paper"|"academic"`, `published_at`, `doi` vào `SourceRepository` và tính điểm uy tín `score_source_credibility`.
- **Trajectory Logging cho Training & DPO (Plan 43.1 & 42.4)**:
  - Ghi trajectory cho bước `query_generation`: lưu goal, plan, danh sách queries, số lượng nguồn liên quan và điểm chất lượng.
  - Ghi trajectory cho bước `evidence_extraction`: lưu từng candidate fact với nhãn verifier đầy đủ (`VERIFIED`, `QUOTE_NOT_FOUND`, `NUMERIC_MISMATCH`, `SUBJECT_MISMATCH`, `UNSUPPORTED_COMPARISON`, `STATEMENT_NOT_GROUNDED`, `SOURCE_UNRESOLVED`). Mẫu VERIFIED sẵn sàng cho SFT; cặp (VERIFIED, bị loại) trên cùng chunk sẵn sàng cho DPO.
  - `TrajectoryRepository.export_partition_jsonl`: xuất toàn bộ trajectory ra các file JSONL có cấu trúc metadata chuẩn mục 42.4.
  - CLI script: `python scripts/export_trajectories.py` xuất dữ liệu ra `data/trajectories/{partition}/{task_type}.jsonl`.

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

## Đã sửa và xác nhận qua 2 lần chạy E2E thật (`sess_9536ce8580de`, `sess_8cd3cfdcc865`)

1. Statement phải được quote hỗ trợ: ≥ 50% từ nội dung của statement có trong quote; chặn thành công các so sánh vượt ngoài quote (`UNSUPPORTED_COMPARISON`).
2. Chọn chunk để trích xuất theo vòng giữa các thực thể được so sánh, tối đa 2 chunk/tài liệu; FastEmbed tối ưu không còn bị nghẽn (SEARCH+RETRIEVE chỉ mất ~37s).
3. Coverage theo evidence tính theo `subject`: phát hiện chính xác `Missing evidence for: postgresql, mysql` và kết thúc PARTIAL đúng mục tiêu.
4. Chuẩn hoá trích dẫn `**E1**`, `(E1)`, `E1`, `[E1, E2]` → `[E1]`: writer sinh trích dẫn chuẩn, 3/3 claim đạt `CITED` và truy vết thành công.
5. Không còn bịa số liệu: writer thừa nhận thiếu metric định lượng so sánh trực tiếp, không tự bịa TPS.

## Đã giải quyết (review 83150eb)

- **Composite Search xen kẽ quota & gọi song song**: Gọi song song `ThreadPoolExecutor` cho tất cả provider (arXiv, OpenAlex, DuckDuckGo) với quota chia đều, xếp xen kẽ round-robin và deduplicate canonical key, không còn hiện tượng một provider nuốt trọn kết quả.
- **arXiv URL trỏ tới `/html/`**: Sinh URL dạng `https://arxiv.org/html/{id}` thay cho `/abs/`, cho phép crawler lấy toàn bộ bài báo, bảng thực nghiệm và số liệu.
- **OpenAlex ưu tiên `oa_url`/`pdf_url`**: Chọn open-access URL trước landing page paywall/JS của nhà xuất bản, tiết kiệm fetch và lấy text chuẩn.
- **Làm sạch Trajectory `query_generation`**: Gắn nhãn `query_origin: "llm" | "open_questions"`, đặt `verified=False` cho query thô, và thêm cờ `llm_only_queries=True` khi xuất JSONL để loại bỏ query lập trình không do LLM sinh.
- **Version hóa prompt trích xuất evidence**: Lưu `prompt_version="v1.2"` cùng toàn văn prompt trong payload của `evidence_extraction` trajectory phục vụ huấn luyện SFT/DPO.
- **Dự trữ thời gian WRITE & dừng sớm EXTRACT**: Dự trữ tối thiểu 90s cho pha WRITE; dừng vòng lặp chunk nếu thời gian còn lại < 120s hoặc khi đã có ≥ 3 evidence bao quát tất cả thực thể so sánh.

## Đã giải quyết (review `a05af31`)

- **arXiv `/html/` tự động fallback sang `/pdf/` khi 404/rỗng**: Trong `run_fetch_phase`, nếu fetch URL `/html/` trả về None hoặc không có text, hệ thống tự động fallback sang fetch bản `/pdf/` tương ứng qua `PDFFetchTool` (kèm test).
- **Dừng sớm EXTRACT hoạt động chuẩn xác với goal có dataset**: Trong `EvidenceExtractor`, chuyển sang dùng `_extract_goal_subject_entities` (loại bỏ context entities như `coco`, `nuscenes`, `tpc-c`) và so khớp linh hoạt `subject_matches_entity` (kèm test).
- **Dự trữ WRITE động và cấu hình qua `ResearchLimits`**: Bổ sung `write_reserved_seconds` vào `ResearchLimits` (mặc định 90.0s) và cộng dồn thời gian đo thực tế của chunk trước (`last_chunk_duration`), đảm bảo EXTRACT ngắt kịp thời trước khi WRITE bị đói thời gian (kèm test).
- **Composite search bù chỗ trống khi provider lỗi**: Gọi đồng thời các provider với `limit` đầy đủ thay vì chia nhỏ; các kết quả được xếp xen kẽ round-robin. Nếu OpenAlex bị 429 hoặc DuckDuckGo bị chặn, các provider còn lại tự động bù đủ số lượng kết quả yêu cầu (kèm test).
- **Tránh trùng lặp tên provider**: Gom kết quả theo index vị trí thay vì `p.name`.
- **Bổ sung 4 regression test mới**: Đạt 95 passed + 1 skipped.

## Vấn đề còn mở (ưu tiên từ trên xuống)

1. Statement đọc sai quote nhưng trùng từ nhiều → cần NLI (plan 23, Tuần 4).
2. Tỉ lệ quote bị loại "not in source chunk" cao → dữ liệu cho Dataset C (plan 43).

## Lịch sử kết quả chạy thật

| Ngày | Goal | Kết quả |
|---|---|---|
| 2026-09-26 | Compare PointPillars and CenterPoint on nuScenes | DONE nhưng nguồn rác (Facebook, WhatsApp…), answer "no information" |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO | DONE, 8 nguồn liên quan nhưng **0 nguồn RT-DETR** |
| 2026-09-26 | (Tuần 3, `sess_ebc602b16c47`) | 0 evidence (quote là ô bảng, statement rỗng) |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO (`sess_a90c2f47fd8e`) | PARTIAL, 264s, 15 source, 177 chunk, **5 evidence nhưng đều từ paper RF-DETR**; lý do: câu tóm tắt nhắc lại "5 AP" không trích dẫn |
| 2026-09-26 | Compare PostgreSQL and MySQL on TPC-C (`sess_84c5aa35ff8f`) | PARTIAL, 212s, 5 evidence về tool BenchBase (không có số liệu hiệu năng); **chặn đúng 2 số bịa** (301,030 / 717,480 TPS); writer viết `**E1**` nên "no valid citations" |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO (`sess_9536ce8580de`) | PARTIAL (665s), 15 sources (điểm uy tín 78–90 cho arXiv/Springer), 3 evidence nguyên văn, chặn triệt để quote diễn đạt lại; dừng an toàn do vượt runtime 600s ở EXTRACT |
| 2026-09-26 | Compare PostgreSQL and MySQL on TPC-C (`sess_8cd3cfdcc865`) | PARTIAL (479s), 15 sources, 5 evidence nguyên văn, 3 claims CITED, report chuẩn không bịa số TPS; kết thúc PARTIAL trung thực vì thiếu metric so sánh đối đầu |
| 2026-09-26 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO (`sess_6dd46d23c476`) | **DONE / COMPLETED** (353s, <600s), 15 sources đa nguồn (arXiv HTML, OpenAlex PDF paper gốc RT-DETR CVPR 2024, GitHub official repo, Roboflow web), 5 evidence nguyên văn có số liệu benchmark (2.2% AP, 108 FPS vs 5 FPS, 1.9% AP, 0.7% AP), 4/4 claims CITED, 0 số bịa, xuất trajectory SFT/DPO v1.2 |


