# Trạng thái hiện tại

> Cập nhật sau mỗi vòng review. Lần cập nhật: 2026-09-27, sau giải quyết review `19b14d0` (clause-aware comparative scoping, metric comparability guard & model variant differentiation).

## Tiến độ

| Tuần | Trạng thái | Ghi chú |
|---|---|---|
| 1 — Core Engine | ~95% | Đã chạy E2E thật với SmolLM3 |
| 2 — Parsing + Retrieval | ~95% | Đã chạy E2E thật (YOLOv8 vs RT-DETR) |
| 3 — Evidence Engine | ~95% | Đã chạy E2E thật xác nhận các bản sửa: quote nguyên văn, chuẩn hoá [E#], claim lineage CITED, không bịa số liệu, dừng trung thực |
| 4 — Verification + UI | 🟢 đang thực hiện | Hoàn thiện NLI Verifier: clause-aware comparative scoping, metric comparability guardrail chống false CONTRADICTED, model variant suffixes differentiation. Còn: writer theo section, UI đầy đủ |
| 5 — Evaluation + Trajectory | ~25% | Đã ghi và xuất trajectory có query_origin và prompt_version |

Test: **126 passed, 0 skipped**.

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

## Đã giải quyết (review `73d8ee9`)

- **Giới hạn context_window 2048 & token clamping**:
  - Đồng bộ cấu hình `context_window: 2048` trên `config/settings.yaml`, `ResearchLimits.llm_context`, `OllamaBackend`, và `scripts/benchmark_ollama.py`.
  - Cắt ngắn văn bản chunk đầu vào tối đa 2400 ký tự (~600 token) trong `EvidenceExtractor` để đảm bảo vừa context window.
  - Phân bổ `num_predict` theo loại tác vụ: EXTRACT (512), PLAN (512), SEARCH (256), WRITE (768).
  - Bổ sung guardrail an toàn trong `OllamaBackend._resolve_context_and_predict`: tự động ước lượng token của prompt + system và clamp `num_predict` sao cho tổng không vượt `num_ctx`, loại bỏ nguy cơ Ollama âm thầm truncate prompt hoặc output JSON bị ngắt giữa chừng.
- **Chuẩn hoá subject matching trong EXTRACT**:
  - Thay thế hàm định nghĩa lỏng `subject_matches_entity` trong `extractor.py` bằng hàm chuẩn từ `backend.core.coverage`, ngăn chặn các tên họ chung như `DETR`, `YOLO`, `v8` thỏa mãn bao phủ sai mục tiêu so sánh cụ thể (`rt-detr`, `yolov8`).
- **Thắt chặt và bổ sung regression test**:
  - Sửa `test_composite_search_fills_quota_when_providers_fail_or_empty`: provider giả tôn trọng tham số `max_results`, fail trên code cũ nếu quota không được bù đủ.
  - Sửa `test_extract_write_reservation_dynamic`: kiểm tra thời gian còn lại 130s (`elapsed_seconds = 470.0`), phân biệt rõ ngưỡng cũ (120s) và ngưỡng động mới (170s).
  - Bổ sung 2 regression test mới: `test_subject_matches_entity_strict_semantics` và `test_ollama_backend_clamps_predict_to_stay_within_context`.

## Đã giải quyết (review `0374bbd`)

- **Sửa triệt để `test_extract_write_reservation_dynamic`**: Gán tường minh `mock_tracker.assert_can_call_llm = MagicMock()`, loại bỏ việc `MagicMock` ném `AttributeError` giả bị `except` nuốt; đồng thời kiểm tra cả 2 chiều:
  - Khi còn 130s (< 170s ngưỡng động): dừng hoàn toàn trước chunk 1 (`call_count == 0`).
  - Khi còn 220s (> 170s ngưỡng động): LLM bắt buộc được gọi (`call_count > 0`).
- **Giảm `max_chunk_tokens` xuống 450 và log cảnh báo khi cắt chunk**:
  - Cập nhật mặc định `max_chunk_tokens = 450` trong `backend/core/limits.py`, `config/settings.yaml`, và `SectionAwareChunker`.
  - Các chunk sau parsing tự nhiên nằm trong ngưỡng an toàn ~1800 ký tự, không bị cụt phần đuôi khi chuyển sang EXTRACT.
  - Thêm log cảnh báo chi tiết nếu có chunk bất thường vượt 2400 ký tự trong `EvidenceExtractor`.
- **Bảo vệ context window 2048 & tính lại token trên retry**:
  - Trong `OllamaBackend._resolve_context_and_predict`: nếu prompt $\ge num\_ctx$ hoặc headroom còn lại $< 64$ token, ném `ModelInferenceError` rõ ràng thay vì để Ollama âm thầm cắt đầu prompt.
  - Trong `structured_generate`: tính lại `num_predict` và kiểm tra context trên từng attempt (bao gồm retry khi output lỗi schema). Giới hạn độ dài thông báo lỗi retry ($\le 400$ ký tự) tránh bùng nổ token.
  - Bổ sung test kiểm tra `ModelInferenceError` khi prompt quá dài.
- **Khởi động NLI Verifier cho Tuần 4 (Plan 23)**:
  - Xây dựng module `backend/verification/`: `NLILabel`, `NLIVerificationResult`, `RuleBasedNLIVerifier` (kiểm tra đối lập phương hướng, nghịch đảo phủ định, token overlap), `LLMNLIVerifier`, `CompositeNLIVerifier`, và `ClaimVerificationPipeline`.
  - Tích hợp phương thức `ResearchEngine.verify_session_claims(session_id)`: kiểm chứng ngữ nghĩa entailment các claim `CITED` với các quote bằng chứng đã trích dẫn, cập nhật trạng thái `SUPPORTED` / `CONTRADICTED` và ghi nhận độ tin cậy cùng lý do.
  - Bổ sung 9 unit test cho toàn bộ pipeline NLI trong `tests/test_verification_nli.py`.

## Đã giải quyết (review `91a2829` — NLI verifier)

- **Nối NLI vào pipeline chính (`run_basic_answer` / `run_week1`)**:
  - Nối trực tiếp `ClaimVerificationPipeline` vào quá trình sinh answer và lưu claim trong `run_basic_answer`. Mỗi câu trích dẫn hợp lệ và khớp số liệu được đánh giá ngữ nghĩa entailment ngay lập tức thay vì bỏ lửng `PENDING`.
- **Cải tổ toàn diện bộ luật `RuleBasedNLIVerifier`**:
  - Bổ sung hàm trích xuất bộ ba so sánh cấu trúc `extract_comparative_triples`: chỉ kết luận `CONTRADICTED` khi cùng cặp thực thể nhưng bị đảo chiều đối tượng hoặc dùng từ so sánh đối lập (`faster` vs `slower`, `higher` vs `lower`).
  - Giải quyết triệt để lỗi false contradiction khi quote có nhiều tính từ đối lập trên các metric khác nhau ("RT-DETR achieves higher AP with lower latency than YOLOv8").
  - Xử lý giới từ phủ định ("removes NMS" vs "without NMS") không còn bị đánh đồng với phủ định logic mâu thuẫn.
  - Khi đảo chiều so sánh ("YOLOv8 is faster than RT-DETR" vs "RT-DETR is faster than YOLOv8"), bắt chính xác `CONTRADICTED`.
  - Khi token overlap thấp hoặc không đủ cơ sở ngữ nghĩa, trả về `NOT_SUPPORTED` thay vì gán nhãn sai.
- **Bảo toàn status số liệu & chuyển session sang `PARTIAL` khi có claim mâu thuẫn**:
  - `claims.status` giữ nguyên theo tầng kiểm tra số liệu (`CITED`, `NUMERIC_MISMATCH`, `UNSUPPORTED`). NLI chỉ ghi nhận kết quả vào dictionary `claims.verification`.
  - `verify_session_claims` bỏ qua các claim `NUMERIC_MISMATCH` / `UNSUPPORTED`, không bao giờ ghi đè tín hiệu sai số.
  - Nếu báo cáo chứa bất kỳ claim nào bị bằng chứng trích dẫn trực tiếp bác bỏ (`CONTRADICTED`), session tự động chuyển sang `PARTIAL` với lý do minh bạch `"Report contains claims contradicted by cited evidence"`.
  - Session Viewer (`backend/ui/index.html`) bổ sung mã màu và badge hiển thị rõ ràng cho `.SUPPORTED`, `.CONTRADICTED`, `.PARTIALLY_SUPPORTED`, `.NOT_SUPPORTED`.
- **Kiểm soát ngân sách token cho các lượt gọi NLI LLM**:
  - `LLMNLIVerifier` và `CompositeNLIVerifier` kiểm tra `budget_tracker.assert_can_call_llm()` trước mỗi lượt gọi và ghi nhận qua `record_llm_call(tokens=..., count=1)`.
  - Tự động fallback tức thì sang `RuleBasedNLIVerifier` khi cạn kiệt ngân sách mà không gây gián đoạn pipeline.
- **Bổ sung regression tests**:
  - Đạt 109 passed + 1 skipped (bổ sung các unit test cho 4 ca corner case của bộ luật NLI, budget enforcement, status preservation và contradiction session transition).

## Đã giải quyết (review `485345a` — NLI verifier & điều kiện DONE)

- **Chuẩn hóa claim và so khớp nguyên văn an toàn**:
  - Tự động lược bỏ các thẻ trích dẫn `[E#]`, `**E#**`, `(E#)` trước khi so khớp chuỗi.
  - Chuẩn hóa text (`_normalize_text_for_match`): giữ nguyên số thập phân (67.3), chuẩn hóa dấu câu và khoảng trắng. Claim trùng nguyên văn quote chứa `[E1]` đạt ngay `SUPPORTED 1.0`.
- **Cải tổ bộ trích xuất so sánh `extract_comparative_triples`**:
  - Dùng word boundary regex (`\b{comp}\b`) cho từ so sánh, loại bỏ hoàn toàn so khớp chuỗi con giả ("more" trong "furthermore", "less" trong "lossless").
  - Quét ngược (reverse token search) từ từ so sánh để xác định chủ ngữ thực thể gần nhất; bổ sung `LINKING_VERBS` (run, achieve, yield, perform...) vào danh sách bỏ qua, giúp "On T4 GPU, YOLOv8 is faster than RT-DETR" nhận diện chính xác chủ ngữ là `yolov8`.
  - Sử dụng `_entities_match` (dựa trên `subject_matches_entity`) cho phép nhận diện biến thể mô hình (`yolov8` vs `yolov8-l`, `rt-detr` vs `rt-detr-r50`), bắt chính xác mâu thuẫn đảo chiều (`CONTRADICTED 0.95`).
  - Khi câu có nhiều từ so sánh trên các metric khác nhau ("higher AP with lower latency"), không còn bị so chéo đối lập giả.
- **Guardrail cho từ phủ định (`VERBAL_NEGATIONS`)**:
  - Khi claim hoặc evidence xuất hiện từ phủ định logic (not, never, cannot...) mà không khớp nguyên văn, bộ luật trả về `NOT_SUPPORTED` với confidence thấp (0.50), không bao giờ tự ý kết luận `SUPPORTED`, nhường quyền phán quyết cho LLM.
- **Nguyên tắc vàng: Rule-based không bao giờ bypass LLM bằng token overlap**:
  - Bộ luật chỉ trả `SUPPORTED` với confidence 1.0 khi khớp nguyên văn/substring chuẩn hóa.
  - Token overlap cao chỉ trả tối đa `PARTIALLY_SUPPORTED (confidence 0.70 < 0.85)`, đảm bảo `CompositeNLIVerifier` luôn hỏi LLM khi có LLM.
- **Thắt chặt điều kiện kết thúc `DONE`**:
  - Phân định rõ `cited_claims_count` và `supported_claims_count` (chỉ đếm các claim được NLI xác nhận `SUPPORTED`).
  - Phiên chỉ được kết thúc `DONE` khi có ít nhất 1 claim được NLI xác nhận `SUPPORTED` (`supported_claims_count >= 1`).
  - Nếu claim có trích dẫn nhưng không claim nào được NLI xác nhận, phiên kết thúc `PARTIAL` kèm lý do minh bạch: `"Report contains cited claims but none were confirmed as fully supported by evidence"`.
- **Regression Tests**:
  - Đạt 114 passed + 1 skipped (bổ sung đầy đủ 6 test case cho các ca kiểm thử thực tế của review).

## Đã giải quyết (review `69c1355` — NLI Asymmetry, Multi-Quote Union & Polarity)

- **Sửa so khớp nguyên văn bất đối xứng (P0)**:
  - Trong `RuleBasedNLIVerifier.verify`, loại bỏ triệt để `e_norm in c_norm`, chỉ giữ `c_norm == e_norm or c_norm in e_norm`.
  - Claim chứa quote kèm thêm nội dung bịa/chưa kiểm chứng (ví dụ `"PointPillars achieves 59.2 NDS and is 10x faster than CenterPoint and all other detectors"`) không còn được đánh dấu `SUPPORTED 1.0`, mà nhận `NOT_SUPPORTED 0.85` / `PARTIALLY_SUPPORTED 0.70` và bắt buộc phải qua LLM.
- **Kiểm tra số liệu theo hợp các quote trích dẫn (P0)**:
  - Trong `RuleBasedNLIVerifier.verify`, bổ sung tham số `all_quotes: Optional[List[str]] = None`. Khi câu có nhiều trích dẫn, các số trong claim (`c_nums`) được đối soát với tập hợp số trong toàn bộ các quote được trích (`union(all_quotes)`), loại bỏ hoàn toàn lỗi fail số giả (`59.2 not found`) khi câu trích 2 nguồn.
  - Trong `ClaimVerificationPipeline.verify_claim_against_quotes`: truyền `all_quotes=valid_quotes` khi kiểm tra từng quote. Đồng thời với câu trích nhiều nguồn (`len(valid_quotes) > 1`), pipeline đánh giá thêm toàn văn câu so sánh với hợp các bằng chứng `combined_evidence = " ".join(valid_quotes)`.
  - Khôi phục câu test gốc `"CenterPoint reaches 67.3 NDS **E1** while PointPillars reaches 59.2 NDS (E2)."` trong `tests/test_evidence.py`, đảm bảo session kết thúc `DONE` tự nhiên mà không cần trick biến đổi câu văn.
- **Nhận diện chiều (Polarity) của Metric so sánh (P1)**:
  - Phân loại rõ `LOWER_IS_BETTER_METRICS` (latency, error, error rate, loss, memory, params, flops...) và `HIGHER_IS_BETTER_METRICS` (accuracy, ap, map, nds, fps, throughput...).
  - Thắt chặt điều kiện kết luận `CONTRADICTED`: chỉ kết luận khi cùng metric (ví dụ cùng nói về `latency` một bên `lower`, một bên `higher`) hoặc các cặp từ so sánh nội hàm metric đối lập trực tiếp (`faster` vs `slower`, `outperforms` vs `underperforms`).
  - Khi hai câu đề cập metric khác nhau (ví dụ `"RT-DETR has a lower error rate than YOLOv8"` vs `"RT-DETR achieves higher accuracy than YOLOv8"`), bộ luật không còn kết luận sai thành `CONTRADICTED`, mà nhường quyền đánh giá ngữ nghĩa cho LLM.
- **Lọc số, hệ số nhân và đơn vị khi tìm chủ ngữ (P1)**:
  - Bổ sung helper `_is_number_or_unit`: nhận diện số thực/nguyên (`1.5`, `59.2`), hệ số nhân (`2x`, `10x`), và đơn vị (`ms`, `fps`, `ap`, `map`, `nds`, `%`...).
  - Thêm `_is_number_or_unit(tok)` vào điều kiện bỏ qua khi quét ngược tìm thực thể chủ ngữ trong `extract_comparative_triples`.
  - `"YOLOv8 is 2x faster than RT-DETR"` trích xuất chính xác chủ ngữ là `yolov8`; `"RT-DETR is 1.5 AP higher than YOLOv8"` trích xuất chính xác `rt-detr`.
- **Ép kiểu an toàn và bẫy lỗi trong `budget.record_llm_call` (P2)**:
  - Dùng `int(count)` và `int(float(tokens))`, ném `ValueError` rõ ràng khi nhận kiểu dữ liệu không hợp lệ thay vì âm thầm bỏ qua.
- **Hỗ trợ Enum schema trong `MockLLMBackend`**:
  - `MockLLMBackend.structured_generate` tự động gán giá trị enum hợp lệ (`list(field_info.annotation)[0].value`) thay vì `None`, giúp các unit test chạy với schema Enum như `NLILabel` không bị ném `ValidationError`.
- **Regression Tests**:
  - Đạt **119 passed + 1 skipped** (bổ sung đầy đủ 5 test case kiểm thử hồi quy tương ứng với 5 vấn đề trong `tests/test_verification_nli.py`).

## Đã giải quyết (sau review `aa5a034` — Mock NLI, Multi-Quote 1-Call & Metric Polarity)

- **Sửa hành vi mặc định Mock NLI (P0)**:
  - Trong `backend/llm/mock.py`: với `NLIStructuredOutputSchema`, mock không còn tự động gán enum đầu tiên (`SUPPORTED`) mà mặc định trả về `NOT_SUPPORTED 0.5`, tránh hiện tượng false pass trong test khi test không khai báo `canned_responses` cho NLI.
  - Khai báo tường minh canned NLI schema trong các integration test `tests/test_evidence.py` khi kiểm tra luồng hoàn chỉnh.
- **Tối ưu hóa Pipeline Multi-Quote: Tối đa 1 LLM Call & tránh False Contradiction (P0)**:
  - Trong `ClaimVerificationPipeline.verify_claim_against_quotes`: khi câu có nhiều trích dẫn (`len(valid_quotes) > 1`), pipeline không còn gọi LLM cho từng quote riêng lẻ (vốn tốn $N+1$ lượt gọi và dễ bị LLM 3B phán nhầm `CONTRADICTED` vì một quote đơn lẻ không chứa đủ vế so sánh).
  - Thay vào đó, pipeline chạy `rule_verifier` trên từng quote (0 LLM call) chỉ để bắt mâu thuẫn đối lập trực tiếp hoặc trùng nguyên văn (`SUPPORTED 1.0`); sau đó gọi `self.verifier` DUY NHẤT 1 LẦN trên văn bản hợp nhất `combined_evidence = " ".join(valid_quotes)`.
- **Lý do PARTIAL minh bạch khi NLI không khả dụng (P1)**:
  - Trong `ResearchEngine.run_basic_answer`: ghi nhận `nli_llm_evaluated_count`.
  - Nếu session kết thúc bằng `PARTIAL` vì claim có trích dẫn nhưng không claim nào đạt `SUPPORTED`:
    - Nếu không có lượt gọi LLM NLI nào thành công (`nli_llm_evaluated_count == 0` do LLM lỗi, Ollama offline, hoặc hết budget nên chỉ qua rule-based), lý do ghi nhận rõ ràng: `"NLI unavailable – claims not verified"`.
    - Nếu LLM NLI đã đánh giá nhưng từ chối: ghi nhận `"Report contains cited claims but none were confirmed as fully supported by evidence"`.
- **Khai thác chiều metric (Polarity) và lọc danh từ chung khi trích xuất chủ ngữ (P1)**:
  - Bổ sung `_resolve_polarity_and_domain` tích hợp `_get_metric_polarity`, phân loại `SPEED_METRIC_TERMS` và `ACCURACY_METRIC_TERMS`. Áp dụng trực tiếp vào bước 3 của `RuleBasedNLIVerifier.verify`, đồng bộ cả kiểm tra đối kháng (`CONTRADICTED`) và kiểm tra đồng thuận (`has_agreeing`).
  - Bổ sung `GENERIC_DOMAIN_NOUNS` (`detector`, `detectors`, `model`, `system`, `real-time`, `sota`...) vào `CONTEXT_EXCLUDED`.
  - Bổ sung tham số `known_entities` (lấy từ research goal) được truyền xuyên suốt qua `BaseNLIVerifier`, `CompositeNLIVerifier`, và `ClaimVerificationPipeline`.
  - Trong `extract_comparative_triples`: ưu tiên 1 so khớp với `known_entities`, ưu tiên 2 bỏ qua `GENERIC_DOMAIN_NOUNS`, giúp câu phức như `"RT-DETR is a real-time detector with higher AP than YOLOv8"` trích xuất chính xác chủ ngữ là `rt-detr`.
- **Regression Tests**:
  - Đạt **123 passed, 0 skipped** (bổ sung 4 unit test mới kiểm thử hồi quy cho generic nouns, polarity, multi-quote 1-call và nli unavailable trong `tests/test_verification_nli.py`).

## Đã giải quyết (sau review `19b14d0` — Clause-Aware Comparative Scoping, Metric Comparability Guard & Model Variant Differentiation)

- **Ngăn chặn mâu thuẫn giả khi khác metric hoặc một bên không có metric (P0)**:
  - Bổ sung `speed`, `efficiency`, `inference speed`, `sample efficiency` vào `HIGHER_IS_BETTER_METRICS` & `ALL_KNOWN_METRICS`.
  - Bổ sung `training time`, `train time`, `inference time`, `size`, `model size` vào `LOWER_IS_BETTER_METRICS`.
  - Bổ sung `inference`, `training time`, `train time`, `inference speed` vào `SPEED_METRIC_TERMS`.
  - Với các quan hệ hoán đổi thực thể (swapped entities, 3a): áp dụng guardrail `is_swapped_comparable`. Chỉ kết luận `CONTRADICTED 0.95` khi hai bên cùng so sánh trên cùng domain cụ thể (`speed` vs `speed` hoặc `accuracy` vs `accuracy`), hoặc chia sẻ metric cụ thể (`shared_metrics`), hoặc cả hai đều không nêu metric. Nếu một bên là `accuracy` và bên kia là `speed`, hoặc một bên có metric và bên kia là so sánh chung, rule-based KHÔNG kết luận `CONTRADICTED` mà chuyển quyền đánh giá ngữ nghĩa cho LLM.
- **Trích xuất quan hệ so sánh theo từng mệnh đề (Clause-Aware Scoping) (P0)**:
  - Trong `extract_comparative_triples`: với mỗi từ khóa `than <Obj>`, chỉ ghép cặp với từ so sánh gần nhất nằm ngay trước nó (sau `prev_than_end`), loại bỏ việc ghép từ so sánh của mệnh đề trước với tân ngữ của mệnh đề sau.
  - Quét ngược tìm chủ ngữ loại trừ toàn bộ tân ngữ của các mệnh đề `than` (`all_than_objects`) và toàn bộ từ so sánh (`ALL_COMPARATIVES`), ngăn chặn việc nhầm lẫn tân ngữ của vế trước thành chủ ngữ của vế sau.
  - Câu phức `"RT-DETR is faster than YOLOv8 but slower than RF-DETR"` trích xuất chính xác 2 triple `(rt-detr, faster, yolov8)` và `(rt-detr, slower, rf-detr)`, hoàn toàn không sinh triple chéo sai `(rt-detr, faster, rf-detr)` hay `(yolov8, slower, rf-detr)`.
  - Hỗ trợ phát hiện đồng thuận khi hoán đổi thực thể đối lập chiều (`"A is slower than B"` đồng thuận với `"B is faster than A"`).
- **Phân biệt hậu tố biến thể model (Model Variant Suffixes) (P1)**:
  - Xây dựng `VARIANT_SUFFIX_PATTERN` bắt các hậu tố kích thước/backbone chuẩn (`-l`, `-n`, `-s`, `-m`, `-x`, `-r50`, `-r101`, `yolov8n`, `resnet50`...). Bảo tồn các tên model họ chung có gạch nối (`rt-detr`, `fast-rcnn`, `mask-rcnn`).
  - Trong `_entities_match(e1, e2)`: nếu hai thực thể cùng có hậu tố biến thể khác nhau (`s1 and s2 and s1 != s2`, ví dụ `yolov8-l` vs `yolov8-n`, hoặc `rt-detr-r50` vs `rt-detr-r101`), coi chúng là hai thực thể khác biệt và trả về `False`.
  - Khẳng định `RT-DETR-L is faster than YOLOv8-L` không bị coi là mâu thuẫn với `YOLOv8-N is faster than RT-DETR-L`.
- **Regression Tests**:
  - Đạt **126 passed, 0 skipped** (bổ sung 3 unit test hồi quy toàn diện cho 3 vấn đề trong `tests/test_verification_nli.py`).

## Vấn đề còn mở (ưu tiên từ trên xuống)

1. Chạy E2E thật lại 2–3 goal để xem NLI hoạt động thực tế với LLM thật (SmolLM3), kiểm tra tỷ lệ latency/budget tiêu thụ cho NLI trên CPU/GPU.
2. Tỉ lệ quote bị loại "not in source chunk" cao → dữ liệu cho Dataset C (plan 43).
3. Tuần 4 tiếp theo: writer theo section (plan 28), UI đầy đủ.

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
| 2026-09-27 | Compare YOLOv8 and RT-DETR accuracy and latency on COCO (`sess_2137844bb549`) | **PARTIAL** (474s, <600s), 15 sources, 4 evidence nguyên văn trích xuất từ arXiv YOLOv10 (1.5 AP, 2.0 AP, 51%/61%, 41%/52%, 46%/62%). **NLI hoạt động thực tế thành công**: 4/4 cited claims được SmolLM3-3B xác nhận **`SUPPORTED` 1.0** với reasoning đầy đủ; 1 claim thiếu citation bị gắn nhãn **`UNSUPPORTED`**; 0 mâu thuẫn giả (`CONTRADICTED`). Dừng trung thực ở PARTIAL do thiếu dữ liệu so sánh trực tiếp YOLOv8 vs RT-DETR |


