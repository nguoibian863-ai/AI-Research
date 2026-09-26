# Week 1 Review — Core Engine

> Đánh giá nghiệm thu Tuần 1 (roadmap mục 37 trong `local_deep_research_agent_senior_plan.md`).
> Ngày: 2026-09-26 · Phần cứng: laptop GPU 4GB VRAM · Model: `hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M`

## 1. Kết luận

- **Tuần 1 đạt theo tiêu chí nghiệm thu**: pipeline `Question → Plan → Search → Fetch → Answer` chạy end-to-end với LLM thật, planner xuất JSON hợp lệ, giới hạn step/budget được enforce, LLM backend thay được.
- **Chất lượng nghiên cứu chưa đạt**: model 3B hiểu sai câu hỏi và sinh query dài dạng câu hỏi nên search trả về nguồn không liên quan. Cần sửa trước khi bắt đầu Tuần 2 (mục 6).

## 2. Tiêu chí nghiệm thu

| Tiêu chí | Kết quả | Bằng chứng |
|---|---|---|
| Một research session chạy end-to-end | ✅ | `POST /api/research/run` → `DONE` sau 42s với Ollama thật |
| Không infinite loop | ✅ | Giới hạn `max_research_steps`, search/fetch/LLM calls, runtime; test `test_multi_step_e2e_terminates_at_max_steps` |
| Planner output valid JSON | ✅ | `ResearchPlanSchema` (schema lồng nhau) parse thành công với SmolLM3 thật |
| Max step được enforce | ✅ | `run_week1` đi qua `EVALUATE → evaluate_next_step`; `budget.steps` đồng bộ với `state.step` |
| Local LLM backend swap được | ⚠️ | `LLMBackend` + `MockLLMBackend` + `OllamaBackend`; chưa có backend llama.cpp, `llm.profiles` trong config chưa được dùng |

## 3. Lịch sử review

| Vòng | Commit | Vấn đề chính phát hiện | Trạng thái |
|---|---|---|---|
| 1 | `3e65e8f` | Ghi raw evidence giả; trajectory `verified=True` giả; gán sai source; lỗi không chuyển `FAILED`; budget chỉ trong RAM; test dùng DB/mạng thật; max step không chạy trong luồng thật | Đã sửa ở `5034266` |
| 2 | `5034266` | `_handle_phase_error` ghi đè state terminal → request sai của client làm hỏng session (`DONE → PARTIAL`); `/run` trả 200 khi lỗi; `KeyError` → 404 toàn cục; `budget.steps` lệch `state.step` | Đã sửa ở `26f0e53` |
| 3 | `26f0e53` | `/health` bỏ qua dependency override (test tạo DB thật); lỗi giữa các phase làm session treo `RUNNING`; không xoá được `error_message`; logging cấu hình lúc import và `handlers.clear()` root logger | Đã sửa ở `33de030` |
| 4 | `a569fb3`, `cac640e` | Tag `smollm3:3b` không tồn tại trên Ollama; SmolLM3 sinh phần suy luận với thẻ `<think>` được prefill (chỉ có `</think>` trong output) | Chuyển sang GGUF từ Hugging Face, `system_prefix: "/no_think"`, cắt phần suy luận |

Test suite: **27/27 pass** (~0.6s, không cần mạng, không cần GPU).

## 4. Benchmark model (`scripts/benchmark_ollama.py`)

| Chỉ số | Kết quả | Nhận xét |
|---|---|---|
| Tốc độ sinh | **68 tokens/s** | Report 1024 token ≈ 15s |
| VRAM của model | **2.19 GB**, 100% trên GPU | Không offload sang CPU |
| Tổng VRAM đang dùng | 3.68 / 4.10 GB | Còn ~0.4 GB → **giữ `num_ctx: 4096`**, chưa nâng lên 6144/8192 |
| JSON schema constrained decoding | Hợp lệ, 0.86s | Nội dung số liệu bịa (`accuracy_nds: 0.95`, thực tế CenterPoint ≈ 0.67) → model chỉ dùng để trích xuất, không dùng để "nhớ" số liệu |
| `/no_think` trong system prompt | ✅ Có hiệu lực | Đặt trong user prompt thì **không** có hiệu lực |

Cài đặt model:

```powershell
ollama pull hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M
python scripts/benchmark_ollama.py
```

## 5. Chạy E2E thật

Goal: `Compare PointPillars and CenterPoint on nuScenes`

| Trường | Giá trị |
|---|---|
| phase / status | `DONE` / `COMPLETED` |
| step | 1 |
| Plan | 5 task, 3 giả thuyết |
| Queries | 4 |
| Sources / documents | 8 / 8 |
| Budget | LLM 3/30 · search 4/12 · fetch 8/20 · 3355 tokens · 42.18s |
| Answer | "There is no information available to compare PointPillars and CenterPoint in the context of nuScenes." |

Điểm tốt: model **không bịa thông tin** khi context không liên quan.

## 6. Vấn đề chất lượng (cần sửa trước Tuần 2)

1. **Planner hiểu sai câu hỏi.** Model coi PointPillars/CenterPoint là đối tượng trong dataset ("number of instances in nuScenes") thay vì mô hình phát hiện 3D; một task đề xuất "user study".
2. **Query dài dạng câu hỏi** (15–20 từ), ví dụ: *"How many PointPillars and CenterPoint instances are present in the nuScenes dataset?"* → DuckDuckGo trả kết quả kém.
3. **Search trả về rác** (Facebook, WhatsApp, Snaptube, YouTube, Chrome) và **không có lọc độ liên quan**, nên cả 8 trang đều bị fetch.
4. **Trạng thái sai**: session `DONE/COMPLETED` dù câu trả lời là "không có thông tin" → nên là `PARTIAL`.

### Đề xuất sửa

| # | Sửa | Tác dụng |
|---|---|---|
| 1 | Prompt planner có ngữ cảnh ("so sánh phương pháp/mô hình; task là tìm số liệu công bố trong paper/benchmark") + 1 ví dụ few-shot | Plan đúng hướng; model 3B bám ví dụ tốt hơn bám mô tả |
| 2 | Query ngắn 3–8 từ dạng từ khoá (ví dụ `CenterPoint nuScenes NDS mAP`), giới hạn `max_length` trong schema | Search liên quan hơn |
| 3 | Lọc độ liên quan bằng Python (title/snippet phải chứa từ khoá chính của goal) trước khi fetch | Chặn rác, tiết kiệm fetch budget, test được |
| 4 | Chuyển `PARTIAL` khi không có nguồn liên quan | Trạng thái phản ánh đúng kết quả |

Kiểm chứng nguyên nhân query:

```powershell
python -c "from ddgs import DDGS; [print(r['title']) for r in DDGS().text('How many PointPillars and CenterPoint instances are present in the nuScenes dataset?', max_results=5)]"
python -c "from ddgs import DDGS; [print(r['title']) for r in DDGS().text('CenterPoint PointPillars nuScenes NDS', max_results=5)]"
```

## 7. Nợ kỹ thuật còn lại

| Mục | Ghi chú | Dự kiến |
|---|---|---|
| Vòng lặp research là stub | Chưa có gì ghi `open_questions` nên luồng thật chỉ chạy 1 vòng | Gap evaluator — Tuần 3 |
| Retry/repair JSON | Chưa có; một lần JSON sai → 502 | Trước Tuần 2 (schema extraction phức tạp hơn) |
| `max_runtime_seconds` | Tính theo từng request, `start_time` không được lưu | Tuần 2 |
| Fetch tuần tự, timeout 20s/trang | Lượt chạy có thể mất vài phút nếu gặp trang chậm | Fetch song song + timeout ngắn — Tuần 2 |
| Fetch không xử lý PDF/content-type | PDF bị đọc như text | Parsing pipeline — Tuần 2 |
| Endpoint `CANCELLED`, backend llama.cpp | Chưa có | P2 |
| `StateTransitionError` phát sinh bên trong engine | Không đưa session về trạng thái lỗi (được coi như lỗi client) | Theo dõi |
| Tiếng Việt | Model 3B nhận nhầm tiếng Việt là tiếng Trung, trả lời tiếng Anh | Kiểm tra nếu report cần tiếng Việt |
