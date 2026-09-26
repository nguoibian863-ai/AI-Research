# Quyết định thiết kế đã chốt

Mỗi mục: quyết định → lý do → thay thế đã loại. Chỉ sửa khi có bằng chứng mới (benchmark, lần chạy thật).

| # | Quyết định | Lý do | Đã loại |
|---|---|---|---|
| D1 | State machine Python quyết định mọi transition | Model 3B không đáng tin để tự quyết vòng lặp; cần budget cứng | LangGraph, ReAct agent, CrewAI |
| D2 | SQLite là source of truth | Local-first, truy vết được, rebuild index được | Chỉ dùng vector DB |
| D3 | LLM: SmolLM3-3B GGUF Q4_K_M qua Ollama (`hf.co/ggml-org/...`) | Tag `smollm3:3b` không có trên Ollama; bản GGUF 1.9GB vừa 4GB VRAM, 68 tok/s | `qwen3:8b`/`14b` (vượt VRAM), `deepseek-r1:8b` |
| D4 | `/no_think` trong system prompt + `strip_think_block` | Reasoning làm hỏng JSON và tốn token; template có thể prefill `<think>` | Để model tự suy luận |
| D5 | `num_ctx` giữ 4096 | Tổng VRAM đã 3.68/4.1GB | Tăng lên 6144/8192 |
| D6 | Embedding FastEmbed `bge-small-en-v1.5` chạy CPU | Semantic thật, 0 VRAM | Hash embedding (chỉ lexical, chỉ dùng trong test) |
| D7 | Hybrid BM25 + FAISS + RRF; boost reranker scale theo 1/(k+1) | BM25 cho tên/metric, vector cho ngữ nghĩa; boost không được lấn át RRF | FAISS-only; boost không scale |
| D8 | Quote phải tìm thấy nguyên văn trong chunk; số/subject kiểm bằng Python | Chặn hallucination mà không cần LLM | Tin output LLM |
| D9 | Không resolve được source → reject evidence | Gán nhầm source nguy hiểm hơn thiếu evidence | Fallback sang source đầu tiên |
| D10 | Claim = câu thật trong report | Lineage phải đi từ report về evidence | Claim = bản sao evidence |
| D11 | Dataset/benchmark (nuScenes, COCO…) là ngữ cảnh, không phải đối thủ so sánh | Tránh reject evidence hợp lệ | Coi mọi core entity là đối thủ |
| D12 | Few-shot prompt dùng domain trung tính (RocksDB/LevelDB) | Tránh lộ đáp án câu hỏi test | Few-shot trùng câu hỏi benchmark |
| D13 | Làm việc trực tiếp trên `main` | Chủ repo yêu cầu | Feature branch |
| D14 | Fine-tuning chỉ sau MVP, QLoRA adapter, bắt đầu từ evidence extraction | Train behavior, không train knowledge; 4GB chỉ đủ QLoRA | Full fine-tune, RL |
