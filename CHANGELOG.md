# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-10

### Added
- `ppb sweep` — throughput benchmark via `llama-bench` subprocess across n_ctx/n_batch/quant combos
- `ppb vram-cliff` — OOM-probe sweep to find max safe context window per quantisation
- `ppb server-bench` — TTFT/ITL latency measurement via llama-server `/completion` SSE streaming
- `ppb loadtest` — auto-discovers max sustainable concurrency (1→2→4→8→…) until error threshold
- `ppb publish` — upload results JSONL to HuggingFace dataset via `huggingface_hub`
- `ppb hw-info` — prints detected GPU name, VRAM, driver version
- `ppb all` — runs vram-cliff → sweep → publish in sequence
- Suite TOML files for model/sweep configuration; gitignored personal configs, example committed
- `scripts/aggregate_results.py` — builds aggregated Parquet with mean/StdDev/95% CI across repeated runs
- `scripts/backfill_flags.py` — backfills schema v0.10.0 fields into older JSONL result files
- Qualitative benchmark suite: context-rot, tool-accuracy, answer-quality, multi-turn memory tracking
- `PPB_MODELS_DIR`, `PPB_RESULTS_FILE`, `PPB_LLAMA_BENCH`, `PPB_LLAMA_SERVER` env var overrides
