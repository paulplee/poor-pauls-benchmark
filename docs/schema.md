# PPB Result Schema

Every benchmark run appends one or more JSON objects (one per line) to
`results.jsonl`. The flattener (`utils/flattener.py`) normalizes nested
runner output into a strict, Arrow-friendly column set. This document is
the canonical reference for every field in that flat row.

The authoritative source is `COLUMN_ORDER` in
[`utils/flattener.py`](../utils/flattener.py). Update this file whenever
that list changes.

- **Schema version:** `0.1.0`
- **Format:** JSON Lines (one record per row)
- **Field convention:** keys not applicable to a given runner are `null`

## Benchmark Identity

| Field         | Type           | Description                                                                                  |
| ------------- | -------------- | -------------------------------------------------------------------------------------------- |
| `model`       | string \| null | Full model path passed to the runner (e.g. `unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf`) |
| `model_base`  | string \| null | Base model name parsed from the GGUF filename (e.g. `Qwen3.5-2B`)                            |
| `quant`       | string \| null | Quantization label parsed from the filename (e.g. `Q4_K_M`, `BF16`)                          |
| `model_org`   | string \| null | Hugging Face org/user (e.g. `unsloth`)                                                       |
| `model_repo`  | string \| null | Hugging Face `org/repo` portion of the model path                                            |
| `runner_type` | string         | Backend used: `llama-bench`, `llama-server`, or `llama-server-loadtest`                      |

## LLM Engine

| Field                | Type           | Description                              |
| -------------------- | -------------- | ---------------------------------------- |
| `llm_engine_name`    | string \| null | Inference engine name (e.g. `llama.cpp`) |
| `llm_engine_version` | string \| null | Engine version / build hash              |

## Hardware

| Field               | Type           | Description                                                                                      |
| ------------------- | -------------- | ------------------------------------------------------------------------------------------------ |
| `gpu_name`          | string \| null | Primary GPU model name (e.g. `NVIDIA GeForce RTX 4090`)                                          |
| `gpu_vram_gb`       | float \| null  | Primary GPU VRAM in GB                                                                           |
| `gpu_driver`        | string \| null | Primary GPU driver version                                                                       |
| `gpu_count`         | integer        | Number of GPUs detected                                                                          |
| `gpu_names`         | string \| null | Comma-separated list of all detected GPU names                                                   |
| `gpu_total_vram_gb` | float \| null  | Total VRAM across all GPUs in GB                                                                 |
| `backends`          | string \| null | Compute backend(s), enriched with CUDA version when available (e.g. `CUDA 13.0`, `Metal`, `CPU`) |
| `cpu_model`         | string \| null | CPU model string                                                                                 |

## Configuration

| Field              | Type            | Description                                            |
| ------------------ | --------------- | ------------------------------------------------------ |
| `n_ctx`            | integer \| null | Context window size (tokens)                           |
| `n_batch`          | integer \| null | Prompt-processing batch size                           |
| `split_mode`       | string \| null  | Multi-GPU split strategy (e.g. `none`, `layer`, `row`) |
| `tensor_split`     | string \| null  | GPU weight ratios for multi-GPU (e.g. `"1,1"`)         |
| `concurrent_users` | integer \| null | Simultaneous inference requests (server runners only)  |

## Workload

| Field            | Type            | Description                                    |
| ---------------- | --------------- | ---------------------------------------------- |
| `task_type`      | string \| null  | Workload type (default: `text-generation`)     |
| `prompt_dataset` | string \| null  | Prompt dataset identifier (e.g. `sharegpt-v3`) |
| `num_prompts`    | integer \| null | Number of prompts sent                         |
| `n_predict`      | integer \| null | Max tokens generated per prompt                |

## Performance — Throughput

| Field              | Type          | Description                                             |
| ------------------ | ------------- | ------------------------------------------------------- |
| `throughput_tok_s` | float \| null | Aggregate tokens per second (primary throughput metric) |

## Performance — Power

| Field         | Type          | Description                                              |
| ------------- | ------------- | -------------------------------------------------------- |
| `avg_power_w` | float \| null | Average power draw in Watts (NVML / RAPL / powermetrics) |
| `max_power_w` | float \| null | Peak power draw in Watts                                 |

## Performance — Thermal

| Field               | Type          | Description                   |
| ------------------- | ------------- | ----------------------------- |
| `avg_gpu_temp_c`    | float \| null | Average GPU temperature in °C |
| `max_gpu_temp_c`    | float \| null | Peak GPU temperature in °C    |
| `avg_cpu_temp_c`    | float \| null | Average CPU temperature in °C |
| `max_cpu_temp_c`    | float \| null | Peak CPU temperature in °C    |
| `avg_fan_speed_rpm` | float \| null | Average fan speed (RPM)       |
| `max_fan_speed_rpm` | float \| null | Peak fan speed (RPM)          |

## Performance — User Experience (server runners)

| Field         | Type          | Description                       |
| ------------- | ------------- | --------------------------------- |
| `avg_ttft_ms` | float \| null | Average time-to-first-token in ms |
| `p50_ttft_ms` | float \| null | Median TTFT in ms                 |
| `p99_ttft_ms` | float \| null | 99th-percentile TTFT in ms        |
| `avg_itl_ms`  | float \| null | Average inter-token latency in ms |
| `p50_itl_ms`  | float \| null | Median ITL in ms                  |
| `p99_itl_ms`  | float \| null | 99th-percentile ITL in ms         |

## Performance — Quality (Quantitative)

| Field           | Type          | Description                                                                   |
| --------------- | ------------- | ----------------------------------------------------------------------------- |
| `quality_score` | float \| null | Reserved for future quantitative quality evaluation (currently always `null`) |

## Qualitative Evaluation (Context-Rot)

The qualitative phase runs a semantic Needle-in-a-Haystack (NIAH) evaluation
over a 6 × 5 grid of haystack lengths × needle depths. Rows produced by this
phase carry `runner_type = "context-rot"` and `run_type = "qualitative"` (or
`"all"` when produced by `ppb all`). Quantitative perf columns
(`avg_throughput_tps`, `p50_ttft_ms`, …) are `null` on these rows.

| Field                            | Type           | Description                                                                       |
| -------------------------------- | -------------- | --------------------------------------------------------------------------------- |
| `task_type`                      | string \| null | `"context-rot-niah"` for qualitative rows                                         |
| `prompt_dataset`                 | string \| null | Source corpus for haystack tokens (e.g. `"sharegpt-v3"`)                          |
| `context_rot_score`              | float \| null  | Mean accuracy across all (length × depth) cases — the headline qualitative metric |
| `context_rot_accuracy_by_length` | string \| null | JSON-encoded `{haystack_length: accuracy}` map (one entry per runnable length)    |
| `context_rot_accuracy_by_depth`  | string \| null | JSON-encoded `{depth_pct: accuracy}` map (one entry per tested depth)             |

Lengths exceeding the model's measured `vram_cliff_tokens` are skipped and
recorded as `null` in `context_rot_accuracy_by_length` rather than failing the
run.

## Qualitative Evaluation (Answer Knowledge-Accuracy & Quality)

Phase 6 scores model answers on three orthogonal dimensions using a
_two-model_ pipeline: the model under test (the "generator") plus a separate,
smaller local "judge" model. The generator and judge MUST be different
models — the judge cannot grade itself. Rows produced by this phase carry
`runner_type = "answer-quality"` and `task_type = "answer-quality"`.

| Field                     | Type          | Description                                                              |
| ------------------------- | ------------- | ------------------------------------------------------------------------ |
| `knowledge_accuracy_mean` | float \| null | Mean fraction of factual claims judged consistent with common knowledge  |
| `knowledge_accuracy_std`  | float \| null | Standard deviation of per-prompt knowledge-accuracy scores               |
| `answer_relevancy_mean`   | float \| null | Mean judge-rated relevancy (1–5 → 0–1)                                   |
| `coherence_mean`          | float \| null | Mean judge-rated coherence (1–10 → 0–1)                                  |
| `quality_composite_score` | float \| null | Mean of the three means above — the headline answer-quality metric       |

NOTE: `knowledge_accuracy` differs from RAGAS-style faithfulness, which
grounds verification against a reference context. PPB has no ground-truth
reference for open-ended ShareGPT prompts. See
[`docs/qualitative-methodology.md`](qualitative-methodology.md).

50 single-turn factual prompts are sampled once from
`anon8231489123/ShareGPT_Vicuna_unfiltered` and cached in
`ppb_quality_prompts_cache.json` so every model in the leaderboard is graded
on identical prompts. The cache file's SHA-256 is published in the row's
`meta.quality_prompts_cache_hash` field.

| Field  | Type           | Description                                                                                                |
| ------ | -------------- | ---------------------------------------------------------------------------------------------------------- |
| `meta` | object \| null | Per-row reproducibility hints. Currently carries only `quality_prompts_cache_hash` (SHA-256 of the cache). |

## Composable Schema and Join Key

PPB's three run modes (`all`, `quantitative`, `qualitative`) all emit rows
that share a stable composite join key so downstream consumers (ppb-mcp,
poorpaul.dev) can stitch them back together:

| Field        | Type           | Description                                            |
| ------------ | -------------- | ------------------------------------------------------ |
| `gpu_name`   | string \| null | Primary GPU model (first GPU in the hardware snapshot) |
| `model_name` | string \| null | Base model name (alias of `model_base`)                |
| `quant`      | string \| null | Quantisation tag (e.g. `Q4_K_M`, `IQ4_XS`)             |
| `run_type`   | string \| null | One of `"all"`, `"quantitative"`, `"qualitative"`      |

`(gpu_name, model_name, quant)` uniquely identifies a model+hardware pairing;
`run_type` distinguishes the phase that produced the row. A `qualitative`
row LEFT-JOINed against the most recent `quantitative` row on this key yields
a unified profile without re-running expensive perf sweeps.

## OS / System Context

| Field               | Type            | Description                                   |
| ------------------- | --------------- | --------------------------------------------- |
| `os_system`         | string \| null  | OS family: `Linux`, `Darwin`, `Windows`       |
| `os_release`        | string \| null  | OS release string                             |
| `os_machine`        | string \| null  | Machine architecture (e.g. `x86_64`, `arm64`) |
| `os_distro`         | string \| null  | Distro name (e.g. `Ubuntu`, `macOS`)          |
| `os_distro_version` | string \| null  | Distro version (e.g. `24.04`, `15.3`)         |
| `cpu_cores`         | integer \| null | Number of logical CPU cores                   |
| `ram_total_gb`      | float \| null   | System RAM in GB                              |

## Submission Metadata

| Field          | Type           | Description                                                       |
| -------------- | -------------- | ----------------------------------------------------------------- |
| `submitter`    | string \| null | Hugging Face username of the submitter                            |
| `timestamp`    | string \| null | UTC time the runner produced the row (ISO 8601)                   |
| `submitted_at` | string \| null | UTC time the row was published to Hugging Face (set by publisher) |

## Provenance / Deduplication

| Field                 | Type           | Description                                                                 |
| --------------------- | -------------- | --------------------------------------------------------------------------- |
| `schema_version`      | string         | Version of this schema (currently `0.1.0`)                                  |
| `benchmark_version`   | string         | Version of `poor-pauls-benchmark` that produced the row                     |
| `suite_run_id`        | string \| null | UUID shared by all rows produced by the same suite invocation               |
| `submission_id`       | string \| null | UUID set by the publisher for one upload batch                              |
| `row_id`              | string         | Per-row UUID (hex)                                                          |
| `machine_fingerprint` | string         | SHA-256 of stable hardware-identity fields                                  |
| `run_fingerprint`     | string         | SHA-256 of benchmark configuration + machine fingerprint                    |
| `result_fingerprint`  | string         | SHA-256 of run identity + measured metrics — uniquely identifies one result |
| `source_file_sha256`  | string \| null | SHA-256 of the JSONL file the row was published from                        |

## Extensibility

| Field         | Type           | Description                                                                                                                                |
| ------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `tags`        | object \| null | Free-form JSON dict for ad-hoc metadata from the suite TOML                                                                                |
| `raw_payload` | string         | The full original nested JSON row, serialized as a string. Included in JSONL/HF uploads for forensic inspection; excluded from CSV output. |

## Schema Versioning

The schema is append-only — new fields are added with `null` defaults so old
readers continue to work. If a breaking change is ever required,
`schema_version` and `benchmark_version` together identify the producing tool.

The canonical schema source is [`utils/flattener.py`](../utils/flattener.py)
(`COLUMN_ORDER`). This document should be updated whenever that list changes.
