---
license: mit
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - benchmarking
  - llama-cpp
  - llm-inference
  - local-llm
  - homelab
  - open-data
  - consumer-hardware
pretty_name: Poor Paul's Benchmark Results
size_categories:
  - 1K<n<10K
---

# Dataset Card for Poor Paul's Benchmark Results

## Dataset Description

Poor Paul's Benchmark (PPB) is an open benchmarking framework for local AI
inference on consumer, prosumer, and small-business hardware.

This dataset contains normalized, community-submitted benchmark results across
models, quantizations, hardware, runtimes, and benchmark settings. Each row
represents one benchmark result and is intended for open benchmarking,
reproducibility, and downstream analysis.

Common use cases include:

- Comparing inference throughput across hardware
- Studying context-length scaling
- Comparing latency metrics such as TTFT and ITL
- Evaluating multi-GPU scaling and split strategies
- Capacity planning via concurrency curves
- Powering dashboards and derived leaderboards such as [poorpaul.dev](https://poorpaul.dev)

Current benchmark runners represented in the dataset include:

- `llama-bench` — raw throughput (tok/s)
- `llama-server` — TTFT and ITL latency with real conversational prompts
- `llama-server-loadtest` — max concurrency discovery and concurrency curves

## Dataset Structure

Each row corresponds to one normalized benchmark result.

The schema is arranged so the most important fields appear first in the Hugging
Face viewer:

1. Model identity and quantization
2. LLM engine
3. Hardware
4. Benchmark settings
5. Workload
6. Performance metrics
7. OS / system context
8. Submission metadata
9. Provenance and deduplication
10. Extensibility

### Column Reference

#### Model identity

| Column | Type | Description |
|---|---|---|
| `model` | string | Full model path (e.g. `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf`) |
| `model_base` | string | Base model name without quant suffix (e.g. `Qwen3.5-9B`) |
| `quant` | string | Quantization format (e.g. `Q8_0`, `Q4_K_M`, `F16`) |
| `model_org` | string | HF organisation (e.g. `unsloth`). `null` for local models |
| `model_repo` | string | Full HF `org/repo` string (e.g. `unsloth/Qwen3.5-9B-GGUF`). `null` for local models |
| `runner_type` | string | Benchmark backend: `llama-bench`, `llama-server`, or `llama-server-loadtest` |

#### LLM engine

| Column | Type | Description |
|---|---|---|
| `llm_engine_name` | string | Inference engine identifier (e.g. `llama.cpp`) |
| `llm_engine_version` | string | Engine version with build hash (e.g. `b5063 (58ab80c3)`) |

#### Hardware

| Column | Type | Description |
|---|---|---|
| `gpu_name` | string | Primary GPU model name |
| `gpu_vram_gb` | float | Primary GPU VRAM in GB |
| `gpu_driver` | string | GPU driver version |
| `gpu_count` | int | Number of GPUs used |
| `gpu_names` | string | Comma-joined list of all GPU names |
| `gpu_total_vram_gb` | float | Sum of VRAM across all GPUs |
| `backends` | string | Compute backend with version (e.g. `CUDA 12.8`, `Metal`) |
| `cpu_model` | string | CPU model name |

#### Configuration

| Column | Type | Description |
|---|---|---|
| `n_ctx` | int | Context window size (tokens) |
| `n_batch` | int | Batch size for prompt processing |
| `split_mode` | string | Multi-GPU split strategy (`layer`, `row`, `none`). `null` for single-GPU |
| `tensor_split` | string | Per-GPU VRAM weight string (e.g. `"1,1"`). `null` for single-GPU |
| `concurrent_users` | int | Number of simulated parallel users |

#### Workload

| Column | Type | Description |
|---|---|---|
| `task_type` | string | Workload category (e.g. `text-generation`) |
| `prompt_dataset` | string | Dataset identifier (e.g. `sharegpt-v3`). `null` for `llama-bench` |
| `num_prompts` | int | Number of prompts sent per run. `null` for `llama-bench` |
| `n_predict` | int | Max tokens generated per prompt. `null` for `llama-bench` |

#### Performance — raw speed

| Column | Type | Description |
|---|---|---|
| `throughput_tok_s` | float | Tokens per second (prompt processing or generation) |

#### Performance — power efficiency

| Column | Type | Description |
|---|---|---|
| `avg_power_w` | float | Average GPU power draw during the run (watts) |
| `max_power_w` | float | Peak GPU power draw during the run (watts) |

#### Performance — thermal

| Column | Type | Description |
|---|---|---|
| `avg_gpu_temp_c` | float | Average GPU temperature (°C) |
| `max_gpu_temp_c` | float | Peak GPU temperature (°C) |
| `avg_cpu_temp_c` | float | Average CPU temperature (°C) |
| `max_cpu_temp_c` | float | Peak CPU temperature (°C) |
| `avg_fan_speed_rpm` | float | Average fan speed (RPM) |
| `max_fan_speed_rpm` | float | Peak fan speed (RPM) |

#### Performance — user experience

| Column | Type | Description |
|---|---|---|
| `avg_ttft_ms` | float | Average Time-To-First-Token (ms). `llama-server` only |
| `p50_ttft_ms` | float | Median TTFT (ms) |
| `p99_ttft_ms` | float | 99th-percentile TTFT (ms) |
| `avg_itl_ms` | float | Average Inter-Token Latency (ms) |
| `p50_itl_ms` | float | Median ITL (ms) |
| `p99_itl_ms` | float | 99th-percentile ITL (ms) |

#### Performance — quality

| Column | Type | Description |
|---|---|---|
| `quality_score` | float | Output quality metric. Reserved for future use, currently `null` |

#### OS / system context

| Column | Type | Description |
|---|---|---|
| `os_system` | string | OS family (`Linux`, `Darwin`, `Windows`) |
| `os_release` | string | Kernel version |
| `os_machine` | string | CPU architecture (e.g. `x86_64`, `arm64`) |
| `os_distro` | string | Linux distribution (e.g. `Ubuntu`), or `macOS` / `Windows` |
| `os_distro_version` | string | Distribution version (e.g. `24.04`, `15.5`) |
| `cpu_cores` | string | CPU core count |
| `ram_total_gb` | float | Total system RAM in GB |

#### Submission metadata

| Column | Type | Description |
|---|---|---|
| `submitter` | string | Optional public display name of the contributor |
| `timestamp` | string | ISO 8601 timestamp of the benchmark run |
| `submitted_at` | string | ISO 8601 timestamp of the upload |

#### Provenance and deduplication

| Column | Type | Description |
|---|---|---|
| `schema_version` | string | Schema version at time of flattening (e.g. `0.1.0`) |
| `benchmark_version` | string | PPB software version |
| `suite_run_id` | string | UUID grouping all records from a single `ppb all` / `ppb sweep` invocation |
| `submission_id` | string | UUID assigned during upload |
| `row_id` | string | UUID uniquely identifying this row |
| `machine_fingerprint` | string | SHA-256 hash of hardware profile (anonymous machine identity) |
| `run_fingerprint` | string | SHA-256 hash of benchmark configuration (model + settings + machine) |
| `result_fingerprint` | string | SHA-256 hash of performance metrics (exact result identity) |
| `source_file_sha256` | string | SHA-256 hash of the source JSONL file |

#### Extensibility

| Column | Type | Description |
|---|---|---|
| `tags` | string | Free-form JSON string for arbitrary metadata (e.g. `{"env": "ci"}`) |

### Notes on null values

Many fields are runner-specific, so `null` values are expected:

- `llama-bench` rows will have `null` for TTFT/ITL, `prompt_dataset`, `num_prompts`, `n_predict`, and `concurrent_users`
- `llama-server` and `llama-server-loadtest` rows will have `null` for `backends` (llama-bench specific)
- Multi-GPU fields (`split_mode`, `tensor_split`) are `null` for single-GPU runs
- `quality_score` is `null` until quality evaluation is implemented
- `tags` is `null` unless the submitter attached custom metadata

The dataset currently uses a single split: `train`.

## Dataset Creation

The source data comes from PPB runs executed locally by users on their own
hardware.

Before upload, raw benchmark outputs are normalized into a flat tabular schema
so they can be previewed on Hugging Face and easily consumed by pandas, DuckDB,
spreadsheets, and downstream dashboards.

## Considerations for Using the Data

This dataset is an append-only raw submission ledger, not a final curated leaderboard.

Important limitations:

- Results come from heterogeneous real-world systems
- Thermals, drivers, background load, and local tuning can affect outcomes
- Some metrics apply only to specific runner types
- Duplicate and repeated submissions may exist by design
- Cost-related metrics are not included

For downstream analysis:

- Use `result_fingerprint` to identify exact duplicate rows
- Use `run_fingerprint` to group repeated runs of the same benchmark identity
- Use `machine_fingerprint` to group results from the same anonymous machine
- Use `llm_engine_name` and `llm_engine_version` to compare across inference backends
- Use `os_distro` and `os_distro_version` to study OS-level performance differences
- Use `task_type` and `prompt_dataset` to ensure like-for-like comparisons

## Additional Information

This dataset is intended to contain benchmark telemetry, not personal
information. Contributors may optionally include a public display name through `submitter`.

License: MIT

Links:

- Project repository: https://github.com/paulplee/poor-pauls-benchmark
- Dataset repository: https://huggingface.co/datasets/paulplee/ppb-results
- Leaderboard: https://poorpaul.dev
