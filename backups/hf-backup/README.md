---
license: mit
pretty_name: Poor Paul's Benchmark Results
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.jsonl
  features:
  - name: model
    dtype: string
  - name: model_base
    dtype: string
  - name: quant
    dtype: string
  - name: runner_type
    dtype: string
  - name: gpu_name
    dtype: string
  - name: gpu_vram_gb
    dtype: float64
  - name: gpu_driver
    dtype: string
  - name: backends
    dtype: string
  - name: cpu_model
    dtype: string
  - name: n_ctx
    dtype: int64
  - name: n_batch
    dtype: int64
  - name: concurrent_users
    dtype: int64
  - name: throughput_tok_s
    dtype: float64
  - name: avg_power_w
    dtype: float64
  - name: max_power_w
    dtype: float64
  - name: avg_gpu_temp_c
    dtype: float64
  - name: max_gpu_temp_c
    dtype: float64
  - name: avg_cpu_temp_c
    dtype: float64
  - name: max_cpu_temp_c
    dtype: float64
  - name: avg_fan_speed_rpm
    dtype: float64
  - name: max_fan_speed_rpm
    dtype: float64
  - name: avg_ttft_ms
    dtype: float64
  - name: p50_ttft_ms
    dtype: float64
  - name: p99_ttft_ms
    dtype: float64
  - name: avg_itl_ms
    dtype: float64
  - name: p50_itl_ms
    dtype: float64
  - name: p99_itl_ms
    dtype: float64
  - name: os_system
    dtype: string
  - name: os_release
    dtype: string
  - name: os_machine
    dtype: string
  - name: cpu_cores
    dtype: string
  - name: ram_total_gb
    dtype: float64
  - name: submitter
    dtype: string
  - name: timestamp
    dtype: string
  - name: submitted_at
    dtype: string
  - name: schema_version
    dtype: int64
  - name: benchmark_version
    dtype: string
  - name: suite_run_id
    dtype: string
  - name: submission_id
    dtype: string
  - name: row_id
    dtype: string
  - name: machine_fingerprint
    dtype: string
  - name: run_fingerprint
    dtype: string
  - name: result_fingerprint
    dtype: string
  - name: source_file_sha256
    dtype: string
  - name: max_sustainable_users
    dtype: int64
tags:
- benchmarking
- llama-cpp
- llm-inference
- local-llm
- homelab
- open-data
- tabular
---

# Dataset Card for Poor Paul's Benchmark Results

## Dataset Description

Poor Paul's Benchmark (PPB) is an open benchmarking framework for local AI inference on consumer, prosumer, and small-business hardware.

This dataset contains normalized, community-submitted benchmark results across models, quantizations, hardware, runtimes, and benchmark settings. Each row represents one benchmark result and is intended for open benchmarking, reproducibility, and downstream analysis.

Common use cases include:
- Comparing inference throughput across hardware
- Studying context-length scaling
- Comparing latency metrics such as TTFT and ITL
- Powering dashboards and derived leaderboards such as `poorpaul.dev`

Current benchmark runners represented in the dataset include:
- `llama-bench`
- `llama-server`

## Dataset Structure

Each row corresponds to one normalized benchmark result.

The schema is arranged so the most important fields appear first in the Hugging Face viewer:
- Model identity and quantization
- Hardware
- Benchmark settings
- Performance metrics
- Secondary system metadata
- Submission and provenance fields

Field groups include:
- **Model and benchmark identity:** `model`, `model_base`, `quant`, `runner_type`
- **Hardware:** `gpu_name`, `gpu_vram_gb`, `gpu_driver`, `backends`, `cpu_model`
- **Benchmark settings:** `n_ctx`, `n_batch`, `concurrent_users`
- **Performance metrics:** `throughput_tok_s`, `avg_ttft_ms`, `p50_ttft_ms`, `p99_ttft_ms`, `avg_itl_ms`, `p50_itl_ms`, `p99_itl_ms`, `avg_power_w`, `max_power_w`, `max_sustainable_users`
- **System metadata:** `os_system`, `os_release`, `os_machine`, `cpu_cores`, `ram_total_gb`
- **Submission and provenance:** `submitter`, `timestamp`, `submitted_at`, `schema_version`, `benchmark_version`, `submission_id`, `row_id`, `machine_fingerprint`, `run_fingerprint`, `result_fingerprint`, `source_file_sha256`

Some fields are runner-specific, so `null` values are expected.

The dataset currently uses a single split:
- `train`

## Dataset Creation

The source data comes from PPB runs executed locally by users on their own hardware.

Before upload, raw benchmark outputs are normalized into a flat tabular schema so they can be previewed on Hugging Face and easily consumed by pandas, DuckDB, spreadsheets, and downstream dashboards.

## Considerations for Using the Data

This dataset is an **append-only raw submission ledger**, not a final curated leaderboard.

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

## Additional Information

This dataset is intended to contain benchmark telemetry, not personal information. Contributors may optionally include a public display name through `submitter`.

License: `MIT`

Links:
- Project repository: `https://github.com/paulplee/poor-pauls-benchmark`
- Dataset repository: `https://huggingface.co/datasets/paulplee/ppb-results`