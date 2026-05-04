---
license: mit
language:
  - en
tags:
  - benchmarking
  - llama-cpp
  - llm-inference
  - local-llm
  - gpu
  - quantization
task_categories:
  - tabular-regression
  - tabular-classification
pretty_name: Poor Paul's Benchmark Results
---

# Poor Paul's Benchmark Results

Community-submitted LLM inference benchmark data from real consumer, prosumer, and small-business hardware.

Each row is one normalized benchmark result: a specific model × quantization × hardware × configuration combination with measured performance metrics.

**Use this to:**
- Compare throughput, latency, and power efficiency across GPU models and quantizations
- Study how concurrency and context length scale on different hardware
- Build leaderboards, dashboards, and data-driven GPU purchasing decisions
- Query via AI: any MCP client connected to `mcp.poorpaul.dev` can answer questions from this data directly

---

## Files

| File | Description |
| ---- | ----------- |
| `data/results_*.jsonl` | Raw benchmark submissions, one file per run, appended continuously |
| `llms.txt` | Machine-readable summary for LLM context injection |

---

## Quick start

```python
import pandas as pd
from datasets import load_dataset

# Stream all rows (recommended for large queries)
ds = load_dataset("paulplee/ppb-results", streaming=True, split="train")
df = pd.DataFrame(iter(ds))

# Filter to throughput rows for a specific GPU
tput = df[
    (df["gpu_name"] == "NVIDIA GeForce RTX 5090") &
    (df["runner_type"] == "llama-bench")
][["model_base", "quant", "n_ctx", "throughput_tok_s"]]
```

Or via DuckDB directly from HuggingFace:

```sql
SELECT gpu_name, model_base, quant, concurrent_users,
       AVG(throughput_tok_s) AS mean_tok_s,
       COUNT(*)              AS n
FROM   read_parquet('hf://datasets/paulplee/ppb-results/data/*.jsonl')
WHERE  runner_type = 'llama-bench'
GROUP  BY ALL
ORDER  BY mean_tok_s DESC;
```

---

## Schema (v0.9.0)

All columns are present on every row. Fields that do not apply to a given runner are `null`.

### Model identity

| Column | Type | Description |
| ------ | ---- | ----------- |
| `run_type` | string | `quantitative`, `qualitative`, or `all` |
| `model` | string | Full model path (e.g. `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf`) |
| `model_base` | string | Base model name without quant suffix (e.g. `Qwen3.5-9B`) |
| `quant` | string | Quantization format (e.g. `Q4_K_M`, `Q8_0`, `BF16`) |
| `model_org` | string\|null | HuggingFace organisation (e.g. `unsloth`); null for local paths |
| `model_repo` | string\|null | Full HF `org/repo` string; null for local paths |
| `runner_type` | string | Benchmark backend: `llama-bench`, `llama-server`, or `llama-server-loadtest` |

### LLM engine

| Column | Type | Description |
| ------ | ---- | ----------- |
| `llm_engine_name` | string\|null | Inference engine (e.g. `llama.cpp`) |
| `llm_engine_version` | string\|null | Engine version with build hash (e.g. `b5063 (58ab80c3)`) |

### Hardware

| Column | Type | Description |
| ------ | ---- | ----------- |
| `gpu_name` | string\|null | Primary GPU model name |
| `gpu_vram_gb` | float\|null | Primary GPU VRAM in GB |
| `gpu_driver` | string\|null | GPU driver version |
| `gpu_count` | int | Number of GPUs used |
| `gpu_names` | string\|null | Comma-joined list of all GPU names (multi-GPU runs) |
| `gpu_total_vram_gb` | float\|null | Total VRAM across all GPUs |
| `unified_memory` | bool\|null | `true` for Apple Silicon — GPU and CPU share the same memory pool |
| `gpu_compute_capability` | string\|null | CUDA compute capability (e.g. `"9.0"` for Blackwell); null for non-CUDA |
| `gpu_pcie_gen` | int\|null | PCIe generation (e.g. `5`); null for unified-memory platforms |
| `gpu_pcie_width` | int\|null | PCIe link width in lanes (e.g. `16`); null for unified-memory platforms |
| `gpu_power_limit_w` | float\|null | Configured TDP limit in Watts (from NVML); null for non-NVIDIA |
| `backends` | string\|null | Compute backend with version (e.g. `CUDA 13.0`, `Metal`, `CPU`) |
| `cpu_model` | string\|null | CPU model name |

### Benchmark configuration

| Column | Type | Description |
| ------ | ---- | ----------- |
| `n_ctx` | int\|null | Context window size in tokens |
| `n_batch` | int\|null | Batch size for prompt processing |
| `split_mode` | string\|null | Multi-GPU split strategy (`layer`, `row`, `none`); null for single-GPU |
| `tensor_split` | string\|null | Per-GPU VRAM weight string (e.g. `"1,1"`); null for single-GPU |
| `concurrent_users` | int\|null | Number of simulated parallel users. For `llama-server-loadtest`, **each row is one concurrency level** from the measured curve. |

### Workload

| Column | Type | Description |
| ------ | ---- | ----------- |
| `task_type` | string\|null | Workload category (e.g. `text-generation`, `context-rot-niah`) |
| `prompt_dataset` | string\|null | Prompt source (e.g. `sharegpt-v3`); null for `llama-bench` |
| `num_prompts` | int\|null | Prompts sent per run; null for `llama-bench` |
| `n_predict` | int\|null | Max tokens generated per prompt; null for `llama-bench` |

### Performance — throughput

| Column | Type | Description |
| ------ | ---- | ----------- |
| `throughput_tok_s` | float\|null | Tokens per second (primary throughput metric) |
| `vram_cliff_tokens` | int\|null | Largest `n_ctx` that loaded without OOM during pre-flight discovery |

### Performance — power

| Column | Type | Description |
| ------ | ---- | ----------- |
| `avg_power_w` | float\|null | Average GPU power draw in Watts |
| `max_power_w` | float\|null | Peak GPU power draw in Watts |

### Performance — thermal

| Column | Type | Description |
| ------ | ---- | ----------- |
| `avg_gpu_temp_c` | float\|null | Average GPU temperature (°C) |
| `max_gpu_temp_c` | float\|null | Peak GPU temperature (°C) |
| `avg_cpu_temp_c` | float\|null | Average CPU temperature (°C) |
| `max_cpu_temp_c` | float\|null | Peak CPU temperature (°C) |
| `avg_fan_speed_rpm` | float\|null | Average fan speed (RPM) |
| `max_fan_speed_rpm` | float\|null | Peak fan speed (RPM) |

### Performance — user experience (server runners only)

| Column | Type | Description |
| ------ | ---- | ----------- |
| `avg_ttft_ms` | float\|null | Average Time-To-First-Token (ms) |
| `p50_ttft_ms` | float\|null | Median TTFT (ms) |
| `p99_ttft_ms` | float\|null | 99th-percentile TTFT (ms) |
| `avg_itl_ms` | float\|null | Average Inter-Token Latency (ms) |
| `p50_itl_ms` | float\|null | Median ITL (ms) |
| `p99_itl_ms` | float\|null | 99th-percentile ITL (ms) |

### Qualitative evaluation

Populated when `run_type` is `qualitative` or `all`. All null for pure `quantitative` runs.

| Column | Type | Description |
| ------ | ---- | ----------- |
| `context_rot_score` | float\|null | Mean accuracy across all (length × depth) long-context recall cases |
| `context_rot_accuracy_by_length` | string\|null | JSON `{haystack_length: accuracy}` map |
| `context_rot_accuracy_by_depth` | string\|null | JSON `{depth_pct: accuracy}` map |
| `tool_selection_accuracy` | float\|null | Fraction of cases with correct tool name selected |
| `parameter_accuracy` | float\|null | Fraction of cases with all required arguments matching ground truth |
| `parameter_hallucination_rate` | float\|null | Fraction of cases with invented arguments not in schema |
| `parse_success_rate` | float\|null | Fraction of cases with parseable tool-call JSON |
| `overall_tool_accuracy` | float\|null | Geometric mean of tool selection × parameter accuracy |
| `knowledge_accuracy_mean` | float\|null | Mean fraction of factual claims judged consistent with common knowledge |
| `knowledge_accuracy_std` | float\|null | Standard deviation of per-prompt knowledge-accuracy scores |
| `answer_relevancy_mean` | float\|null | Mean judge-rated response relevancy (0–1) |
| `coherence_mean` | float\|null | Mean judge-rated coherence (0–1) |
| `quality_composite_score` | float\|null | Mean of knowledge accuracy, relevancy, and coherence |
| `memory_accuracy` | float\|null | LongMemEval recall accuracy (0–1); null when MT-Bench mode was used |
| `mt_bench_score` | float\|null | MT-Bench score (1–10 scale); null when LongMemEval mode was used |
| `cases_evaluated` | int\|null | Number of evaluation cases that completed |
| `cases_skipped_context` | int\|null | Cases skipped because context exceeded `vram_cliff_tokens` |

### Blob columns

| Column | Type | Description |
| ------ | ---- | ----------- |
| `qualitative` | string\|null | Full qualitative result payload as JSON string |
| `quantitative` | string\|null | Full quantitative result payload as JSON string |
| `meta` | string\|null | Reproducibility hints (e.g. `quality_prompts_cache_hash`) as JSON string |

### OS / system context

| Column | Type | Description |
| ------ | ---- | ----------- |
| `os_system` | string\|null | OS family: `Linux`, `Darwin`, `Windows` |
| `os_release` | string\|null | Kernel / OS release string |
| `os_machine` | string\|null | CPU architecture (e.g. `x86_64`, `arm64`) |
| `os_distro` | string\|null | Distribution name (e.g. `Ubuntu`, `macOS`) |
| `os_distro_version` | string\|null | Distribution version (e.g. `24.04`, `15.5`) |
| `cpu_cores` | int\|null | Number of logical CPU cores |
| `ram_total_gb` | float\|null | Total system RAM in GB |

### Submission metadata

| Column | Type | Description |
| ------ | ---- | ----------- |
| `submitter` | string\|null | Optional public display name of the contributor |
| `timestamp` | string\|null | ISO 8601 UTC time the benchmark run produced the row |
| `submitted_at` | string\|null | ISO 8601 UTC time the row was uploaded |

### Provenance and deduplication

| Column | Type | Description |
| ------ | ---- | ----------- |
| `schema_version` | string | Schema version at time of flattening (`0.9.0`) |
| `benchmark_version` | string | PPB software version that produced the row |
| `suite_run_id` | string\|null | UUID shared by all rows from the same `ppb` invocation |
| `submission_id` | string\|null | UUID assigned during upload |
| `row_id` | string | UUID uniquely identifying this row |
| `machine_fingerprint` | string | SHA-256 of hardware profile fields (anonymous machine identity) |
| `run_fingerprint` | string | SHA-256 of benchmark configuration + machine fingerprint |
| `result_fingerprint` | string | SHA-256 of run identity + measured metrics — uniquely identifies one result |
| `source_file_sha256` | string\|null | SHA-256 of the source JSONL file |

### Extensibility

| Column | Type | Description |
| ------ | ---- | ----------- |
| `tags` | string\|null | Free-form JSON string for arbitrary metadata from the suite TOML |

---

## Null value guide

Many columns are runner-specific. Expected nulls by runner type:

| Column group | `llama-bench` | `llama-server` | `llama-server-loadtest` |
| --- | --- | --- | --- |
| TTFT / ITL metrics | null | populated | populated |
| `prompt_dataset`, `num_prompts`, `n_predict` | null | populated | populated |
| `concurrent_users` | null | populated | populated (one row per level) |
| `gpu_pcie_gen`, `gpu_pcie_width` | null on Apple Silicon | null on Apple Silicon | null on Apple Silicon |
| `unified_memory` | null on NVIDIA | null on NVIDIA | null on NVIDIA |
| Qualitative columns | null | null | null |

Qualitative columns are populated only when `run_type` is `qualitative` or `all`.

---

## Deduplication

Use fingerprints to control for duplicates in analysis:

```python
# Exact duplicate rows (same result, same machine, same run)
df.drop_duplicates(subset=["result_fingerprint"], inplace=True)

# Latest run per (gpu, model, quant, n_ctx, concurrent_users) config
latest = (
    df.sort_values("timestamp")
      .drop_duplicates(subset=["run_fingerprint"], keep="last")
)
```

---

## Ecosystem

| | |
| --- | --- |
| **Benchmark tool** | [poor-pauls-benchmark](https://github.com/paulplee/poor-pauls-benchmark) — run benchmarks and contribute results |
| **MCP server** | [ppb-mcp](https://github.com/paulplee/ppb-mcp) — lets any MCP-compatible LLM client query this dataset directly |
| **Analytics** | [poorpaul.dev/insights](https://poorpaul.dev/insights) — leaderboard and visual analysis |

Connect any MCP client to `https://mcp.poorpaul.dev/mcp` to query this data conversationally.

---

## Contributing results

1. Clone [poor-pauls-benchmark](https://github.com/paulplee/poor-pauls-benchmark)
2. Configure `suites/my_gpu.toml` with your hardware and models
3. Run: `uv run ppb.py all suites/my_gpu.toml`
4. Results are pushed here automatically — no PR required

No hardware contribution is too small. Every GPU tier that's missing from this dataset is a blind spot for the community.

---

## License

Dataset content: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — contributions are attributed to their submitters.

Tooling: MIT — see the [benchmark repository](https://github.com/paulplee/poor-pauls-benchmark).

Third-party evaluation data included in rows:
- BFCL v4 evaluation cases © UC Berkeley, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- MT-Bench questions © LMSYS, [MIT licence](https://github.com/lm-sys/FastChat/blob/main/LICENSE)
- ShareGPT prompts under their [original licence](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
