# Poor Paul's Benchmark (PPB)

Find the absolute limit of your local AI hardware without the enterprise budget.

AI-driven GPU and RAM prices got you down? It certain got me feeling a lot poorer. I can't turn back the clock to the "normal" times, but I can try to get myself more informed in how I can best use the little hardware that I can muster.

Poor Paul's Benchmark is an automated evaluation framework for local LLM inference. It uses a **pluggable runner architecture** — built-in runners wrap `llama.cpp`'s `llama-bench` (raw throughput) and `llama-server` (real-world UX latency), and new backends (vLLM, Stable Diffusion, …) can be added without touching core code.

My goal is to build the definitive public leaderboard for cost-effective AI setups, helping homelabbers (me), prosumers (me), and small businesses (and me) answer the question: _What is the most efficient hardware / model / settings for my AI workload?_

## The Problem

Evaluating local LLM performance across heterogeneous hardware (Apple Silicon, discrete consumer Nvidia GPUs, mixed rigs) is tedious. Finding the exact maximum context window (`n_ctx`) before your system hits the "VRAM Cliff" (crashing with an Out-of-Memory error or degrading into system swap) usually requires hours of manual trial and error. Furthermore, self-reported benchmarks on forums often lack crucial hardware context (driver versions, background tasks, exact quantization).

## The Solution

PPB automates the tedious parts of benchmarking so you can focus on studying the results and agonizing over what to sacrifice.

### Key Features

- **Pluggable Runner Architecture:** Benchmark backends are implemented as plugins inheriting from `BaseRunner`. Swap `llama-bench` (implemented) for `llama-server` (implemented), `vllm` (not yet implemented), or your own backend by setting `runner_type` in the sweep config.
- **Auto-Discover VRAM Limits:** Using a binary search algorithm, PPB automatically probes your hardware to find the exact maximum context size a specific model can handle before triggering an OOM error.
- **Declarative Parameter Sweeps:** Define your test matrices in a simple TOML file. PPB will automatically iterate through every combination of GGUF models, batch size, context length, and GPU layers.
- **Real-World UX Metrics:** The `llama-server` runner streams completions from real ShareGPT conversational prompts, measuring **Time-To-First-Token (TTFT)** and **Inter-Token Latency (ITL)** — the numbers that matter for interactive applications.
- **Integrated Model Downloader:** Native integration with Hugging Face Hub to download, cache, and symlink GGUF models directly via the CLI.
- **Automated Hardware Fingerprinting:** PPB automatically detects your OS, RAM, CPU architecture, and GPU details (via `pynvml` on Linux/Windows or `system_profiler` on macOS). Hardware profiles are embedded in every result record and can be viewed any time with `ppb hw-info`.
- **Stable Result Envelope:** Every JSONL record includes `runner_type`, `timestamp`, and `hardware` — so results from different runners or years apart remain comparable.

## Project Structure

```
ppb.py                 # CLI entry point (Typer app)
runners/
  __init__.py          # Runner registry (register_runner / get_runner)
  base.py              # BaseRunner ABC — contract for all backends
  llama_bench.py       # Built-in llama-bench runner (raw throughput)
  llama_server.py      # Built-in llama-server runner (TTFT / ITL latency)
datasets/
  __init__.py          # Dataset download & loading helpers
  sharegpt.py          # ShareGPT download (HF Hub) + prompt extraction
  data/                # Downloaded dataset cache (gitignored)
tests/
  conftest.py          # Shared fixtures + FakeRunner
  test_runners.py      # Runner ABC, registry, and LlamaBenchRunner tests
  test_llama_server.py # LlamaServerRunner, SSE parsing, dataset tests
  test_config.py       # SweepConfig, BenchCombo, _write_result tests
  test_orchestration.py # Sweep & auto-limit integration tests
suites/
  suite.example.toml   # Starter benchmark suite (copy → suites/my_gpu.toml)
  .gitignore           # Ignores user suite files, keeps examples
results/               # Benchmark output directory (gitignored)
  results.jsonl        # ← files land here automatically
```

## Installation

> **Note:** PPB is currently in active development.

Requires **Python ≥ 3.13**.

```bash
git clone https://github.com/paulplee/poor-pauls-benchmark.git
cd poor-pauls-benchmark
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

> **NVIDIA GPU detection:** `pynvml` is included in the dependencies and will be installed automatically. If no NVIDIA hardware is present it is simply unused. On systems without `pynvml`, PPB falls back to parsing `nvidia-smi` output.

You must also have `llama-bench` and/or `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) compiled and accessible in your system PATH (or point to them with `PPB_LLAMA_BENCH` / `PPB_LLAMA_SERVER`).

> **Conversational dataset:** The `llama-server` runner uses real conversational prompts from the [ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) (~700 MB) by default. It is downloaded automatically on the first `llama-server` run, or you can pre-fetch it with `python ppb.py download-dataset`. You can also point to any HF-hosted dataset with `--repo` and `--filename` — see [§3](#3-run-a-parameter-sweep) for details.

### Running Tests

```bash
pip install pytest   # or: uv sync --group dev
python -m pytest tests/ -v
```

## Usage Guide

### 1. Download a Model

* Note: This is a only a convenience command. If you already have models sitting somewhere, just specify the `model_path` when you run the benchmark.

Easily fetch GGUF files from Hugging Face:

```bash
# Download all the Q4 variants of the unsloth/Qwen3.5-0.8B-GGUF series
python ppb.py download unsloth/Qwen3.5-0.8B-GGUF "*Q4*.gguf"
```

### 2. Find Your VRAM Limit (`auto-limit`)

PPB can automatically discover the maximum context window your hardware supports before running out of memory, using a binary search algorithm.

**CLI-only mode:**

```bash
python ppb.py auto-limit --model ./models/Qwen3.5-0.8B-Q4_K_M.gguf
# Or a whole directory / glob pattern:
python ppb.py auto-limit --model ./models/
python ppb.py auto-limit --model "./models/*Q4*.gguf"
```

**TOML-driven mode** (reads the `[auto-limit]` section from a suite file):

```bash
python ppb.py auto-limit suites/my_gpu.toml
```

**Full set of options:**

```bash
python ppb.py auto-limit \
  --model ~/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --min-ctx 2048 \
  --max-ctx 131072 \
  --tolerance 1024 \
  --runner llama-bench
```

When driven by a TOML file, CLI flags override TOML values.

#### How it works

1. Instantiates the requested runner via the registry.
2. Sets `lo = min_ctx`, `hi = max_ctx`.
3. Probes the midpoint `mid = (lo + hi) / 2` by calling `runner.probe_ctx()` (for llama-bench this runs with `-n 0`, allocation-only).
4. **Pass** (no OOM) → `lo = mid + 1`, records `mid` as the last known-good value.
5. **Fail** (non-zero exit code, or output contains `"out of memory"` / `"bad alloc"` / etc.) → `hi = mid - 1`.
6. Stops when `hi - lo < tolerance` and prints the largest safe context size.

Sample output (each iteration shows per-probe duration):

```
  iter  1: n_ctx= 66,560  ✓ pass  → window [66,561, 131,071]  (2.3s)
  iter  2: n_ctx= 98,815  ✗ OOM   → window [66,561, 98,814]   (0.8s)
  iter  3: n_ctx= 82,687  ✓ pass  → window [82,688, 98,814]   (3.1s)
  ...

✓ Maximum safe context for Llama-3-8B-Instruct.Q4_K_M.gguf

    90,111 tokens
```

#### auto-limit options

| Flag          | Default        | Description                                                  |
| ------------- | -------------- | ------------------------------------------------------------ |
| `CONFIG`      | _(optional)_   | Path to a TOML suite file containing an `[auto-limit]` section. |
| `--model`     | _(from TOML)_  | Path to a GGUF file, directory, or glob pattern (overrides TOML). |
| `--min-ctx`   | `2048`         | Lower bound for the binary search.                           |
| `--max-ctx`   | `131072`       | Upper bound for the binary search.                           |
| `--tolerance` | `1024`         | Stop searching when the remaining window is this narrow.     |
| `--runner`    | `llama-bench`  | Runner backend to use for probing.                           |

#### auto-limit TOML section

```toml
model_path  = "~/models/Qwen3.5-0.8B-Q4_K_M.gguf"   # single file, dir, or glob
runner_type = "llama-bench"  # optional (shared across sections)

[auto-limit]
min_ctx    = 2048       # optional (default: 2048)
max_ctx    = 131072     # optional (default: 131072)
tolerance  = 1024       # optional (default: 1024)
```

When `model_path` points to a directory or glob pattern, auto-limit probes **each matched model independently** and reports per-model results.

### 3. Run a Parameter Sweep

Create a `suite.toml` file to define your test matrix (a starter config is included at [`suites/suite.example.toml`](suites/suite.example.toml)):

```toml
model_path = "./models/Qwen3.5-0.8B-Q4_K_M.gguf"

[sweep]
n_ctx    = [8192, 16384, 32768]
n_batch  = [512, 1024]
```

**TOML-driven:**

```bash
python ppb.py sweep suites/my_gpu.toml
python ppb.py sweep suites/my_gpu.toml --results results/my_results.jsonl
```

**Pure-CLI mode** (no TOML file needed):

```bash
python ppb.py sweep \
  --model ./models/Llama-3-8B-Q4_K_M.gguf \
  --n-ctx 8192,16384,32768 \
  --n-batch 512,1024
```

**CLI flags override TOML values** when both are provided:

```bash
python ppb.py sweep suites/my_gpu.toml --n-ctx 4096,8192  # only test these two contexts
```

Sample sweep output (per-combo duration shown):

```
  ✓ [1/6] model.gguf ctx=8192 batch=512   42.0 tok/s  (12.4s)
  ✓ [2/6] model.gguf ctx=8192 batch=1024  38.7 tok/s  (11.9s)
  ✓ [3/6] model.gguf ctx=16384 batch=512  31.2 tok/s  (14.1s)
  ...
```

#### Sweep config reference

Shared fields (`model_path`, `runner_type`, `runner_params`) can live at the **top level** and are inherited by both `[auto-limit]` and `[sweep]`. Section-level values override the root. Sweep-specific fields:

| Key             | Type             | Required | Default         | Description |
| --------------- | ---------------- | -------- | --------------- | ----------- |
| `model_path`    | string (path)    | ✅       |                 | Path to a `.gguf` file, directory of `.gguf` files, or a glob pattern. Relative paths are resolved from the working directory. |
| `n_ctx`         | list of integers | ✅       |                 | Context sizes to test. Controls KV cache depth — larger values stress longer-context workloads, e.g. `[8192, 16384, 32768]`. |
| `n_batch`       | list of integers | ✅       |                 | Batch sizes for prompt processing, e.g. `[512, 1024]`. Used by `llama-bench`; accepted but unused by `llama-server` (it manages its own batching). |
| `runner_type`   | string           |          | `"llama-bench"` | Which benchmark backend to use. See [Runner Plugins](#runner-plugins) below. |
| `runner_params` | table            |          | `{}`            | Runner-specific overrides; see `[sweep.runner_params]` below. |

PPB computes the full Cartesian product of `model_paths × n_ctx × n_batch`, so 2 models × 3 contexts × 2 batches = 12 combinations.

##### Example: exhaustive sweep across multiple quantisations

```toml
[sweep]
model_path = "~/models/Llama-3*Q4*.gguf"   # all Q4 quants in the folder
n_ctx   = [8192, 16384, 32768]
n_batch = [256, 512, 1024]
```

This tests every matched model at all 9 `(n_ctx, n_batch)` combinations.

##### Example: using a custom runner and params

```toml
[sweep]
runner_type = "llama-bench"
model_path  = "~/models/"
n_ctx       = [8192]
n_batch     = [512]

[sweep.runner_params]
llama_bench_cmd = "/opt/llama.cpp/build/bin/llama-bench"
```

#### Choosing a runner

PPB ships with two runners — pick the one that matches your evaluation goal:

| Runner | Measures | Best for |
| --- | --- | --- |
| `llama-bench` *(default)* | Raw throughput (tok/s) | Finding peak hardware performance, comparing quants |
| `llama-server` | **TTFT** and **ITL** latency | UX-relevant benchmarks for interactive / chat applications |

Both runners use the same `ppb sweep` command — just set `runner_type` in your TOML.

##### Setting up `llama-server` sweeps

**1. Ensure `llama-server` is available:**

```bash
# On your PATH after building llama.cpp
llama-server --version

# Or point to it explicitly
export PPB_LLAMA_SERVER=/path/to/llama-server
```

**2. (Optional) Pre-download the conversational dataset:**

The runner sends real human prompts from a conversational dataset (default: [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered), ~700 MB). It auto-downloads on first run, but you can pre-fetch it:

```bash
python ppb.py download-dataset

# Or use a different HF dataset:
python ppb.py download-dataset --repo "my-org/my-dataset" --filename "convos.json"
```

**3. Create a TOML suite:**

```toml
# suites/my_server.toml
model_path  = "./models/Llama-3-8B-Instruct.Q4_K_M.gguf"
runner_type = "llama-server"

[sweep]
n_ctx   = [8192, 16384]
n_batch = [512]          # required by PPB but unused by llama-server

[sweep.runner_params]
num_prompts = 10         # prompts to send per run (default: 10)
n_predict   = 256        # max tokens per prompt (default: 256)
shuffle     = true       # randomise prompt order across runs
```

**4. Run:**

```bash
python ppb.py sweep suites/my_server.toml
```

The JSONL output includes per-run TTFT and ITL statistics:

```
avg_ttft_ms: 142.5    p50_ttft_ms: 138.2    p99_ttft_ms: 210.7
avg_itl_ms:   12.3    p50_itl_ms:   11.8    p99_itl_ms:   18.4
```

##### `llama-server` runner_params reference

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `llama_server_cmd` | string | `llama-server` | Path or name of the binary. Falls back to `PPB_LLAMA_SERVER` env → `$PATH`. |
| `num_prompts` | int | `10` | Number of prompts to send per run. |
| `n_predict` | int | `256` | Maximum tokens to generate per prompt. |
| `health_timeout` | int/float | `120` | Seconds to wait for `/health` to return 200. |
| `dataset_dir` | string | `datasets/data/` | Directory for cached dataset files. |
| `dataset_repo` | string | `anon8231489123/ShareGPT_Vicuna_unfiltered` | HF Hub repository ID for the prompt dataset. |
| `dataset_filename` | string | `ShareGPT_V3_unfiltered_cleaned_split.json` | File to download from the repository. |
| `shuffle` | bool | `false` | Randomise prompt order so repeated runs use different workloads. |
| `seed` | int | _(none)_ | RNG seed for reproducible shuffling. |

##### Sample `llama-server` results

The `results` object in the JSONL envelope contains:

```json
{
  "num_prompts_attempted": 10,
  "num_prompts_succeeded": 10,
  "n_predict": 256,
  "total_tokens": 2048,
  "total_duration_s": 34.567,
  "throughput_tok_s": 59.26,
  "avg_ttft_ms": 142.5,
  "p50_ttft_ms": 138.2,
  "p99_ttft_ms": 210.7,
  "avg_itl_ms": 12.3,
  "p50_itl_ms": 11.8,
  "p99_itl_ms": 18.4
}
```

##### Results format

Each line written to the JSONL file is a self-contained record with a **stable envelope** that stays consistent across runners and PPB versions:

```json
{
  "timestamp": "2026-03-04T10:00:00+00:00",
  "runner_type": "llama-bench",
  "model_path": "/abs/path/to/model.gguf",
  "n_ctx": 8192,
  "n_batch": 512,
  "hardware": {
    "os": { "system": "Linux", "release": "6.8.0", "machine": "x86_64" },
    "cpu": { "model": "AMD Ryzen 9 7950X", "cores": "32" },
    "ram": { "total_gb": 63.9 },
    "gpus": [
      {
        "name": "NVIDIA GeForce RTX 4090",
        "driver": "560.35.03",
        "cuda_version": "12.4",
        "compute_capability": "8.9",
        "power_limit_w": 450,
        "pcie_gen": 4,
        "pcie_width": 16,
        "vram_total_gb": 24.0
      }
    ],
    "runtime": {
      "python_version": "3.13.2",
      "llama_bench": "version: b5063 (58ab80c3)"
    }
  },
  "results": [
    /* raw runner-specific JSON output */
  ]
}
```

The `runner_type` field ensures results from different backends can coexist in the same file and be compared meaningfully.

#### Auto-generated results filenames

When no `--results` flag is passed and the TOML file has no `results` key, PPB auto-generates a filename from the config name and current UTC time:

```
suite_20250714_1830.jsonl
```

Format: `<config_stem>_YYYYMMDD_HHMM.jsonl`.

You can set a fixed output path in the TOML root:

```toml
results = "my_results.jsonl"

[sweep]
# ...
```

The `--results` CLI flag always takes priority over the TOML value.

##### Environment overrides

| Variable            | Default           | Description                                       |
| ------------------- | ----------------- | ------------------------------------------------- |
| `PPB_LLAMA_BENCH`   | `llama-bench`     | Path or name of the `llama-bench` binary.         |
| `PPB_LLAMA_SERVER`  | `llama-server`    | Path or name of the `llama-server` binary.        |
| `PPB_MODELS_DIR`    | `./models`        | Default directory used by the `download` command.  |

### 4. Run the Full Suite (`all`)

The `all` command combines **auto-limit** and **sweep** into a single invocation:

1. **Phase 1 — auto-limit:** Discovers the max safe context window for each model.
2. **Phase 2 — sweep:** Runs the parameter sweep, automatically skipping any combo whose `n_ctx` exceeds the per-model limit found in Phase 1.

```bash
python ppb.py all suites/my_gpu.toml
python ppb.py all suites/my_gpu.toml --results results/my_run.jsonl
```

If the TOML has no `[auto-limit]` section, Phase 1 is skipped and the sweep runs unmodified.

#### Example suite TOML

```toml
# Shared — declared once, inherited by both sections
model_path  = "~/models/"                    # file, dir, or glob — probes each model
results     = "results/my_benchmark.jsonl"   # optional

[auto-limit]
min_ctx    = 2048
max_ctx    = 131072
tolerance  = 1024

[sweep]
n_ctx      = [8192, 16384, 32768, 65536, 131072]
n_batch    = [512, 1024]
```

When multiple models are matched, Phase 1 probes each one independently.
The per-model caps are then passed to Phase 2, which skips combos that exceed each model's limit — no manual editing required.

### 5. View Your Hardware Profile

Quickly check what PPB detects about your system:

```bash
uv run ppb.py hw-info
```

Sample output:

```
  Hardware Profile
    OS          : Linux 6.8.0  (x86_64)
    CPU         : AMD Ryzen 9 7950X  (32 cores)
    RAM         : 63.9 GB
    GPU [0]     : NVIDIA GeForce RTX 5090  31.8 GB VRAM  sm_120  CUDA 13.0  driver 580.126.09  600 W TDP  PCIe 5.0 x16
    Python      : 3.13.2
    llama-bench : version: b5063 (58ab80c3)
```

**Fields collected per GPU (where available):**

| Field | Source | Why it matters |
|---|---|---|
| `name` | pynvml / nvidia-smi / system_profiler | Model identification |
| `driver` | pynvml / nvidia-smi | Affects performance and supported features |
| `cuda_version` | pynvml | Max CUDA version; determines available kernels (Flash Attn 2, etc.) |
| `compute_capability` | pynvml / nvidia-smi | SM version (e.g. `8.9` = RTX 4090, `12.0` = RTX 5090) |
| `power_limit_w` | pynvml | TDP cap; enables tokens-per-watt comparisons |
| `pcie_gen` / `pcie_width` | pynvml | PCIe bandwidth ceiling for large model loading |
| `vram_total_gb` | pynvml / nvidia-smi | VRAM capacity |
| `metal_version` *(macOS)* | system_profiler | Metal API generation |
| `gpu_cores` *(macOS)* | system_profiler | Apple Silicon GPU core count |

This same profile is automatically included in every benchmark record written to `results.jsonl`.

## Runner Plugins

PPB uses a **pluggable runner architecture** so new benchmark backends can be added without modifying core code.

### Built-in runners

| `runner_type`   | Module                   | Description |
| --------------- | ------------------------ | ----------- |
| `llama-bench`   | `runners/llama_bench.py`  | Default. Wraps llama.cpp's `llama-bench` CLI via subprocess. Measures raw throughput (tok/s). Supports OOM probing for `auto-limit`. |
| `llama-server`  | `runners/llama_server.py` | Starts `llama-server` as a subprocess, streams real ShareGPT conversational prompts via SSE, and records **TTFT** and **ITL** latency metrics. Supports `auto-limit`. |

### Creating a custom runner

1. Create a new file (e.g. `runners/my_runner.py`).
2. Subclass `BaseRunner` from `runners.base` and implement `setup()`, `run()`, and `teardown()`.
3. Optionally override `probe_ctx()` if your backend supports OOM probing.
4. Register it in `runners/__init__.py`:

```python
from .my_runner import MyRunner
register_runner("my-runner", MyRunner)
```

5. Use it in your sweep config:

```toml
[sweep]
runner_type = "my-runner"
model_path  = "~/models/model.gguf"
n_ctx       = [8192]
n_batch     = [512]

[sweep.runner_params]
custom_option = "value"
```

### Runner contract

| Method | Required | Description |
| --- | --- | --- |
| `setup(runner_params)` | ✅ | Called once before the sweep. Receives `[sweep.runner_params]` from TOML. |
| `run(config) → dict \| None` | ✅ | Execute one benchmark. `config` always has `"model_path"`. Return `{"results": ...}` or `None` on failure. Must NOT write files — the orchestrator handles JSONL output. |
| `teardown()` | ✅ | Called once after the sweep (guaranteed via `try/finally`). |
| `probe_ctx(model_path, n_ctx) → bool` | Optional | Override to support `auto-limit`. Default raises `NotImplementedError`. |

## Contributing to the Leaderboard

We are crowdsourcing a definitive database of Tokens-per-Second and Tokens-per-Watt across consumer hardware.

To add your machine to the public database:

1. Run a benchmark sweep: `python ppb.py all suites/my_gpu.toml`.
2. Open an Issue in this repository titled "Benchmark Submission: [Your Hardware]".
3. Attach or paste the contents of your results file from the `results/` directory.

The public leaderboard is hosted at: [poorpaul.dev](https://poorpaul.dev)

## About the Maintainers

Poor Paul's Benchmark is an open-source project maintained by the team at [Ximplar](https://ximplar.com).

While PPB is built for the community to test and stretch consumer hardware to its limits, Ximplar specializes in taking those insights and deploying cost-effective, high-ROI AI models for enterprise environments. If you need help architecting your company's local AI infrastructure, reach out to us.
