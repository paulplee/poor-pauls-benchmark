# Poor Paul's Benchmark (PPB)

Find the absolute limit of your local AI hardware without the enterprise budget.

AI-driven GPU and RAM prices got you down? It certain got me feeling a lot poorer. I can't turn back the clock and go back to normal prices. About the only thing I can do is to get myself more informed in how I can best use the little hardware that I can muster.

Poor Paul's Benchmark is an automated evaluation framework for local LLM inference. It ships with a **pluggable runner architecture** — the built-in runner wraps `llama.cpp`'s `llama-bench`, and new backends (llama-server, vLLM, Stable Diffusion, …) can be added without touching core code.

My goal is to build the definitive public leaderboard for cost-effective AI setups, helping homelabbers, prosumers, and small businesses answer the question: _What is the most efficient hardware for my AI workload?_

## The Problem

Evaluating local LLM performance across heterogeneous hardware (Apple Silicon, discrete consumer Nvidia GPUs, mixed rigs) is tedious. Finding the exact maximum context window (`n_ctx`) before your system hits the "VRAM Cliff" (crashing with an Out-of-Memory error or degrading into system swap) usually requires hours of manual trial and error. Furthermore, self-reported benchmarks on forums often lack crucial hardware context (driver versions, background tasks, exact quantization).

## The Solution

PPB automates the tedious parts of benchmarking so you can focus on the data.

### Key Features

- **Pluggable Runner Architecture:** Benchmark backends are implemented as plugins inheriting from `BaseRunner`. Swap `llama-bench` for `llama-server`, `vllm`, or your own backend by setting `runner_type` in the sweep config.
- **Declarative Parameter Sweeps:** Define your test matrices in a simple TOML file. PPB will automatically iterate through every combination of batch size, context length, and GPU layers.
- **Auto-Discover VRAM Limits:** Using a binary search algorithm, PPB automatically probes your hardware to find the exact maximum context size a specific model can handle before triggering an OOM error.
- **Integrated Model Downloader:** Native integration with Hugging Face Hub to download, cache, and symlink GGUF models directly via the CLI.
- **Automated Hardware Fingerprinting:** PPB automatically detects your OS, RAM, CPU architecture, and GPU details (via `pynvml` on Linux/Windows or `system_profiler` on macOS). Hardware profiles are embedded in every result record and can be viewed any time with `ppb hw-info`.
- **Stable Result Envelope:** Every JSONL record includes `runner_type`, `timestamp`, and `hardware` — so results from different runners or years apart remain comparable.

## Project Structure

```
ppb.py                 # CLI entry point (Typer app)
runners/
  __init__.py          # Runner registry (register_runner / get_runner)
  base.py              # BaseRunner ABC — contract for all backends
  llama_bench.py       # Built-in llama-bench runner
tests/
  conftest.py          # Shared fixtures + FakeRunner
  test_runners.py      # Runner ABC, registry, and LlamaBenchRunner tests
  test_config.py       # SweepConfig, BenchCombo, _write_result tests
  test_orchestration.py # Sweep & auto-limit integration tests
sweep.example.toml     # Starter sweep config (copy → sweep.toml)
results.jsonl          # Benchmark output (JSONL)
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

You must also have the `llama-bench` binary compiled and accessible in your system PATH (or point to it with `PPB_LLAMA_BENCH`).

### Running Tests

```bash
pip install pytest   # or: uv sync --group dev
python -m pytest tests/ -v
```

## Usage Guide

### 1. Download a Model

Easily fetch GGUF files from Hugging Face:

```bash
python ppb.py download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF "*Q4_K_M.gguf"
```

### 2. Run a Parameter Sweep

Create a `sweep.toml` file to define your test matrix (a starter config is included at [`sweep.example.toml`](sweep.example.toml)):

```toml
[sweep]
model_path = "./models/Llama-3-8B-Instruct.Q4_K_M.gguf"
n_ctx    = [8192, 16384, 32768]
n_batch  = [512, 1024]
```

Execute the sweep:

```bash
python ppb.py sweep sweep.toml
# optionally pick a custom output file
python ppb.py sweep sweep.toml --results my_results.jsonl
```

#### Sweep config reference

All fields live inside a single `[sweep]` section.

| Key             | Type             | Required | Default         | Description                                                                                                                                                           |
| --------------- | ---------------- | -------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_path`    | string (path)    | ✅       |                 | Path to a `.gguf` file, directory of `.gguf` files, or a glob pattern. Relative paths are resolved from the working directory.                                        |
| `n_ctx`         | list of integers | ✅       |                 | Prompt token counts to test (passed as `-p` to `llama-bench`). Controls KV cache depth — larger values stress longer-context throughput. e.g. `[8192, 16384, 32768]`. |
| `n_batch`       | list of integers | ✅       |                 | Batch sizes (passed as `-b` to `llama-bench`) for prompt-processing throughput tests, e.g. `[512, 1024]`.                                                             |
| `runner_type`   | string           |          | `"llama-bench"` | Which benchmark backend to use. See [Runner Plugins](#runner-plugins) below.                                                                                          |
| `runner_params` | table            |          | `{}`            | Runner-specific overrides; see `[sweep.runner_params]` below.                                                                                                         |

PPB computes the full Cartesian product of `model_paths × n_ctx × n_batch`, so 2 models × 3 contexts × 2 batches = 12 combinations.

##### Example: exhaustive sweep across multiple quantisations

```toml
[sweep]
model_path = "~/models/Llama-3*Q4*.gguf"   # all Q4 quants in the folder
n_ctx   = [8192, 16384, 32768]
n_batch = [256, 512, 1024]
```

This tests every matched model at all 9 `(n_ctx, n_batch)` combinations and appends every result as a JSON line to `results.jsonl`.

##### Example: using a custom runner and params

```toml
[sweep]
runner_type = "llama-bench"           # default; swap for future runners
model_path  = "~/models/"
n_ctx       = [8192]
n_batch     = [512]

[sweep.runner_params]
llama_bench_cmd = "/opt/llama.cpp/build/bin/llama-bench"
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

##### Environment overrides

| Variable           | Default           | Description                                       |
| ------------------ | ----------------- | ------------------------------------------------- |
| `PPB_LLAMA_BENCH`  | `llama-bench`     | Path or name of the `llama-bench` binary.         |
| `PPB_RESULTS_FILE` | `./results.jsonl` | Default output file (overridden by `--results`).  |
| `PPB_MODELS_DIR`   | `./models`        | Default directory used by the `download` command. |

### 3. Find Your VRAM Limit

PPB can automatically discover the maximum context window your hardware supports before running out of memory, using a binary search algorithm.

```bash
python ppb.py auto-limit --model ./models/Llama-3-8B-Instruct.Q4_K_M.gguf
```

The full set of options:

```bash
python ppb.py auto-limit \
  --model ~/models/Llama-3-8B-Instruct.Q4_K_M.gguf \
  --min-ctx 2048 \
  --max-ctx 131072 \
  --tolerance 1024 \
  --runner llama-bench   # use a different runner backend (default: llama-bench)
```

#### How it works

1. Instantiates the requested runner via the registry.
2. Sets `lo = min_ctx`, `hi = max_ctx`.
3. Probes the midpoint `mid = (lo + hi) / 2` by calling `runner.probe_ctx()` (for llama-bench this runs with `-n 0`, allocation-only).
4. **Pass** (no OOM) → `lo = mid + 1`, records `mid` as the last known-good value.
5. **Fail** (non-zero exit code, or output contains `"out of memory"` / `"bad alloc"` / etc.) → `hi = mid - 1`.
6. Stops when `hi - lo < tolerance` and prints the largest safe context size.

Sample output:

```
  iter  1: n_ctx= 66,560  ✓ pass  → window [66,561, 131,071]
  iter  2: n_ctx= 98,815  ✗ OOM   → window [66,561, 98,814]
  iter  3: n_ctx= 82,687  ✓ pass  → window [82,688, 98,814]
  ...

✓ Maximum safe context for Llama-3-8B-Instruct.Q4_K_M.gguf

    90,111 tokens
```

#### auto-limit options

| Flag          | Default        | Description                                               |
| ------------- | -------------- | --------------------------------------------------------- |
| `--model`     | _(required)_   | Path to the GGUF model file.                              |
| `--min-ctx`   | `2048`         | Lower bound for the binary search.                        |
| `--max-ctx`   | `131072`       | Upper bound for the binary search.                        |
| `--tolerance` | `1024`         | Stop searching when the remaining window is this narrow.  |
| `--runner`    | `llama-bench`  | Runner backend to use for probing.                        |

### 4. View Your Hardware Profile

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

This same profile is automatically included in every benchmark record written
to `results.jsonl`.

## Runner Plugins

PPB uses a **pluggable runner architecture** so new benchmark backends can be added without modifying core code.

### Built-in runners

| `runner_type`  | Module                  | Description |
| -------------- | ----------------------- | ----------- |
| `llama-bench`  | `runners/llama_bench.py` | Default. Wraps llama.cpp's `llama-bench` CLI via subprocess. Supports OOM probing for `auto-limit`. |

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

1. Run a benchmark sweep: `python ppb.py sweep sweep.toml`.
2. Open an Issue in this repository titled "Benchmark Submission: [Your Hardware]".
3. Attach or paste the contents of your `results.jsonl`.

The public leaderboard is hosted at: [poorpaul.dev](https://poorpaul.dev)

## About the Maintainers

Poor Paul's Benchmark is an open-source project maintained by the team at [Ximplar](https://ximplar.com).

While PPB is built for the community to test and stretch consumer hardware to its limits, Ximplar specializes in taking those insights and deploying cost-effective, high-ROI AI models for enterprise environments. If you need help architecting your company's local AI infrastructure, reach out to us.
