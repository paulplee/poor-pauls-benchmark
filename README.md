# Poor Paul's Benchmark (PPB)

Find the absolute limit of your local AI hardware without the enterprise budget.

AI-driven GPU and RAM prices got you down? It certain got me feeling a lot poorer. I can't turn back the clock and go back to normal prices. About the only thing I can do is to get myself more informed in how I can best use the little hardware that I can muster.

Poor Paul's Benchmark is an automated evaluation framework for local LLM inference. It wraps `llama.cpp`'s `llama-bench` to provide declarative parameter sweeps, automated VRAM limit discovery, and standardized hardware fingerprinting.

My goal is to build the definitive public leaderboard for cost-effective AI setups, helping homelabbers, prosumers, and small businesses answer the question: _What is the most efficient hardware for my AI workload?_

## The Problem

Evaluating local LLM performance across heterogeneous hardware (Apple Silicon, discrete consumer Nvidia GPUs, mixed rigs) is tedious. Finding the exact maximum context window (`n_ctx`) before your system hits the "VRAM Cliff" (crashing with an Out-of-Memory error or degrading into system swap) usually requires hours of manual trial and error. Furthermore, self-reported benchmarks on forums often lack crucial hardware context (driver versions, background tasks, exact quantization).

## The Solution

PPB automates the tedious parts of benchmarking so you can focus on the data.

### Key Features

- **Declarative Parameter Sweeps:** Define your test matrices in a simple TOML file. PPB will automatically iterate through every combination of batch size, context length, and GPU layers.
- **Auto-Discover VRAM Limits:** Using a binary search algorithm, PPB automatically probes your hardware to find the exact maximum context size a specific model can handle before triggering an OOM error.
- **Integrated Model Downloader:** Native integration with Hugging Face Hub to download, cache, and symlink GGUF models directly via the CLI.
- **Automated Hardware Fingerprinting:** PPB automatically detects your OS, RAM, CPU architecture, and GPU details (via `pynvml` on Linux/Windows or `system_profiler` on macOS) to ensure benchmark submissions are accurate.
- **Standardized Leaderboard Export:** Generates clean Markdown tables and JSONL logs ready for submission to the public leaderboard.

## Installation

> **Note:** PPB is currently in active development. These instructions represent the target CLI interface.

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/paulplee/poor-pauls-benchmark.git
cd poor-pauls-benchmark
pip install -r requirements.txt
```

You must also have the `llama-bench` binary compiled and accessible in your system PATH or local directory.

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

| Key          | Type             | Required | Description                                                                                                                                                           |
| ------------ | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_path` | string (path)    | ✅       | Path to the GGUF model file. Relative paths are resolved from the working directory.                                                                                  |
| `n_ctx`      | list of integers | ✅       | Prompt token counts to test (passed as `-p` to `llama-bench`). Controls KV cache depth — larger values stress longer-context throughput. e.g. `[8192, 16384, 32768]`. |
| `n_batch`    | list of integers | ✅       | Batch sizes (passed as `-b` to `llama-bench`) for prompt-processing throughput tests, e.g. `[512, 1024]`.                                                             |

PPB computes the full Cartesian product of `n_ctx × n_batch`, so `3 × 2 = 6` combinations in the example above.

##### Example: exhaustive sweep across multiple quantisations

```toml
[sweep]
model_path = "~/models/Llama-3*Q4*.gguf"   # all Q4 quants in the folder
n_ctx   = [8192, 16384, 32768]
n_batch = [256, 512, 1024]
```

This tests every matched model at all 9 `(n_ctx, n_batch)` combinations and appends every result as a JSON line to `results.jsonl`.

##### Results format

Each line written to the JSONL file is a self-contained record:

```json
{
  "timestamp": "2026-03-04T10:00:00+00:00",
  "model_path": "/abs/path/to/model.gguf",
  "n_ctx": 8192,
  "n_batch": 512,
  "results": [
    /* raw llama-bench JSON output */
  ]
}
```

##### Environment overrides

| Variable           | Default           | Description                                       |
| ------------------ | ----------------- | ------------------------------------------------- |
| `PPB_LLAMA_BENCH`  | `llama-bench`     | Path or name of the `llama-bench` binary.         |
| `PPB_RESULTS_FILE` | `./results.jsonl` | Default output file (overridden by `--results`).  |
| `PPB_MODELS_DIR`   | `./models`        | Default directory used by the `download` command. |

### 3. Find Your VRAM Limit

PPB can automatically discover the maximum context window your hardware supports before running out of memory, using a binary search algorithm.

```bash
uv run ppb.py auto-limit --model ./models/Llama-3-8B-Instruct.Q4_K_M.gguf
```

The full set of options:

```bash
uv run ppb.py auto-limit \
  --model ~/models/Llama-3-8B-Instruct.Q4_K_M.gguf \
  --min-ctx 2048      \ # lower bound (default: 2048)
  --max-ctx 131072    \ # upper bound (default: 131072)
  --tolerance 1024      # stop when hi-lo < this (default: 1024)
```

#### How it works

1. Sets `lo = min_ctx`, `hi = max_ctx`.
2. Probes the midpoint `mid = (lo + hi) / 2` by running `llama-bench` with `-n 0` (allocation-only, no generation).
3. **Pass** (exit 0, no OOM marker in output) → `lo = mid + 1`, records `mid` as the last known-good value.
4. **Fail** (non-zero exit code, or output contains `"out of memory"` / `"bad alloc"` / etc.) → `hi = mid - 1`.
5. Stops when `hi - lo < tolerance` and prints the largest safe context size.

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

| Flag          | Default   | Description                                               |
| ------------- | --------- | --------------------------------------------------------- |
| `--model`     | _(required)_ | Path to the GGUF model file.                           |
| `--min-ctx`   | `2048`    | Lower bound for the binary search.                        |
| `--max-ctx`   | `131072`  | Upper bound for the binary search.                        |
| `--tolerance` | `1024`    | Stop searching when the remaining window is this narrow.  |

## Contributing to the Leaderboard

We are crowdsourcing a definitive database of Tokens-per-Second and Tokens-per-Watt across consumer hardware.

When you run a test suite, PPB generates a `results.md` file containing your hardware fingerprint and performance metrics. To add your machine to the public database:

1. Run a standardized benchmark suite (e.g., `ppb sweep suites/standard_llama3_8b.toml`).
2. Open an Issue in this repository titled "Benchmark Submission: [Your Hardware]".
3. Paste the contents of `results.md` into the issue.

The public leaderboard is hosted at: [poorpaul.dev](https://poorpaul.dev)

## About the Maintainers

Poor Paul's Benchmark is an open-source project maintained by the team at [B2AIN](https://b2ain.com).

While PPB is built for the community to test and stretch consumer hardware to its limits, B2AIN specializes in taking those insights and deploying cost-effective, high-ROI "Second Brain" AI models for enterprise environments. If you need help architecting your company's local AI infrastructure, reach out to us.
