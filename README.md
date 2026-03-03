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

_(Note: PPB is currently in active development. These instructions represent the target CLI interface.)_

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

Create a `sweep.toml` file to define your test matrix:

```toml
[sweep]
model_path = "./models/Llama-3-8B-Instruct.Q4_K_M.gguf"
n_gpu_layers = [-1]
n_batch = [512, 1024]
n_ctx = [8192, 16384, 32768, 65536]
```

Execute the sweep:

```bash
python ppb.py sweep sweep.toml
```

### 3. Find Your VRAM Limit

Let PPB find the maximum context window your hardware can support without crashing:

```bash
python ppb.py auto-limit --model ./models/Llama-3-8B-Instruct.Q4_K_M.gguf --min-ctx 2048 --max-ctx 128000
```

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
