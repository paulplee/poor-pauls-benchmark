# Poor Paul's Benchmark (PPB)

Find the absolute limit of your local AI hardware without the enterprise budget.

AI-driven GPU and RAM prices got you down? It certain got me feeling a lot poorer. I can't turn back the clock to the "normal" times, but I can try to get myself more informed in how I can best use the little hardware that I can muster.

Poor Paul's Benchmark is an automated evaluation framework for local LLM inference. It uses a **pluggable runner architecture** — built-in runners wrap `llama.cpp`'s `llama-bench` (raw throughput) and `llama-server` (real-world UX latency), and new backends (vLLM, Stable Diffusion, …) can be added without touching core code.

My goal is to build the definitive public leaderboard for cost-effective AI setups, helping homelabbers (me), prosumers (me), and small businesses (and me) answer the question: _What is the most efficient hardware / model / settings for my AI workload?_

## The PPB Ecosystem

PPB is one part of a three-component platform:

| Component                                                                        | What it does                                        |
| -------------------------------------------------------------------------------- | --------------------------------------------------- |
| **poor-pauls-benchmark** _(this repo)_                                           | Runs benchmarks on your hardware; publishes results |
| **[paulplee/ppb-results](https://huggingface.co/datasets/paulplee/ppb-results)** | Public HuggingFace dataset of community results     |
| **[ppb-mcp](https://github.com/paulplee/ppb-mcp)**                               | MCP server — lets LLMs query the dataset directly   |
| **[poorpaul.dev](https://poorpaul.dev)**                                         | Visual analytics and leaderboard                    |

Once you submit results, any MCP-compatible LLM client (Claude Desktop, Cursor,
Windsurf, etc.) can answer questions like:

> _"What's the best quantization for my RTX 4090 running 4 concurrent users?"_

using your real benchmark data, via `https://mcp.poorpaul.dev`.

## The Problem

Evaluating local LLM performance across heterogeneous hardware (Apple Silicon, discrete consumer Nvidia GPUs, mixed rigs) is tedious. Finding the exact maximum context window (`n_ctx`) before your system hits the "VRAM Cliff" (crashing with an Out-of-Memory error or degrading into system swap) usually requires hours of manual trial and error. Furthermore, self-reported benchmarks on forums often lack crucial hardware context (driver versions, background tasks, exact quantization).

## The Solution

PPB automates the tedious parts of benchmarking so you can focus on studying the results and agonizing over what to sacrifice.

### Key Features

- **Pluggable Runner Architecture:** Benchmark backends are implemented as plugins inheriting from `BaseRunner`. Swap `llama-bench` (implemented) for `llama-server` (implemented), `vllm` (not yet implemented), or your own backend by setting `runner_type` in the sweep config.
- **Auto-Discover VRAM Limits:** Using a binary search algorithm, PPB automatically probes your hardware to find the exact maximum context size a specific model can handle before triggering an OOM error.
- **Declarative Parameter Sweeps:** Define your test matrices in a simple TOML file. PPB will automatically iterate through every combination of GGUF models, batch size, context length, and GPU layers.
- **Real-World UX Metrics:** The `llama-server` runner streams completions from real ShareGPT conversational prompts, measuring **Time-To-First-Token (TTFT)** and **Inter-Token Latency (ITL)** — the numbers that matter for interactive applications.
- **Integrated Model Downloader:** Native integration with Hugging Face Hub to automatically download, cache, and verify GGUF models as part of any benchmark run.
- **Automated Hardware Fingerprinting:** PPB automatically detects your OS, RAM, CPU architecture, and GPU details (via `pynvml` on Linux/Windows or `system_profiler` on macOS). Hardware profiles are embedded in every result record and can be viewed any time with `ppb hw-info`.
- **Concurrent User Simulation:** Measure how latency degrades under load by setting `concurrent_users = [1, 2, 4, 8, 16, 32]` (or any values you like — there is no hard upper limit) in your sweep config. The `llama-server-loadtest` runner auto-discovers maximum sustainable concurrency.
- **LLM Engine Detection:** Every record captures `llm_engine_name` and `llm_engine_version` (including build hashes) so results from different inference backends remain comparable.
- **OS Distro Detection:** Automatically identifies the Linux distribution (Ubuntu, Fedora, etc.), macOS version, or Windows version alongside the kernel info.
- **Workload Classification:** Records include `task_type`, `prompt_dataset`, `num_prompts`, and `n_predict` so readers know exactly what workload produced each result.
- **Quality Scoring:** An optional `quality_score` field (defaults to `null`) is reserved for future output-quality evaluation.
- **Extensible Tags:** A free-form `tags` JSON column lets you attach arbitrary metadata (CI run IDs, experiment labels, etc.) without schema changes.
- **Stable Result Envelope:** Every JSONL record includes `runner_type`, `timestamp`, and `hardware` — so results from different runners or years apart remain comparable.

## Project Structure

```text
ppb.py                 # CLI entry point (Typer app)
ppb_context_rot.py     # context-rot / NIAH qualitative evaluation
ppb_tool_accuracy.py   # tool-call accuracy (BFCL **v4** single-turn splits + PPB-native; emits no_call_accuracy on the irrelevance split)
ppb_answer_quality.py  # answer knowledge-accuracy / quality (judge-model pipeline)
ppb_multiturn.py       # multi-turn memory & coherence (LongMemEval / MT-Bench)
ppb_quality_prompts_cache.json  # frozen 50-prompt evaluation set (auto-generated)
runners/
  __init__.py          # Runner registry (register_runner / get_runner)
  base.py              # BaseRunner ABC — contract for all backends
  _server_mixin.py     # Shared server start/stop/health-check logic
  llama_bench.py       # Built-in llama-bench runner (raw throughput)
  llama_server.py      # Built-in llama-server runner (TTFT / ITL latency)
  llama_server_loadtest.py  # Load-test runner (max concurrency discovery)
utils/
  __init__.py          # Package init
  flattener.py         # Normalize nested JSONL → flat, Arrow-friendly dicts
  gguf_metadata.py     # Read GGUF header metadata for VRAM pre-flight checks
  publisher.py         # Upload flattened results to HF leaderboard repo
datasets/
  __init__.py          # Dataset download & loading helpers
  sharegpt.py          # ShareGPT download (HF Hub) + prompt extraction
  data/                # Downloaded dataset cache (gitignored)
tests/
  conftest.py          # Shared fixtures + FakeRunner
  test_runners.py      # Runner ABC, registry, and LlamaBenchRunner tests
  test_llama_server.py # LlamaServerRunner, SSE parsing, dataset tests
  test_config.py       # SweepConfig, BenchCombo, _write_result tests
  test_orchestration.py # Sweep, vram-cliff, & VRAM preflight tests
  test_gguf_metadata.py # GGUF header parser & VRAM estimation tests
suites/
  suite.example.toml        # Starter benchmark suite (copy → suites/my_gpu.toml)
  qualitative_example.toml  # Starter suite with context-rot enabled
  .gitignore                # Ignores user suite files, keeps examples
results/               # Benchmark output directory (gitignored)
  results.jsonl        # ← files land here automatically
docs/
  building-llama-cpp.md # How to build and upgrade llama.cpp
```

## Installation

> **Note:** PPB is currently in active development.

Requires **Python ≥ 3.11**.

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

#### Context-rot (optional)

The qualitative context-rot evaluation (`ppb_context_rot.py`) requires `llama-cpp-python`, which is **not installed by default** because it needs a platform-specific GPU-enabled build. A plain `pip install llama-cpp-python` gives a CPU-only binary that is impractically slow for long-context evaluation.

Install the GPU-accelerated variant **once** before running a qualitative suite:

```bash
# CUDA (Linux / Windows)
CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python>=0.3.0"

# Metal (macOS Apple Silicon)
CMAKE_ARGS="-DGGML_METAL=on" pip install "llama-cpp-python>=0.3.0"

# Or via uv with a pre-built wheel index (CUDA 12.4 example)
uv pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Or use the pyproject.toml optional group (compiles from source)
uv pip install -e ".[qualitative]"
```

Pre-built wheels for common CUDA / ROCm / Metal targets are published by the llama-cpp-python project at <https://github.com/abetlen/llama-cpp-python/releases>.

You must also have `llama-bench` and/or `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) compiled and accessible in your system PATH (or point to them with `PPB_LLAMA_BENCH` / `PPB_LLAMA_SERVER`). See **[Building and Upgrading llama.cpp](docs/building-llama-cpp.md)** for step-by-step instructions.

> **llama.cpp version:** Build **b8688+** is recommended. Older builds may lack support for newer model architectures (e.g. Gemma 4 requires ≥ b8688). PPB will detect unsupported architectures and report a clear error instead of silently failing.

> **Conversational dataset:** The `llama-server` runner uses real conversational prompts from the [ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) (~700 MB) by default. It is downloaded automatically on the first `llama-server` run, or you can pre-fetch it with `python ppb.py download-dataset`. You can also point to any HF-hosted dataset with `--repo` and `--filename` — see [§2](#2-run-a-parameter-sweep) for details.

### Running Tests

```bash
pip install pytest   # or: uv sync --group dev
python -m pytest tests/ -v
```

## Share Your Results

PPB is a **community leaderboard**. Every result you submit makes the dataset
more useful for everyone — including LLM clients querying [ppb-mcp](https://github.com/paulplee/ppb-mcp).

### 1. Get a Hugging Face token

Log in at [huggingface.co](https://huggingface.co) → Settings → Access Tokens →
create a token with **Write** access to `paulplee/ppb-results`.

```bash
export HF_TOKEN=hf_your_token_here
# Or add it to your .env file (see .env.example)
```

### 2. Run your benchmark

```bash
# Run a full sweep (vram-cliff → parameter sweep → publish)
uv run ppb.py all suites/my_gpu.toml

# Or run a sweep only and publish manually afterward
uv run ppb.py sweep suites/my_gpu.toml
```

### 3. Publish your results

```bash
uv run ppb.py publish --results results/my_results.jsonl
```

Results appear in the [public dataset](https://huggingface.co/datasets/paulplee/ppb-results)
and on [poorpaul.dev/insights](https://poorpaul.dev/insights) within minutes.

> **Incremental publishing:** When using `ppb all` with a `[publish]` section in your
> suite file, results are published after _each model_ completes — so a long run is
> never lost to a late crash. See `suites/suite.example.toml` for the `[publish]`
> configuration block.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide including hardware metadata
best practices and the result schema.

## Run Modes

PPB has three top-level run modes that map directly to columns of the
published HuggingFace dataset.

### Commands

```bash
# Quantitative only — vram-cliff + parameter sweep (throughput, TTFT, VRAM).
uv run ppb.py quantitative suites/my_gpu.toml

# Qualitative only — context-rot + tool-call accuracy + answer-quality + multi-turn.
# Use this when the quantitative row already exists on Hugging Face and
# you only want to (re)score qualitative metrics. The orchestrator looks
# up the prior `vram_cliff_tokens` from the published row and uses it to
# skip context-rot lengths and LongMemEval cases that would OOM.
uv run ppb.py qualitative suites/qualitative_example.toml

# Everything — quantitative followed by qualitative for the same models.
# In `all` mode the judge `Llama` instance loaded for Phase 6 (answer
# quality) is reused by Phase 7 (multi-turn) so the judge is only
# loaded into VRAM once.
uv run ppb.py all suites/my_gpu.toml
```

### Composable join key

Every published row carries the four-field join key
**`(gpu_name, model_name, quantization, run_type)`**. Quantitative and
qualitative runs are independent: you can publish a quant sweep today and a
qualitative pass next week, and downstream consumers (`ppb-mcp`,
[poorpaul.dev](https://poorpaul.dev/insights)) will JOIN them into a single
model profile by matching on this tuple. `ppb-mcp` performs an outer
JOIN: rows with `run_type == "all"` already contain both blocks,
while `"quantitative"` and `"qualitative"` rows are stitched together
on the join tuple to produce the same merged profile.

### Qualitative block schema

Every published row includes a nested `qualitative` JSON column. Phases that
did not run for the row carry `null` for their respective keys. The schema
below is the **canonical final shape** — do not add keys without a schema
version bump.

| Key                              | Owning phase                          | Type         |
| -------------------------------- | ------------------------------------- | ------------ |
| `context_rot_score`              | Phase 4 — Context-Rot (Semantic NIAH) | float        |
| `context_rot_accuracy_by_length` | Phase 4 — Context-Rot (Semantic NIAH) | object       |
| `context_rot_accuracy_by_depth`  | Phase 4 — Context-Rot (Semantic NIAH) | object       |
| `tool_selection_accuracy`        | Phase 5 — Tool-Call Accuracy          | float        |
| `parameter_accuracy`             | Phase 5 — Tool-Call Accuracy          | float        |
| `parameter_hallucination_rate`   | Phase 5 — Tool-Call Accuracy          | float        |
| `parse_success_rate`             | Phase 5 — Tool-Call Accuracy          | float        |
| `overall_tool_accuracy`          | Phase 5 — Tool-Call Accuracy          | float        |
| `knowledge_accuracy_mean`        | Phase 6 — Answer Quality              | float        |
| `knowledge_accuracy_std`         | Phase 6 — Answer Quality              | float        |
| `answer_relevancy_mean`          | Phase 6 — Answer Quality              | float        |
| `coherence_mean`                 | Phase 6 — Answer Quality              | float        |
| `quality_composite_score`        | Phase 6 — Answer Quality              | float        |
| `memory_accuracy`                | Phase 7 — Multi-Turn (LongMemEval)    | float        |
| `mt_bench_score`                 | Phase 7 — Multi-Turn (MT-Bench quick) | float (1–10) |
| `cases_evaluated`                | Phase 7 — Multi-Turn                  | int          |
| `cases_skipped_context`          | Phase 7 — Multi-Turn                  | int          |

In each multi-turn run only one of `memory_accuracy` / `mt_bench_score`
is populated — the other is `null` — according to which `multiturn_mode`
was selected.

> **Note:** `mt_bench_score` is on a 1–10 scale (MT-Bench community
> standard). All other float metrics are in the range 0–1. Downstream
> consumers should normalise before building composite scores:
> `mt_bench_score_norm = (mt_bench_score - 1) / 9`.

### Setting up a judge model

Phase 6 (answer quality) and Phase 7 (multi-turn) both use a separate,
locally-loaded "judge" GGUF to grade responses. The judge **MUST** be a
different model from the one under test — otherwise the model is
grading itself, which produces self-enhancement bias and inflates
leaderboard scores. PPB enforces this with a runtime check that
compares resolved paths.

Recommended judges: a small, well-aligned model in the 3–7B range such
as `Qwen3.5-4B-Q4_K_M` or `Llama-3-8B-Instruct-Q4_K_M`. Set
`judge_model_path` once in the `[qualitative]` block of your suite
TOML; the same instance is shared between Phase 6 and Phase 7 in
`ppb all` mode so the judge is only loaded into VRAM once.

When `run_type == "qualitative"`, the sibling `quantitative` column is
explicitly `null` so consumers don't conflate stale quant numbers with a
fresh qualitative measurement.

Each published row also carries an opaque `meta` JSON column. Phase 6
(answer quality) populates `meta.quality_prompts_cache_hash` with the
SHA-256 of `ppb_quality_prompts_cache.json` so downstream tools can
detect drift in the 50-prompt evaluation set across runs.

## Usage Guide

### 1. Find Your VRAM Cliff (`vram-cliff`)

PPB can automatically discover the maximum context window your hardware supports before running out of memory, using a binary search algorithm.

**CLI-only mode:**

```bash
python ppb.py vram-cliff \
  --repo-id unsloth/Qwen3.5-0.8B-GGUF \
  --filename "Qwen3.5-0.8B-Q4_K_M.gguf"
# Or a glob pattern to test multiple quants:
python ppb.py vram-cliff \
  --repo-id unsloth/Qwen3.5-0.8B-GGUF \
  --filename "*Q4*.gguf"
```

**TOML-driven mode** (reads the `[vram-cliff]` section from a suite file):

```bash
python ppb.py vram-cliff suites/my_gpu.toml
```

**Full set of options:**

```bash
python ppb.py vram-cliff \
  --repo-id unsloth/Qwen3.5-0.8B-GGUF \
  --filename "Qwen3.5-0.8B-Q4_K_M.gguf" \
  --models-dir ~/models \
  --min_ctx 2048 \
  --max_ctx 131072 \
  --tolerance 1024 \
  --runner llama-bench
```

When driven by a TOML file, CLI flags override TOML values.

#### How it works

1. Instantiates the requested runner via the registry.
2. **Sanity check:** Probes `min_ctx` first. If even the minimum context fails, PPB aborts immediately with the real error (e.g. "unknown model architecture") instead of binary-searching down to nothing.
3. Sets `lo = min_ctx`, `hi = max_ctx`.
4. Probes the midpoint `mid = (lo + hi) / 2` by calling `runner.probe_ctx()` (for llama-bench this runs with `-n 0`, allocation-only).
5. **Pass** (no OOM) → `lo = mid + 1`, records `mid` as the last known-good value.
6. **Fail** (non-zero exit code, or output contains `"out of memory"` / `"bad alloc"` / etc.) → `hi = mid - 1`.
7. Stops when `hi - lo < tolerance` and prints the largest safe context size.

Sample output (each iteration shows per-probe duration):

```text
  iter  1: n_ctx= 66,560  ✓ pass  → window [66,561, 131,071]  (2.3s)
  iter  2: n_ctx= 98,815  ✗ OOM   → window [66,561, 98,814]   (0.8s)
  iter  3: n_ctx= 82,687  ✓ pass  → window [82,688, 98,814]   (3.1s)
  ...

✓ Maximum safe context for unsloth/Qwen3.5-0.8B-Q4_K_M.gguf

    90,111 tokens
```

#### vram-cliff options

| Flag           | Default       | Description                                                    |
| -------------- | ------------- | -------------------------------------------------------------- |
| `CONFIG`       | _(optional)_  | Path to a TOML suite file containing a `[vram-cliff]` section. |
| `--repo-id`    | _(from TOML)_ | HF repository ID, e.g. `unsloth/Qwen3.5-0.8B-GGUF`.            |
| `--filename`   | _(from TOML)_ | GGUF filename or glob pattern, e.g. `"*Q4*.gguf"`.             |
| `--models-dir` | `./models`    | Local directory to cache downloaded models.                    |
| `--min_ctx`    | `2048`        | Lower bound for the binary search.                             |
| `--max_ctx`    | `131072`      | Upper bound for the binary search.                             |
| `--tolerance`  | `1024`        | Stop searching when the remaining window is this narrow.       |
| `--runner`     | `llama-bench` | Runner backend to use for probing.                             |

#### vram-cliff TOML section

```toml
repo_id     = "unsloth/Qwen3.5-0.8B-GGUF"             # HF repo
filename    = "Qwen3.5-0.8B-Q4_K_M.gguf"              # exact file or glob
models_dir  = "./models"                               # local cache dir
runner_type = "llama-bench"  # optional (shared across sections)

[vram-cliff]
min_ctx    = 2048       # optional (default: 2048)
max_ctx    = 131072     # optional (default: 131072)
tolerance  = 1024       # optional (default: 1024)
```

When `filename` is a glob pattern (e.g. `"*Q4*.gguf"`), vram-cliff probes **each matched model independently** and reports per-model results.

### 2. Run a Parameter Sweep

Create a `suite.toml` file to define your test matrix (a starter config is included at [`suites/suite.example.toml`](suites/suite.example.toml)):

```toml
repo_id    = "unsloth/Qwen3.5-0.8B-GGUF"
filename   = "Qwen3.5-0.8B-Q4_K_M.gguf"
models_dir = "./models"

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
  --repo-id unsloth/Qwen3.5-0.8B-GGUF \
  --filename "Qwen3.5-0.8B-Q4_K_M.gguf" \
  --n-ctx 8192,16384,32768 \
  --n-batch 512,1024
```

**CLI flags override TOML values** when both are provided:

```bash
python ppb.py sweep suites/my_gpu.toml --n-ctx 4096,8192  # only test these two contexts
```

Sample sweep output (per-combo duration shown):

```text
  ✓ [1/6] model.gguf ctx=8192 batch=512   42.0 tok/s  (12.4s)
  ✓ [2/6] model.gguf ctx=8192 batch=1024  38.7 tok/s  (11.9s)
  ✓ [3/6] model.gguf ctx=16384 batch=512  31.2 tok/s  (14.1s)
  ...
```

#### Sweep config reference

Shared fields (`repo_id`, `filename`, `models_dir`, `runner_type`, `runner_params`) can live at the **top level** and are inherited by both `[vram-cliff]` and `[sweep]`. Section-level values override the root. Sweep-specific fields:

| Key                | Type             | Required | Default         | Description                                                                                                                                                                                                                     |
| ------------------ | ---------------- | -------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `repo_id`          | string           | ✅       |                 | Hugging Face repository ID, e.g. `unsloth/Qwen3.5-0.8B-GGUF`.                                                                                                                                                                   |
| `filename`         | string           | ✅       |                 | GGUF filename or glob pattern (e.g. `"*Q4*.gguf"`). Models are downloaded automatically.                                                                                                                                        |
| `models_dir`       | string (path)    |          | `"./models"`    | Local directory to cache downloaded models.                                                                                                                                                                                     |
| `n_ctx`            | list of integers | ✅       |                 | Context sizes to test. Controls KV cache depth — larger values stress longer-context workloads, e.g. `[8192, 16384, 32768]`.                                                                                                    |
| `n_batch`          | list of integers | ✅       |                 | Batch sizes for prompt processing, e.g. `[512, 1024]`. Used by `llama-bench`; accepted but unused by `llama-server` (it manages its own batching).                                                                              |
| `concurrent_users` | list of integers |          | `[1]`           | Number of simulated users sending requests in parallel. Only meaningful for `llama-server` and `llama-server-loadtest` runners. Any positive integers are accepted — e.g. `[1, 2, 4, 8, 16, 32]`. There is no hard upper limit. |
| `runner_type`      | string           |          | `"llama-bench"` | Which benchmark backend to use. See [Runner Plugins](#runner-plugins) below.                                                                                                                                                    |
| `runner_params`    | table            |          | `{}`            | Runner-specific overrides; see `[sweep.runner_params]` below.                                                                                                                                                                   |

PPB computes the full Cartesian product of `models × n_ctx × n_batch × concurrent_users`, so 2 models × 3 contexts × 2 batches × 3 user counts = 36 combinations. When `concurrent_users` is omitted (defaults to `[1]`), the product stays the same as before.

##### Example: exhaustive sweep across multiple quantisations

```toml
[sweep]
repo_id  = "unsloth/Qwen3.5-0.8B-GGUF"
filename = "*Q4*.gguf"   # all Q4 quants in the repo
n_ctx    = [8192, 16384, 32768]
n_batch  = [256, 512, 1024]
```

This tests every matched model at all 9 `(n_ctx, n_batch)` combinations.

##### Example: using a custom runner and params

```toml
[sweep]
runner_type = "llama-bench"
repo_id     = "unsloth/Qwen3.5-0.8B-GGUF"
filename    = "Qwen3.5-0.8B-Q4_K_M.gguf"
n_ctx       = [8192]
n_batch     = [512]

[sweep.runner_params]
llama_bench_cmd = "/opt/llama.cpp/build/bin/llama-bench"
```

#### Choosing a runner

PPB ships with three runners — pick the one that matches your evaluation goal:

| Runner                    | Measures                                | Best for                                                   |
| ------------------------- | --------------------------------------- | ---------------------------------------------------------- |
| `llama-bench` _(default)_ | Raw throughput (tok/s)                  | Finding peak hardware performance, comparing quants        |
| `llama-server`            | **TTFT** and **ITL** latency            | UX-relevant benchmarks for interactive / chat applications |
| `llama-server-loadtest`   | Max concurrent users, concurrency curve | Capacity planning — how many users your hardware can serve |

All three runners use the same `ppb sweep` command — just set `runner_type` in your TOML.

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
repo_id     = "unsloth/Qwen3.5-0.8B-GGUF"
filename    = "Qwen3.5-0.8B-Q4_K_M.gguf"
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

```text
avg_ttft_ms: 142.5    p50_ttft_ms: 138.2    p99_ttft_ms: 210.7
avg_itl_ms:   12.3    p50_itl_ms:   11.8    p99_itl_ms:   18.4
```

##### `llama-server` runner_params reference

| Key                   | Type      | Default                                     | Description                                                                                                                                                                   |
| --------------------- | --------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llama_server_cmd`    | string    | `llama-server`                              | Path or name of the binary. Falls back to `PPB_LLAMA_SERVER` env → `$PATH`.                                                                                                   |
| `num_prompts`         | int       | `10`                                        | Number of prompts to send per run.                                                                                                                                            |
| `n_predict`           | int       | `256`                                       | Maximum tokens to generate per prompt.                                                                                                                                        |
| `health_timeout`      | int/float | `120`                                       | Seconds to wait for `/health` to return 200.                                                                                                                                  |
| `dataset_dir`         | string    | `datasets/data/`                            | Directory for cached dataset files.                                                                                                                                           |
| `dataset_repo`        | string    | `anon8231489123/ShareGPT_Vicuna_unfiltered` | HF Hub repository ID for the prompt dataset.                                                                                                                                  |
| `dataset_filename`    | string    | `ShareGPT_V3_unfiltered_cleaned_split.json` | File to download from the repository.                                                                                                                                         |
| `shuffle`             | bool      | `false`                                     | Randomise prompt order so repeated runs use different workloads.                                                                                                              |
| `seed`                | int       | _(none)_                                    | RNG seed for reproducible shuffling.                                                                                                                                          |
| `prompt_distribution` | string    | `"shared"`                                  | How prompts are assigned when `concurrent_users > 1`. `"shared"` = all users get the same prompts; `"split"` = prompts are divided among users (each gets `num_prompts / N`). |

##### `llama-server-loadtest` runner_params reference

| Key                   | Type         | Default                                     | Description                                                      |
| --------------------- | ------------ | ------------------------------------------- | ---------------------------------------------------------------- |
| `max_users`           | int          | `64`                                        | Stop escalating when this concurrency level is reached.          |
| `user_steps`          | list of ints | powers of 2 up to `max_users`               | Explicit concurrency levels to test, e.g. `[1, 2, 4, 8, 16]`.    |
| `error_threshold`     | float        | `0.1`                                       | Fraction of failed requests that marks a level as unsustainable. |
| `ramp_delay_s`        | float        | `2.0`                                       | Seconds to pause between concurrency levels.                     |
| `num_prompts`         | int          | `10`                                        | Number of prompts each user sends per level.                     |
| `n_predict`           | int          | `256`                                       | Max tokens to generate per prompt.                               |
| `prompt_distribution` | string       | `"shared"`                                  | Same as `llama-server` — `"shared"` or `"split"`.                |
| `llama_server_cmd`    | string       | `llama-server`                              | Path or name of the binary.                                      |
| `health_timeout`      | int/float    | `120`                                       | Seconds to wait for `/health` to return 200.                     |
| `dataset_dir`         | string       | `datasets/data/`                            | Directory for cached dataset files.                              |
| `dataset_repo`        | string       | `anon8231489123/ShareGPT_Vicuna_unfiltered` | HF Hub repository ID for the prompt dataset.                     |
| `dataset_filename`    | string       | `ShareGPT_V3_unfiltered_cleaned_split.json` | File to download from the repository.                            |
| `shuffle`             | bool         | `false`                                     | Randomise prompt order.                                          |
| `seed`                | int          | _(none)_                                    | RNG seed for reproducible shuffling.                             |

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
  "model": "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
  "n_ctx": 8192,
  "n_batch": 512,
  "llm_engine_name": "llama.cpp",
  "llm_engine_version": "b5063 (58ab80c3)",
  "task_type": "text-generation",
  "prompt_dataset": null,
  "hardware": {
    "os": {
      "system": "Linux",
      "release": "6.8.0",
      "machine": "x86_64",
      "distro": "Ubuntu",
      "distro_version": "24.04"
    },
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

```text
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

| Variable           | Default        | Description                                      |
| ------------------ | -------------- | ------------------------------------------------ |
| `PPB_LLAMA_BENCH`  | `llama-bench`  | Path or name of the `llama-bench` binary.        |
| `PPB_LLAMA_SERVER` | `llama-server` | Path or name of the `llama-server` binary.       |
| `PPB_MODELS_DIR`   | `./models`     | Default directory for caching downloaded models. |

### 3. Run the Full Suite (`all`)

The `all` command combines **vram-cliff**, **sweep**, and (optionally) **publish** into a single invocation:

1. **Phase 1 — vram-cliff:** Discovers the max safe context window for each model.
2. **Phase 1.5 — VRAM pre-flight check:** Before sweeping, PPB reads GGUF metadata from each model and estimates worst-case VRAM usage (max `n_ctx` × max `concurrent_users`). If any model is likely to OOM, a warning table is shown and you can choose to **a**uto-cap `n_ctx`, **p**roceed anyway, or **q**uit.
3. **Phase 2 — sweep:** Runs the parameter sweep, automatically skipping any combo whose `n_ctx` exceeds the per-model limit found in Phase 1 (or the auto-cap from Phase 1.5).
4. **Phase 3 — context-rot** _(optional)_: If the TOML has a `[qualitative]` section with `context_rot_enabled = true`, runs the semantic Needle-in-a-Haystack evaluation after each model's sweep. Results are appended to the same JSONL file with `runner_type = "context-rot"`. See [§5 — Context-Rot](#5-run-context-rot-qualitative) below.
5. **Publish** _(optional)_: If the TOML has a `[publish]` section, flattens the results to a local CSV and (when `upload = true`) uploads them to the central PPB leaderboard on Hugging Face.

> **Result rows:** `ppb all` writes _separate rows_ per phase to the same JSONL/CSV — one quantitative row (`runner_type = "llama-bench"`) per sweep configuration plus one qualitative row (`runner_type = "context-rot"`) per model. Every row carries the composite join key `(gpu_name, model_name, quant, run_type)` so downstream tools can stitch them together. The `run_type` column is `"all"` for rows produced by this command; the standalone `ppb quantitative` and `ppb qualitative` subcommands tag their rows `"quantitative"` and `"qualitative"` respectively.

```bash
python ppb.py all suites/my_gpu.toml
python ppb.py all suites/my_gpu.toml --results results/my_run.jsonl
```

If the TOML has no `[vram-cliff]` section, Phase 1 is skipped and the sweep runs unmodified. If there is no `[publish]` section, Phase 3 is skipped.

#### Example suite TOML

```toml
# Shared — declared once, inherited by both sections
repo_id     = "unsloth/Qwen3.5-0.8B-GGUF"
filename    = "*Q4*.gguf"                    # glob — tests every matched quant
models_dir  = "./models"
results     = "results/my_benchmark.jsonl"   # optional

[vram-cliff]
min_ctx    = 2048
max_ctx    = 131072
tolerance  = 1024

[sweep]
n_ctx      = [8192, 16384, 32768, 65536, 131072]
n_batch    = [512, 1024]

# Optional: auto-publish after the sweep completes
[publish]
submitter = "Your Name"
# upload  = true   # set false to write CSV only (default: true)
```

When the filename glob matches multiple models, Phase 1 probes each one independently.
The per-model caps are then passed to Phase 2, which skips combos that exceed each model's limit — no manual editing required.

#### Resuming an interrupted run

Long benchmark suites (lots of quants × concurrency levels × context sizes) can take many hours. If a run is interrupted — Ctrl-C, a crash, a power cut, a wedged HF download — you don't have to start over.

**Auto-resume (default).** When you re-run `ppb all` on the same suite, PPB scans `results/` for the most recent file matching `<config_stem>_*.jsonl` and **appends to it**, skipping every model that already has a complete set of rows:

```bash
uv run ppb.py all suites/my_gpu.toml
```

You'll see a banner like:

```text
🔄 RESUMING previous run from my_gpu_20260421_1543.jsonl
   19 of 22 model(s) already done — 3 remaining
   ⏭  Skipping unsloth/.../Qwen3.6-35B-A3B-MXFP4_MOE.gguf — already completed
   ...
```

**Resume from a specific file.** Pass `--results` (a.k.a. `-r`) to point at any prior JSONL:

```bash
uv run ppb.py all suites/my_gpu.toml -r results/my_gpu_20260421_1543.jsonl
```

The `--results` flag always wins over auto-detection.

**Force a fresh run.** Pass `--no-resume` to ignore prior results files and start a new timestamped JSONL:

```bash
uv run ppb.py all suites/my_gpu.toml --no-resume
```

**How "completed" is determined.** A model is considered done when the JSONL contains all `len(n_ctx) × len(n_batch) × len(concurrent_users)` expected rows for it (after applying any per-model `vram-cliff` cap). Resume granularity is **per model**, not per combo — a model that was interrupted mid-sweep will be re-run from its first combo, so a few duplicate rows for that one model can appear. The `suite_run_id` is preserved from the original file so the publish step keeps a single submission id across the resumed run.

> **Caveat:** auto-resume keys on the HF id (`repo_id/filename.gguf`) and the expected combo count from your current TOML. If you change `n_ctx`, `n_batch`, `concurrent_users`, `repo_id`, or `filename` between runs, models previously marked complete may no longer match and will be re-benchmarked.

## Run Modes — Quantitative, Qualitative, or Both

PPB exposes three top-level commands that all consume the **same suite TOML**:

| Command                    | Phases run                                       | When to use                                                                                                             |
| -------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `ppb all <suite>`          | vram-cliff + sweep + context-rot                 | First-time benchmark of a new model on new hardware.                                                                    |
| `ppb quantitative <suite>` | vram-cliff + sweep                               | Refresh perf numbers (e.g. after a llama.cpp upgrade) without re-running expensive qualitative evals.                   |
| `ppb qualitative <suite>`  | context-rot (and future qualitative phases) only | Add qualitative scores to a model that already has published quantitative results. Skips vram-cliff and sweep entirely. |

`ppb all <suite> --mode {all,quantitative,qualitative}` is the canonical form; `quantitative` and `qualitative` are convenience subcommands that pin `--mode` for you. Existing scripts that call `ppb all` keep working unchanged.

### Composable result schema

Every published row carries a four-field **join key** so quantitative-only and qualitative-only runs can be stitched together downstream by `ppb-mcp` and `poorpaul.dev`:

| Column         | Meaning                                        |
| -------------- | ---------------------------------------------- |
| `gpu_name`     | e.g. `NVIDIA GeForce RTX 4090`                 |
| `model_name`   | base model name (alias of `model_base`)        |
| `quantization` | quant tag (alias of `quant`)                   |
| `run_type`     | `"all"` \| `"quantitative"` \| `"qualitative"` |

Each row also carries a **quantitative block** (`vram_used_gb`, `vram_cliff_tokens`, `tokens_per_sec_prompt`, `tokens_per_sec_generation`, `throughput_tok_s`, …) and a **qualitative block** (`context_rot_score`, `context_rot_accuracy_by_length`, `context_rot_accuracy_by_depth`, plus nullable placeholders for forthcoming `tool_selection_accuracy`, `quality_composite_score`, `mt_bench_score`, etc.). Whichever block isn't relevant to a given run is left `null`.

### How `qualitative` mode finds the VRAM cliff

When you run `ppb qualitative <suite>`, the runner queries the central `paulplee/ppb-results` Hugging Face dataset for the most recent row matching `(gpu_name, model_name, quantization)` with `run_type ∈ {"all", "quantitative"}`. If a match exists its `vram_cliff_tokens` is reused to filter context-rot haystack lengths:

```text
[PPB] Run mode: qualitative | Suite: my_gpu.toml
  ℹ Using existing quantitative result: vram_cliff_tokens=32768
```

If no match is found:

```text
  ⚠ No existing quantitative result found for this config.
    context_rot will not skip lengths based on VRAM cliff.
```

Qualitative results are **not** merged into the quantitative row — they are published as a separate row with `run_type="qualitative"` and `quantitative=null` columns. The JOIN happens downstream.

### Backward compatibility

The existing dataset Parquet schema is preserved: every new column is **nullable** and old columns (`model`, `quant`, `throughput_tok_s`, `context_rot_score`, …) are unchanged. Legacy rows without an explicit `run_type` are treated as `"quantitative"` by the lookup logic.

### 5. Run Context-Rot (Qualitative)

**Context rot** measures how a model's answer accuracy degrades as context length grows — a qualitative complement to the raw-throughput numbers from the standard sweep.

PPB implements a semantic [Needle-in-a-Haystack (NIAH)](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) evaluation inspired by [NVIDIA RULER](https://github.com/NVIDIA/RULER):

1. **Haystack** — real ShareGPT conversations are concatenated to build distractors of target token lengths `[4096, 8192, 16384, 32768, 65536, 131072]`.
2. **Needle** — a synthetic factual statement not present in any ShareGPT conversation (e.g. `"The secret launch code for Project Nightingale is ALPHA-7-DELTA."`) is inserted at five depth positions: 10 %, 30 %, 50 %, 70 %, 90 %.
3. **Query** — `temperature=0, max_tokens=20`, exact-match scoring.
4. **30 test cases** (6 lengths × 5 depths) per model/quantisation.

#### Prerequisites

Install a GPU-enabled `llama-cpp-python` — see the [context-rot installation note above](#context-rot-optional).

#### Enabling context-rot in a suite

Add a `[qualitative]` section to any suite TOML:

```toml
[qualitative]
context_rot_enabled = true

haystack_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
depths_pct       = [10, 30, 50, 70, 90]

needle_text   = "The secret launch code for Project Nightingale is ALPHA-7-DELTA."
needle_query  = "Based only on the text provided, what is the secret launch code for Project Nightingale? Answer with just the code."
needle_answer = "ALPHA-7-DELTA"

n_gpu_layers  = -1   # -1 = offload all layers to GPU
```

A ready-to-copy example is at [`suites/qualitative_example.toml`](suites/qualitative_example.toml).

Run with:

```bash
uv run ppb.py all suites/qualitative_example.toml
```

Haystack lengths that exceed the per-model VRAM cliff cap are skipped automatically and recorded as `null` in the results.

#### Progress output

```text
[context-rot] 4 length(s) × 5 depth(s) = 20 case(s)  [skipped 2 > VRAM cap 32768]
  ✓ [1/20] ctx=4096  depth=10% pass (0.4s)
  ✓ [2/20] ctx=4096  depth=30% pass (0.4s)
  ...
  ✓ [20/20] ctx=32768 depth=90% fail (1.8s)
[context-rot] overall score: 0.750  (by length: 4096=1.00, 8192=0.80, ...)
```

#### Result columns

Context-rot results are appended to the same JSONL file as regular benchmark rows (`runner_type = "context-rot"`) and flattened to three new CSV/HF columns:

| Column                           | Type        | Description                       |
| -------------------------------- | ----------- | --------------------------------- |
| `context_rot_score`              | float       | Mean accuracy across all 30 cases |
| `context_rot_accuracy_by_length` | JSON string | `{"4096": 1.0, "8192": 0.8, …}`   |
| `context_rot_accuracy_by_depth`  | JSON string | `{"10": 0.9, "30": 0.85, …}`      |

#### `[qualitative]` TOML reference

| Key                   | Type         | Default                                     | Description                                                |
| --------------------- | ------------ | ------------------------------------------- | ---------------------------------------------------------- |
| `context_rot_enabled` | bool         | `false`                                     | Master switch — must be `true` to run the phase.           |
| `haystack_lengths`    | list of ints | `[4096, 8192, 16384, 32768, 65536, 131072]` | Target token-length haystacks.                             |
| `depths_pct`          | list of ints | `[10, 30, 50, 70, 90]`                      | Needle insertion depths (percent of haystack).             |
| `needle_text`         | string       | (built-in)                                  | The synthetic fact injected into the haystack.             |
| `needle_query`        | string       | (built-in)                                  | Prompt sent to the model after the haystack.               |
| `needle_answer`       | string       | `"ALPHA-7-DELTA"`                           | Expected answer (exact substring match, case-insensitive). |
| `n_gpu_layers`        | int          | `-1`                                        | GPU layers to offload (`-1` = all).                        |

### 6. View Your Hardware Profile

Quickly check what PPB detects about your system:

```bash
uv run ppb.py hw-info
```

Sample output:

```text
  Hardware Profile
    OS          : Linux 6.8.0  (x86_64)
    Distro      : Ubuntu 24.04
    CPU         : AMD Ryzen 9 7950X  (32 cores)
    RAM         : 63.9 GB
    GPU [0]     : NVIDIA GeForce RTX 5090  31.8 GB VRAM  sm_120  CUDA 13.0  driver 580.126.09  600 W TDP  PCIe 5.0 x16
    Python      : 3.13.2
    llama-bench : version: b5063 (58ab80c3)
```

**Fields collected per GPU (where available):**

| Field                     | Source                                | Why it matters                                                      |
| ------------------------- | ------------------------------------- | ------------------------------------------------------------------- |
| `name`                    | pynvml / nvidia-smi / system_profiler | Model identification                                                |
| `driver`                  | pynvml / nvidia-smi                   | Affects performance and supported features                          |
| `cuda_version`            | pynvml                                | Max CUDA version; determines available kernels (Flash Attn 2, etc.) |
| `compute_capability`      | pynvml / nvidia-smi                   | SM version (e.g. `8.9` = RTX 4090, `12.0` = RTX 5090)               |
| `power_limit_w`           | pynvml                                | TDP cap; enables tokens-per-watt comparisons                        |
| `pcie_gen` / `pcie_width` | pynvml                                | PCIe bandwidth ceiling for large model loading                      |
| `vram_total_gb`           | pynvml / nvidia-smi                   | VRAM capacity                                                       |
| `metal_version` _(macOS)_ | system_profiler                       | Metal API generation                                                |
| `gpu_cores` _(macOS)_     | system_profiler                       | Apple Silicon GPU core count                                        |

This same profile is automatically included in every benchmark record written to `results.jsonl`.

### 5. Export Results

Convert raw JSONL benchmark files into flat CSV or normalized JSONL — ready for spreadsheets, Pandas, or Arrow ingestion:

```bash
# Export to CSV
python ppb.py export --input results/my_run.jsonl --output results/my_run.csv

# Export to flat JSONL (one row per result)
python ppb.py export -i results/my_run.jsonl -o results/my_run_flat.jsonl
```

Nested fields (hardware, per-GPU info, runner-specific metrics) are flattened into top-level columns. `llama-bench` rows with multiple result items are exploded — each item becomes its own row. The original raw record is preserved in a `raw_payload` column.

Output format is inferred from the file extension: `.csv` → CSV, anything else → JSONL.

### 6. Publish to Leaderboard

Flatten your raw JSONL results to a local CSV and optionally upload to the central PPB dataset on Hugging Face:

```bash
# Step 1 — flatten to a local CSV (review in Excel first)
python ppb.py publish results/my_run.jsonl

# Step 2 — when you're happy, upload to the leaderboard
python ppb.py publish results/my_run.jsonl --upload

# Publish all JSONL files at once
python ppb.py publish results/*.jsonl --upload
```

The CSV is written alongside the input file (e.g. `results/my_run.csv`).

Authentication for `--upload` uses (in order):

1. `--token` flag
2. `HF_TOKEN` environment variable (recommended — set in `.env`)
3. Cached `huggingface-cli login` credential

Results land in `data/results_<timestamp>_<uuid>.jsonl` inside the [`poor-paul/ppb-results`](https://huggingface.co/datasets/paulplee/ppb-results) repository.

For a fully automated "shoot and forget" workflow, add a `[publish]` section to your suite TOML and use `ppb all` — see [§3](#3-run-the-full-suite-all).

### 7. Migrate Legacy Results to the Current Schema

PPB extended its result schema to add **model provenance**, **multi-GPU**, **LLM engine**, **workload**, **quality**, and **OS distro** fields. Existing raw `.jsonl` files written before this change already contain all the data needed — only the flat `.csv` exports need to be regenerated.

> **Note:** New runs produce the full schema automatically. This step is only needed for historical result files.

#### New columns

| Column               | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `model_org`          | HF organisation extracted from the model path, e.g. `unsloth`                   |
| `model_repo`         | Full HF `org/repo` string, e.g. `unsloth/Qwen3.5-2B-GGUF`                       |
| `gpu_count`          | Number of GPUs used in the run                                                  |
| `gpu_names`          | Comma-joined list of GPU names (`gpu_name` for single-GPU runs)                 |
| `gpu_total_vram_gb`  | Sum of VRAM across all GPUs in GB                                               |
| `split_mode`         | llama.cpp layer-split strategy (`layer`, `row`, `none`) — `null` for single-GPU |
| `tensor_split`       | Per-GPU VRAM weight string (e.g. `"1,1"`) — `null` for single-GPU runs          |
| `llm_engine_name`    | Inference engine identifier, e.g. `llama.cpp`                                   |
| `llm_engine_version` | Engine version with build hash, e.g. `b5063 (58ab80c3)`                         |
| `os_distro`          | Linux distribution name (e.g. `Ubuntu`), or `macOS` / `Windows`                 |
| `os_distro_version`  | Distribution version string (e.g. `24.04`, `15.5`)                              |
| `task_type`          | Workload category, e.g. `text-generation` (backfilled for legacy runs)          |
| `prompt_dataset`     | Dataset identifier, e.g. `sharegpt-v3` (server runners) or `null` (bench)       |
| `num_prompts`        | Number of prompts sent per run (`null` for `llama-bench`)                       |
| `n_predict`          | Max tokens per prompt (`null` for `llama-bench`)                                |
| `quality_score`      | Output quality metric — reserved for future use, defaults to `null`             |
| `tags`               | Free-form JSON string for arbitrary metadata (e.g. `{"env": "ci"}`)             |

#### Prerequisites

1. **Finish** all active benchmark runs before migrating.
2. **Back up** your results directory (the script refuses to run without this):

```bash
mkdir results/backup
cp results/*.* results/backup/
```

#### Run the migration

```bash
# 1. Dry run — reads every *.jsonl and prints what would change (no files written)
uv run scripts/migrate_schema.py --dry-run

# 2. Apply — overwrites *.csv files in-place with the new schema
uv run scripts/migrate_schema.py
```

The script leaves raw `*.jsonl` source files untouched; only the flat `*.csv` files are overwritten.

Sample dry-run output:

```text
Found 14 JSONL file(s) to migrate.

  Qwen3.5-2B_20260313_1005.jsonl: 128 row(s)
    new fields: {'model_org': 'unsloth', 'model_repo': 'unsloth/Qwen3.5-2B-GGUF',
                 'gpu_count': 1, 'gpu_names': 'NVIDIA GeForce RTX 4060 Ti',
                 'gpu_total_vram_gb': 16.0, 'split_mode': None, 'tensor_split': None}
    [dry-run] would write Qwen3.5-2B_20260313_1005.csv
  ...

[dry-run] Migration complete — 3817 total flat rows across 14 file(s).
Re-run without --dry-run to overwrite CSV files.
```

#### Re-publish to Hugging Face

After verifying the regenerated CSVs look correct, back up the existing HF dataset, wipe the old data, and re-upload everything:

```bash
# 1. Back up the existing HF dataset locally before wiping
uv run scripts/download_hf_dataset.py
# Downloads to results/hf-backup/ by default. Custom destination:
# uv run scripts/download_hf_dataset.py --local-dir results/my-backup

# 2. Delete all files in paulplee/ppb-results/data/ via the HF web UI, then:

# 3. Publish all JSONL files in one command (shell glob expands to a list)
uv run ppb.py publish results/*.jsonl --upload

# Or with an explicit token if HF_TOKEN is not set in your environment
uv run ppb.py publish results/*.jsonl --upload --token $HF_TOKEN
```

PPB batches all matching files into a single upload session.

##### `download_hf_dataset.py` options

| Flag          | Default                | Description                                                                       |
| ------------- | ---------------------- | --------------------------------------------------------------------------------- |
| `--repo-id`   | `paulplee/ppb-results` | HF dataset repo to download                                                       |
| `--local-dir` | `results/hf-backup`    | Local destination directory                                                       |
| `--token`     | _(env / cached)_       | HF API token (falls back to `HF_TOKEN` env var or cached `huggingface-cli login`) |

#### Cleanup

Once Hugging Face is verified:

```bash
rm scripts/migrate_schema.py
rm -rf results/backup/
```

---

## Concurrent User Benchmarking

PPB can simulate multiple users hitting the server at the same time so you can see how latency degrades under load.

### Option A: `concurrent_users` sweep axis (controlled, repeatable)

Add `concurrent_users` to your sweep config to test exact user counts:

```toml
[sweep]
repo_id      = "unsloth/Qwen3.5-9B-GGUF"
filename     = "Qwen3.5-9B-Q4_K_M.gguf"
n_ctx        = [8192]
n_batch      = [512]
runner_type  = "llama-server"
# Any positive integers are valid — there is no hard upper limit.
concurrent_users = [1, 2, 4, 8, 16, 32]

[sweep.runner_params]
n_predict = 256
prompt_distribution = "shared"   # all users get the same prompt (default)
# prompt_distribution = "split"  # each user gets a different prompt
```

Or from the CLI:

```bash
uv run ppb.py sweep \
    --repo-id unsloth/Qwen3.5-9B-GGUF --filename "*Q4*.gguf" \
    --n-ctx 8192 --n-batch 512 \
    --runner llama-server --concurrent-users 1,2,4,8,16,32
```

Each JSONL record at `concurrent_users > 1` includes extra fields:

| Field                    | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| `concurrent_users`       | Number of simulated parallel users                         |
| `ttft_per_user_ms`       | List of TTFT values — one per user                         |
| `itl_per_user_ms`        | List of median ITL values — one per user                   |
| `queue_time_per_user_ms` | Time each user waited before receiving the first SSE event |
| `aggregate_tok_s`        | Combined token throughput across all users                 |

### Option B: `llama-server-loadtest` runner (auto-discovery)

The load-test runner **automatically escalates** concurrency until it finds the breaking point:

```toml
[sweep]
repo_id      = "unsloth/Qwen3.5-9B-GGUF"
filename     = "Qwen3.5-9B-Q4_K_M.gguf"
n_ctx        = [8192]
n_batch      = [512]
runner_type  = "llama-server-loadtest"

[sweep.runner_params]
max_users        = 64    # stop escalation at this level
error_threshold  = 0.1   # >10% errors = failure
ramp_delay_s     = 2.0   # pause between levels
# user_steps     = [1, 2, 4, 8, 16, 32, 64]  # custom; default = powers of 2
```

Results include:

```json
{
  "max_sustainable_users": 8,
  "error_threshold": 0.1,
  "concurrency_curve": [
    {"concurrent_users": 1, "error_rate": 0.0, "ttft_p50_ms": 42, ...},
    {"concurrent_users": 2, "error_rate": 0.0, "ttft_p50_ms": 65, ...},
    {"concurrent_users": 4, "error_rate": 0.0, "ttft_p50_ms": 120, ...},
    {"concurrent_users": 8, "error_rate": 0.05, "ttft_p50_ms": 310, ...},
    {"concurrent_users": 16, "error_rate": 0.25}
  ]
}
```

## Runner Plugins

PPB uses a **pluggable runner architecture** so new benchmark backends can be added without modifying core code.

### Built-in runners

| `runner_type`           | Module                             | Description                                                                                                                                                                                              |
| ----------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llama-bench`           | `runners/llama_bench.py`           | Default. Wraps llama.cpp's `llama-bench` CLI via subprocess. Measures raw throughput (tok/s). Supports OOM probing for `vram-cliff`.                                                                     |
| `llama-server`          | `runners/llama_server.py`          | Starts `llama-server` as a subprocess, streams real ShareGPT conversational prompts via SSE, and records **TTFT** and **ITL** latency metrics. Supports `vram-cliff` and **concurrent user simulation**. |
| `llama-server-loadtest` | `runners/llama_server_loadtest.py` | Escalates concurrent users (1 → 2 → 4 → …) againt a single server instance. Reports the max sustainable user count and a full **concurrency curve** with TTFT/ITL/queue metrics at each level.           |

### Creating a custom runner

1. Create a new file (e.g. `runners/my_runner.py`).
2. Subclass `BaseRunner` from `runners.base` and implement `setup()`, `run()`, and `teardown()`.
3. Optionally override `probe_ctx()` if your backend supports OOM probing.
4. Register it in `runners/__init__.py`:

```python
from .my_runner import MyRunner
register_runner("my-runner", MyRunner)
```

1. Use it in your sweep config:

```toml
[sweep]
runner_type = "my-runner"
repo_id     = "my-org/my-model-GGUF"
filename    = "model.gguf"
n_ctx       = [8192]
n_batch     = [512]

[sweep.runner_params]
custom_option = "value"
```

### Runner contract

| Method                                | Required | Description                                                                                                                                                              |
| ------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `setup(runner_params)`                | ✅       | Called once before the sweep. Receives `[sweep.runner_params]` from TOML.                                                                                                |
| `run(config) → dict \| None`          | ✅       | Execute one benchmark. `config` always has `"model_path"`. Return `{"results": ...}` or `None` on failure. Must NOT write files — the orchestrator handles JSONL output. |
| `teardown()`                          | ✅       | Called once after the sweep (guaranteed via `try/finally`).                                                                                                              |
| `probe_ctx(model_path, n_ctx) → bool` | Optional | Override to support `vram-cliff`. Default raises `NotImplementedError`. When returning `False`, set `self.last_probe_error` to the stderr/error message for diagnostics. |

## Contributing to the Leaderboard

We are crowdsourcing a definitive database of Tokens-per-Second and Tokens-per-Watt across consumer hardware.

The fastest way to contribute:

```bash
# One command — benchmark + publish
python ppb.py all suites/my_gpu.toml   # add [publish] section to auto-upload
```

Or manually:

1. Run a benchmark sweep: `python ppb.py all suites/my_gpu.toml`.
2. Publish your results: `python ppb.py publish results/my_run.jsonl --upload`.

You can also open an Issue titled "Benchmark Submission: [Your Hardware]" and attach your results file.

The public leaderboard is hosted at: [poorpaul.dev](https://poorpaul.dev)

## About the Maintainers

Poor Paul's Benchmark is an open-source project maintained by the team at [Ximplar](https://ximplar.com).

While PPB is built for the community to test and stretch consumer hardware to its limits, Ximplar specializes in taking those insights and deploying cost-effective, high-ROI AI models for enterprise environments. If you need help architecting your company's local AI infrastructure, reach out to us.
