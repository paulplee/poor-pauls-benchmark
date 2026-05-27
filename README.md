# Poor Paul's Benchmark

> Empirical LLM inference benchmarks for the GPU you actually own.

[![Dataset](https://img.shields.io/badge/🤗%20dataset-paulplee%2Fppb--results-blue)](https://huggingface.co/datasets/paulplee/ppb-results)
[![Analytics](https://img.shields.io/badge/analytics-poorpaul.dev-teal)](https://poorpaul.dev)
[![MCP Server](https://img.shields.io/badge/MCP-mcp.poorpaul.dev-green)](https://github.com/paulplee/ppb-mcp)
[![CI](https://img.shields.io/github/actions/workflow/status/paulplee/poor-pauls-benchmark/ci.yml?branch=main&label=CI)](https://github.com/paulplee/poor-pauls-benchmark/actions)
[![License](https://img.shields.io/github/license/paulplee/poor-pauls-benchmark)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)

Find the absolute limit of your local AI hardware without the enterprise budget.

AI-driven GPU and RAM prices got you down? Same here. Poor Paul's Benchmark exists to answer a simple question with real data: what is the most efficient hardware, model, and quantization setup for the budget you actually have?

Poor Paul's Benchmark is an automated evaluation framework for local LLM inference. It combines throughput, latency, VRAM headroom, long-context recall, tool-call accuracy, answer quality, and multi-turn memory into one open benchmarking workflow.

The goal is simple: build the most useful public benchmark for cost-effective local AI — for hobbyists, homelabbers, prosumers, and small teams who need honest numbers, not marketing claims.

No cloud credits. No black boxes. No benchmark theatre. Just your GPU, real models, and honest numbers.

---

## The Platform

PPB is the benchmark-runtime layer of a four-layer open platform. Every result you contribute flows through the full stack automatically.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4 — Human interface     poorpaul.dev                     │
│            Analytics, leaderboard, blog, editorial              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — LLM interface       mcp.poorpaul.dev                 │
│            Any MCP client can query benchmark data directly      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — Data                huggingface.co/datasets/         │
│            paulplee/ppb-results  (Parquet, versioned, public)   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 — Benchmark runtime   poor-pauls-benchmark  ← here    │
│            Contributor-run; results pushed to Layer 2           │
└─────────────────────────────────────────────────────────────────┘
```

Once you push results, any MCP-compatible AI client (Claude Desktop, Cursor, Windsurf, etc.) connected to `mcp.poorpaul.dev` can answer questions like:

> _"What's the best quantization for an RTX 4090 running 4 concurrent users?"_

using real community benchmark data — including yours.

---

## What PPB Measures

PPB produces two complementary result types for each model and hardware configuration.

**Performance benchmarks** measure raw hardware capability: how fast the model runs, how much VRAM it uses, and how many concurrent users it can handle.

**Quality benchmarks** measure what the model actually does with that compute: whether it retrieves facts accurately from long contexts, calls tools correctly, produces reliable answers, and maintains coherence across a conversation.

| Benchmark type  | What it covers                                                   | Key metrics                                                                                                           |
| --------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Performance** | VRAM limits, throughput, latency, concurrency                    | `tokens_per_sec`, `vram_used_mb`, `ttft_ms`, `itl_ms`, `max_concurrent_users`                                         |
| **Quality**     | Context recall, tool accuracy, answer quality, multi-turn memory | `context_rot_score`, `overall_tool_accuracy`, `knowledge_accuracy_mean`, `quality_composite_score`, `memory_accuracy` |

Performance and quality runs are **independent and composable**: run a throughput sweep today, add quality scoring next week. Results are joined in the dataset by `(gpu_name, model_name, quantization, run_type)`.

---

## Quick Start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), `llama-bench` and/or `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) on your `$PATH`, and a CUDA- or Metal-capable GPU.

```bash
# 1. Clone and install
git clone https://github.com/paulplee/poor-pauls-benchmark
cd poor-pauls-benchmark
uv sync

# 2. Copy the starter suite and edit for your GPU
cp suites/suite.example.toml suites/my_gpu.toml
# → set repo_id, filename, n_ctx, n_batch

# 3. Run
uv run ppb.py quantitative suites/my_gpu.toml
```

Results are written to `results/results.jsonl`. See [Publishing Results](#publishing-results) to contribute them to the shared dataset.

> **llama.cpp version:** Build `b8688` or later is required. See [docs/building-llama-cpp.md](docs/building-llama-cpp.md) for step-by-step build instructions for CUDA, ROCm, Metal, and CPU.

---

## Installation

### Core (performance benchmarking)

```bash
uv sync
```

### Quality benchmarking

All quality evaluations run local models via `llama-cpp-python`, which requires a GPU-accelerated build. A plain `pip install llama-cpp-python` gives a CPU-only binary that is impractically slow for long-context evaluation.

Install the GPU-accelerated build once before running quality benchmarks:

```bash
# CUDA — x86_64 Linux / Windows: pre-built wheel (fastest path)
uv pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# CUDA — x86_64 Linux / Windows: build from source
CMAKE_ARGS="-DGGML_CUDA=ON" uv pip install "llama-cpp-python>=0.3.0"

# CUDA — ARM64/aarch64 (Grace Blackwell GB10, Jetson, etc.)
# No CUDA-enabled wheels exist for aarch64; PyPI has a CPU-only aarch64 wheel.
# --no-binary forces a source build. CMAKE_CUDA_ARCHITECTURES=native auto-detects the GPU.
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native" \
  uv pip install "llama-cpp-python>=0.3.0" --no-binary llama-cpp-python

# Metal (macOS Apple Silicon)
CMAKE_ARGS="-DGGML_METAL=on" uv pip install "llama-cpp-python>=0.3.0"

# Or compile via the pyproject extras group
uv pip install -e ".[qualitative]"
```

> **Note — ARM64 / aarch64 machines (including Lenovo ThinkStation PGX GB10, Jetson AGX Orin, etc.):**
> The pre-built wheel index at `abetlen.github.io/llama-cpp-python/whl/cu124` only publishes
> `linux_x86_64` and `win_amd64` wheels. Passing `--extra-index-url` on an aarch64 host will
> fail with a "no matching platform tag" error. You must build from source as shown above.
> The `native` architecture flag auto-detects your GPU's compute capability at compile time,
> so you do not need to look it up manually.

Pre-built wheels for common targets: [github.com/abetlen/llama-cpp-python/releases](https://github.com/abetlen/llama-cpp-python/releases)

### Verifying the llama-cpp-python build

Run this one-liner after installing. No model file needed.

```bash
uv run python - <<'EOF'
from llama_cpp import llama_supports_gpu_offload, llama_backend_init, llama_backend_free
from llama_cpp.llama_cpp import llama_print_system_info
llama_backend_init()
print(llama_print_system_info().decode())
gpu = llama_supports_gpu_offload()
print("GPU offload:", "OK ✓" if gpu else "FAIL — CPU-only build (qualitative benchmarks will be unusably slow)")
llama_backend_free()
EOF
```

**Healthy CUDA output** — look for the `ggml_cuda_init` header and a `CUDA :` line in the system info:

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 122502 MiB):
  Device 0: NVIDIA GB10, compute capability 12.1, VMM: yes, VRAM: 122502 MiB
CUDA : ARCHS = 1210 | USE_GRAPHS = 1 | ...
GPU offload: OK ✓
```

**If `GPU offload: FAIL` or no `CUDA :` line appears**, the wheel was built without GPU support. Reinstall with the source-build command for your platform from the section above, making sure to pass `--no-binary llama-cpp-python` so uv doesn't silently reuse the CPU-only wheel from PyPI:

```bash
# Remove the CPU-only install first
uv pip uninstall llama-cpp-python

# Then reinstall from source (substitute the correct CMAKE_ARGS for your platform)
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native" \
  uv pip install "llama-cpp-python>=0.3.0" --no-binary llama-cpp-python
```

**If the GPU isn't visible at all**, check the OS level first:

```bash
nvidia-smi                    # GPU visible and driver loaded?
nvcc --version                # CUDA toolkit installed?
echo $CUDA_HOME               # CUDA_HOME set? (needed for source builds)
```

### Building with Flash Attention (recommended for Gemma 4 and other hybrid-attention models)

Some models — notably **Gemma 4** and other hybrid sparse-attention architectures — have variable-sized V embeddings across transformer layers. Without Flash Attention, llama.cpp pads the entire V cache to the maximum layer size, which inflates VRAM usage significantly at larger context windows. This can trigger a native `ggml_cuda_pool_vmm::alloc` crash (SIGABRT from the CUDA VMM allocator) that Python cannot catch, killing the whole benchmark process mid-run.

Adding `-DGGML_CUDA_FA=on` at build time enables the CUDA Flash Attention kernel, which eliminates the padding requirement and reduces peak VRAM substantially. There is no pre-built wheel with FA enabled, so you must compile from source:

```bash
# Uninstall the existing wheel first to prevent uv from reusing it
uv pip uninstall llama-cpp-python

# Build from source with CUDA + Flash Attention
# --no-binary prevents uv from silently falling back to the CPU-only PyPI wheel
CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_CUDA_FA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
  uv pip install "llama-cpp-python>=0.3.0" \
    --no-binary llama-cpp-python \
    --no-build-isolation
```

> `--no-build-isolation` is required because the vendored llama.cpp sources need the CUDA toolkit visible to the compiler. If CMake can't find `nvcc`, prepend `PATH=/usr/local/cuda/bin:$PATH` to the command. Build time is approximately 10–15 minutes.

**Verify Flash Attention is active** by loading a model with `flash_attn=True` and checking the startup log:

```bash
uv run python - <<'EOF'
from llama_cpp import Llama
llm = Llama(
    model_path="/path/to/model.gguf",
    n_ctx=8192, n_gpu_layers=-1, flash_attn=True, verbose=True
)
# Look for: "llama_kv_cache: using Flash Attention"
# Absence of: "padding V cache to 2048" warns that FA is not active
EOF
```

A successful FA build replaces the warning line:
```
# Without FA (bad):
llama_kv_cache: the V embeddings have different sizes across layers and FA is
not enabled - padding V cache to 2048

# With FA (good):
llama_kv_cache: using Flash Attention
```

Once the Flash Attention build is confirmed, restore full context-rot coverage in your suite TOML by uncommenting or removing the `haystack_lengths` cap:

```toml
[qualitative]
context_rot_enabled = true
# haystack_lengths defaults to [4096, 8192, 16384, 32768, 65536, 131072]
```

---

## Run Modes

```bash
# Performance benchmarks only
uv run ppb.py quantitative suites/my_gpu.toml

# Quality benchmarks only
# (auto-reads your VRAM limit from an existing result row to skip OOM cases)
uv run ppb.py qualitative suites/my_gpu.toml

# Everything together — most efficient; judge model loaded only once
uv run ppb.py all suites/my_gpu.toml
```

---

## Performance Benchmarking

### VRAM Limit Discovery

Before running a sweep, PPB automatically discovers the maximum context window (`n_ctx`) your hardware supports before hitting an Out-of-Memory error, using a binary search. Run it once per model/quantization and the result is reused:

```bash
uv run ppb.py vram-cliff suites/my_gpu.toml
```

A VRAM pre-flight check also runs automatically before any sweep, reading GGUF metadata to estimate worst-case memory usage across your parameter matrix. If any combination would likely OOM, you are offered three options: **a**uto-cap `n_ctx`, **p**roceed anyway, or **q**uit.

### Performance Sweep

Runs the full `models × n_ctx × n_batch × concurrent_users` Cartesian product, automatically skipping any combination that exceeds the VRAM limit. Three runner backends are available:

| Runner                    | Measures               | Best for                                     |
| ------------------------- | ---------------------- | -------------------------------------------- |
| `llama-bench` _(default)_ | Raw throughput (tok/s) | Peak hardware performance, quant comparisons |
| `llama-server`            | TTFT and ITL latency   | UX-relevant interactive / chat benchmarks    |
| `llama-server-loadtest`   | Max concurrent users   | Capacity planning for multi-user deployments |

Set `runner_type` in your suite TOML to switch backends.

---

## Quality Benchmarking

All quality evaluations require a GPU-accelerated `llama-cpp-python` build. The **long-context recall** and **answer quality** evaluations use [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) conversations as their prompt source — real human exchanges that reflect how models are actually used.

### Long-Context Recall

Tests whether a model can retrieve specific facts from long contexts — and whether it gets _confused_ by them. PPB plants 15 diverse factual needles (covering alphanumeric codes, dates, currency amounts, chemical formulae, place names, and more) into ShareGPT haystacks at 6 context lengths × 5 depth positions (30 cells total). Needle selection is deterministic per model while rotating across all 15 needles so no single fact type dominates the score.

Enable **multi-needle mode** (`multi_needle_enabled = true`) to measure context confusion specifically: whether the model can correctly reason across 2–3 simultaneously planted facts, not just retrieve a single phrase.

### Tool-Call Accuracy

Dual-track evaluation using established benchmarks and PPB-native cases:

- **[BFCL v3](https://gorilla.cs.berkeley.edu/leaderboard.html)** (Berkeley Function Calling Leaderboard) — four single-turn splits (`simple`, `multiple`, `parallel`, `irrelevance`) covering straightforward calls, multi-candidate tool selection, parallel/batched requests, and cases where the model must correctly _decline_ to call any tool. The irrelevance split directly measures hallucination avoidance.

- **PPB-native ground truth** — 100 cases covering all four `ppb-mcp` tools (`recommend_quantization`, `query_ppb_results`, `get_gpu_headroom`, `list_tested_configs`) with realistic hobbyist-AI prompts.

Scores: `tool_selection_accuracy`, `parameter_accuracy`, `parameter_hallucination_rate`, `no_call_accuracy` (from BFCL irrelevance), `parse_success_rate`, and `overall_tool_accuracy` (geometric mean of selection and parameter accuracy).

### Answer Quality

A judge model scores 50 responses sampled from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) across three dimensions:

- **Knowledge accuracy** — fraction of factual claims the judge considers consistent with common knowledge. Note: this measures claim plausibility against the judge's parametric knowledge, not RAGAS-style reference-grounded faithfulness. See [docs/qualitative-methodology.md](docs/qualitative-methodology.md) for the distinction and rationale.
- **Answer relevancy** — how directly the response addresses the question
- **Coherence** — logical flow and internal consistency

A `quality_composite_score` combines all three. The 50-prompt evaluation set is frozen by a SHA-256 hash stored in every published row so results remain comparable across models.

### Multi-Turn Memory

Two modes, one selected per suite run:

- **LongMemEval** _(default)_ — 50 questions from the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark (ICLR 2025) embedded in conversation histories up to 500K tokens. Tests recall, temporal reasoning, and knowledge-update handling across turns.
- **MT-Bench** — all 80 canonical MT-Bench questions (bundled at `ppb_datasets/data/mt_bench_questions.json`, MIT-licensed). Fast signal for development cycles. Score is on the standard 1–10 MT-Bench scale.

Context-length cases that would exceed the model's VRAM limit are automatically skipped and reported in `cases_skipped_context`.

### Judge Model Setup

Answer quality and multi-turn evaluation both use a separate judge GGUF. **The judge must be a different model from the one under test** — PPB enforces this with a runtime path comparison. Self-grading inflates scores and defeats the purpose.

Recommended: a small, well-aligned 3–7B model such as `Qwen3.5-4B-Q4_K_M` or `Llama-3-8B-Instruct-Q4_K_M`. In `ppb all` mode the judge is loaded once and shared across both evaluations.

```toml
[qualitative]
judge_model_path = "/path/to/judge-model-Q4_K_M.gguf"
```

---

## Suite Configuration

All benchmark parameters live in a TOML file. Copy `suites/suite.example.toml` or `suites/qualitative_example.toml` as your starting point.

### Minimal performance suite

```toml
# suites/my_gpu.toml
repo_id  = "unsloth/Qwen3.5-8B-GGUF"
filename = "Qwen3.5-8B-Q4_K_M.gguf"

[vram-cliff]
min_ctx = 2048
max_ctx = 131072

[sweep]
n_ctx    = [8192, 32768, 65536]
n_batch  = [512, 1024]

[publish]
hf_dataset = "paulplee/ppb-results"
# hf_token = "hf_..."  # or export HF_TOKEN
```

### Adding quality benchmarks

```toml
[qualitative]
context_rot_enabled    = true
tool_accuracy_enabled  = true
answer_quality_enabled = true
multiturn_enabled      = true
multiturn_mode         = "longmemeval"   # or "mt_bench"
multi_needle_enabled   = false           # true to test context confusion

judge_model_path = "/path/to/judge-model-Q4_K_M.gguf"

# Optional
needle_seed  = 20260426   # deterministic needle selection per model
sample_size  = 50         # LongMemEval cases to evaluate
```

### Key sweep parameters

| Key                | Type     | Default         | Description                                               |
| ------------------ | -------- | --------------- | --------------------------------------------------------- |
| `repo_id`          | string   | —               | HuggingFace repo, e.g. `unsloth/Qwen3.5-8B-GGUF`          |
| `filename`         | string   | —               | GGUF filename or glob (`"*Q4*.gguf"`)                     |
| `models_dir`       | string   | `"./models"`    | Local model cache directory                               |
| `n_ctx`            | int list | —               | Context lengths to test                                   |
| `n_batch`          | int list | —               | Batch sizes (`llama-bench`)                               |
| `concurrent_users` | int list | `[1]`           | Parallel users (`llama-server` runners)                   |
| `runner_type`      | string   | `"llama-bench"` | `llama-bench`, `llama-server`, or `llama-server-loadtest` |

Full configuration reference: see `suites/suite.example.toml`.

---

## Results Schema

Results are written to `results/results.jsonl`. Every row carries a nested `qualitative` JSON column; evaluations that did not run carry `null` for their keys.

### Quality metrics

| Key                               | Evaluation                              | Type               |
| --------------------------------- | --------------------------------------- | ------------------ |
| `context_rot_score`               | Long-Context Recall                     | float              |
| `context_rot_accuracy_by_length`  | Long-Context Recall                     | object             |
| `context_rot_accuracy_by_depth`   | Long-Context Recall                     | object             |
| `context_rot_accuracy_by_needle`  | Long-Context Recall                     | object \| null     |
| `multi_needle_score`              | Long-Context Recall (multi-needle mode) | float \| null      |
| `multi_needle_accuracy_by_length` | Long-Context Recall (multi-needle mode) | object \| null     |
| `tool_selection_accuracy`         | Tool-Call Accuracy                      | float              |
| `parameter_accuracy`              | Tool-Call Accuracy                      | float              |
| `parameter_hallucination_rate`    | Tool-Call Accuracy                      | float              |
| `no_call_accuracy`                | Tool-Call Accuracy (irrelevance split)  | float \| null      |
| `parse_success_rate`              | Tool-Call Accuracy                      | float              |
| `overall_tool_accuracy`           | Tool-Call Accuracy                      | float              |
| `knowledge_accuracy_mean`         | Answer Quality                          | float              |
| `knowledge_accuracy_std`          | Answer Quality                          | float              |
| `answer_relevancy_mean`           | Answer Quality                          | float              |
| `coherence_mean`                  | Answer Quality                          | float              |
| `quality_composite_score`         | Answer Quality                          | float              |
| `memory_accuracy`                 | Multi-Turn Memory (LongMemEval)         | float \| null      |
| `mt_bench_score`                  | Multi-Turn Memory (MT-Bench)            | float 1–10 \| null |
| `cases_evaluated`                 | Multi-Turn Memory                       | int                |
| `cases_skipped_context`           | Multi-Turn Memory                       | int                |

> `multi_needle_score` and `multi_needle_accuracy_by_length` are `null` unless `multi_needle_enabled = true` is set.
>
> `mt_bench_score` uses the 1–10 MT-Bench scale. All other float metrics are 0–1. To normalise: `mt_bench_score_norm = (mt_bench_score − 1) / 9`.

Full schema documentation: [docs/schema.md](docs/schema.md)

---

## Publishing Results

Every result you submit makes the shared dataset more useful — and more accurate for every LLM querying it via ppb-mcp.

### 1. Get a HuggingFace write token

[huggingface.co](https://huggingface.co) → Settings → Access Tokens → create a token with **Write** access.

```bash
export HF_TOKEN=hf_your_token_here
```

### 2. Add a publish block to your suite

```toml
[publish]
hf_dataset = "paulplee/ppb-results"
```

### 3. Run

```bash
uv run ppb.py all suites/my_gpu.toml
```

Results are pushed to HuggingFace after each model completes — so a multi-model run is never lost to a late crash.

### How data flows from HuggingFace to the MCP and website

Understanding the propagation path explains why a freshly pushed row doesn't appear instantly everywhere:

```
Your machine → HuggingFace (paulplee/ppb-results)
                       │
                       ├──► mcp.poorpaul.dev  (pulls on demand, see below)
                       │
                       └──► poorpaul.dev      (pulls at deploy time, see below)
```

**MCP (`mcp.poorpaul.dev`)** — the live REST API pulls data from HuggingFace automatically, but **only when the dataset's git commit SHA has changed** since its last sync. It checks the SHA on every request cycle; once a change is detected it downloads only the new/changed shards and upserts them into its local SQLite cache. The default refresh interval is **1 hour** (controlled by the `REFRESH_INTERVAL_HOURS` environment variable on the server). In practice this means new results appear in the MCP within an hour of being pushed to HuggingFace. To force an immediate refresh, restart the Docker container (`docker compose restart` on the server runs `update.sh`).

**Website (`poorpaul.dev`)** — the site is a **static Next.js export**. The data files it serves (including `/data/results.json`, the file the `/insights` page reads for its charts) are baked in at **deploy time** by running `npx tsx scripts/fetch-data.ts` followed by `npm run build`. This means new HuggingFace rows do **not** appear on the website until a new deployment is triggered. Deployments happen automatically on every push to the `main` branch of `poorpaul-dev` via GitHub Actions, or can be triggered manually via `workflow_dispatch`. If you push benchmark results to HuggingFace but no site deployment follows, the insights page will be stale.

**In short:** after pushing results to HuggingFace, wait up to 1 hour for the MCP to pick them up. For the website, a new deployment is required — either merge a change to `poorpaul-dev/main` or trigger the workflow manually.

---

## Auditing Published Coverage

`scripts/audit_published_coverage.py` is an end-to-end correctness check that verifies every `(model, GPU)` pair in the HuggingFace dataset is also visible in the MCP REST API and the website's static snapshot. Run it whenever you suspect results are missing from the insights page.

### What it checks

1. **HuggingFace → source of truth.** Downloads the full dataset Parquet and enumerates every `(model_base, gpu_name)` pair, noting how many quantitative and qualitative rows each has.
2. **MCP REST API.** Queries `https://mcp.poorpaul.dev/api/v1/results` and `/qualitative` for each pair and flags any that return zero rows when the dataset has data.
3. **Static snapshot.** Fetches `https://poorpaul.dev/data/results.json` (the file the `/insights` page reads) and flags any pair missing from it — the most common cause of empty charts on the site.
4. **Headless browser (optional).** Drives Chromium against `/insights?gpu=…&model=…`, clicks each qualitative tab (Context Rot / Tools / Quality / Multi-Turn), and flags tabs that render no charts.

### Running the audit

```bash
# Fast smoke test — 5 random pairs, no browser (good for a quick sanity check)
uv run python scripts/audit_published_coverage.py --limit-pairs 5

# Full audit of all pairs (takes ~3 minutes at the default 0.9 req/s throttle)
uv run python scripts/audit_published_coverage.py

# Also drive a headless browser against /insights for a sample of pairs
# (requires playwright: uv pip install playwright && playwright install chromium)
uv run python scripts/audit_published_coverage.py --with-browser

# Point at a local dev MCP/site instead of production
uv run python scripts/audit_published_coverage.py \
    --mcp-base http://localhost:8000/api/v1 \
    --site-base http://localhost:3000
```

The report is written to `audit_report.md` (override with `--report PATH`). The script exits with code 0 when everything is clean, 1 when findings are reported, and 2 on a fatal error — suitable for CI.

### Reading the report

Findings are grouped by surface and sorted by severity:

| Surface | What it means |
| --- | --- |
| `mcp-results` | A pair has quantitative rows in HF but none returned by the MCP `/results` endpoint. The MCP cache may be stale — wait up to 1 hour or restart the server. |
| `mcp-qualitative` | A pair has qualitative rows in HF but none returned by `/qualitative`. Same cause as above; qualitative rows that were recently pushed are the most common culprit. |
| `static-snapshot` | A pair is absent from `/data/results.json`. The website needs a new deployment (`npm run build` + deploy, or push any change to `poorpaul-dev/main` to trigger GitHub Actions). |
| `insights-page` | The `/insights` page loaded but rendered no charts for that pair. Usually indicates the static snapshot is stale (same fix) or a chart component has a filtering bug. |

A `severity: error` row means the request itself failed (e.g. HTTP 429 from rate limiting or a network error); a `severity: missing` row means the request succeeded but returned no data.

### Unit tests

The pure-function logic (pair enumeration, snapshot indexing, report rendering) is covered by offline unit tests:

```bash
uv run pytest tests/test_audit_published_coverage.py -v
```

---

## Contributing

The benchmark is only as good as the hardware coverage. If your GPU isn't in the dataset, the community is flying blind for your tier.

### Run a benchmark and push results

```bash
git clone https://github.com/your-username/poor-pauls-benchmark
cd poor-pauls-benchmark
uv sync
cp suites/suite.example.toml suites/rtx-4070-ti.toml
# edit: repo_id, filename, n_ctx, n_batch, [publish] block
HF_TOKEN=hf_... uv run ppb.py all suites/rtx-4070-ti.toml
```

No PR needed to contribute data — results go directly to the shared dataset.

### Extend the PPB-native tool-accuracy ground truth

The ground truth lives in `ppb_mcp_ground_truth.json`. Add cases for new tools, edge cases, or underrepresented prompt styles. Open a PR.

### Add a runner backend

Implement `BaseRunner` from `runners/base.py`, register with `@register_runner("your-backend")`, open a PR. See `runners/llama_bench.py` for the reference implementation.

### Run the test suite

```bash
uv sync --group dev
pytest tests/ -v

# Quality unit tests only (no GPU required)
pytest tests/test_qualitative_fixes.py tests/test_context_rot.py \
       tests/test_tool_accuracy.py tests/test_multiturn.py -v
```

---

## Project Structure

<details>
<summary>Expand file tree</summary>

```
ppb.py                        # CLI entry point (Typer app)
ppb_context_rot.py            # Long-Context Recall (Semantic NIAH)
ppb_tool_accuracy.py          # Tool-Call Accuracy (BFCL v3: simple/multiple/parallel/irrelevance + PPB-native)
ppb_answer_quality.py         # Answer Quality (judge-model pipeline)
ppb_multiturn.py              # Multi-Turn Memory (LongMemEval / MT-Bench)
ppb_mcp_ground_truth.json     # 100 PPB-native tool-call evaluation cases
ppb_quality_prompts_cache.json  # Frozen 50-prompt evaluation set (auto-generated)
runners/
  base.py                     # BaseRunner ABC
  llama_bench.py              # llama-bench runner (throughput)
  llama_server.py             # llama-server runner (TTFT / ITL latency)
  llama_server_loadtest.py    # Load-test runner (max concurrency)
ppb_datasets/
  sharegpt.py                 # ShareGPT download + prompt extraction
  data/
    mt_bench_questions.json   # 80 canonical MT-Bench questions (MIT licence)
utils/
  flattener.py                # Normalise nested JSONL → flat Arrow-friendly dicts (schema v0.9.0)
  gguf_metadata.py            # Read GGUF headers for VRAM estimation
  publisher.py                # Push results to HuggingFace
scripts/
  aggregate_results.py        # Group Parquet rows by key dimensions; compute mean/CI95; optional HF upload
  audit_published_coverage.py # End-to-end coverage audit: HF dataset → MCP → website snapshot → browser
suites/
  suite.example.toml          # Starter performance suite
  qualitative_example.toml    # Starter quality suite
docs/
  schema.md                   # Full results schema reference (v0.9.0)
  qualitative-methodology.md  # Knowledge accuracy vs. reference-grounded faithfulness
  building-llama-cpp.md       # Build llama.cpp for CUDA / Metal / ROCm
tests/                        # pytest suite (no GPU required for unit tests)
results/                      # Benchmark output — gitignored
models/                       # Downloaded GGUF cache — gitignored
```

</details>

---

## Principles

**Radical honesty.** Benchmarks run on real hardware under real conditions. Results are published unfiltered — no cherry-picking, no vendor-sponsored averages.

**Practitioner empathy.** PPB is designed by and for people running on limited GPU budgets. Every decision — from automatic VRAM limit discovery to pre-flight safety checks — exists because hardware is expensive and crashes waste time.

**Open by default.** Code, data, and methodology are all public. The brand earns trust through transparency, not access restrictions.

**Compounding contribution.** Every benchmark row, every ground-truth case, every runner plugin makes the resource more useful for the next person. The flywheel only spins if people contribute.

---

## Ecosystem

| Project                                                                          | What it does                                           |
| -------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **poor-pauls-benchmark** _(this repo)_                                           | Benchmark runner — contribute results here             |
| **[paulplee/ppb-results](https://huggingface.co/datasets/paulplee/ppb-results)** | Public HuggingFace dataset — canonical source of truth |
| **[ppb-mcp](https://github.com/paulplee/ppb-mcp)**                               | MCP server — lets LLMs query the dataset directly      |
| **[poorpaul.dev](https://poorpaul.dev)**                                         | Visual analytics, leaderboard, and editorial           |

---

## License

[MIT](LICENSE) — code and tooling.

Benchmark results contributed to `paulplee/ppb-results` are published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Contributions remain attributed to their authors.

BFCL v3 evaluation data © UC Berkeley, used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
MT-Bench questions © LMSYS, used under [MIT licence](https://github.com/lm-sys/FastChat/blob/main/LICENSE).
ShareGPT dataset used under its [original licence](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
