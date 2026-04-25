# Contributing to Poor Paul's Benchmark

Thank you for running PPB on your hardware. Every result strengthens the
community leaderboard and makes the dataset more useful for everyone.

## The Contribution Loop

```
Clone → Configure suite → Run benchmarks → Publish results → See them on poorpaul.dev
```

## Step-by-Step Guide

### 1. Fork and clone

```bash
git clone https://github.com/paulplee/poor-pauls-benchmark.git
cd poor-pauls-benchmark
uv sync
```

### 2. Configure your suite

```bash
cp suites/suite.example.toml suites/my_gpu.toml
# Edit suites/my_gpu.toml with your model and settings
```

Suite files are gitignored (everything except `suite.example.toml`), so your
personal configs will never be accidentally committed.

### 3. Verify your hardware is detected

```bash
uv run ppb.py hw-info
```

Review the output — especially GPU name and VRAM. This metadata is embedded
in every result record.

### 4. Run your benchmarks

```bash
# Recommended: full run (vram-cliff → sweep → publish)
uv run ppb.py all suites/my_gpu.toml

# Or step by step:
uv run ppb.py vram-cliff suites/my_gpu.toml
uv run ppb.py sweep      suites/my_gpu.toml
uv run ppb.py publish    --results results/my_results.jsonl
```

### 5. Publish

Set your HF token (write access to `paulplee/ppb-results`):

```bash
export HF_TOKEN=hf_your_token_here
uv run ppb.py publish --results results/my_results.jsonl
```

Results appear in the [public dataset](https://huggingface.co/datasets/paulplee/ppb-results)
and on [poorpaul.dev/insights](https://poorpaul.dev/insights) within minutes.

## Hardware Metadata Best Practices

PPB auto-detects hardware, but you can improve result quality:

- **Close background apps** before benchmarking — other GPU workloads affect
  throughput measurements.
- **Note your cooling setup** using the `tags` field in your suite:
  ```toml
  tags = '{"cooling": "liquid", "psu": "1000W", "overclock": false}'
  ```
- **Run at least 3 context sizes** in `[sweep]` so the dataset captures
  how your GPU behaves at different KV cache pressures.
- **Include multiple concurrency levels** — `concurrent_users = [1, 2, 4, 8]`
  is the minimum useful set for the ppb-mcp recommendation engine.

## Result Schema

See [docs/schema.md](docs/schema.md) for the full JSONL output schema —
field names, types, and semantics for every column written to `results.jsonl`.

## Reporting Issues

- **Benchmark bugs** (wrong results, crashes): open a GitHub issue with your
  `ppb hw-info` output and the suite TOML you used.
- **Dataset issues**: open an issue on the
  [HuggingFace dataset page](https://huggingface.co/datasets/paulplee/ppb-results/discussions).
- **MCP server issues**: open an issue on
  [ppb-mcp](https://github.com/paulplee/ppb-mcp/issues).

## Code Contributions

1. Open an issue first to discuss significant changes.
2. Run `python -m pytest tests/ -v` before submitting a PR.
3. New runners must inherit from `BaseRunner` (see `runners/base.py`) and
   include tests in `tests/`.
4. New result schema fields must be added to both `utils/flattener.py` and
   `docs/schema.md`.
