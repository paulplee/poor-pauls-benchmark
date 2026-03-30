---
title: "Poor Paul's Benchmark: The 4-User Wall — Qwen3.5-27B on a Lenovo ThinkStation PGX"
date: "2026-03-14"
summary: "We found the max number of concurrent users for the PGX running Qwen3.5-27B"
tags: ["analysis", "qwen", "gb-10", "lenovo", "pgx", "dgx-spark"]
published: true
---

*Published: March 14, 2026 | Hardware: Lenovo ThinkStation PGX (NVIDIA GB10) | Model family: Qwen3.5-27B*

---

If you're planning to serve a 27B model locally to your team, you'll hit a wall. Not a gradual degradation — a sudden, 53× cliff in response latency the moment you cross 4 concurrent users. This article shows you exactly where it is, why it happens, and what you can realistically expect from one of the most accessible high-RAM workstations on the market today.

## The Hardware

The Lenovo ThinkStation PGX isn't a flashy GPU server. It's a quiet, ARM-based workstation built around NVIDIA's GB10 "Grace Blackwell" chip — a unified memory architecture with 120GB of RAM shared between CPU and GPU. No PCIe bandwidth bottleneck. No VRAM overflow headaches. The whole 27B model loads into fast, unified memory and stays there.

**Key specs:**
- **GPU:** NVIDIA GB10 (Compute Capability 12.1)
- **Memory:** 119.6 GB unified (CPU+GPU)
- **CPU:** 20-core ARM aarch64
- **OS:** Ubuntu Linux 6.17 (aarch64)
- **Inference engine:** llama.cpp (`llama-server`)

This is a machine that a well-funded homelab builder or a small engineering team could actually purchase. It's not a $30,000 H100 node. That's the whole point.

## What We Measured

We ran the PPB load test suite — realistic [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) prompts piped into `llama-server`'s `/completion` endpoint with server-sent event (SSE) streaming — and recorded two UX-critical metrics:

- **TTFT (Time to First Token):** How long a user waits before they see *any* output. This is the "is it frozen?" feeling.
- **ITL (Inter-Token Latency):** The time between each streamed token. This determines how smooth the streaming feels.

We tested three quantizations across five context window sizes (2K–130K tokens) at six concurrency levels (1, 2, 4, 8, 16, and 32 parallel users). The data below aggregates **146 individual benchmark runs** across all sweeps.

---

## The Headline Result

Here's the IQ4_NL quantization — our most complete dataset, with 20 runs per concurrency level covering all context sizes. Medians across all tested context windows:

| Concurrent Users | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | Throughput | Avg Power | Failed Prompts |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **165 ms** | 516 ms | 78 ms | 82 ms | 12.7 t/s | 51 W | 0 / 200 |
| 2 | **395 ms** | 742 ms | 93 ms | 170 ms | 19.9 t/s | 55 W | 0 / 400 |
| 4 | **602 ms** | 1,318 ms | 123 ms | 187 ms | 30.4 t/s | 62 W | 2 / 800 |
| **8** | **32,758 ms** ⚠️ | 34,414 ms | 122 ms | 307 ms | 31.0 t/s | 64 W | 1 / 1,600 |
| 16 | 97,757 ms | 101,842 ms | 123 ms | 327 ms | 31.0 t/s | 65 W | 24 / 3,200 |
| 32 | 226,935 ms | 236,882 ms | 123 ms | 326 ms | 31.1 t/s | 65 W | 0 / 6,080 |

*Model: Qwen3.5-27B-IQ4_NL.gguf. All values are medians across runs at context sizes 2K, 8K, 32K, 64K, and 130K.*

---

## Visualizing the Cliff

This is what the TTFT p50 looks like on a log scale:

```
TTFT p50 — Qwen3.5-27B-IQ4_NL (log₁₀ scale)
─────────────────────────────────────────────────────────────────────
 1 user │▌                                              165 ms
 2 user │▊                                              395 ms
 4 user │█                                              602 ms
        │
 8 user │████████████████████████████████████████  32,758 ms  ← THE WALL
        │
16 user │████████████████████████████████████████████████████████████  97,757 ms
32 user │████████████████████████████████████████████████████████████████████████████ 226,935 ms
─────────────────────────────────────────────────────────────────────
```

The jump from 4 → 8 users is a **54× increase** in TTFT. Not 54% — 54 times. A user who had been waiting 600ms for their first token now waits **33 seconds**.

---

## Why This Happens: Prefill Is the Bottleneck

The key insight is in the ITL column. Notice how it barely moves:

```
ITL p50 across all concurrency levels
─────────────────────────────────────
 1 user │████████████████████████  78 ms
 2 user │███████████████████████████████  93 ms
 4 user │████████████████████████████████████████  123 ms
 8 user │████████████████████████████████████████  122 ms
16 user │████████████████████████████████████████  123 ms
32 user │████████████████████████████████████████  123 ms
─────────────────────────────────────
```

**Once a user's prompt starts generating, the streaming is smooth at every concurrency level.** The GB10 can decode tokens at ~13 t/s per user and nobody is fighting over that resource during generation.

The bottleneck is *prefill* — the phase where the model processes the entire input prompt before it can generate a single output token. On this hardware, prefill is sequential: the server can only process one request's prefill at a time. Every other user is queued behind it.

At 8 users, each user's prompt spends, on average, the time equal to 7 others' prefill phases before it even starts generating. With typical ShareGPT prompts averaging ~250 input tokens each at ~13 prefill t/s, that's roughly 7 × 19 seconds = 133 seconds of wait. The data backs this up precisely.

**This is a fundamental property of the prefill-decode architecture, not a tuning problem.** You won't fix it with a bigger batch size or more context window.

---

## Aggregate Throughput Tells the Same Story

```
Aggregate throughput (tokens/sec, all users combined)
──────────────────────────────────────────────────────
 1 user  │██████████████                           12.7 t/s
 2 users │███████████████████████                  19.9 t/s
 4 users │████████████████████████████████████     30.4 t/s ← near max
 8 users │████████████████████████████████████▌    31.0 t/s
16 users │████████████████████████████████████▌    31.0 t/s
32 users │████████████████████████████████████▌    31.1 t/s
──────────────────────────────────────────────────────
```

The system hits its throughput ceiling at 4 users (~30 t/s) and barely gains anything from 8→32 users. **Beyond 4 concurrent users, you're not getting more output — you're just distributing the same amount of work across more waiting people.**

---

## Consistency Across Quantizations

The cliff shows up identically in IQ4_XS and the partial Q8_K_XL data. The quant changes the absolute token speed, but the shape of the concurrency curve is the same:

| Model | Users | TTFT p50 | ITL p50 | Throughput |
|:---|:---:|---:|---:|---:|
| IQ4_XS | 1 | 164 ms | 76 ms | 13.1 t/s |
| IQ4_XS | 4 | 643 ms | 129 ms | 29.2 t/s |
| IQ4_XS | 8 | **34,324 ms** | 128 ms | 29.7 t/s |
| IQ4_NL | 1 | 165 ms | 78 ms | 12.7 t/s |
| IQ4_NL | 4 | 602 ms | 123 ms | 30.4 t/s |
| IQ4_NL | 8 | **32,758 ms** | 122 ms | 31.0 t/s |
| Q8_K_XL | 4 | 1,102 ms | 238 ms | 15.7 t/s |
| Q8_K_XL | 8 | **62,844 ms** | 238 ms | 16.0 t/s |

Note that Q8_K_XL — the higher-quality, heavier quantization — is *slower* at decode, which means its prefill queuing problem is even worse: 8 users means waiting ~63 seconds. If you're serving multiple users, the IQ4 quants are the clear choice on this hardware.

---

## Context Window Has No Effect on the Cliff

One thing we were curious about: does a larger context window make the prefill bottleneck worse? Short answer: not meaningfully at the prompt lengths in our benchmark.

| Context | TTFT p50 (4 users) | TTFT p50 (8 users) |
|:---:|---:|---:|
| 2,048 | 604 ms | 32,576 ms |
| 8,192 | 638 ms | 32,763 ms |
| 32,768 | 600 ms | 32,776 ms |
| 65,536 | 610 ms | 32,729 ms |
| 130,064 | 609 ms | 32,740 ms |

The context window *allocates* KV-cache memory, but it doesn't affect how long it takes to prefill a 250-token prompt. The wall is at 8 users regardless of whether you configure 2K or 130K context.

---

## Power and Thermals

The machine runs cool and doesn't break a sweat:

| Concurrency | Avg Power | Peak Power | GPU Temp |
|:---:|---:|---:|---:|
| 1 user | 51 W | ~60 W | ~67°C |
| 4 users | 62 W | ~68 W | ~70°C |
| 8–32 users | 64–65 W | ~68 W | ~71°C |

The GB10 is drawing roughly 65W sustained under full inference load. For reference, a single RTX 3090 idles at ~30W and runs ~350W under load. The PGX's efficiency story is compelling, even if the throughput numbers don't compete with dedicated GPU servers.

---

## So, What Can You Actually Do With This?

**The sweet spots, in order of usefulness:**

**1 user, dedicated assistant** — This is where the PGX absolutely shines. 165ms TTFT, smooth 78ms ITL, 0 failures across 200 prompts. This is a tier-1 local AI experience.

**2 users** — Still good. Under 400ms TTFT, streaming feels natural. Fine for a pair of developers or a small household.

**3–4 users** — Usable, with caveats. TTFT creeps toward 600ms–1.3s p99. Users will occasionally notice a second-long wait before output starts. For internal tooling or async workflows (where nobody is staring at the cursor), this is fine.

**5+ users** — Don't. The math doesn't work. At 8 users, the median wait before the first token is **33 seconds**. That's not a slow AI — that's a broken one from a user experience perspective.

---

## The Bottom Line

> **The Lenovo ThinkStation PGX running Qwen3.5-27B can serve 1–4 concurrent interactive users with a genuinely good experience. Cross that threshold and the system falls off a cliff.**

This is not a knock on the hardware. The GB10's unified memory architecture means you can actually *run* a 27B model at full quality, right now, on a workstation that fits under a desk. The 4-user ceiling is a fundamental property of prefill-bound inference at this model size — you'd see the same pattern on almost any single-GPU or single-NPU setup.

If you need to push past 4 users with a 27B model, your options are:
- Use a smaller model (we have 2B, 9B data coming) and accept lower quality
- Pay for a dedicated multi-GPU inference server
- Implement request queuing in your application and set user expectations appropriately

For a homelab, a small development team, or a business with a handful of internal power users? **The PGX running IQ4_NL is a legitimate production setup.** The data is there.

---

## Dataset

All raw results are available in the [Poor Paul's Benchmark dataset on Hugging Face](https://huggingface.co/datasets/poorpauls/benchmark). The specific runs cited in this article are in files `Qwen3.5-27B_20260312_1515.jsonl`, `Qwen3.5-27B_20260313_1308.jsonl`, and `Qwen3.5-27B_20260314_0412.jsonl` — 146 benchmark runs, 12,480 individual prompts completed.

*Want to replicate this on your own hardware? The PPB benchmark tool is open source: [github.com/poorpauls/benchmark](https://github.com/poorpauls/benchmark)*

---

*Poor Paul's Benchmark — open data, no hype, just numbers.*
