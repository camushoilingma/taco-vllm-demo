# TACO-X vs vLLM: Benchmark Analysis

## Test Setup

- **Model**: Qwen3-32B FP16
- **Hardware**: 4x NVIDIA L20 48GB GPUs (Tencent Cloud PNV5b.48XLARGE768)
- **Tensor Parallelism**: TP=4
- **Benchmark**: 10 requests per configuration, short/medium/long input prompts (~50/200/500 tokens), 256 max output tokens, concurrency levels 1/5/10, plus a 5-turn multi-turn chat test
- **Date**: March 1, 2026

### Engine Configurations

Each engine was tested in its best-performing mode:

- **TACO-X opt-level 3**: Tencent's C++ inference engine with TileLang JIT-compiled kernels. Opt-level 3 enables aggressive kernel fusion — attention, MLP, and normalization operations are compiled into fewer, larger GPU kernels that minimize memory round-trips. TACO-X does not use CUDA graphs; its JIT kernels replace that need.

- **vLLM with CUDA graphs**: The open-source vLLM engine (v0.16.0) with CUDA graph capture enabled. CUDA graphs pre-record GPU kernel launch sequences and replay them, eliminating Python-side launch overhead. Also enabled: chunked prefill, prefix caching, and `--gpu-memory-utilization 0.85`.

---

## Results

Data source: `results-tp4/best-comparison/` (best-of runs, March 1, 2026)

### Throughput (tokens/s — higher is better)

| Prompt | Concurrency | TACO-X (opt-level 3) | vLLM (CUDA graphs) | TACO-X Advantage |
|--------|-------------|----------------------|---------------------|------------------|
| short  | 1           | **972**              | 348                 | 2.8x             |
| short  | 5           | **418**              | 322                 | 1.3x             |
| short  | 10          | 296                  | **307**             | 0.96x (vLLM wins)|
| medium | 1           | **927**              | 349                 | 2.7x             |
| medium | 5           | **406**              | 323                 | 1.3x             |
| medium | 10          | 273                  | **305**             | 0.89x (vLLM wins)|
| long   | 1           | **1,052**            | 348                 | 3.0x             |
| long   | 5           | **404**              | 319                 | 1.3x             |
| long   | 10          | 303                  | **298**             | 1.02x (~tied)    |

### Per-Token Latency — TPOT p50 (ms — lower is better)

| Prompt | Concurrency | TACO-X (opt-level 3) | vLLM (CUDA graphs) | TACO-X Advantage |
|--------|-------------|----------------------|---------------------|------------------|
| short  | 1           | **6.8**              | 26.5                | 3.9x faster      |
| short  | 5           | **20.6**             | 28.7                | 1.4x faster      |
| short  | 10          | **27.4**             | 30.1                | 1.1x faster      |
| medium | 1           | **8.0**              | 26.5                | 3.3x faster      |
| medium | 5           | **21.9**             | 28.7                | 1.3x faster      |
| medium | 10          | **27.0**             | 30.3                | 1.1x faster      |
| long   | 1           | **6.8**              | 26.6                | 3.9x faster      |
| long   | 5           | **21.2**             | 29.0                | 1.4x faster      |
| long   | 10          | **29.8**             | 31.0                | 1.04x faster     |

### Time-to-First-Token — TTFT p50 (ms — lower is better)

| Prompt | Concurrency | TACO-X (opt-level 3) | vLLM (CUDA graphs) |
|--------|-------------|----------------------|---------------------|
| short  | 1           | 761                  | **586**             |
| short  | 5           | 857                  | **636**             |
| short  | 10          | 808                  | **657**             |
| medium | 1           | 677                  | **575**             |
| medium | 5           | 728                  | **620**             |
| medium | 10          | 815                  | **667**             |
| long   | 1           | 677                  | **580**             |
| long   | 5           | 905                  | **632**             |
| long   | 10          | 802                  | **687**             |

### End-to-End Latency — p50 (seconds — lower is better)

| Prompt | Concurrency | TACO-X (opt-level 3) | vLLM (CUDA graphs) |
|--------|-------------|----------------------|---------------------|
| short  | 1           | **2.54**             | 7.35                |
| short  | 5           | **6.12**             | 7.96                |
| short  | 10          | 8.52                 | **8.34**            |
| medium | 1           | **2.71**             | 7.34                |
| medium | 5           | **6.30**             | 7.94                |
| medium | 10          | **8.02**             | 8.39                |
| long   | 1           | **2.42**             | 7.36                |
| long   | 5           | **6.33**             | 8.02                |
| long   | 10          | **8.40**             | 8.60                |

### Multi-Turn Chat (single user, 256 output tokens per turn)

| Turn | Context Tokens | TACO-X (tok/s) | vLLM (tok/s) | TACO-X Advantage |
|------|----------------|----------------|--------------|------------------|
| 1    | 29             | **76.8**       | 34.9         | 2.2x             |
| 2    | 310            | **45.8**       | 34.7         | 1.3x             |
| 3    | 587            | **35.4**       | 34.7         | 1.02x (~tied)    |
| 4    | 872            | **39.1**       | 34.6         | 1.1x             |
| 5    | 1,156          | 33.4           | 34.7         | 0.96x (~tied)    |

---

## Why TACO-X Is Faster at Low Concurrency

At concurrency=1, TACO-X achieves ~2.8–3x throughput and ~3.9x lower TPOT. This comes from **TileLang kernel fusion**.

In a standard transformer decode step, the GPU executes a sequence of small operations: RMSNorm → QKV projection → rotary embedding → attention → output projection → gate/up projection → SiLU → down projection. Each is a separate CUDA kernel with its own launch overhead and memory read/write cycle. vLLM (even with CUDA graphs) runs these as individual operations — CUDA graphs only eliminate the Python-side launch overhead, not the memory traffic between kernels.

TACO-X's TileLang compiler fuses multiple operations into a single kernel that keeps intermediate results in GPU registers/shared memory instead of writing them back to HBM (high-bandwidth memory) between steps. This dramatically reduces memory traffic. Since LLM decode is memory-bandwidth-bound (one token at a time, reading the full KV cache), reducing memory round-trips translates directly into lower latency.

The numbers confirm this: TACO-X's 6.8ms TPOT vs vLLM's 26.5ms means TACO-X is doing ~4x less memory traffic per decode step.

## Why vLLM Catches Up at High Concurrency

At concurrency=10, throughput converges (~296 vs 307 tok/s for short prompts), and vLLM wins on medium prompts (305 vs 273 tok/s).

With 10 concurrent requests, the GPU is processing batches of 10 sequences simultaneously. The bottleneck shifts from **per-token kernel efficiency** to **total memory bandwidth** — reading 10 KV caches in parallel saturates HBM bandwidth regardless of how fused the kernels are. Both engines hit the same memory wall.

Additionally, vLLM's scheduler is more mature at high concurrency:

- **Continuous batching**: vLLM dynamically adds/removes sequences from the running batch every iteration, maximizing GPU utilization
- **PagedAttention**: Efficiently manages fragmented KV cache memory across many sequences
- **Chunked prefill**: Overlaps prefill of new requests with decode of existing ones

TACO-X's scheduler appears to handle contention less gracefully — its TTFT P90 spikes to 7,272ms at concurrency=10 (vs vLLM's 665ms), suggesting requests queue up rather than being interleaved.

## Why TTFT Favors vLLM

Time-to-first-token measures how quickly the engine processes the input prompt (prefill) before generating the first output token. vLLM consistently wins TTFT across all configurations (586ms vs 761ms at short/c=1, 636ms vs 857ms at short/c=5).

Prefill is a **compute-bound** operation (large matrix-matrix multiplications across all input tokens simultaneously), not a memory-bound one like decode. vLLM uses FlashAttention for prefill, which is highly optimized for this workload. TACO-X's TileLang kernels are primarily optimized for the decode phase (matrix-vector operations), so its prefill path doesn't have the same advantage.

## Why Multi-Turn Converges

At turn 1 (29 tokens of context), TACO-X achieves 76.8 tok/s vs vLLM's 34.9 — a 2.2x advantage. By turn 5 (1,156 tokens), vLLM is marginally ahead at 34.7 vs 33.4 tok/s.

As context grows, each decode step must read a larger KV cache from HBM. At ~900 tokens of context, the KV cache read becomes the dominant cost per step, and both engines are limited by the same L20 memory bandwidth (~1 TB/s). TACO-X's kernel fusion advantage is overwhelmed by the sheer volume of KV cache data that must be read regardless of kernel design.

---

## Use Cases

### When TACO-X Is the Better Choice

| Use Case | Why |
|----------|-----|
| **Interactive developer assistant** (code completion, pair programming) | Single user waiting for streaming tokens — 3x faster perceived response |
| **Real-time chatbot** with 1–3 concurrent users | Low concurrency is TACO-X's sweet spot |
| **Long-form generation** (essays, reports, code files) | At c=1, a 2048-token response takes ~14s on TACO-X vs ~54s on vLLM |
| **Agentic workflows with sequential LLM calls** | Each call is effectively c=1; lower TPOT compounds across chain |
| **Latency-sensitive applications** where users notice token speed | 6.8ms vs 26.5ms between tokens is visible in streaming UIs |

### When vLLM Is the Better Choice

| Use Case | Why |
|----------|-----|
| **Multi-tenant API serving** (10+ concurrent users) | Throughput parity + better tail latency + mature scheduler |
| **Batch processing** (many requests in parallel) | vLLM's continuous batching maximizes GPU utilization under load |
| **Prefill-heavy workloads** (long inputs, short outputs — e.g., classification, extraction) | vLLM's FlashAttention prefill is faster |
| **Production services with SLA requirements** | Predictable P99 latency — TACO-X's P90 TTFT spikes to 7.3s at c=10 |
| **Workloads needing LoRA, speculative decoding, or quantization variety** | vLLM has a much larger feature ecosystem |
| **Rapid prototyping and model experimentation** | `pip install vllm && vllm serve` vs Docker + config files |

---

## Scaling Considerations

### More GPUs (Higher TP)

Moving from TP=4 to TP=8 would:
- Improve prefill speed for both engines (more compute for matrix multiplications)
- Modestly improve decode latency — but all-reduce communication between 8 GPUs becomes a larger fraction of each step, yielding diminishing returns
- Likely shrink TACO-X's per-token advantage, since the inter-GPU sync overhead (NCCL, same for both) becomes the dominant cost rather than kernel efficiency
- Give vLLM more KV cache capacity to serve additional concurrent requests

### Faster GPUs (e.g., H20 96GB HBM3)

H20 offers ~4x the memory bandwidth of L20 (3.9 TB/s vs ~1 TB/s):
- Both engines would see dramatic decode speedups, roughly proportional to the bandwidth increase
- TACO-X's advantage might widen on H20 — its fused kernels would better exploit higher bandwidth since they reduce total memory operations, while vLLM's unfused kernels would still bottleneck on read/write patterns between operations
- TP=2 on H20 (2x 96GB = 192GB) would be sufficient for Qwen3-32B, reducing all-reduce overhead compared to TP=4 on L20
- The multi-turn convergence point would shift later (higher bandwidth means kernel efficiency matters for longer before bandwidth saturates)

### Horizontal Scaling

For serving many users, the most practical approach is running multiple independent instances behind a load balancer rather than scaling up TP. Each instance handles a subset of users at low concurrency — this plays to TACO-X's strength while avoiding its high-concurrency weakness.

---

## Summary

TACO-X's advantage is **compiler-driven kernel fusion** that reduces GPU memory traffic during token generation. This produces a large speedup when the GPU is serving one or a few users (the kernel efficiency regime). vLLM's advantage is **scheduling maturity and ecosystem breadth** — it handles concurrent load gracefully and offers more features for production deployment. The two engines converge at high concurrency because memory bandwidth becomes the shared bottleneck regardless of kernel design.

For organizations that primarily serve individual users (developer tools, chatbots, internal assistants), TACO-X delivers a meaningfully faster experience on the same hardware. For organizations running multi-tenant APIs under load, vLLM remains the safer choice.
