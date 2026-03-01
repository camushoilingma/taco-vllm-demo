# How vLLM and TACO-X Optimize Differently

Both engines serve the same model (Qwen3-32B) on the same hardware (4x L20 PCIe) but take fundamentally different approaches to making inference fast. This document explains what each engine does, where their optimizations apply, and why the benchmark results look the way they do.

---

## The Problem Both Engines Solve

Generating one token from a 32B-parameter transformer involves:

1. **Prefill** (once per request): Process all input tokens in parallel. This is a compute-bound matrix-matrix multiplication — the GPU's FLOPS are the bottleneck.

2. **Decode** (repeated per output token): Generate one token at a time. Each step reads the model weights (~61GB) and the KV cache from GPU memory, but only does a small matrix-vector multiply. This is memory-bandwidth-bound — the GPU spends most of its time waiting for data, not computing.

3. **Scheduling**: When multiple requests arrive, the engine must decide how to batch them, when to preempt, and how to manage limited KV cache memory.

4. **Communication**: With TP=4 over PCIe, every decode step requires an all-reduce across 4 GPUs. PCIe 4.0 x16 provides ~32 GB/s bidirectional — roughly 28x slower than NVLink 4.0 (900 GB/s). This overhead is the same for both engines (both use NCCL).

Each engine focuses its optimization effort on different parts of this pipeline.

---

## vLLM's Approach: Optimize the Scheduler and Memory System

vLLM is a Python/PyTorch system that leaves the GPU kernels mostly to existing libraries (FlashAttention, CUTLASS, torch.compile) and instead innovates on **how requests are managed**.

### PagedAttention

The key vLLM innovation. Traditional systems pre-allocate a contiguous block of GPU memory for each request's KV cache up to the maximum sequence length. This wastes memory — a request generating 100 tokens has 4000 tokens of empty reserved space if max_seq_len is 4096.

PagedAttention manages KV cache like virtual memory pages: small fixed-size blocks (typically 16 tokens) allocated on demand. This eliminates fragmentation and lets vLLM fit 2-5x more concurrent sequences in the same GPU memory, which directly increases throughput under load.

### Continuous Batching

Traditional batching waits for a batch of requests to all finish before starting the next batch. vLLM adds and removes individual sequences from the running batch at every decode iteration. When one sequence finishes, a waiting request is immediately added — the GPU never idles waiting for the longest sequence to complete.

### Chunked Prefill

Long input prompts can stall decode for all other sequences while the GPU processes the prefill. vLLM splits prefill into chunks and interleaves them with decode steps, keeping existing sequences generating tokens while new requests are being processed. This reduces TTFT variance under concurrent load.

### Prefix Caching

When multiple requests share the same system prompt or prefix, vLLM reuses the computed KV cache blocks instead of recomputing prefill. This saves both compute and memory for repetitive workloads.

### CUDA Graphs (Optional)

vLLM can capture GPU execution into CUDA graphs that replay fixed kernel sequences without Python-side dispatch. **In our benchmark, this made no difference** (348.2 vs 348.3 tok/s). This reveals that Python dispatch overhead is not the bottleneck on this hardware — the PCIe all-reduce and memory bandwidth dominate.

Additionally, vLLM's full CUDA graph capture (default 51 sizes) with torch.compile **hung indefinitely** on 4x L20 PCIe. We could only make it work by limiting capture to 4 sizes (`--cudagraph-capture-sizes 1 2 4 8`), with custom all-reduce disabled and P2P disabled (`NCCL_P2P_DISABLE=1`). This suggests vLLM's CUDA graph implementation has edge cases with PCIe topologies.

### Where vLLM's Optimizations Show Up

| Optimization | What It Helps | Benchmark Evidence |
|---|---|---|
| PagedAttention | More concurrent sequences, less memory waste | vLLM uses `gpu_memory_utilization=0.85` effectively |
| Continuous batching | Throughput at c=5, c=10 | vLLM matches TACO-X at c=10 despite slower kernels |
| Chunked prefill | TTFT under load | vLLM TTFT stays at 657ms at c=10; TACO-X spikes to 1549ms |
| Prefix caching | Repeated prefixes | Not directly measured but enabled |
| CUDA graphs | Kernel launch overhead | No measurable impact on PCIe L20s |

### What vLLM Does NOT Optimize

vLLM does not fuse the individual transformer operations. Each decode step still runs separate kernels for RMSNorm, QKV projection, attention, MLP, etc. Each kernel writes its output to HBM and the next kernel reads it back. This per-kernel memory round-trip is the main reason vLLM's TPOT is 26.5ms — the GPU cycles through ~15-20 kernel launches and HBM writes/reads per token.

---

## TACO-X's Approach: Optimize the GPU Kernels Themselves

TACO-X is a C++ engine that focuses its innovation on **making each decode step use the GPU more efficiently**. It replaces PyTorch's kernel dispatch with TileLang JIT-compiled kernels that fuse operations together.

### TileLang Kernel Fusion (opt-level 3)

The core optimization. Instead of running separate CUDA kernels for each transformer operation, TACO-X's TileLang compiler analyzes the compute graph and fuses multiple operations into a single kernel.

What this means in practice: where vLLM runs a sequence like:

```
RMSNorm kernel → write to HBM → read from HBM → QKV projection kernel → write to HBM → read from HBM → attention kernel → ...
```

TACO-X fuses these into fewer kernels that keep intermediate results in GPU registers or shared memory (L1/L2 cache), avoiding the HBM round-trip:

```
Fused(RMSNorm + QKV projection + RoPE) → Fused(Attention + Output projection) → Fused(Gate + Up + SiLU + Down)
```

Each HBM round-trip avoided saves ~1-3ms on L20 hardware. Fusing 10+ operations saves the bulk of the 19.7ms gap between vLLM's 26.5ms and TACO-X's 6.8ms TPOT.

At startup, we see TACO-X JIT-compile 28 kernels across 7 rounds (4 kernels per round, ~6-7 seconds each). These are the fused kernels being generated for the specific model architecture and GPU hardware. This takes ~70 seconds but only happens once (or could be cached).

### C++ Runtime

TACO-X manages the entire inference pipeline in C++ — HTTP serving, request scheduling, memory management, kernel dispatch. This eliminates:
- Python GIL contention
- PyTorch dispatcher overhead
- Python↔C++ boundary crossings on every operation

The impact of this is visible in the TACO-X eager (no optimization) results: even without kernel fusion, TACO-X eager achieves 39.4ms TPOT. This is slower than vLLM's 26.5ms, which means the C++ runtime alone doesn't explain the speedup — it's the kernel fusion that matters. The C++ runtime contributes when combined with fused kernels, eliminating the overhead that would otherwise exist between fused kernel launches.

### Opt-Level Progression

TACO-X's `--opt-level` flag controls how aggressively kernels are fused:

| Mode | TPOT (c=1) | Tok/s (c=1) | What It Does |
|---|---|---|---|
| Eager (no opt) | 39.4ms | 230 | No fusion — individual kernels like PyTorch, but from C++ |
| Opt-level 3 | 6.8ms | 1,102 | Aggressive fusion — 4.8x faster than its own eager mode |

The 4.8x internal speedup (eager → opt-3) isolates the kernel fusion benefit. The gap between TACO-X eager (39.4ms) and vLLM (26.5ms) shows that without fusion, TACO-X's C++ runtime is actually slower than vLLM's optimized PyTorch path — likely because vLLM benefits from FlashAttention and CUTLASS kernels that are individually well-tuned, while TACO-X eager uses less optimized standalone kernels.

### Where TACO-X's Optimizations Show Up

| Optimization | What It Helps | Benchmark Evidence |
|---|---|---|
| TileLang kernel fusion | Per-token decode latency | 6.8ms vs 26.5ms TPOT at c=1 |
| C++ runtime | Eliminates Python overhead | Contributes when combined with fusion |
| TP rewrite (64 blocks) | Model sharding optimization | 64 optimization blocks processed at startup |

### What TACO-X Does NOT Optimize (Yet)

**Scheduling under load**: TACO-X's current configuration is constrained — `max_num_seqs=8` and `gpu_memory_utilization=0.2` for KV cache (vs vLLM's 0.85). This means:
- Maximum 8 sequences in a batch (vLLM has no hard limit, bounded by memory)
- Only 20% of GPU memory for KV cache (vLLM uses 85%)

These constraints explain why TACO-X degrades steeply at c=10:
- 832 tok/s at c=1 → 262 tok/s at c=10 (68% drop)
- vLLM: 348 tok/s at c=1 → 307 tok/s at c=10 (12% drop)

The TTFT spike to 1549ms at c=10 (vs vLLM's 657ms) suggests requests are queuing behind the `max_num_seqs=8` limit rather than being batched.

**Prefill efficiency**: TACO-X's TTFT at c=1 (596-702ms) is slightly worse than vLLM's (572-586ms). The TileLang kernels are optimized for decode (memory-bound, matrix-vector) rather than prefill (compute-bound, matrix-matrix). vLLM's FlashAttention handles prefill efficiently.

**Note**: These are constraints of the current TACO-X configuration and version — not fundamental architectural limitations. The TACO-X team is actively working on scheduler improvements, and the `max_num_seqs` and KV cache utilization parameters may change in future releases.

---

## Side-by-Side: Optimization Strategy Comparison

```
                    vLLM                                TACO-X
                    ────                                ──────

KERNEL LEVEL        Standard kernels                    TileLang JIT-fused kernels
                    (FlashAttention, CUTLASS)           (multiple ops → single kernel)
                    Each op reads/writes HBM            Intermediates stay in registers/SRAM
                    TPOT: 26.5ms                        TPOT: 6.8ms

MEMORY MGMT         PagedAttention                      Block allocation
                    Virtual memory-style paging          Fixed block_size=16
                    gpu_mem_util: 0.85                  gpu_mem_util: 0.20
                    Fits many concurrent seqs            Limited by config

SCHEDULING          Continuous batching                  Batch scheduling
                    Dynamic add/remove per iteration     max_num_seqs=8
                    Chunked prefill interleaving         Sequential prefill
                    No hard seq limit                   Hard seq limit

PREFILL             FlashAttention                      TileLang (decode-optimized)
                    Fused QKV + softmax + output         Less optimized for prefill
                    Chunked for concurrency             Not chunked

RUNTIME             Python + PyTorch                    C++
                    GIL, dispatcher overhead             Native dispatch
                    Mitigated by CUDA graphs             No Python in hot path

GRAPH CAPTURE       CUDA graphs (optional)              Not used
                    Replays fixed kernel sequences       JIT kernels replace the need
                    No impact on PCIe L20s               N/A

COMMUNICATION       NCCL all-reduce                     NCCL all-reduce
                    Custom all-reduce disabled           Custom all-reduce enabled
                    (due to PCIe compatibility)
```

---

## Why the Results Look the Way They Do

### Concurrency=1: TACO-X Dominates (3.2x throughput)

At c=1, there is one sequence decoding. The GPU is idle most of each step, waiting for HBM reads. TACO-X's fused kernels reduce the number of HBM reads from ~15-20 per step to ~3-5, slashing idle time. vLLM's scheduling advantages (PagedAttention, continuous batching) don't help because there's nothing to schedule — it's just one request.

### Concurrency=5: TACO-X Still Ahead (1.3x throughput)

At c=5, vLLM's scheduler starts to show value — it interleaves 5 sequences efficiently, keeping the GPU busier. But TACO-X's kernel efficiency still wins overall because HBM bandwidth isn't fully saturated yet. TACO-X's TTFT starts to degrade (820ms vs 636ms) as the limited scheduler queues requests.

### Concurrency=10: Convergence (~1x throughput)

At c=10, two things happen simultaneously:
1. **HBM bandwidth saturates**: Reading 10 KV caches per step consumes nearly all available bandwidth. Fused kernels can't help if the bottleneck is raw bytes/second from HBM.
2. **Scheduling becomes critical**: vLLM's continuous batching and chunked prefill handle 10 sequences gracefully. TACO-X's `max_num_seqs=8` cap means 2 requests must wait, and the TTFT spikes to 1549ms.

### TACO-X Eager Is Slowest (39.4ms TPOT)

TACO-X without optimizations runs individual unfused kernels from its C++ runtime. This is slower than vLLM (26.5ms) because vLLM's kernels (FlashAttention, CUTLASS) are individually more optimized than TACO-X's default kernels. TACO-X's speed comes from fusion, not from having better individual kernels. This is an important insight: the TileLang compiler is doing the heavy lifting, not the C++ runtime.

### CUDA Graphs Made No Difference for vLLM

CUDA graphs eliminate Python-side kernel launch overhead (~10-50μs per launch). With TP=4 over PCIe, the all-reduce alone costs ~500-1000μs per step. The launch overhead is <1% of step time, so eliminating it has no measurable effect. CUDA graphs would matter more on a single GPU (no communication overhead) or with NVLink (faster communication, making launch overhead a larger fraction).

---

## What This Means for Choosing an Engine

The two engines are optimized for different bottlenecks:

| Bottleneck | Who Wins | Typical Scenario |
|---|---|---|
| **Decode kernel efficiency** (memory traffic per token) | TACO-X | Single user, interactive chat, streaming |
| **Request scheduling** (batching, preemption, memory management) | vLLM | Multi-tenant API, 10+ concurrent users |
| **Prefill speed** (TTFT) | vLLM | Long-input workloads, concurrency > 1 |
| **PCIe communication** | Neither (same NCCL) | All TP=4 PCIe configs |
| **HBM bandwidth saturation** | Neither (same hardware limit) | High concurrency on any engine |

TACO-X's approach is fundamentally about **making each token cheaper to generate**. vLLM's approach is about **serving more users on the same hardware**. They're solving different problems, and the benchmark shows exactly where each solution matters.

---

## Potential for Improvement

### TACO-X

The current benchmark runs TACO-X in a constrained configuration (`max_num_seqs=8`, `gpu_memory_utilization=0.2`). A fully unconstrained TACO-X with:
- Higher `max_num_seqs` (32+)
- Higher KV cache utilization (0.80+)
- Chunked or interleaved prefill
- Warmup-aware startup (blocking until JIT complete)

...could potentially maintain its decode advantage while closing the scheduling gap at higher concurrency. The kernel fusion benefit is real and significant — the limitation is currently in the surrounding infrastructure, not the core optimization.

### vLLM

vLLM's path to faster decode would require:
- `torch.compile` with operator fusion (partially available, but doesn't match TileLang's fusion depth)
- Custom fused kernels for specific model architectures
- Better CUDA graph support for PCIe topologies (currently hangs with default settings)

These are harder problems because vLLM's architecture separates kernel implementation from the serving framework — it relies on upstream libraries (FlashAttention, CUTLASS) that are general-purpose rather than model-specific.
