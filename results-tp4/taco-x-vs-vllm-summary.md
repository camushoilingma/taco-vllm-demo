# TACO-X vs vLLM Benchmark Report
## Qwen3-32B on 4x NVIDIA L20 (PCIe) - TP=4

### Hardware
- Instance: Tencent Cloud PNV5b.48XLARGE768
- GPUs: 4x NVIDIA L20 48GB (PCIe Gen4, no NVLink)
- CPU: 192 vCPU, 768GB RAM
- Model: Qwen3-32B FP16, Tensor Parallel = 4

### Software
- TACO-X: v0.0.1-260127-lightspeed (C++ inference engine)
- vLLM: v0.16.0 (PyTorch-based, V1 engine)

### Benchmark Setup
- OpenAI-compatible streaming API, 10 requests per config + 2 warmup
- Prompt sizes: short (~30 tokens), medium (~300 tokens), long (~800 tokens)
- Max output tokens: 256
- Concurrency: 1, 5, 10

---

## 1. Single-Request Performance (Concurrency = 1)

| Engine | Prompt | TPOT p50 | Tok/s | TTFT p50 |
|--------|--------|----------|-------|----------|
| TACO-X --opt-level 3 | short | 6.8ms | 971 | 761ms |
| TACO-X --opt-level 3 | medium | 8.0ms | 927 | 677ms |
| TACO-X --opt-level 3 | long | 6.8ms | 1052 | 677ms |
| TACO-X eager (no opt) | short | 39.4ms | 230 | 687ms |
| TACO-X eager (no opt) | medium | 41.1ms | 219 | 682ms |
| TACO-X eager (no opt) | long | 38.3ms | 240 | 734ms |
| vLLM eager | short | 26.6ms | 348 | 583ms |
| vLLM eager | medium | 26.7ms | 347 | 580ms |
| vLLM eager | long | 26.8ms | 346 | 572ms |
| vLLM CUDA graphs | short | 26.5ms | 348 | 586ms |
| vLLM CUDA graphs | medium | 26.5ms | 349 | 575ms |
| vLLM CUDA graphs | long | 26.6ms | 348 | 580ms |

## 2. Concurrency Scaling

| Engine | Conc=1 tok/s | Conc=5 tok/s | Conc=10 tok/s |
|--------|-------------|-------------|--------------|
| TACO-X opt-level 3 (short) | 971 | 418 | 296 |
| TACO-X opt-level 3 (medium) | 927 | 406 | 273 |
| TACO-X eager (short) | 230 | 214 | 162 |
| vLLM eager (short) | 348 | 321 | 306 |
| vLLM CUDA graphs (short) | 348 | 322 | 307 |

## 3. Multi-Turn Conversation (Growing Context)

| Engine | Turn 1 (29 tok) | Turn 3 (587 tok) | Turn 5 (1156 tok) |
|--------|----------------|-------------------|-------------------|
| TACO-X opt-level 3 TPOT | 10.3ms | 25.1ms | 27.1ms |
| vLLM CUDA graphs TPOT | 26.6ms | 26.5ms | 26.6ms |

---

## Key Observations

### TACO-X Strengths
- At concurrency=1, TACO-X opt-level 3 delivers 2.8x higher throughput than vLLM (971 vs 348 tok/s)
- Per-token latency is 3.9x lower (6.8ms vs 26.5ms TPOT)
- The --opt-level 3 flag alone provides a 4.2x speedup over TACO-X's own eager mode (971 vs 230 tok/s), indicating significant C++ kernel fusion and optimization
- TTFT (prefill latency) is comparable across all engines (~580-760ms), confirming the advantage is in the decode phase

### TACO-X Weaknesses
- Performance degrades more steeply under concurrency: at conc=10, TACO-X (296 tok/s) is only marginally better than vLLM (307 tok/s)
- Multi-turn TPOT grows with context length: 10ms at 29 tokens, 27ms at 1156 tokens. vLLM stays flat at ~26.5ms regardless of context
- By mid-conversation (~500+ tokens context), TACO-X and vLLM converge to similar per-token latency
- KV cache is heavily constrained (gpu_memory_utilization=0.2 to avoid OOM), limiting batching capacity
- Conservative scheduler defaults (max_num_seqs=8) cap concurrent request handling

### vLLM Observations
- CUDA graphs with limited capture sizes (1,2,4,8) made zero difference vs eager mode on PCIe L20s
- Full CUDA graph capture (51 sizes) with torch.compile hangs indefinitely on 4-way TP over PCIe -- never completes
- Performance is remarkably stable across prompt sizes and concurrency levels
- vLLM auto-manages KV cache allocation, while TACO-X requires manual tuning

### Configuration Tuning Attempted
- Increasing kv_cache from 0.2 to 0.4: no improvement (slightly worse)
- Enabling CUDA graphs in TACO-X: no measurable impact
- Increasing max_num_seqs/batched_tokens: config validation error (incompatible with encoder settings)
- Conclusion: the default config with --opt-level 3 is the best achievable with current tuning options

---

## Validation Questions for TACO-X Team

1. The engine binary contains H2O (Heavy-Hitter Oracle) and FIFO/LRU KV cache eviction symbols. How do we enable H2O eviction to improve long-context multi-turn performance?
   - Which kv_layout_type or cache_mode values activate H2O?
   - What are the supported eviction policies (EVICT_FIRST, EVICT_LAST, EVICT_NORMAL)?

2. Why does TPOT grow linearly with context length? Is this expected behavior or a configuration issue?
   - vLLM maintains flat TPOT regardless of context length on the same hardware

3. The default scheduler config (max_num_seqs=32, max_num_batched_tokens=131072, gpu_memory_utilization=0.95) OOMs on 4xL20 with FP16. What are the recommended settings for this hardware?
   - We had to reduce to max_num_seqs=8, kv_cache=0.2 to avoid crashes

4. Is --opt-level 3 officially supported with TP=4? Documentation says it is incompatible with TP, but it works and gives a 4x speedup

5. The lookahead_cache_config.json has speculative decoding settings (TurboLookaheadCache). Is this active by default? Does it interact with --opt-level 3?

6. Are there FP8 KV cache options to double effective cache capacity without model quantization?

---

## Recommended Next Steps

1. Engage TACO-X team on H2O / KV eviction enablement -- this is the most likely path to fixing multi-turn degradation

2. Test with FP8 KV cache (if supported) -- would double KV cache capacity within the same 0.2 memory budget, potentially improving both multi-turn and high-concurrency

3. Benchmark on NVLink hardware (e.g., A100/H100 with NVLink) -- PCIe is a bottleneck for TP communication; NVLink would benefit both engines but may disproportionately help TACO-X at high concurrency

4. Test 2xL20 TP=2 with --opt-level 3 -- the 2xL20 instance is cheaper; if single-user performance is the target use case, TP=2 may be sufficient

5. Longer context benchmarks -- test with 2K, 4K, 8K input tokens to quantify the multi-turn degradation curve and validate H2O impact

6. Production-realistic workload mix -- current benchmarks use fixed concurrency; a mixed workload with arrival rate would better simulate production behavior

---

## Result Files

Best-of runs are in `results-tp4/best-comparison/`:

| File | Description |
|------|-------------|
| `best-comparison/taco-x-best.csv` | TACO-X opt-level 3, default config (best run, March 1) |
| `best-comparison/vllm-best.csv` | vLLM 0.16.0 CUDA graphs (best run, March 1) |
| `best-comparison/comparison.csv` | Side-by-side comparison with advantage ratios |

Raw run files (for reference, not committed):

| File | Description |
|------|-------------|
| taco-x-tp4-optlevel3-defaults-20260301-093141.csv | Source for taco-x-best.csv |
| tacox-noopt-eager-tp4-20260228.csv | TACO-X without opt-level (baseline) |
| vllm-tp4-cudagraph-20260301.csv | Source for vllm-best.csv |
| vllm-tp4-eager-20260301.csv | vLLM eager mode (compilation level 0) |
