# TACO-X vs vLLM Benchmark

Benchmark infrastructure for comparing TACO-X and vLLM serving Qwen3-32B on Tencent Cloud L20 GPUs.

## What is TACO-X?

TACO-X is Tencent's proprietary LLM inference engine, built as a high-performance alternative to open-source serving frameworks like vLLM. Where vLLM is a Python-first project that relies on PyTorch, torch.compile, and community-maintained CUDA kernels, TACO-X is a C++ engine with a static compute graph for nanosecond-level operator scheduling. Key technologies: TurboAttention (fused PagedAttention + FlashAttention), TileLang JIT-compiled kernels, reference-counted zero-copy memory management, and Lookahead Cache (training-free speculative decoding, up to 6x+ throughput, average 2.7x). Startup is 70% faster than vLLM (18s vs 66s).

TACO-X is distributed as a private Docker image — contact your Tencent Cloud representative for access. It exposes an OpenAI-compatible API, so benchmarks and client code work unchanged between the two engines.

## Benchmark Results (2026-03-03, TP=4, vLLM v0.16.0)

### Summary

- TACO-X achieves **~4x higher throughput** and **~3.8x lower per-token latency** than vLLM at concurrency 1
- At concurrency 5, TACO-X maintains **~1.4x throughput advantage**
- At concurrency 10, performance is comparable with TACO-X winning on medium/long prompts. TACO-X's TP all-reduce is more sensitive to PCIe bandwidth than vLLM's — NVLink-connected GPUs (H100, H20) should eliminate this gap
- TTFT is **12x faster** at concurrency 1 (63ms vs 761ms for short prompts)

Each engine is shown in its best configuration: **TACO-X with opt-level 3** (TileLang JIT-compiled kernels + lookahead speculative decoding) and **vLLM with CUDA graphs** (captures and replays GPU kernel sequences to reduce launch overhead, the recommended production mode for vLLM).

### Throughput (tok/s, higher is better)

| Prompt | Conc | TACO-X (opt-level 3) | vLLM (CUDA graphs) | Speedup |
|--------|------|----------------------|---------------------|---------|
| short  | 1    | **1,444**            | 348                 | 4.2x    |
| short  | 5    | **461**              | 322                 | 1.4x    |
| short  | 10   | **308**              | 307                 | 1.0x    |
| medium | 1    | **1,393**            | 349                 | 4.0x    |
| medium | 5    | **466**              | 323                 | 1.4x    |
| medium | 10   | **382**              | 305                 | 1.3x    |
| long   | 1    | **1,410**            | 348                 | 4.1x    |
| long   | 5    | **462**              | 319                 | 1.4x    |
| long   | 10   | **299**              | 298                 | 1.0x    |

### Latency — TPOT p50 (ms, lower is better)

| Prompt | Conc | TACO-X (opt-level 3) | vLLM (CUDA graphs) |
|--------|------|----------------------|---------------------|
| short  | 1    | **6.7**              | 26.5                |
| short  | 5    | **20.8**             | 28.7                |
| short  | 10   | **34.3**             | 30.1                |
| medium | 1    | **7.1**              | 26.5                |
| medium | 5    | **21.3**             | 28.7                |
| medium | 10   | **25.9**             | 30.3                |
| long   | 1    | **6.9**              | 26.6                |
| long   | 5    | **21.0**             | 29.0                |
| long   | 10   | **34.3**             | 31.0                |

### Latency — TTFT p50 (ms, lower is better)

| Prompt | Conc | TACO-X (opt-level 3) | vLLM (CUDA graphs) |
|--------|------|----------------------|---------------------|
| short  | 1    | **63**               | 761                 |
| short  | 5    | **234**              | 763                 |
| short  | 10   | **236**              | 787                 |
| medium | 1    | **40**               | 773                 |
| long   | 1    | **57**               | 762                 |

### Methodology

Each engine served Qwen3-32B FP16 with TP=4 on 4x L20 48GB GPUs. The benchmark script (`benchmark.py`) sends 10 requests per configuration using short (~30 tok), medium (~300 tok), and long (~800 tok) prompts at varying concurrency levels (1, 5, 10), with max output of 256 tokens. It measures streaming throughput (tokens/s), time-to-first-token (TTFT), and time-per-output-token (TPOT) at p50/p90/p99 percentiles.

### Architecture Comparison

```
Layer                vLLM                              TACO-X
─────────────────────────────────────────────────────────────────────────
1. HTTP Server       FastAPI + Uvicorn (Python)         Uvicorn (Python) + C++ core engine

2. Request Manager   Tokenizer + chat template          Tokenizer + chat template
                     parsing, validation                parsing, validation

3. Scheduler         Continuous batching,               Producer-consumer async,
                     preemption                         static compute graph scheduling

4. KV Cache Manager  PagedAttention block table,        Block allocation + Lookahead Cache,
                     allocation, eviction,              multi-level (VRAM/RAM/SSD),
                     prefix caching                     global prefix caching

5. Model Runner      torch.compile, CUDA graphs         TileLang JIT kernels,
                     or eager execution                 C++ static graph execution

6. Attention Backend FlashAttention / FlashInfer        TurboAttention
                                                        (PagedAttn + FlashAttn fused)

7. Linear Kernels    CUTLASS, Marlin (for quant)        TileLang + auto-tuned cublas/cutlass

8. GPU Communication NCCL (for TP sync)                 NCCL (async TP dispatch)

9. Memory Mgmt       PyTorch allocator                  Ref-counted zero-copy,
                                                        tensor views, memory pool
```

### Key Differences

| Aspect | vLLM | TACO-X |
|--------|------|--------|
| Language | Python + PyTorch | C++ + TileLang (static compute graph) |
| Attention | FlashAttention / FlashInfer | TurboAttention (PagedAttn + FlashAttn fused) |
| Speculative decode | Requires draft model | Lookahead Cache (training-free, up to 6x+) |
| Memory mgmt | PyTorch allocator | Ref-counted zero-copy, tensor views, memory pool |
| Image size | ~8 GB | ~39 GB compressed / 62 GB on disk |
| Startup time | ~66s (torch.compile warmup) | ~18s (70% faster, no Python GC) |
| Model support | Wide HuggingFace ecosystem | Common models with pre-built configs |
| Quantization | GPTQ, AWQ, FP8 (native) | FP8/AutoRound on H20 best supported |

## Configurations

One set of Terraform files, multiple presets via `configs/*.tfvars`:

| Config | Instance Type | GPUs | vCPU | RAM | Disk |
|--------|---------------|------|------|-----|------|
| `vllm-4xl20` | PNV5b.48XLARGE768 | 4x L20 | 192 | 768 GB | 500 GB |
| `tacox-4xl20` | PNV5b.48XLARGE768 | 4x L20 | 192 | 768 GB | 500 GB |

## Quick Start

### 1. Provision

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your credentials, VPC, subnet, IP

terraform init

# Pick a configuration:
terraform apply -var-file=configs/vllm-4xl20.tfvars
# or: terraform apply -var-file=configs/tacox-4xl20.tfvars
```

### 2. Configure

```bash
cp .env.example .env
# Set REMOTE_IP from: terraform output instance_ip
```

### 3. Deploy & Setup

```bash
./deploy.sh
```

Uploads `setup.sh`, `benchmark.py`, `chat.py` to the instance.

Then SSH in and run setup for your chosen engine:

```bash
ssh -i <your-ssh-key> ubuntu@<IP>

# For vLLM:
bash setup.sh --vllm

# For TACO-X (image URL from your Tencent Cloud rep):
export TACO_X_IMAGE="<ask your Tencent Cloud representative for the image URL>"
bash setup.sh --tacox
```

Setup takes ~25-45 min: mounts data disk, installs NVIDIA drivers, creates Python venv, installs the engine (vLLM pip package or Docker + TACO-X image), and downloads the model (~65 GB).

### 4a. vLLM

`setup.sh --vllm` automatically launches vLLM in a tmux session. Monitor logs:

```bash
tmux attach -t vllm
```

Wait for `Application startup complete`, then proceed to benchmark.

### 4b. TACO-X

`setup.sh --tacox` automatically launches the TACO-X container after installation. Monitor logs:

```bash
docker logs -f taco_x
```

Wait for `Application startup complete`, then proceed to benchmark.

### 5. Run Benchmark

```bash
source /data/venv/bin/activate

# vLLM
python3 benchmark.py --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-32B --concurrency 1,5,10 --save

# TACO-X (use full model path as model name)
python3 benchmark.py --base-url http://localhost:18080/v1 \
    --model /data/hf_cache/models--Qwen--Qwen3-32B/snapshots/<hash>/ \
    --concurrency 1,5,10 --save

# With warmup file (pre-populates lookahead cache before benchmark)
python3 benchmark.py --base-url http://localhost:18080/v1 \
    --model <model_path> --concurrency 1,5,10 \
    --warmup-file warmup-prompts.json --save

# Longer outputs (2048 tokens)
python3 benchmark.py --base-url http://localhost:18080/v1 \
    --model <model_path> --concurrency 1,5,10 \
    --max-tokens 2048 --save
```

## Known Issues

### CUDA Graph Capture OOM (vLLM)

Qwen3-32B FP16 on L20 GPUs can leave very little free VRAM after weight loading + KV cache allocation. CUDA graph capture may fail at high `--gpu-memory-utilization`. Use `--enforce-eager`, or drop to 0.85 with `--max-model-len 4096`.

### Tensor Parallelism Required

Qwen3-32B FP16 needs ~61 GB VRAM. A single L20 (48 GB) cannot fit it. Always use `--tensor-parallel-size 4`.

## Files

| File | Description |
|------|-------------|
| `main.tf` | Terraform: GPU instance + security group |
| `variables.tf` | Terraform variable definitions |
| `outputs.tf` | Terraform outputs (instance IP, SSH command) |
| `configs/*.tfvars` | Per-setup presets (vllm-4xl20, tacox-4xl20) |
| `terraform.tfvars.example` | Base config template (credentials, network) |
| `setup.sh` | Instance bootstrap: `--vllm` or `--tacox` (drivers, Python, engine, model, launch) |
| `deploy.sh` | Upload scripts and run setup remotely |
| `benchmark.py` | Concurrent benchmark with percentile reporting, COS upload |
| `warmup-prompts.json` | Warmup prompts for pre-populating lookahead cache |
| `chat.py` | Interactive multi-turn chat client |
| `.env.example` | SSH connection template |


## Cleanup

```bash
terraform destroy -var-file=configs/vllm-4xl20.tfvars
```

## TACO-X Configuration Reference

TACO-X documentation is sparse. Below is every configuration option we've found across the delivery doc, product docs, and the TACO Kit website.

### CLI Arguments (`python3 -m taco_x.api_server`)

| Flag | Description |
|------|-------------|
| `--model_dir` | Model weights directory (required) |
| `--model_type` | Model type: `qwen3_32b`, `qwen2_5_vl_7b`, `intern2_5_vl_2b`, `qwen3_vl_8b` |
| `--config_dir` | Directory containing scheduler, kv_cache, and lookahead config JSONs |
| `--port` | Server port (default 18080) |
| `--opt-level 3` | Enables (a) lookahead decoding, (b) const folding, (c) all kernel fusion, (d) best-performance kernels. Without it, TACO-X gets 230 tok/s (slower than vLLM's 348) |
| `--tp` | Tensor parallel size (1,2,4,8). **Docs say incompatible with --opt-level, but it works in practice** |
| `--tool-call-parser hermes` | Function calling parser |
| `--enable-auto-tool-choice` | Enable auto mode function calling |
| `--reasoning-parser qwen3` | Reasoning/thinking parser |

### CLI Arguments (shared with TACO-LLM engine)

| Flag | Description |
|------|-------------|
| `--max-num-batched-tokens N` | Max tokens per batch across all sequences |
| `--max-num-seqs N` | Max concurrent sequences. Too high = slow startup + OOM. Too low = caps throughput |
| `--enforce-eager` | Disable CUDA graphs, use eager mode |
| `--num-scheduler-steps N` | Multi-step scheduling [1-8], overlaps CPU work onto GPU during decode |
| `--gpu-memory-utilization F` | GPU memory fraction [0.1-0.95] for weights + activations + KV cache |
| `--conservative-dry-run` | Extra memory safety margin for quantization scenarios |
| `--speculative-model PATH` | Path to draft model for traditional speculative decoding |
| `--num-speculative-tokens N` | Tokens per speculative step (e.g. 3) |
| `--trust-remote-code` | Trust remote model code from HuggingFace |
| `--enable-prefix-caching` | Enable automatic prefix caching (APC) |
| `--enable-prefix-cache-offload` | Offload evicted prefix cache to CPU memory |
| `--cpu-prefill-memory-utilization F` | CPU memory fraction for prefix cache offload (default 0.3) |
| `--lookahead-cache-config-dir DIR` | Directory containing `lookahead_cache_config.json` |
| `--cpu-decoding-memory-utilization F` | CPU memory fraction for lookahead cache (default 0.15) |
| `--max-seq-len-to-capture N` | Max sequence length covered by CUDA graphs |
| `--swap-space N` | CPU swap space in GiB per GPU |
| `--cpu-offload-gb N` | CPU memory in GiB for weight offloading |
| `--quantization TYPE` | Quantization: `awq`, `gptq`, `fp8` (experimental). FP8 requires Hopper GPUs (H100/H20) |

### `lookahead_cache_config.json`

This is TACO-X's proprietary lookahead speculative decoding config. Ships enabled by default in the container.

| Field | Default | Description |
|-------|---------|-------------|
| `cache_mode` | 2 | 0=RawLookaheadCache (N-gram/LLMA based), 1=TurboLookaheadCache (redesign, 1.4-2x higher hit rate), **2=hybrid (recommended)** |
| `cache_size` | 5000000 | Cache size. Can also set via `--cpu-decoding-memory-utilization` |
| `copy_length` | 7 | Lookahead window length. Reduce to 4-6 for large batch sizes (>=32) |
| `match_length` | 2 | Min match length to trigger lookahead |
| `turbo_match_length` | 7 | TurboLookaheadCache max match length |
| `min_match_length` | 2 | TurboLookaheadCache min match length |
| `cell_max_size` | 16 | Secondary cache LRU size. Range [8,32], lower = faster cache turnover |
| `voc_size` | 200000 | Tokenizer vocab size (auto from model config) |
| `max_seq_len` | 32768 | Max sequence length (auto from model config) |
| `eos_token_id` | 2 | EOS token (auto from model config) |
| `top_k` | 1 | Top-k items from secondary cache by frequency |
| `threshold` | 2.0 | Frequency threshold for secondary cache eviction |
| `decay` | 0.5 | Frequency decay when secondary cache is full |
| `is_hybrid` | true | Mix multiple match-lengths for RawLookaheadCache |
| `is_debug` | false | Enable cache hit-rate logging |
| `log_interval` | 3000 | Log print frequency |
| `target_parallelism` | 512 | Max sum(seq_lens) before penalizing copy_length. Only for cache_mode=0 |
| `top_k_in_cell` | 16 | TurboLookaheadCache: tokens returned per secondary cache lookup |
| `token_paths_top_k` | 2 | Beam search width. Try 2-4 for low hit rates |
| `start_freq` | 10.0 | Local cache initial weight. Only when token_paths_top_k > 1 |
| `num_threads` | 8 | TurboLookaheadCache concurrency |
| `global_cache_switch` | true | true=cross-request global cache, false=per-request only |
| `ignore_prompt` | false | Ignore prompt tokens (for translation / unrelated input-output scenarios) |
| `warmup_file` | - | Path to warmup data JSON: `[{"prompt": [ids], "output": [ids]}]` |

### `scheduler_config.json` / `kv_cache_config.json`

Located in `--config_dir`. Key fields:

| File | Field | Our Setting | Default | Notes |
|------|-------|-------------|---------|-------|
| scheduler | `gpu_memory_utilization` | 0.85 | 0.95 | Reduced to avoid OOM during CUDA graph capture |
| scheduler | `max_num_seqs` | 32 | 32 | Default works with tuned memory settings |
| scheduler | `max_num_batched_tokens` | 131072 | 131072 | Default |
| kv_cache | `gpu_memory_utilization` | 0.4 | 0.95 | Reduced to leave room for CUDA graphs + activations |

### Lookahead Tuning Guide (from docs)

- Expected speedup with default config: **1.7x-3x+**
- Use **greedy sampling** for best performance. If not possible, minimize temperature
- `global_average_hit_len + 1` = effective tokens per decode iteration (ideal speedup)
- If `global_average_hit_len < 0.8`: enable MultiPath (`token_paths_top_k: 2`), adjust `cell_max_size`
- If hit rate OK but perf bad at large batch (>=32): reduce `copy_length` to 4-6
- Low-compute GPUs (L20/PNV5b): batch degradation starts earlier (~bs=16)
- Short outputs (<=32 tokens) or unrelated input/output (speech, multimodal): lookahead has limited benefit

## Disclaimers

**Hardware**: These benchmarks were run on **4x L20 48GB GPUs connected via PCIe 4.0** (no NVLink). Production LLM deployments typically use NVLink-connected GPUs (H100, H20, A100 SXM), which provide ~28x higher inter-GPU bandwidth. The PCIe bottleneck in our setup adds latency to every decode step's all-reduce and may disproportionately affect both engines' results compared to NVLink hardware. CUDA graphs also failed to provide any benefit here and hung with default settings — this is likely PCIe-specific and may not apply to NVLink systems.

**TACO-X configuration**: TACO-X was run with `max_num_seqs=32`, `scheduler gpu_memory_utilization=0.85`, `kv_cache gpu_memory_utilization=0.4`. vLLM was run with `gpu_memory_utilization=0.85`, chunked prefill, and prefix caching enabled. Both engines used their recommended production settings for this hardware.

**Model support**: TACO-X logged a warning that `ModelType: qwen3` is not directly supported and fell back to `DefaultWeightsProcessor`. This means the benchmark may not reflect TACO-X's full optimization potential for Qwen3-32B specifically.

**Sample size**: Each configuration was tested with only **10 requests** (plus 2 warmup). This is sufficient for directional comparison but too small for statistically rigorous P99 measurements. Production benchmarks should use 100+ requests.

**Output length**: All tests used `--max-tokens 256`. TACO-X's decode advantage would be more pronounced with longer outputs (1024-2048 tokens), since the faster per-token generation compounds over more tokens. The current benchmark under-represents TACO-X's value for long-form generation use cases.

**Single model, single hardware**: Results are for Qwen3-32B FP16 on L20 GPUs only. Performance characteristics may differ significantly on other models (7B, 70B, MoE), other precisions (FP8, INT4), or other GPU families (H100, H20, A100).

**TP=4 only**: This benchmark has only been tested to run consistently with TP=4. TP=2 configurations were attempted but produced unreliable results and are not supported.
