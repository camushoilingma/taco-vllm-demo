# TACO-X vs vLLM Benchmark

Benchmark infrastructure for comparing TACO-X and vLLM serving Qwen3-32B on Tencent Cloud L20 GPUs.

## What is TACO-X?

TACO-X is Tencent's proprietary LLM inference engine, built as a high-performance alternative to open-source serving frameworks like vLLM. Where vLLM is a Python-first project that relies on PyTorch, torch.compile, and community-maintained CUDA kernels, TACO-X is a C++ engine that uses TileLang JIT-compiled kernels and a custom runtime to maximize GPU utilization.

TACO-X is distributed as a private Docker image — contact your Tencent Cloud representative for access. It exposes an OpenAI-compatible API, so benchmarks and client code work unchanged between the two engines.

## Benchmark Results (2026-03-01, TP=4, vLLM v0.16.0)

### Summary

- TACO-X achieves **~2.8–3x higher throughput** and **~3.9x lower per-token latency** than vLLM at low concurrency
- At high concurrency (10), the two engines converge to similar throughput
- TTFT is comparable at concurrency 1 (~580-760ms); vLLM wins TTFT at all concurrency levels

Each engine is shown in its best configuration: **TACO-X with opt-level 3** (TileLang JIT-compiled kernels — TACO-X does not use CUDA graphs, its optimized kernels replace that need) and **vLLM with CUDA graphs** (captures and replays GPU kernel sequences to reduce launch overhead, the recommended production mode for vLLM).

### Throughput (tok/s, higher is better)

| Prompt | Conc | TACO-X (opt-level 3) | vLLM (CUDA graphs) |
|--------|------|----------------------|---------------------|
| short  | 1    | **972**              | 348                 |
| short  | 5    | **418**              | 322                 |
| short  | 10   | 296                  | **307**             |
| medium | 1    | **927**              | 349                 |
| medium | 5    | **406**              | 323                 |
| medium | 10   | 273                  | **305**             |
| long   | 1    | **1,052**            | 348                 |
| long   | 5    | **404**              | 319                 |
| long   | 10   | 303                  | **298**             |

### Latency — TPOT p50 (ms, lower is better)

| Prompt | Conc | TACO-X (opt-level 3) | vLLM (CUDA graphs) |
|--------|------|----------------------|---------------------|
| short  | 1    | **6.8**              | 26.5                |
| short  | 5    | **20.6**             | 28.7                |
| short  | 10   | **27.4**             | 30.1                |
| medium | 1    | **8.0**              | 26.5                |
| medium | 5    | **21.9**             | 28.7                |
| medium | 10   | **27.0**             | 30.3                |
| long   | 1    | **6.8**              | 26.6                |
| long   | 5    | **21.2**             | 29.0                |
| long   | 10   | **29.8**             | 31.0                |

### Methodology

Each engine served Qwen3-32B FP16 with TP=4 on 4x L20 48GB GPUs. The benchmark script (`benchmark.py`) sends 10 requests per configuration using short (~35 tok), medium (~290 tok), and long (~1000 tok) prompts at varying concurrency levels (1, 5, 10). It measures streaming throughput (tokens/s), time-to-first-token (TTFT), and time-per-output-token (TPOT) at p50/p90/p99 percentiles. vLLM was tested in both eager and CUDA graph modes.

### Architecture Comparison

```
Layer                vLLM                              TACO-X
─────────────────────────────────────────────────────────────────────────
1. HTTP Server       FastAPI + Uvicorn (Python)         C++ HTTP server

2. Request Manager   Tokenizer + chat template          Tokenizer + chat template
                     parsing, validation                parsing, validation

3. Scheduler         Continuous batching,               Batching,
                     preemption                         scheduling

4. KV Cache Manager  PagedAttention block table,        Block allocation,
                     allocation, eviction,              lookahead cache config
                     prefix caching

5. Model Runner      torch.compile, CUDA graphs         TileLang JIT kernels,
                     or eager execution                 C++ execution

6. Attention Backend FlashAttention / FlashInfer        Custom C++ attention

7. Linear Kernels    CUTLASS, Marlin (for quant)        TileLang, naive dequant
                                                        (for unsupported quants)

8. GPU Communication NCCL (for TP sync)                 NCCL (for TP sync)

9. Response Streaming SSE via FastAPI                   SSE via C++ server
```

### Key Differences

| Aspect | vLLM | TACO-X |
|--------|------|--------|
| Language | Python + PyTorch | C++ + TileLang |
| Image size | ~8 GB | ~39 GB compressed / 62 GB on disk |
| Model support | Wide HuggingFace ecosystem | Common models with pre-built configs |
| Quantization | GPTQ, AWQ, FP8 (native) | FP8/AutoRound on H20 best supported |
| Tensor Parallelism | Works for all models | Works for FP16 |
| Startup | `pip install vllm && vllm serve` | Docker pull + config dir + flags |

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
export TACO_X_IMAGE="ccr.ccs.tencentyun.com/taco/taco_x_prod:<tag>"
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
| `benchmark.py` | Concurrent benchmark with percentile reporting |
| `chat.py` | Interactive multi-turn chat client |
| `.env.example` | SSH connection template |
| `results-tp4/best-comparison/` | Best-of TP=4 benchmark CSVs and comparison |

## Cleanup

```bash
terraform destroy -var-file=configs/vllm-4xl20.tfvars
```

## Disclaimers

**Hardware**: These benchmarks were run on **4x L20 48GB GPUs connected via PCIe 4.0** (no NVLink). Production LLM deployments typically use NVLink-connected GPUs (H100, H20, A100 SXM), which provide ~28x higher inter-GPU bandwidth. The PCIe bottleneck in our setup adds latency to every decode step's all-reduce and may disproportionately affect both engines' results compared to NVLink hardware. CUDA graphs also failed to provide any benefit here and hung with default settings — this is likely PCIe-specific and may not apply to NVLink systems.

**TACO-X configuration**: TACO-X was run with **constrained scheduler settings** (`max_num_seqs=8`, `gpu_memory_utilization=0.2` for KV cache) — not a fully tuned production configuration. vLLM was run with `gpu_memory_utilization=0.85`, chunked prefill, and prefix caching enabled. The high-concurrency results (c=10) where TACO-X's throughput degrades and TTFT P90 spikes to 7,272ms likely reflect these scheduler constraints rather than a fundamental engine limitation. A fully unconstrained TACO-X configuration may perform differently.

**Model support**: TACO-X logged a warning that `ModelType: qwen3` is not directly supported and fell back to `DefaultWeightsProcessor`. This means the benchmark may not reflect TACO-X's full optimization potential for Qwen3-32B specifically.

**Sample size**: Each configuration was tested with only **10 requests** (plus 2 warmup). This is sufficient for directional comparison but too small for statistically rigorous P99 measurements. Production benchmarks should use 100+ requests.

**Output length**: All tests used `--max-tokens 256`. TACO-X's decode advantage would be more pronounced with longer outputs (1024-2048 tokens), since the faster per-token generation compounds over more tokens. The current benchmark under-represents TACO-X's value for long-form generation use cases.

**Single model, single hardware**: Results are for Qwen3-32B FP16 on L20 GPUs only. Performance characteristics may differ significantly on other models (7B, 70B, MoE), other precisions (FP8, INT4), or other GPU families (H100, H20, A100).

**TP=4 only**: This benchmark has only been tested to run consistently with TP=4. TP=2 configurations were attempted but produced unreliable results and are not supported.
