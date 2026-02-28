# TACO-X vs vLLM Benchmark

Benchmark infrastructure for comparing TACO-X and vLLM serving Qwen3-32B on Tencent Cloud L20 GPUs.

## What is TACO-X?

TACO-X is Tencent's proprietary LLM inference engine, built as a high-performance alternative to open-source serving frameworks like vLLM. Where vLLM is a Python-first project that relies on PyTorch, torch.compile, and community-maintained CUDA kernels, TACO-X is a C++ engine that uses TileLang JIT-compiled kernels and a custom runtime to maximize GPU utilization.

TACO-X is distributed as a private Docker image — contact your Tencent Cloud representative for access. It exposes an OpenAI-compatible API, so benchmarks and client code work unchanged between the two engines.

## Benchmark Results (2026-02-28, vLLM v0.16.0)

### Summary

- TACO-X achieves **~4.5x higher throughput** and **~6x lower per-token latency** than vLLM
- CUDA graphs provide only ~2-3% improvement in vLLM — the bottleneck is not kernel launch overhead
- TTFT is comparable across all three (~55-62ms at concurrency 1)

### Throughput (tok/s, higher is better)

| Prompt | Conc | TACO-X | vLLM (enforce-eager) | vLLM (CUDA graphs) |
|--------|------|--------|----------------------|---------------------|
| short | 1 | **1,003** | 216 | 221 |
| short | 5 | **594** | 201 | 208 |
| short | 10 | **864** | 197 | 200 |
| medium | 1 | **1,103** | 215 | 220 |
| medium | 5 | **429** | 200 | 207 |
| medium | 10 | **833** | 196 | 201 |
| long | 1 | **912** | 215 | 220 |
| long | 5 | **894** | 197 | 204 |
| long | 10 | **828** | 191 | 196 |

### Latency — TPOT p50 (ms, lower is better)

| Prompt | Conc | TACO-X | vLLM (enforce-eager) | vLLM (CUDA graphs) |
|--------|------|--------|----------------------|---------------------|
| short | 1 | **7.5** | 46.3 | 45.3 |
| short | 5 | **15.9** | 49.4 | 47.7 |
| short | 10 | **11.1** | 50.2 | 49.4 |
| medium | 1 | **8.5** | 46.4 | 45.3 |
| medium | 5 | **23.0** | 49.6 | 47.9 |
| medium | 10 | **11.6** | 50.7 | 49.4 |
| long | 1 | **9.8** | 46.6 | 45.4 |
| long | 5 | **10.8** | 50.3 | 48.5 |
| long | 10 | **11.5** | 51.7 | 50.4 |

### Methodology

Each engine served Qwen3-32B FP16 with TP=2 on 2x L20 48GB GPUs. The benchmark script (`benchmark.py`) sends 10 requests per configuration using short (~50 tok), medium (~200 tok), and long (~500 tok) prompts at varying concurrency levels (1, 5, 10). It measures streaming throughput (tokens/s), time-to-first-token (TTFT), and time-per-output-token (TPOT) at p50/p90/p99 percentiles. vLLM was tested in both enforce-eager and CUDA graph modes.

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
| `vllm-2xl20` | PNV5b.16XLARGE192 | 2x L20 | 64 | 192 GB | 200 GB |
| `vllm-4xl20` | PNV5b.48XLARGE768 | 4x L20 | 192 | 768 GB | 500 GB |
| `tacox-2xl20` | PNV5b.16XLARGE192 | 2x L20 | 64 | 192 GB | 200 GB |
| `tacox-4xl20` | PNV5b.48XLARGE768 | 4x L20 | 192 | 768 GB | 500 GB |

## Quick Start

### 1. Provision

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your credentials, VPC, subnet, IP

terraform init

# Pick a configuration:
terraform apply -var-file=configs/vllm-2xl20.tfvars
# or: terraform apply -var-file=configs/tacox-4xl20.tfvars
```

### 2. Configure

```bash
cp .env.example .env
# Set REMOTE_IP from: terraform output instance_ip
```

### 3. Deploy

```bash
./deploy.sh
```

Uploads `setup.sh`, `benchmark.py`, `chat.py` and runs setup (~25-45 min).

### 4a. Start vLLM

```bash
ssh -i <your-ssh-key> ubuntu@<IP>
source /data/venv/bin/activate
export HF_HUB_CACHE=/data/hf_cache
tmux new -s vllm

# 2x L20:
vllm serve Qwen/Qwen3-32B \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --port 8000

# 4x L20:
vllm serve Qwen/Qwen3-32B \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --port 8000
```

### 4b. Start TACO-X (alternative)

The TACO-X Docker image is private. Contact your Tencent Cloud representative to obtain access.

```bash
export TACO_X_IMAGE="<image-url-from-tencent>"
bash taco_x_start.sh          # auto-detects GPU count for TP
```

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

Qwen3-32B FP16 on 2x L20 leaves only ~58 MiB free after weight loading + KV cache allocation. CUDA graph capture fails at `--gpu-memory-utilization` 0.95 and 0.90. Use `--enforce-eager`, or drop to 0.85 with `--max-model-len 4096`.

### Tensor Parallelism Required

Qwen3-32B FP16 needs ~61 GB VRAM. A single L20 (48 GB) cannot fit it. Always use `--tensor-parallel-size 2` or higher.

## Files

| File | Description |
|------|-------------|
| `main.tf` | Terraform: GPU instance + security group |
| `variables.tf` | Terraform variable definitions |
| `outputs.tf` | Terraform outputs (instance IP, SSH command) |
| `configs/*.tfvars` | Per-setup presets (vllm-2xl20, tacox-4xl20, etc.) |
| `terraform.tfvars.example` | Base config template (credentials, network) |
| `setup.sh` | Instance bootstrap (venv, vLLM, model download) |
| `taco_x_start.sh` | TACO-X container launcher |
| `deploy.sh` | Upload scripts and run setup remotely |
| `benchmark.py` | Concurrent benchmark with percentile reporting |
| `chat.py` | Interactive multi-turn chat client |
| `.env.example` | SSH connection template |
| `results/` | Benchmark result CSVs |

## Cleanup

```bash
terraform destroy -var-file=configs/vllm-2xl20.tfvars
```
