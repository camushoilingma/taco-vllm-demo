#!/usr/bin/env bash
# vLLM Baseline Setup — Qwen3-32B on 2× L20 (PNV5b.16XLARGE192)
# Run on a fresh Tencent Cloud Ubuntu 22.04 instance.
set -euo pipefail

MODEL="Qwen/Qwen3-32B"

echo "============================================"
echo "  vLLM Baseline — Instance Setup"
echo "  Model: ${MODEL}"
echo "  Expected: 2× NVIDIA L20 48GB"
echo "============================================"
echo ""
echo "  Full run typically takes 25–45 min"
echo "  (vLLM install + ~65 GB model download)."
echo ""

# ── [1/7] Mount data disk ─────────────────────────────────────────
if lsblk | grep -q vdb; then
    echo "[1/7] Mounting data disk..."
    if ! mount | grep -q /data; then
        sudo mkfs.ext4 -F /dev/vdb
        sudo mkdir -p /data
        sudo mount /dev/vdb /data
        echo '/dev/vdb /data ext4 defaults 0 0' | sudo tee -a /etc/fstab
        sudo chown -R "$USER:$USER" /data
    fi
    echo "  Data disk mounted at /data"
else
    echo "[1/7] No data disk found, using system disk"
    sudo mkdir -p /data
    sudo chown -R "$USER:$USER" /data
fi

# ── [2/7] System packages ─────────────────────────────────────────
echo ""
echo "[2/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv tmux htop > /dev/null 2>&1
echo "  Done"

# ── [3/7] NVIDIA driver ──────────────────────────────────────────
echo ""
echo "[3/7] Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "  NVIDIA driver already installed (${GPU_COUNT} GPU(s)):"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "  Installing NVIDIA driver..."
    sudo apt-get install -y -qq linux-headers-$(uname -r) > /dev/null 2>&1
    sudo apt-get install -y -qq nvidia-driver-535 > /dev/null 2>&1
    echo "  Driver installed. A REBOOT is required."
    echo "  After reboot, run this script again."
    echo ""
    echo "  To reboot: sudo reboot"
    exit 0
fi

# ── [4/7] Python virtual environment ─────────────────────────────
echo ""
echo "[4/7] Setting up Python environment..."
python3 -m venv /data/venv
source /data/venv/bin/activate
pip install --upgrade pip -q

# ── [5/7] Install vLLM and dependencies ──────────────────────────
echo ""
echo "[5/7] Installing vLLM and OpenAI SDK..."
pip install vllm openai huggingface_hub -q
echo "  vLLM version: $(python3 -c 'import vllm; print(vllm.__version__)')"

# ── [6/7] Download model ─────────────────────────────────────────
echo ""
echo "[6/7] Downloading model: ${MODEL} (~65 GB)..."
echo "  This will take 15-30 min depending on bandwidth."
export HF_HUB_CACHE=/data/hf_cache
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}', cache_dir='/data/hf_cache')
print('  Model downloaded successfully')
"

# ── [7/7] Verification ───────────────────────────────────────────
echo ""
echo "[7/7] Verification"
echo "============================================"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) devices"
echo "  VRAM:    $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1) per GPU"
echo "  Python:  $(python3 --version)"
echo "  vLLM:    $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "  Model:   ${MODEL}"
echo "============================================"
echo ""
echo "Setup complete! To start the vLLM server:"
echo ""
echo "  source /data/venv/bin/activate"
echo "  export HF_HUB_CACHE=/data/hf_cache"
echo ""
echo "  # Start in tmux so the server persists after disconnect"
echo "  tmux new -s vllm"
echo ""
echo "  # RECOMMENDED: enforce-eager mode (stable, avoids CUDA graph OOM)"
echo "  vllm serve ${MODEL} \\"
echo "    --dtype bfloat16 \\"
echo "    --tensor-parallel-size 2 \\"
echo "    --max-model-len 8192 \\"
echo "    --gpu-memory-utilization 0.95 \\"
echo "    --enforce-eager \\"
echo "    --enable-prefix-caching \\"
echo "    --enable-chunked-prefill \\"
echo "    --port 8000"
echo ""
echo "  # ALTERNATIVE: CUDA graphs mode (requires lower memory settings)"
echo "  # CUDA graph capture needs extra GPU headroom — 0.95/0.90 both OOM."
echo "  vllm serve ${MODEL} \\"
echo "    --dtype bfloat16 \\"
echo "    --tensor-parallel-size 2 \\"
echo "    --max-model-len 4096 \\"
echo "    --gpu-memory-utilization 0.85 \\"
echo "    --enable-prefix-caching \\"
echo "    --enable-chunked-prefill \\"
echo "    --port 8000"
echo ""
echo "  # Run benchmark (from another terminal):"
echo "  source /data/venv/bin/activate"
echo "  python3 benchmark.py --base-url http://localhost:8000/v1 \\"
echo "    --model ${MODEL} --concurrency 1,5,10 --save"
