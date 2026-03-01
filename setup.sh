#!/usr/bin/env bash
# Instance Setup — Qwen3-32B on L20 GPUs
# Run on a fresh Tencent Cloud Ubuntu 22.04 instance.
#
# Usage:
#   bash setup.sh --vllm                        # setup for vLLM
#   bash setup.sh --tacox                       # setup for TACO-X (requires TACO_X_IMAGE)
set -euo pipefail

MODEL="Qwen/Qwen3-32B"

# ── Parse flags ──────────────────────────────────────────────────
ENGINE=""
case "${1:-}" in
    --vllm)  ENGINE="vllm" ;;
    --tacox) ENGINE="tacox" ;;
    *)
        echo "Usage: bash setup.sh --vllm | --tacox"
        echo ""
        echo "  --vllm    Install vLLM serving engine"
        echo "  --tacox   Install Docker + TACO-X container"
        exit 1
        ;;
esac

if [ "$ENGINE" = "tacox" ] && [ -z "${TACO_X_IMAGE:-}" ]; then
    echo "ERROR: TACO_X_IMAGE is not set."
    echo "  Contact your Tencent Cloud representative for the image URL, then:"
    echo "  export TACO_X_IMAGE=\"ccr.ccs.tencentyun.com/taco/taco_x_prod:<tag>\""
    echo "  bash setup.sh --tacox"
    exit 1
fi

echo "============================================"
echo "  Instance Setup"
echo "  Model:  ${MODEL}"
echo "  Engine: ${ENGINE}"
echo "============================================"
echo ""
echo "  Full run typically takes 25–45 min"
echo "  (dependencies + ~65 GB model download)."
echo ""

STEP=0

# ── Mount data disk ──────────────────────────────────────────────
STEP=$((STEP + 1))
if lsblk | grep -q vdb; then
    echo "[${STEP}] Mounting data disk..."
    if ! mount | grep -q /data; then
        sudo mkfs.ext4 -F /dev/vdb
        sudo mkdir -p /data
        sudo mount /dev/vdb /data
        echo '/dev/vdb /data ext4 defaults 0 0' | sudo tee -a /etc/fstab
        sudo chown -R "$USER:$USER" /data
    fi
    echo "  Data disk mounted at /data"
else
    echo "[${STEP}] No data disk found, using system disk"
    sudo mkdir -p /data
    sudo chown -R "$USER:$USER" /data
fi

# ── System packages ──────────────────────────────────────────────
STEP=$((STEP + 1))
echo ""
echo "[${STEP}] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv tmux htop > /dev/null 2>&1
echo "  Done"

# ── NVIDIA driver ────────────────────────────────────────────────
STEP=$((STEP + 1))
echo ""
echo "[${STEP}] Checking NVIDIA driver..."
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

# ── Python virtual environment ───────────────────────────────────
STEP=$((STEP + 1))
echo ""
echo "[${STEP}] Setting up Python environment..."
python3 -m venv /data/venv
source /data/venv/bin/activate
pip install --upgrade pip -q
pip install openai huggingface_hub -q

# ── Engine-specific install ──────────────────────────────────────
STEP=$((STEP + 1))
echo ""
if [ "$ENGINE" = "vllm" ]; then
    echo "[${STEP}] Installing vLLM..."
    pip install vllm -q
    echo "  vLLM version: $(python3 -c 'import vllm; print(vllm.__version__)')"
else
    echo "[${STEP}] Installing Docker + TACO-X..."

    if command -v docker &> /dev/null; then
        echo "  Docker already installed: $(docker --version)"
    else
        echo "  Installing Docker via Tencent mirror..."
        curl -s -L http://mirrors.tencent.com/install/GPU/taco/get-docker.sh | sudo bash
        echo "  Docker installed: $(docker --version)"
    fi

    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo "  nvidia runtime already available"
    else
        echo "  Installing nvidia-docker2 via Tencent mirror..."
        curl -s -L http://mirrors.tencent.com/install/GPU/taco/get-nvidiadocker2.sh | sudo bash
        sudo systemctl restart docker
        echo "  nvidia-docker2 installed"
    fi

    echo "  Pulling TACO-X image (first pull ~5 min)..."
    docker pull "$TACO_X_IMAGE"
    echo "  TACO-X image ready"
fi

# ── Download model ───────────────────────────────────────────────
STEP=$((STEP + 1))
echo ""
echo "[${STEP}] Downloading model: ${MODEL} (~65 GB)..."
echo "  This will take 15-30 min depending on bandwidth."
export HF_HUB_CACHE=/data/hf_cache
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}', cache_dir='/data/hf_cache')
print('  Model downloaded successfully')
"

# ── Verification ─────────────────────────────────────────────────
STEP=$((STEP + 1))
echo ""
echo "[${STEP}] Verification"
echo "============================================"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) devices"
echo "  VRAM:    $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1) per GPU"
echo "  Python:  $(python3 --version)"
if [ "$ENGINE" = "vllm" ]; then
    echo "  vLLM:    $(python3 -c 'import vllm; print(vllm.__version__)')"
else
    echo "  Docker:  $(docker --version)"
    echo "  Image:   $TACO_X_IMAGE"
fi
echo "  Model:   ${MODEL}"
echo "============================================"
echo ""
echo "Setup complete!"
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$ENGINE" = "vllm" ]; then
    # ── Launch vLLM server ───────────────────────────────────────
    STEP=$((STEP + 1))
    echo ""
    echo "[${STEP}] Launching vLLM in tmux session 'vllm'..."

    # Kill existing session if any
    tmux kill-session -t vllm 2>/dev/null || true

    VLLM_CMD="source /data/venv/bin/activate && export HF_HUB_CACHE=/data/hf_cache && vllm serve ${MODEL} \
        --dtype bfloat16 \
        --tensor-parallel-size ${GPU_COUNT} \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --cudagraph-capture-sizes 1 2 4 8 \
        --disable-custom-all-reduce \
        --port 8000"

    tmux new-session -d -s vllm "$VLLM_CMD"

    echo ""
    echo "============================================"
    echo "  vLLM launching on port 8000"
    echo "============================================"
    echo ""
    echo "  Model:  ${MODEL}"
    echo "  TP:     ${GPU_COUNT}"
    echo "  Port:   8000"
    echo ""
    echo "  Monitor startup logs:"
    echo "    tmux attach -t vllm"
    echo ""
    echo "  Wait for 'Application startup complete', then run benchmark:"
    echo ""
    echo "  source /data/venv/bin/activate"
    echo "  python3 benchmark.py --base-url http://localhost:8000/v1 \\"
    echo "    --model ${MODEL} --concurrency 1,5,10 --save"
else
    # ── Launch TACO-X container ──────────────────────────────────
    STEP=$((STEP + 1))
    echo ""
    echo "[${STEP}] Launching TACO-X container..."

    MODEL_TYPE="qwen3_32b"
    CONFIG_DIR="/workspace/qwen3_32b_taco_x_config"
    CONTAINER_NAME="taco_x"
    PORT=18080

    # Stop existing vLLM if running
    tmux kill-session -t vllm 2>/dev/null && echo "  Stopped vLLM tmux session" || true

    # Locate model snapshot
    MODEL_DIR=$(ls -d /data/hf_cache/models--Qwen--Qwen3-32B/snapshots/*/ 2>/dev/null | head -1)
    if [ -z "${MODEL_DIR:-}" ]; then
        echo "  ERROR: Model not found at /data/hf_cache/models--Qwen--Qwen3-32B/"
        exit 1
    fi
    echo "  Model found: $MODEL_DIR"

    # Remove existing container if any
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "  Removing existing container '$CONTAINER_NAME'..."
        docker rm -f "$CONTAINER_NAME"
    fi

    # Build docker run command
    DOCKER_ARGS=(
        -itd
        --gpus all
        --privileged --cap-add=IPC_LOCK
        --ulimit memlock=-1 --ulimit stack=67108864
        --net=host --ipc=host
        -v /data:/data
        --name="$CONTAINER_NAME"
        --entrypoint python3
    )

    TACO_ARGS=(
        -m taco_x.api_server
        --model_dir "$MODEL_DIR"
        --model_type "$MODEL_TYPE"
        --config_dir "$CONFIG_DIR"
        --port "$PORT"
        --opt-level 3
    )

    # Auto-detect GPU count for TP
    if [ "$GPU_COUNT" -gt 1 ]; then
        TACO_ARGS+=(--tp "$GPU_COUNT")
        echo "  Tensor parallelism: $GPU_COUNT (auto-detected)"
    fi

    echo "  Launching container..."
    docker run "${DOCKER_ARGS[@]}" "$TACO_X_IMAGE" "${TACO_ARGS[@]}"

    echo ""
    echo "============================================"
    echo "  TACO-X launched on port $PORT"
    echo "============================================"
    echo ""
    echo "  Model:      $MODEL_DIR"
    echo "  Model type: $MODEL_TYPE"
    echo "  Port:       $PORT"
    echo ""
    echo "  Monitor startup logs:"
    echo "    docker logs -f $CONTAINER_NAME"
    echo ""
    echo "  Wait for 'Application startup complete', then run benchmark:"
    echo ""
    echo "  source /data/venv/bin/activate"
    echo "  python3 benchmark.py --base-url http://localhost:${PORT}/v1 \\"
    echo "    --model $MODEL_DIR --concurrency 1,5,10 --save"
fi
