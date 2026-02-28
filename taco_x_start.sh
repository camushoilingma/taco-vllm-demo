#!/usr/bin/env bash
# TACO-X — Start the TACO-X inference server.
# The Docker image is private. Contact Tencent Cloud to obtain access.
#
# Usage:
#   export TACO_X_IMAGE="ccr.ccs.tencentyun.com/taco/taco_x_prod:<tag>"
#   bash taco_x_start.sh                    # Qwen3-32B, auto-detect TP
#   bash taco_x_start.sh --tp 4             # Force TP=4
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
if [ -z "${TACO_X_IMAGE:-}" ]; then
    echo "ERROR: TACO_X_IMAGE is not set."
    echo "  Contact your Tencent Cloud representative for the image URL, then:"
    echo "  export TACO_X_IMAGE=\"ccr.ccs.tencentyun.com/taco/taco_x_prod:<tag>\""
    exit 1
fi

MODEL_TYPE="qwen3_32b"
CONFIG_DIR="/workspace/qwen3_32b_taco_x_config"
CONTAINER_NAME="taco_x"
PORT=18080
TP_SIZE="${1:-}"   # optional: --tp N

echo "============================================"
echo "  TACO-X — Instance Setup"
echo "  Image: $TACO_X_IMAGE"
echo "============================================"
echo ""

# ── [1/5] Stop existing vLLM ─────────────────────────────────────
echo "[1/5] Stopping vLLM (if running)..."
tmux kill-session -t vllm 2>/dev/null && echo "  vLLM tmux session killed" || echo "  No vLLM session running"

# ── [2/5] Install Docker ─────────────────────────────────────────
echo ""
echo "[2/5] Checking Docker..."
if command -v docker &> /dev/null; then
    echo "  Docker already installed: $(docker --version)"
else
    echo "  Installing Docker via Tencent mirror..."
    curl -s -L http://mirrors.tencent.com/install/GPU/taco/get-docker.sh | sudo bash
    echo "  Docker installed: $(docker --version)"
fi

# ── [3/5] Install nvidia-docker ──────────────────────────────────
echo ""
echo "[3/5] Checking nvidia-docker2..."
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "  nvidia runtime already available"
else
    echo "  Installing nvidia-docker2 via Tencent mirror..."
    curl -s -L http://mirrors.tencent.com/install/GPU/taco/get-nvidiadocker2.sh | sudo bash
    sudo systemctl restart docker
    echo "  nvidia-docker2 installed"
fi

# ── [4/5] Locate model ──────────────────────────────────────────
echo ""
echo "[4/5] Locating Qwen3-32B in /data/hf_cache..."
MODEL_DIR=$(ls -d /data/hf_cache/models--Qwen--Qwen3-32B/snapshots/*/ 2>/dev/null | head -1)
if [ -z "${MODEL_DIR:-}" ]; then
    echo "  ERROR: Model not found at /data/hf_cache/models--Qwen--Qwen3-32B/"
    echo "  Run setup.sh first to download the model."
    exit 1
fi
echo "  Model found: $MODEL_DIR"

# ── [5/5] Pull image and start TACO-X ───────────────────────────
echo ""
echo "[5/5] Starting TACO-X container..."

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "  Removing existing container '$CONTAINER_NAME'..."
    docker rm -f "$CONTAINER_NAME"
fi

echo "  Pulling TACO-X image (first pull ~5 min)..."
docker pull "$TACO_X_IMAGE"

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
)

# Add TP if specified
if [ "$TP_SIZE" = "--tp" ] && [ -n "${2:-}" ]; then
    TACO_ARGS+=(--tp "$2")
    echo "  Tensor parallelism: $2"
else
    # Auto-detect GPU count
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        TACO_ARGS+=(--tp "$GPU_COUNT")
        echo "  Tensor parallelism: $GPU_COUNT (auto-detected)"
    fi
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
echo "  Wait for 'Application startup complete', then:"
echo ""
echo "  source /data/venv/bin/activate"
echo "  python3 benchmark.py --base-url http://localhost:${PORT}/v1 \\"
echo "    --model $MODEL_DIR --concurrency 1,5,10 --save"
