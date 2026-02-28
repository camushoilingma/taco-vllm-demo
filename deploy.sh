#!/usr/bin/env bash
# Upload scripts to the remote GPU instance and run setup.
# Reads connection details from .env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.env"

echo "Uploading to $REMOTE_USER@$REMOTE_IP ..."
scp -i "$SSH_KEY" \
    "$SCRIPT_DIR/setup.sh" \
    "$SCRIPT_DIR/benchmark.py" \
    "$SCRIPT_DIR/chat.py" \
    "$REMOTE_USER@$REMOTE_IP:~/"

echo ""
echo "Running setup.sh on remote instance ..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_IP" "bash setup.sh"
