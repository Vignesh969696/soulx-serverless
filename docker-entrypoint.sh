#!/bin/bash
set -e

echo "Checking models..."

if [ ! -d "/workspace/models/SoulX-FlashHead-1_3B" ]; then
    echo "ERROR: SoulX model not found"
    exit 1
fi

if [ ! -d "/workspace/models/wav2vec2-base-960h" ]; then
    echo "ERROR: wav2vec model not found"
    exit 1
fi

mkdir -p /workspace/outputs

echo "Starting RunPod Serverless Worker..."

python dual-mode-worker/handler.py
