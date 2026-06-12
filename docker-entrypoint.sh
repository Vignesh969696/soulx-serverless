#!/bin/bash
set -e

mkdir -p /workspace/models

if [ ! -d "/workspace/models/SoulX-FlashHead-1_3B" ]; then
    echo "Downloading SoulX model..."
    hf download \
        Soul-AILab/SoulX-FlashHead-1_3B \
        --local-dir /workspace/models/SoulX-FlashHead-1_3B
fi

if [ ! -d "/workspace/models/wav2vec2-base-960h" ]; then
    echo "Downloading wav2vec model..."
    hf download \
        facebook/wav2vec2-base-960h \
        --local-dir /workspace/models/wav2vec2-base-960h
fi

echo "Starting FastAPI..."

uvicorn dual-mode-worker.fastapi_server:app \
    --host 0.0.0.0 \
    --port 8000
