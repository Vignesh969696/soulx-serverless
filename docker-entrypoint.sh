#!/bin/bash
set -e

echo "=== DEBUG START ==="

which hf || true

hf --help || true

pip show huggingface_hub || true

echo "=== DEBUG END ==="

sleep infinity
