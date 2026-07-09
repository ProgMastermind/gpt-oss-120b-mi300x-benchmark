#!/bin/bash
set -e

# Base ATOM server configuration for openai/gpt-oss-120b on a single MI300X.
# This mirrors the official ROCm/ATOM recipe exactly.

export HF_HOME=/workspace/.cache/huggingface
export SAFETENSORS_FAST_GPU=1

# ROCm environment knobs that are safe for MI300X and commonly used with vLLM/ATOM.
export HSA_NO_SCRATCH_RECLAIM=1
export AMDGCN_USE_BUFFER_OPS=0

mkdir -p "$HF_HOME"

echo "Starting ATOM server (base config) for openai/gpt-oss-120b on port 8000..."

python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 \
  --server-port 8000
