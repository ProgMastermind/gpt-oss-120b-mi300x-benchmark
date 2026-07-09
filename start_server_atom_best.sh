#!/bin/bash
set -e

# Optimized ATOM server configuration for openai/gpt-oss-120b on a single MI300X.
#
# Differences from the base recipe:
#   - max-model-len clamped to 12000 (covers the 10K-in / 1.5K-out benchmark)
#   - larger block-size (64) to reduce block-manager overhead
#   - max-num-seqs and cudagraph-capture-sizes tuned for single/low-concurrency
#   - prefix caching disabled (unique prompts in benchmark, no benefit)
#   - level 3 piecewise compilation with targeted CUDA graph capture
#
# Note: ATOM's CLI does not expose --cudagraph-mode. If you want to try
# FULL_AND_PIECEWISE (full decode graphs + piecewise prefill), you currently
# need to patch atom.config.Config.__post_init__ or use the programmatic API.

export HF_HOME=/workspace/.cache/huggingface
export SAFETENSORS_FAST_GPU=1

# ROCm environment knobs that are safe for MI300X.
export HSA_NO_SCRATCH_RECLAIM=1
export AMDGCN_USE_BUFFER_OPS=0

mkdir -p "$HF_HOME"

echo "Starting ATOM server (best config) for openai/gpt-oss-120b on port 8000..."

python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 12000 \
  --block-size 64 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 12000 \
  --level 3 \
  --cudagraph-capture-sizes "[1,2,4,8,16,32,64,128]" \
  --no-enable_prefix_caching \
  --host 0.0.0.0 \
  --server-port 8000
