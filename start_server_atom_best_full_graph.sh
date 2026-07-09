#!/bin/bash
set -e

# Advanced "best" ATOM server configuration that forces cudagraph_mode = FULL_AND_PIECEWISE.
# This uses a Python wrapper because the ATOM CLI does not expose --cudagraph-mode.
#
# If the attention backend fails to capture full CUDA graphs for gpt-oss-120b,
# fall back to start_server_atom_best.sh (CLI-only piecewise).

export HF_HOME=/workspace/.cache/huggingface
export SAFETENSORS_FAST_GPU=1

export HSA_NO_SCRATCH_RECLAIM=1
export AMDGCN_USE_BUFFER_OPS=0

mkdir -p "$HF_HOME"

echo "Starting ATOM server (best + FULL_AND_PIECEWISE) for openai/gpt-oss-120b on port 8000..."

python "$(dirname "$0")/start_server_atom_best_full_graph.py" \
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
