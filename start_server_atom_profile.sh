#!/bin/bash
set -e

# Start ATOM server with torch profiling enabled.
# Run this in Terminal 1.
# In Terminal 2 run: bash profile_atom_manual.sh

export HF_HOME=/workspace/.cache/huggingface
export SAFETENSORS_FAST_GPU=1

export HSA_NO_SCRATCH_RECLAIM=1
export AMDGCN_USE_BUFFER_OPS=0

# Required for GPT-OSS's native MXFP4 MoE weights on MI300X (gfx942).
export ATOM_USE_TRITON_MOE=1

# Keeps profiler overhead low (no record_shapes, with_stack, profile_memory).
export ATOM_PROFILER_MORE=0

mkdir -p "$HF_HOME"

echo "Starting ATOM server (base config, profiling enabled) on port 8000..."

python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 \
  --server-port 8000 \
  --torch-profiler-dir /workspace/atom_profile_base \
  --mark-trace
