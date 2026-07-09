#!/bin/bash
set -e

# Experimental: Eagle 3 speculative decoding with Nebius draft model
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export AMDGCN_USE_BUFFER_OPS=0
export HSA_NO_SCRATCH_RECLAIM=1
export HF_HOME=/workspace/.cache/huggingface
export SAFETENSORS_FAST_GPU=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1

mkdir -p /workspace/.cache/huggingface

echo "Starting vLLM server with Nebius Eagle 3 draft model..."

vllm serve openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --no-enable-prefix-caching \
  --no-enable-log-requests \
  --gpu-memory-utilization 0.85 \
  --max-model-len 12000 \
  --max-num-seqs 256 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method": "eagle3", "model": "nebius/EAGLE3-gpt-oss-120b", "num_speculative_tokens": 6, "draft_tensor_parallel_size": 1}'
