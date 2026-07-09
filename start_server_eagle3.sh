#!/bin/bash
set -e

# Experimental: Eagle 3 speculative decoding for gpt-oss-120b on AMD MI300X
# WARNING: This is NOT verified on AMD/ROCm. It may fail, crash, or produce wrong output.
# If it fails, use start_server.sh instead.

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

echo "Starting vLLM server with Eagle 3 speculative decoding for openai/gpt-oss-120b..."

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
  --speculative-config '{"method": "eagle3", "model": "RedHatAI/gpt-oss-120b-speculator.eagle3", "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1}'
