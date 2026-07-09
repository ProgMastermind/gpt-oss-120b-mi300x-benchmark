#!/bin/bash
set -e

# Environment variables for vLLM on AMD MI300X
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_MHA=1
export AMDGCN_USE_BUFFER_OPS=0
export HSA_NO_SCRATCH_RECLAIM=1
export HF_HOME=/workspace/.cache/huggingface
export SAFETENSORS_FAST_GPU=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1

mkdir -p /workspace/.cache/huggingface

echo "Starting vLLM server for openai/gpt-oss-120b on port 8000..."

vllm serve openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --attention-backend ROCM_AITER_FA \
  --no-enable-prefix-caching \
  --no-enable-log-requests \
  --gpu-memory-utilization 0.85 \
  --max-model-len 12000 \
  --max-num-seqs 256 \
  --enable-chunked-prefill=False \
  --block-size 64 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "pass_config": {"fuse_rope_kvcache": true}}'
