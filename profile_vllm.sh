#!/bin/bash
set -e

# vLLM PyTorch profiling run for openai/gpt-oss-120b on a single MI300X.
# The trace is viewable in https://ui.perfetto.dev/.

# vLLM/ROCm environment knobs that were used for the ~74 tok/s baseline.
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export AMDGCN_USE_BUFFER_OPS=0
export HSA_NO_SCRATCH_RECLAIM=1
export HF_HOME=/workspace/.cache/huggingface
export SAFETENSORS_FAST_GPU=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1

PROFILE_DIR=/workspace/vllm_profile
mkdir -p "$PROFILE_DIR"

VLLM_LOG="vllm_profile_server.log"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server (PID $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting vLLM server with torch profiler..."
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
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROFILE_DIR\"}" \
  > "$VLLM_LOG" 2>&1 &

SERVER_PID=$!
echo "vLLM server PID: $SERVER_PID"

echo "Waiting for server to become healthy (this may take a while while weights load)..."
until curl -fs http://localhost:8000/health >/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died. Check $VLLM_LOG for details."
        exit 1
    fi
    sleep 5
done
echo "Server is ready."

echo "Starting torch profiler..."
curl -X POST http://localhost:8000/start_profile

echo "Running benchmark..."
python benchmark_gptoss.py

echo "Stopping profiler and flushing trace (this may take a few minutes)..."
curl -X POST http://localhost:8000/stop_profile

echo "Trace files written to $PROFILE_DIR"
echo "Open the .pt.trace.json.gz file in https://ui.perfetto.dev/ to inspect."
