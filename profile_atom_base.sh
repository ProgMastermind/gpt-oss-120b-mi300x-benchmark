#!/bin/bash
set -e

# ATOM PyTorch profiling run for openai/gpt-oss-120b on a single MI300X.
# Uses the base ATOM config (simple, easy to interpret in Perfetto).
# The trace is written to /workspace/atom_profile_base and viewable in
# https://ui.perfetto.dev/.

# ATOM / ROCm environment knobs from start_server_atom_base.sh.
export HF_HOME=/workspace/.cache/huggingface
export SAFETENSORS_FAST_GPU=1
export HSA_NO_SCRATCH_RECLAIM=1
export AMDGCN_USE_BUFFER_OPS=0

# Required for GPT-OSS's native MXFP4 MoE weights on MI300X (gfx942).
export ATOM_USE_TRITON_MOE=1

# Optional: set ATOM_PROFILER_MORE=1 to record shapes, stacks, and memory.
# This adds overhead; leave at 0 for a lighter trace.
export ATOM_PROFILER_MORE=0

PROFILE_DIR=/workspace/atom_profile_base
mkdir -p "$PROFILE_DIR"

ATOM_LOG="atom_profile_base_server.log"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping ATOM base server (PID $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting ATOM base server with torch profiler..."
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 \
  --server-port 8000 \
  --torch-profiler-dir "$PROFILE_DIR" \
  > "$ATOM_LOG" 2>&1 &

SERVER_PID=$!
echo "ATOM base server PID: $SERVER_PID"

echo "Waiting for server to become healthy (this may take a while while weights load)..."
until curl -fs http://localhost:8000/health >/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died. Check $ATOM_LOG for details."
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

echo "Trace files written to $PROFILE_DIR (likely under $PROFILE_DIR/rank_0/)"
echo "Open the .pt.trace.json.gz file in https://ui.perfetto.dev/ to inspect."
