#!/bin/bash
set -e

# Manual profiling workflow for ATOM (base config).
# Terminal 1: bash start_server_atom_profile.sh
# Terminal 2: bash profile_atom_manual.sh

echo "Waiting for ATOM server to become healthy..."
until curl -fs http://localhost:8000/health >/dev/null; do
    sleep 5
done
echo "Server is ready."

echo "Starting torch profiler..."
curl -X POST http://localhost:8000/start_profile

echo "Running benchmark..."
python benchmark_gptoss.py

echo "Stopping profiler and flushing trace (this may take a few minutes)..."
curl -X POST http://localhost:8000/stop_profile

echo ""
echo "Trace files should be under /workspace/atom_profile_base/rank_0/"
echo "Open the .pt.trace.json.gz file in https://ui.perfetto.dev/ to inspect."
