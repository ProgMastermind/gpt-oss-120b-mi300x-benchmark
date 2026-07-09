#!/bin/bash
# Convenience wrapper for the single-stream 10K-in / 1.5K-out benchmark.
# Run this in a second terminal after the ATOM server is up on port 8000.

set -e

cd "$(dirname "$0")"

python benchmark_gptoss.py
