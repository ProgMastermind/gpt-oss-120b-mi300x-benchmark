#!/bin/bash
set -e

# ATOM setup for a ROCm PyTorch base image.
# Recommended: use the official rocm/atom image instead of running this script
# (see runpod_template.md). This script is only needed if you start from a raw
# rocm/pytorch container.

# Base image tested by ATOM upstream:
#   rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0
# Newer images such as rocm7.2.4 + torch2.10 may work with the matching AITER wheel.

echo "Installing ATOM from upstream..."

# Make sure ROCm libraries are visible.
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Optional GPU_ARCHS for AITER source builds.
export GPU_ARCHS="gfx942"

# Clone ATOM source and install in develop mode.
# If the clone already exists, remove it to avoid stale state.
if [ -d "/workspace/ATOM" ]; then
    echo "Removing existing /workspace/ATOM clone..."
    rm -rf /workspace/ATOM
fi

git clone --recursive https://github.com/ROCm/ATOM.git /workspace/ATOM
cd /workspace/ATOM

pip install -r requirements.txt
python3 setup.py develop

echo "Verifying ATOM installation..."
python - <<'PY'
import atom
print("ATOM installed at:", atom.__file__)
PY

echo "Done. ATOM is installed from source."
