#!/bin/bash
set -e

# Install vLLM on a ROCm PyTorch base image.
# Tested with: rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

echo "Installing vLLM 0.24.0 for ROCm..."

# ROCm 7.2.3 variant is the current wheel for vLLM 0.24.0; it works on rocm7.2.4 too.
pip install vllm==0.24.0+rocm723 --extra-index-url https://wheels.vllm.ai/rocm/0.24.0/rocm723

echo "Verifying installation..."
vllm --version

echo "Done."
