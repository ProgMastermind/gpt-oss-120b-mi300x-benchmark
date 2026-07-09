#!/bin/bash
set -e

# Install vLLM on a ROCm PyTorch base image.
# Tested with: rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

echo "Installing vLLM 0.24.0 for ROCm..."

pip install vllm==0.24.0+rocm721 --extra-index-url https://wheels.vllm.ai/rocm/0.24.0/rocm721

echo "Verifying installation..."
vllm --version

echo "Done."
