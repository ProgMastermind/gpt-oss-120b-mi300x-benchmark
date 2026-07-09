#!/bin/bash
set -e

# Install vLLM on a ROCm PyTorch base image.
# Tested with: rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

echo "Installing vLLM 0.24.0 for ROCm..."

# Pin torch/torchvision/torchaudio to the official ROCm 7.2 builds so the
# vllm wheel does not pull a non-ROCm or mismatched torch from its own index.
TMPDIR=$(mktemp -d)
cat > "$TMPDIR/vllm_constraints.txt" <<EOF
torch==2.11.0+rocm7.2
torchvision==0.26.0+rocm7.2
torchaudio==2.11.0+rocm7.2
EOF

pip install vllm==0.24.0+rocm723 \
  --extra-index-url https://wheels.vllm.ai/rocm/0.24.0/rocm723 \
  --extra-index-url https://download.pytorch.org/whl/rocm7.2 \
  -c "$TMPDIR/vllm_constraints.txt"

# If atom is also installed in this venv, disable its vLLM plugin to avoid
# loading a newer AITER than the vllm wheel expects.
export ATOM_DISABLE_VLLM_PLUGIN=1

echo "Verifying installation..."
vllm --version

echo "Done."
