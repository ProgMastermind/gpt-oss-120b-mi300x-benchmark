# RunPod Template for gpt-oss-120b on MI300X

## Option A: Official ATOM ROCm Image (Recommended)

| Field | Value |
|-------|-------|
| Container image | `rocm/atom-dev:atom0.1.5-aiter0.1.16` |
| Container start command (JSON) | `{"entrypoint": ["/bin/bash"], "cmd": ["-c", "sleep infinity"]}` |
| Container disk | 50 GB |
| Volume disk | 150 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8000` |
| Expose TCP ports | `22` |

ATOM is pre-installed and the `python -m atom.entrypoints.openai_server` command is available.

> **Important:** The `rocm/atom:latest` (atom 0.1.4) and `rocm/atom:rocm7.2.4_..._atom0.1.4` tags have a known ATOM/AITER mismatch that causes `ImportError: cannot import name 'swizzle_scales' from 'aiter.ops.triton.moe.moe_op_gemm_a8w4'` when loading GPT-OSS. Use the paired `rocm/atom-dev:atom0.1.5-aiter0.1.16` image instead.

## Option B: ROCm PyTorch Base Image

| Field | Value |
|-------|-------|
| Container image | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` |
| Container start command | `sleep infinity` |
| Container disk | 50 GB |
| Volume disk | 150 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8000` |
| Expose TCP ports | `22` |

For Option B, run `setup_atom.sh` after connecting to install ATOM from source.

## Option C: Legacy vLLM ROCm Image

| Field | Value |
|-------|-------|
| Container image | `vllm/vllm-openai-rocm:v0.24.0` |
| Container start command (JSON) | `{"entrypoint": ["/bin/bash"], "cmd": ["-c", "sleep infinity"]}` |
| Container disk | 50 GB |
| Volume disk | 150 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8000` |
| Expose TCP ports | `22` |

For Option C, the vLLM scripts `setup.sh`, `start_server.sh` and `start_server_eagle3*.sh` can be used.

## Environment Variables

```
HSA_NO_SCRATCH_RECLAIM=1
AMDGCN_USE_BUFFER_OPS=0
ATOM_USE_TRITON_MOE=1
HF_HOME=/workspace/.cache/huggingface
SAFETENSORS_FAST_GPU=1
HF_TOKEN=hf_your_token_here
```
