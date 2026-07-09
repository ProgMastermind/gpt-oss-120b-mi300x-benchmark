# RunPod Template for gpt-oss-120b on MI300X

## Option A: Official vLLM ROCm Image (Recommended)

| Field | Value |
|-------|-------|
| Container image | `vllm/vllm-openai-rocm:v0.24.0` |
| Container start command (JSON) | `{"entrypoint": ["/bin/bash"], "cmd": ["-c", "sleep infinity"]}` |
| Container disk | 50 GB |
| Volume disk | 150 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8000` |
| Expose TCP ports | `22` |

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

For Option B, run `setup.sh` after connecting to install vLLM.

## Environment Variables

```
VLLM_ROCM_USE_AITER=1
VLLM_USE_AITER_UNIFIED_ATTENTION=1
VLLM_ROCM_USE_AITER_MHA=0
AMDGCN_USE_BUFFER_OPS=0
HSA_NO_SCRATCH_RECLAIM=1
HF_HOME=/workspace/.cache/huggingface
SAFETENSORS_FAST_GPU=1
HF_TOKEN=hf_your_token_here
```
