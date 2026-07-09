# RunPod Template for gpt-oss-120b on MI300X

## Option A: Official ATOM ROCm Image (Recommended)

| Field | Value |
|-------|-------|
| Container image | `rocm/atom:latest` |
| Container start command (JSON) | `{"entrypoint": ["/bin/bash"], "cmd": ["-c", "sleep infinity"]}` |
| Container disk | 50 GB |
| Volume disk | 150 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8000` |
| Expose TCP ports | `22` |

ATOM is pre-installed and the `python -m atom.entrypoints.openai_server` command is available.

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
