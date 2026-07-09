# gpt-oss-120b on AMD MI300X

This repo contains scripts to run and benchmark `openai/gpt-oss-120b` on a single AMD MI300X GPU using **ATOM** (the ROCm-optimized inference engine). The older vLLM scripts are still present for comparison but are no longer the recommended path.

## Files

| File | Purpose |
|------|---------|
| `start_server_atom_base.sh` | Start the ATOM server with the **base** recipe config |
| `start_server_atom_best.sh` | Start the ATOM server with the **optimized** CLI config |
| `start_server_atom_best_full_graph.sh` | Advanced: same as best but forces `cudagraph_mode = FULL_AND_PIECEWISE` |
| `start_server_atom_best_full_graph.py` | Python wrapper used by `start_server_atom_best_full_graph.sh` |
| `setup_atom.sh` | Install ATOM from source on a ROCm PyTorch base image (not needed if you use `rocm/atom`) |
| `benchmark_gptoss.py` | Single-stream 10K-in / 1.5K-out decode-throughput benchmark (works with ATOM and vLLM) |
| `benchmark_atom.sh` | Convenience wrapper for `benchmark_gptoss.py` |
| `benchmark_wafer.py` | Non-streaming raw-completions benchmark (legacy) |
| `benchmark_chat.py` | Chat-completions benchmark (legacy, better for Eagle 3) |
| `start_server.sh` / `start_server_eagle3*.sh` | Legacy vLLM server scripts |
| `runpod_template.md` | RunPod template configuration for ATOM (and legacy vLLM) |

## Quick Start on RunPod (ATOM)

### 1. Create the pod

Use the ATOM image in `runpod_template.md` (Option A). If you prefer to reuse a raw `rocm/pytorch` image, use Option B and run `setup_atom.sh` first.

### 2. Start the server

Base config (official recipe):

```bash
cd /workspace
git clone https://github.com/ProgMastermind/gpt-oss-120b-mi300x-benchmark.git gpt-oss
cd gpt-oss
bash start_server_atom_base.sh
```

Optimized/best config (CLI, safe):

```bash
bash start_server_atom_best.sh
```

Advanced best config (decode full CUDA graphs, may fail if the attention backend does not support it):

```bash
bash start_server_atom_best_full_graph.sh
```

The server will bind to `0.0.0.0:8000` and serve `openai/gpt-oss-120b` with an OpenAI-compatible `/v1/completions` endpoint.

### 3. Run the benchmark

In a second terminal:

```bash
cd /workspace/gpt-oss
bash benchmark_atom.sh
```

or directly:

```bash
python benchmark_gptoss.py
```

To benchmark a different model name loaded by the server, override the env variable:

```bash
BENCHMARK_MODEL="openai/gpt-oss-120b" python benchmark_gptoss.py
```

## Benchmark Methodology

The benchmark matches the Artificial Analysis single-stream workload:

- **Input:** ~10,000 tokens
- **Output:** 1,500 tokens
- **Concurrency:** 1
- **Metric:** Median decode throughput (output tokens / decode time, excluding TTFT)

## Important Notes

- **ATOM is the recommended engine** for `gpt-oss-120b` on ROCm. It uses AITER kernels, supports FP8 KV cache, and has a native `GptOssForCausalLM` implementation.
- **Base config** mirrors the official ROCm/ATOM recipe: `openai/gpt-oss-120b`, `--kv_cache_dtype fp8`, `--gpu-memory-utilization 0.5`.
- **Best config** adds:
  - `--max-model-len 12000` (fits the benchmark context)
  - `--block-size 64`
  - `--max-num-seqs 128` with `--cudagraph-capture-sizes [1,2,4,8,16,32,64,128]`
  - `--no-enable_prefix_caching` (unique prompts, no cache benefit)
  - `--level 3` piecewise compilation
- **ATOM CLI does not expose `--cudagraph-mode`**. The default for level 3 is `PIECEWISE`. The `FULL_AND_PIECEWISE` mode is the theoretical fastest for decode; `start_server_atom_best_full_graph.sh` enables it via a Python wrapper. If it crashes during CUDA graph capture, fall back to `start_server_atom_best.sh`.
- **Use the paired `rocm/atom-dev:atom0.1.5-aiter0.1.16` image.** The `rocm/atom:latest` (atom 0.1.4) tag has a known ATOM/AITER version mismatch and fails to load GPT-OSS with `ImportError: cannot import name 'swizzle_scales' from 'aiter.ops.triton.moe.moe_op_gemm_a8w4'`.
- **ATOM_USE_TRITON_MOE=1 is required** on the 0.1.5 dev image to load GPT-OSS's native MXFP4 MoE weights on MI300X (gfx942). Without it, `_swizzle_mxfp4` asserts and `ModelRunner` dies during model initialization.
- **Expected single-stream throughput on 1x MI300X:** unknown until measured; vLLM baseline was ~74 tok/s. Fireworks' 746 tok/s uses multi-GPU + custom kernels/speculative decoding and is not reproducible on a single MI300X with open-source tooling.
- First model download is ~63GB and may take 10-30 minutes.

## Troubleshooting

### `ImportError: cannot import name 'swizzle_scales' from 'aiter.ops.triton.moe.moe_op_gemm_a8w4'`

This means the container's `atom` and `aiter` packages are not paired correctly. The `rocm/atom:latest` / `rocm/atom:rocm7.2.4_..._atom0.1.4` images have this issue for GPT-OSS. The clean fix is to recreate the pod with `rocm/atom-dev:atom0.1.5-aiter0.1.16`.

If you must stay on the current pod, you can try forcing the non-Triton AITER/CK MoE path (slower, but may load):

```bash
export ATOM_USE_TRITON_MOE=0
bash start_server_atom_base.sh
```

Note that `start_server_atom_*.sh` currently sets `ATOM_USE_TRITON_MOE=1`; edit the script or run the server command manually with `ATOM_USE_TRITON_MOE=0` in the environment.
