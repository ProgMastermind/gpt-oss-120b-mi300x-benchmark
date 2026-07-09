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
| `profile_vllm.sh` | vLLM run with torch profiler enabled; produces a trace for `https://ui.perfetto.dev/` |
| `profile_atom.sh` | ATOM run with torch profiler enabled (best config) |
| `profile_atom_base.sh` | ATOM run with torch profiler enabled (base config, all-in-one script) |
| `start_server_atom_profile.sh` | Start ATOM server with profiling enabled (use with `profile_atom_manual.sh`) |
| `profile_atom_manual.sh` | Terminal-2 profiling script for the two-terminal ATOM workflow |
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

## Benchmark Results

Single-stream `openai/gpt-oss-120b` on 1x MI300X (median decode tok/s, excluding TTFT):

| Engine | Image / Config | Throughput | Notes |
|--------|----------------|------------|-------|
| vLLM 0.24.0 (ROCm) | `vllm/vllm-openai-rocm:v0.24.0`, AITER unified attention, full/piecewise CUDA graphs | ~74 tok/s | Baseline measured earlier in the day; profiling run produced a Perfetto trace |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16`, **base** config | 194.6 tok/s | Mirrors the official ROCm/ATOM recipe |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16`, **best + full graph** config | 196.1 tok/s | `FULL_AND_PIECEWISE` decode CUDA graphs |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16`, **best CLI-only** config | 200.6 tok/s | Highest single-stream throughput measured |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16`, **base** with profiling enabled | 167.6 tok/s | Profiling overhead lowers the number; trace analyzed separately |

The key take-away is that ATOM on the paired `0.1.5` image is roughly **2.6-2.7x faster** than the vLLM 0.24.0 baseline on the same single-MI300X setup.

## RunPod Images and Entry Commands

For all three paths, set the **container start command** (or `entrypoint` JSON) to `sleep infinity` so the pod stays alive while you connect and run scripts:

- **ATOM (recommended):** `rocm/atom-dev:atom0.1.5-aiter0.1.16` — start command `sleep infinity` (or `{"entrypoint": ["/bin/bash"], "cmd": ["-c", "sleep infinity"]}`)
- **ROCm PyTorch base:** `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` — start command `sleep infinity`, then run `setup_atom.sh`
- **Legacy vLLM:** `vllm/vllm-openai-rocm:v0.24.0` — start command `sleep infinity` (or `{"entrypoint": ["/bin/bash"], "cmd": ["-c", "sleep infinity"]}`)

Do **not** use `rocm/atom:latest` or `rocm/atom:rocm7.2.4_..._atom0.1.4` for GPT-OSS; those tags have an ATOM/AITER mismatch that causes `ImportError: cannot import name 'swizzle_scales' from 'aiter.ops.triton.moe.moe_op_gemm_a8w4'`.

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
- **ATOM_USE_TRITON_MOE=1 is required** on the 0.1.5 dev image to load GPT-OSS's native MXFP4 MoE weights on MI300X (gfx942). Without it, `_swizzle_mxfp4` asserts and `ModelRunner` dies during model initialization.
- **Profiling adds overhead** and should not be used for final throughput numbers.
- First model download is ~63GB and may take 10-30 minutes.

## Profiling

### Profiling details (what is required)

1. **Start the server with a profiler output directory.**  
   - vLLM: pass `--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/workspace/vllm_profile"}'`.  
   - ATOM: pass `--torch-profiler-dir /workspace/atom_profile_base` **and** `--mark-trace`. Without `--mark-trace` the ATOM server will not emit the runtime trace.
2. **Begin the trace window.** For both engines `POST /start_profile` opens the recording window. In ATOM, `--mark-trace` tells the engine to record the `capture_graph` and runtime phases.
3. **Run the benchmark inside the window.** Use `python benchmark_gptoss.py` (or `bash benchmark_atom.sh`). The workload is the same 10K-in / 1.5K-out single-stream decode used for throughput numbers.
4. **Stop the profiler and collect the trace.** `POST /stop_profile` flushes the trace. For ATOM the files are under `/workspace/atom_profile_base/rank_0/`; for vLLM under `/workspace/vllm_profile/`. Open the `.pt.trace.json.gz` in `https://ui.perfetto.dev/`.

### vLLM → Perfetto

Run `profile_vllm.sh`. It starts `vllm serve` with `--profiler-config` and the torch profiler, calls `/start_profile`, runs `benchmark_gptoss.py`, then `/stop_profile`. The resulting `.pt.trace.json.gz` in `/workspace/vllm_profile` can be opened in `https://ui.perfetto.dev/`.

### ATOM → Perfetto

All-in-one:

- `profile_atom.sh` (best config)
- `profile_atom_base.sh` (base config)

Two-terminal (recommended for transparency, because the all-in-one scripts can hang during trace flush and hide server output):

1. `bash start_server_atom_profile.sh` in Terminal 1
2. `bash profile_atom_manual.sh` in Terminal 2

The second terminal waits for `/health`, calls `/start_profile`, runs `benchmark_gptoss.py`, then `/stop_profile`. The resulting `.pt.trace.json.gz` is written under `/workspace/atom_profile_base/rank_0/` and can be opened in `https://ui.perfetto.dev/`.

### Profile analysis snapshot (ATOM base, 2026-07-09)

A 60-second trace of the ATOM base config was captured and summarized. The dominant findings were:

- `Memcpy HtoD (Host -> Device)` — 65.46 s total across 7,393 calls. This was the single largest time consumer in the trace, highlighting host-to-device copy overhead on each decode step.
- `hipEventSynchronize` — 34.35 s total. The CPU spent a significant amount of time waiting for GPU events.
- `decode` loop (GPU) — 42.88 s total. The actual decode work on the GPU.
- `_moe_gemm_a16w4` — 18.69 s total. The MXFP4 MoE GEMM is the largest kernel category.
- `paged_attention_decode_sliding_window` — 2.77 s total. Attention is a small fraction of the decode step.
- `cpu_op` total — 2.47 s. CPU-side ATen operations are not the bottleneck.

So the primary optimization targets on this single-MI300X setup are **host-to-device memory copies and CPU/GPU synchronization**, not the attention or MoE compute kernels themselves.

## Troubleshooting

### `ImportError: cannot import name 'swizzle_scales' from 'aiter.ops.triton.moe.moe_op_gemm_a8w4'`

This means the container's `atom` and `aiter` packages are not paired correctly. The `rocm/atom:latest` / `rocm/atom:rocm7.2.4_..._atom0.1.4` images have this issue for GPT-OSS. The clean fix is to recreate the pod with `rocm/atom-dev:atom0.1.5-aiter0.1.16`.

If you must stay on the current pod, you can try forcing the non-Triton AITER/CK MoE path (slower, but may load):

```bash
export ATOM_USE_TRITON_MOE=0
bash start_server_atom_base.sh
```

Note that `start_server_atom_*.sh` currently sets `ATOM_USE_TRITON_MOE=1`; edit the script or run the server command manually with `ATOM_USE_TRITON_MOE=0` in the environment.

## Core Contributor

This project is maintained by **[ProgMastermind](https://github.com/ProgMastermind)**.
