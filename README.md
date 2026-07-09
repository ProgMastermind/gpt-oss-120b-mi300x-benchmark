# gpt-oss-120b on AMD MI300X

Single-stream benchmark and profiling results for `openai/gpt-oss-120b` on one AMD MI300X GPU using vLLM and ATOM.

## RunPod Images and Entry Commands Used

| Path | Container image | Entry command |
|------|-----------------|---------------|
| vLLM | `vllm/vllm-openai-rocm:v0.24.0` | `sleep infinity` (or `{"entrypoint": ["/bin/bash"], "cmd": ["-c", "sleep infinity"]}`) |
| ATOM (recommended) | `rocm/atom-dev:atom0.1.5-aiter0.1.16` | `sleep infinity` (or `{"entrypoint": ["/bin/bash"], "cmd": ["-c", "sleep infinity"]}`) |
| ROCm PyTorch base | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` | `sleep infinity` |

`rocm/atom:latest` (atom 0.1.4) has an ATOM/AITER mismatch and will not load GPT-OSS.

## Benchmark Methodology

- **Input:** ~10,000 tokens
- **Output:** 1,500 tokens
- **Concurrency:** 1
- **Metric:** Median decode throughput (output tokens / decode time, excluding TTFT)

## Benchmark Results

Single-stream `openai/gpt-oss-120b` on 1x MI300X:

| Engine | Image | Config | Throughput (tok/s) | Notes |
|--------|-------|--------|-------------------|-------|
| vLLM 0.24.0 | `vllm/vllm-openai-rocm:v0.24.0` | AITER unified attention, full/piecewise CUDA graphs, `--gpu-memory-utilization 0.85`, `--max-num-seqs 256` | ~74 | Baseline without profiling |
| vLLM 0.24.0 | `vllm/vllm-openai-rocm:v0.24.0` | Same as baseline + torch profiler | ~67.8 | Profiling overhead lowers throughput |
| vLLM 0.24.0 | `vllm/vllm-openai-rocm:v0.24.0` | Best speculative-decoding run (Eagle 3) | ~81 | Unstable; not reproducible as a steady state |
| vLLM 0.24.0 | `vllm/vllm-openai-rocm:v0.24.0` | Eagle 3 with other draft models (RedHatAI, Nebius, NVIDIA) | slower than baseline | Mostly detrimental; did not improve on 1x MI300X |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16` | **base** (official recipe) | 194.6 | Best stable baseline on 1x MI300X |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16` | **best + full graph** (`FULL_AND_PIECEWISE` decode graphs) | 196.1 | Small gain over base |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16` | **best CLI** (piecewise CUDA graphs) | 200.6 | Highest measured stable throughput |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16` | **base** with profiling enabled | 167.6 | Profiling overhead lowers throughput |

ATOM on the paired `0.1.5` image is ~2.5-2.7x faster than the stable vLLM 0.24.0 baseline on the same single-MI300X setup. Speculative decoding on vLLM for this model on ROCm did not yield a stable improvement.

## Profiling Results

### ATOM base trace summary (60 s, 10.6 M events)

| Category | Operation | Total time (s) | Calls |
|----------|-----------|----------------|-------|
| GPU memory | `Memcpy HtoD (Host -> Device)` | 65.46 | 7,393 |
| CPU wait | `hipEventSynchronize` | 34.35 | 7,383 |
| GPU annotation | `decode[bs=1 tok=1 d=1]` | 42.88 | 7,378 |
| GPU kernel | `_moe_gemm_a16w4` | 18.69 | 531,548 |
| GPU kernel | `paged_attention_decode_sliding_window` | 2.77 | 265,594 |
| CPU operations | `cpu_op` (all ATen ops) | 2.47 | 593,251 |

The main bottleneck is **host-to-device memory copies and CPU/GPU synchronization**, not the MoE or attention compute itself. The attention kernel takes only 2.77 s, while the MoE GEMM takes 18.69 s; the much larger consumers are `Memcpy HtoD` (65.46 s) and `hipEventSynchronize` (34.35 s).

## Files

- `start_server_atom_base.sh` / `start_server_atom_best.sh` / `start_server_atom_best_full_graph.sh` — ATOM server configs
- `start_server.sh` / `start_server_eagle3*.sh` — legacy vLLM server configs
- `benchmark_gptoss.py` — single-stream 10K-in / 1.5K-out decode benchmark
- `profile_atom.sh` / `profile_atom_base.sh` / `profile_atom_manual.sh` / `start_server_atom_profile.sh` — ATOM profiling scripts
- `profile_vllm.sh` — vLLM profiling script
- `runpod_template.md` — RunPod template with images and entry commands

## Core Contributor

Maintained by [ProgMastermind](https://github.com/ProgMastermind).
