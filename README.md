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

### Eagle3 speculative decoding (ATOM + aiter v0.1.16.post3)

| Engine | Image | Config | Throughput (tok/s) | Notes |
|--------|-------|--------|-------------------|-------|
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16` | Eagle3, `num_spec=1`, `nvidia/gpt-oss-120b-Eagle3-throughput`, `--level 3` | **291.6** | Best stable throughput; +45% over ATOM best CLI |
| ATOM 0.1.5 | `rocm/atom-dev:atom0.1.5-aiter0.1.16` | Same, without `ATOM_USE_TRITON_MOE=1` | 208.4 | Triton MoE kernel is the key enabler |

### DigitalOcean AMD Developer Cloud (MI300X, `rocm/atom` image)

| Engine | Image | Config | Throughput (tok/s) | Notes |
|--------|-------|--------|-------------------|-------|
| ATOM 0.1.5 | `rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4_202607091539` | Eagle3, `num_spec=1`, `nvidia/gpt-oss-120b-Eagle3-throughput`, `--level 3`, `ATOM_USE_TRITON_MOE=1` | **291.6** | Matches RunPod result on same image + aiter v0.1.16.post3 |
| ATOM pr branch | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` | Eagle3, `num_spec=1`, aiter main (not v0.1.16.post3) | 158.5 | `aiter main` less tuned for gfx942; `shuffle_scale_moe` patch needed |

ATOM on the paired `0.1.5` image is ~2.5-2.7x faster than the stable vLLM 0.24.0 baseline on the same single-MI300X setup. Speculative decoding on vLLM for this model on ROCm did not yield a stable improvement.

With Eagle3 speculative decoding and `ATOM_USE_TRITON_MOE=1`, ATOM reaches **~292 tok/s** — a **~45% improvement** over the non-speculative ATOM best CLI (200.6 tok/s) and **~3.9x faster** than vLLM baseline.

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

## Eagle3 speculative decoding setup

### Source code changes (ATOM `test/without-pr3` branch)

The Eagle3 support required three changes to ATOM source code:

1. **`atom/models/eagle3_llama.py`** — Eagle3 draft model fixes:
   - Pass `dtype` and `rope_scaling` to the draft RoPE.
   - Add `input_norm` (global RMSNorm over concatenated aux hidden states) gated by `norm_before_fc` in the checkpoint config. GPT-OSS-120b sets `norm_before_fc=True`, so the full `[N, target_hidden_size * num_aux]` tensor is normalized before `fc`.
   - Fix `fc_norm` chunking to use the contiguous concatenated tensor.

2. **`atom/models/gpt_oss.py`** — target model aux hidden state capture:
   - Capture the final target layer's `aux_hidden_state` and feed it to the Eagle3 draft.

3. **`atom/spec_decode/eagle.py`** — embedding/lm_head sharing:
   - Share target `embed_tokens` and `lm_head` with the Eagle3 draft when the draft checkpoint does not include them (NVIDIA's `gpt-oss-120b-Eagle3-*` checkpoints omit these).
   - Guard `embed_tokens` sharing against pipeline parallelism (only share when `get_pp_group().world_size == 1`).

### Key environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `ATOM_USE_TRITON_MOE` | `1` | Use Triton MoE kernels for GPT-OSS-120b MoE layers. Without this, throughput drops from ~292 to ~208 tok/s. |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Prevents ROCm scratch buffer reclaim overhead. |
| `NCCL_P2P_DISABLE` | `1` | Disables NCCL P2P on single-GPU (avoids unnecessary overhead). |
| `AITER_LOG_LEVEL` | `WARNING` | Suppresses aiter kernel log flooding. |
| `HF_HOME` | `/root/huggingface_cache` | Model download location (avoids root disk filling up). |
| `HF_HUB_DISABLE_XET` | `1` | Disables xet download backend (avoids ENOSPC errors). |

### Server command

```bash
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --method eagle3 \
  --num-speculative-tokens 1 \
  --draft-model nvidia/gpt-oss-120b-Eagle3-throughput \
  --level 3
```

### aiter version

- **`v0.1.16.post3`** (tuned for gfx942/MI300X) — gives ~292 tok/s.
- **`main`** (less tuned for gfx942, requires `shuffle_scale_moe` patch) — gives ~158 tok/s on `rocm/pytorch` image.

### Benchmark result (5 runs, 2026-07-15)

```
Run 1: 9592 in, 1500 out, 6.02s total, 0.923s TTFT, 5.10s decode, 294.2 tok/s
Run 2: 9592 in, 1501 out, 6.16s total, 0.923s TTFT, 5.24s decode, 286.7 tok/s
Run 3: 9592 in, 1500 out, 6.28s total, 0.923s TTFT, 5.35s decode, 280.2 tok/s
Run 4: 9592 in, 1500 out, 6.00s total, 0.922s TTFT, 5.08s decode, 295.3 tok/s
Run 5: 9592 in, 1500 out, 5.90s total, 0.928s TTFT, 4.97s decode, 301.7 tok/s

Average decode throughput: 291.6 tok/s
Median decode throughput:  294.2 tok/s
Min/Max:                   280.2 / 301.7 tok/s
```

## Core Contributor

Maintained by [ProgMastermind](https://github.com/ProgMastermind).
