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
- **Benchmark script:** `benchmark_gptoss.py` (single-stream 10K-in / 1.5K-out)

## Benchmark Results

Single-stream `openai/gpt-oss-120b` on 1x MI300X:

| Engine | Config | Throughput (tok/s) | Notes |
|--------|--------|-------------------|-------|
| vLLM 0.24.0 | AITER unified attention, full/piecewise CUDA graphs, `--gpu-memory-utilization 0.85`, `--max-num-seqs 256` | ~74 | Baseline without profiling |
| vLLM 0.24.0 | Same as baseline + torch profiler | ~67.8 | Profiling overhead lowers throughput |
| vLLM 0.24.0 | Best speculative-decoding run (Eagle 3) | ~81 | Unstable; not reproducible as a steady state |
| vLLM 0.24.0 | Eagle 3 with other draft models (RedHatAI, Nebius, NVIDIA) | slower than baseline | Mostly detrimental; did not improve on 1x MI300X |
| ATOM 0.1.5 | **base** (official recipe) | 194.6 | Best stable baseline on 1x MI300X |
| ATOM 0.1.5 | **best + full graph** (`FULL_AND_PIECEWISE` decode graphs) | 196.1 | Small gain over base |
| ATOM 0.1.5 | **best CLI** (piecewise CUDA graphs) | 200.6 | Highest non-spec throughput |
| ATOM 0.1.5 | **base** with profiling enabled | 167.6 | Profiling overhead lowers throughput |
| ATOM 0.1.5 | **no Eagle3** (base model, `benchmark_gptoss.py`) | 209.5 | True baseline with same benchmark script |
| ATOM 0.1.5 | **Eagle3** `num_spec=1`, `nvidia/gpt-oss-120b-Eagle3-throughput` | **291.6** | Best stable throughput; +39% over no-spec baseline |
| ATOM 0.1.5 | **Eagle3** `num_spec=2` | 275.9 | Worse than `num_spec=1`; higher variance |
| ATOM 0.1.5 | **Eagle3** `num_spec=1` + Triton fusion flags | 290.2 | No gain; fusion flags are no-ops on single GPU |

### Key takeaway

Eagle3 speculative decoding with `num_spec=1` is the single biggest optimization for GPT-OSS-120b on MI300X:

- **+39% over no-spec baseline** (209.5 → 291.6 tok/s)
- **+45% over ATOM best CLI** (200.6 → 291.6 tok/s)
- **~3.9x faster than vLLM baseline** (74 → 291.6 tok/s)

`num_spec=2` hurts throughput due to higher draft rejection rate. Triton kernel fusion flags (`ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT`, `ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT`, `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION`) are no-ops on single GPU and do not improve throughput.

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
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Prevents ROCm scratch buffer reclaim overhead. |
| `NCCL_P2P_DISABLE` | `1` | Disables NCCL P2P on single-GPU (avoids unnecessary overhead). |
| `AITER_LOG_LEVEL` | `WARNING` | Suppresses aiter kernel log flooding. |
| `HF_HOME` | `/root/huggingface_cache` | Model download location (avoids root disk filling up). |
| `HF_HUB_DISABLE_XET` | `1` | Disables xet download backend (avoids ENOSPC errors). |

### Server command (with Eagle3)

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

### Server command (without Eagle3, baseline)

```bash
python -m atom.entrypoints.openai_server \
  --model openai/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --level 3
```

### aiter version

- **`v0.1.16.post3`** (tuned for gfx942/MI300X) — recommended, gives ~292 tok/s with Eagle3.
- **`main`** — less tuned for gfx942; CK MoE path crashes with LLVM error on MI300X.

### Benchmark result — Eagle3 `num_spec=1` (5 runs, 2026-07-15)

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

### Benchmark result — no Eagle3 baseline (5 runs, 2026-07-15)

```
Run 1: 9592 in, 1500 out, 8.06s total, 0.908s TTFT, 7.15s decode, 209.7 tok/s
Run 2: 9592 in, 1500 out, 8.07s total, 0.907s TTFT, 7.16s decode, 209.5 tok/s
Run 3: 9592 in, 1500 out, 8.07s total, 0.908s TTFT, 7.16s decode, 209.4 tok/s
Run 4: 9592 in, 1500 out, 8.07s total, 0.907s TTFT, 7.17s decode, 209.3 tok/s
Run 5: 9592 in, 1500 out, 8.07s total, 0.906s TTFT, 7.17s decode, 209.3 tok/s

Average decode throughput: 209.5 tok/s
Median decode throughput:  209.4 tok/s
Min/Max:                   209.3 / 209.7 tok/s
```

## Files

- `start_server_atom_base.sh` / `start_server_atom_best.sh` / `start_server_atom_best_full_graph.sh` — ATOM server configs
- `start_server.sh` / `start_server_eagle3*.sh` — legacy vLLM server configs
- `benchmark_gptoss.py` — single-stream 10K-in / 1.5K-out decode benchmark
- `profile_atom.sh` / `profile_atom_base.sh` / `profile_atom_manual.sh` / `start_server_atom_profile.sh` — ATOM profiling scripts
- `profile_vllm.sh` — vLLM profiling script
- `runpod_template.md` — RunPod template with images and entry commands

## Core Contributor

Maintained by [ProgMastermind](https://github.com/ProgMastermind).
