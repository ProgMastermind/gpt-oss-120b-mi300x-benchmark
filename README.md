# gpt-oss-120b on AMD MI300X

This repo contains scripts to run and benchmark `openai/gpt-oss-120b` on a single AMD MI300X GPU using vLLM.

## Files

| File | Purpose |
|------|---------|
| `start_server.sh` | Start the vLLM server on port 8000 |
| `start_server_eagle3.sh` | Start vLLM server with Eagle 3 speculative decoding |
| `setup.sh` | Install vLLM on a ROCm PyTorch base image (Option B) |
| `benchmark_gptoss.py` | Benchmark 10K input / 1.5K output using raw completions |
| `benchmark_wafer.py` | Non-streaming benchmark using raw completions |
| `benchmark_chat.py` | Benchmark 10K input / 1.5K output using chat completions (better for Eagle 3) |
| `runpod_template.md` | Exact RunPod template configuration |

## Quick Start on RunPod

### 1. Create the pod

Use the template in `runpod_template.md`. Choose Option A (official vLLM image) or Option B (PyTorch base image).

### 2. Connect and start the server

```bash
cd /workspace
git clone https://github.com/<user>/<repo>.git
cd <repo>
bash start_server.sh
```

For Option B, run setup first:
```bash
bash setup.sh
bash start_server.sh
```

### 3. Run the benchmark

In a second terminal:

```bash
cd /workspace/<repo>
python benchmark_gptoss.py
```

## Benchmark Methodology

The benchmark matches the Artificial Analysis single-stream workload:
- **Input:** ~10,000 tokens
- **Output:** 1,500 tokens
- **Concurrency:** 1
- **Metric:** Median decode throughput (output tokens / decode time, excluding TTFT)

## Important Notes

- **Expected single-stream throughput on 1x MI300X:** ~70-100 tok/s without speculative decoding. Chinmay Hebbal's MI300X benchmark reports ~63 tok/s per stream for gpt-oss-120b. Fireworks' 746 tok/s uses Eagle 3 speculative decoding + custom CUDA kernels, which is **not available on AMD/ROCm for gpt-oss-120b**.
- `FP8 KV cache` is not recommended for gpt-oss due to sliding-window attention layers.
- First model download is ~63GB and may take 10-30 minutes.
- `benchmark_chat.py` uses the chat completions API and a natural prompt. This may give better Eagle 3 acceptance rates than raw completions.
