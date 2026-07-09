"""
Benchmark script for gpt-oss-120b on AMD MI300X.

Workload: 10K input tokens / 1.5K output tokens (matches Artificial Analysis).
Method: Single-stream decode throughput, excluding TTFT.

Usage:
    1. Start the vLLM/ATOM server on port 8000
    2. Run: python benchmark_gptoss.py

Environment variables:
    - BENCHMARK_URL:  OpenAI completions endpoint (default: http://localhost:8000/v1/completions)
    - BENCHMARK_MODEL: model name to send in requests (default: openai/gpt-oss-120b)

Notes:
    - Uses the OpenAI streaming completions API to accurately measure TTFT.
    - Decode throughput = output_tokens / (total_time - ttft).
    - Unique prompts prevent prefix-cache speedups.
    - One warmup request is run first to avoid JIT compilation skew.
"""

import os
import requests
import time
import json
import sys

URL = os.environ.get("BENCHMARK_URL", "http://localhost:8000/v1/completions")
MODEL = os.environ.get("BENCHMARK_MODEL", "openai/gpt-oss-120b")

# Base text (~70 tokens per repeat). 143 repeats gives ~10,000 tokens.
BASE_TEXT = (
    "The history of artificial intelligence spans several decades, beginning with early "
    "theoretical work in the 1940s and 1950s. Researchers like Alan Turing proposed fundamental "
    "questions about machine intelligence, while others developed the first neural network models. "
    "Over time, advances in computing power, data availability, and algorithmic improvements have "
    "transformed the field. "
)
REPEATS = 143
OUTPUT_TOKENS = 1500
NUM_RUNS = 5


def make_prompt(tag: str) -> str:
    return f"{tag}: " + BASE_TEXT * REPEATS + f" End marker {tag}."


def send_streaming_request(prompt: str):
    """
    Send a streaming completion request and return:
      - input_tokens
      - output_tokens
      - ttft (seconds)
      - decode_time (seconds)
      - total_time (seconds)
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    response = requests.post(URL, json=payload, stream=True)
    response.raise_for_status()

    t_first = None
    output_tokens = 0
    input_tokens = 0
    final_chunk_time = None

    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        if "choices" in chunk and chunk["choices"]:
            if t_first is None:
                t_first = time.perf_counter()
            if chunk["choices"][0].get("finish_reason") in ("length", "stop"):
                final_chunk_time = time.perf_counter()

        if "usage" in chunk and chunk["usage"] is not None:
            input_tokens = chunk["usage"].get("prompt_tokens", input_tokens)
            output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

    # If usage was not returned, fallback to requested output length
    if output_tokens == 0:
        output_tokens = OUTPUT_TOKENS

    t_end = final_chunk_time or time.perf_counter()

    total_time = t_end - t0
    ttft = (t_first - t0) if t_first else 0.0
    decode_time = total_time - ttft

    return input_tokens, output_tokens, ttft, decode_time, total_time


def main():
    print(f"URL: {URL}")
    print(f"Model: {MODEL}")
    print(f"Target input tokens: ~{REPEATS * 70}")
    print(f"Target output tokens: {OUTPUT_TOKENS}")
    print(f"Number of runs: {NUM_RUNS}\n")

    # Warmup
    print("Warmup run...")
    try:
        send_streaming_request(make_prompt("Warmup"))
        print("Warmup done.\n")
    except requests.exceptions.RequestException as e:
        print(f"Warmup failed: {e}")
        sys.exit(1)

    results = []
    ttfts = []
    input_tokens_list = []
    output_tokens_list = []

    for i in range(1, NUM_RUNS + 1):
        prompt = make_prompt(f"Run {i}")
        try:
            input_tokens, output_tokens, ttft, decode_time, total_time = send_streaming_request(prompt)
        except requests.exceptions.RequestException as e:
            print(f"Run {i} failed: {e}")
            continue

        tps = output_tokens / decode_time if decode_time > 0 else 0.0

        results.append(tps)
        ttfts.append(ttft)
        input_tokens_list.append(input_tokens)
        output_tokens_list.append(output_tokens)

        print(
            f"Run {i}: {input_tokens} in, {output_tokens} out, "
            f"{total_time:.2f}s total, {ttft:.3f}s TTFT, {decode_time:.2f}s decode, "
            f"{tps:.1f} tok/s"
        )

    if not results:
        print("No successful benchmark runs.")
        sys.exit(1)

    avg_input = sum(input_tokens_list) / len(input_tokens_list)
    avg_output = sum(output_tokens_list) / len(output_tokens_list)
    avg_ttft = sum(ttfts) / len(ttfts)
    avg_tps = sum(results) / len(results)
    median_tps = sorted(results)[len(results) // 2]

    print(f"\nAverage input tokens: {avg_input:.0f}")
    print(f"Average output tokens: {avg_output:.0f}")
    print(f"Average TTFT: {avg_ttft:.3f}s")
    print(f"Average decode throughput: {avg_tps:.1f} tok/s")
    print(f"Median decode throughput: {median_tps:.1f} tok/s")
    print(f"Min/Max decode throughput: {min(results):.1f} / {max(results):.1f} tok/s")


if __name__ == "__main__":
    main()
