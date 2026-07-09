"""
Non-streaming benchmark script for gpt-oss-120b on AMD MI300X.

Workload: 10K input tokens / 1.5K output tokens (matches Artificial Analysis).

Usage:
    1. Start the vLLM server on port 8000
    2. Run: python benchmark_wafer.py

Notes:
    - Uses the OpenAI non-streaming completions API.
    - Measures total time and decode throughput.
    - If the server returns ttft_s in usage, it will be subtracted to show decode-only tok/s.
    - Unique prompts prevent prefix-cache speedups.
"""

import requests
import time

url = 'http://localhost:8000/v1/completions'

# Base text (~70 tokens per repeat, 143 repeats = ~10,020 tokens)
base_text = (
    'The history of artificial intelligence spans several decades, beginning with early theoretical '
    'work in the 1940s and 1950s. Researchers like Alan Turing proposed fundamental questions about '
    'machine intelligence, while others developed the first neural network models. Over time, advances '
    'in computing power, data availability, and algorithmic improvements have transformed the field. '
)

# Warmup run - triggers AITER JIT kernel compilation
print('Warmup run...')
warmup_prompt = 'Warmup: ' + base_text * 143 + ' End.'
payload = {
    'model': 'openai/gpt-oss-120b',
    'prompt': warmup_prompt,
    'max_tokens': 1500,
    'ignore_eos': True,
    'temperature': 0
}
requests.post(url, json=payload)
print('Warmup done\n')

# Benchmark runs with unique prompts
results = []
ttfts = []
for i in range(5):
    unique_prompt = f'Run {i}: ' + base_text * 143 + f' Unique marker {i}.'
    payload = {
        'model': 'openai/gpt-oss-120b',
        'prompt': unique_prompt,
        'max_tokens': 1500,
        'ignore_eos': True,
        'temperature': 0
    }

    start = time.time()
    response = requests.post(url, json=payload)
    end = time.time()
    data = response.json()
    output_tokens = data['usage']['completion_tokens']
    input_tokens = data['usage']['prompt_tokens']
    total_time = end - start
    ttft = data['usage'].get('ttft_s', 0)
    decode_time = total_time - ttft
    tps = output_tokens / decode_time if decode_time > 0 else output_tokens / total_time
    results.append(tps)
    ttfts.append(ttft)
    print(f'Run {i+1}: {input_tokens} in, {output_tokens} out, {total_time:.2f}s total, {tps:.1f} tok/s, TTFT {ttft:.3f}s')

avg = sum(results) / len(results)
avg_ttft = sum(ttfts) / len(ttfts)
print(f'\nAverage decode throughput: {avg:.1f} tok/s')
print(f'Average TTFT: {avg_ttft:.3f}s')
print(f'Min/Max throughput: {min(results):.1f} / {max(results):.1f} tok/s')
