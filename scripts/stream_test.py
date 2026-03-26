#!/usr/bin/env python3
"""
Stream a chat completion from the gateway and print tokens as they arrive.
Reports time-to-first-token (TTFT) and inter-token latency stats at the end.

Usage:
    python scripts/stream_test.py                          # defaults
    python scripts/stream_test.py --url http://host:8000   # custom gateway
    python scripts/stream_test.py --model mymodel          # custom model
    python scripts/stream_test.py --prompt "Explain gravity in 3 sentences"
"""

import argparse
import json
import sys
import time

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_PROMPT = "Explain how large language models work in 5 sentences."


def stream_chat(base_url: str, model: str, prompt: str, max_tokens: int) -> None:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }

    token_times: list[float] = []
    first_token_time: float | None = None
    request_start = time.perf_counter()

    print(f"Model : {model}")
    print(f"Prompt: {prompt}")
    print(f"URL   : {url}")
    print("-" * 60)

    with httpx.Client(timeout=httpx.Timeout(connect=10, read=None, write=10, pool=10)) as client:
        with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                print(f"Error: HTTP {resp.status_code}")
                print(resp.read().decode())
                sys.exit(1)

            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content is None:
                    continue

                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)

                # Print token immediately, no newline buffering
                sys.stdout.write(content)
                sys.stdout.flush()

    request_end = time.perf_counter()
    print()
    print("-" * 60)

    # --- Metrics ---
    total_time = request_end - request_start
    n_tokens = len(token_times)

    if n_tokens == 0:
        print("No tokens received.")
        return

    ttft = first_token_time - request_start

    inter_token_latencies: list[float] = []
    for i in range(1, len(token_times)):
        inter_token_latencies.append(token_times[i] - token_times[i - 1])

    # Prefill ≈ TTFT (request sent → first token received; includes network RTT)
    # Decode  = first token → last token (pure autoregressive generation phase)
    decode_time = token_times[-1] - first_token_time if n_tokens > 1 else 0.0

    print(f"Tokens received      : {n_tokens}")
    print(f"Total time (e2e)     : {total_time * 1000:.1f} ms")
    print()
    print(f"Prefill latency*     : {ttft * 1000:.1f} ms   (≈ TTFT)")
    print(f"Decode latency       : {decode_time * 1000:.1f} ms   ({n_tokens - 1} decode steps)")

    if inter_token_latencies:
        avg_itl = sum(inter_token_latencies) / len(inter_token_latencies)
        min_itl = min(inter_token_latencies)
        max_itl = max(inter_token_latencies)
        sorted_itl = sorted(inter_token_latencies)
        p50 = sorted_itl[len(sorted_itl) // 2]
        p90 = sorted_itl[min(int(len(sorted_itl) * 0.9), len(sorted_itl) - 1)]
        p99 = sorted_itl[min(int(len(sorted_itl) * 0.99), len(sorted_itl) - 1)]
        print(f"Inter-token latency  : avg {avg_itl * 1000:.1f} ms | "
              f"p50 {p50 * 1000:.1f} ms | p90 {p90 * 1000:.1f} ms | "
              f"p99 {p99 * 1000:.1f} ms")
        print(f"  (range)            : min {min_itl * 1000:.1f} ms | max {max_itl * 1000:.1f} ms")
        tps = (n_tokens - 1) / decode_time if decode_time > 0 else float("inf")
        print(f"Decode throughput    : {tps:.1f} tokens/s")

    print()
    print("* Prefill latency is measured client-side (TTFT) and includes network RTT."
          " For server-side prefill/decode, check Prometheus metrics"
          " (engine_prefill_seconds, engine_decode_tokens_per_second).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream chat completion and measure latency")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Gateway base URL (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    args = parser.parse_args()

    stream_chat(args.url, args.model, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()
