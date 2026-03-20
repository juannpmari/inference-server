#!/usr/bin/env python3
"""
Benchmark script for the inference engine.

Sends N queries to the engine and reports mean/median latency.
Supports both mock and real engine modes.

Usage:
    # Against mock engine (standalone, no server needed)
    python scripts/benchmark.py --mode mock --n 10

    # Against a running engine server
    python scripts/benchmark.py --mode server --n 10 --url http://localhost:8080

    # With custom prompts
    python scripts/benchmark.py --mode mock --n 5 --prompt "Explain quantum computing"
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

# Add project root to path so we can import engine modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SAMPLE_PROMPTS = [
    "hello, how are you?",
    "who created this model?",
    "what is the meaning of life?",
    "test the inference pipeline",
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean.",
    "Summarize the key ideas behind transformers in deep learning.",
    "What are the main differences between Python and Rust?",
    "Describe the process of photosynthesis.",
    "How does a compiler differ from an interpreter?",
]


async def benchmark_mock(n: int, prompt: str | None, max_tokens: int, temperature: float) -> list[float]:
    """Run benchmark directly against the mock engine (no server needed)."""
    from data_plane.inference.engine.config import EngineConfig
    from data_plane.inference.engine.mock_engine import MockLLMEngine

    config = EngineConfig(enable_engine_mock=True)
    engine = MockLLMEngine(config)
    loop_task = asyncio.create_task(engine.continuous_batching_loop())

    latencies: list[float] = []
    try:
        for i in range(n):
            p = prompt if prompt else SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            t0 = time.perf_counter()
            result = await engine.add_request(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            print(f"  [{i+1:>{len(str(n))}}/{n}] {elapsed*1000:8.2f} ms | {result[:60]}...")
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    return latencies


async def benchmark_server(n: int, prompt: str | None, url: str, max_tokens: int, temperature: float) -> list[float]:
    """Run benchmark against a running engine server via HTTP."""
    import httpx

    latencies: list[float] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Check server health first
        try:
            resp = await client.get(f"{url}/health")
            resp.raise_for_status()
        except httpx.RequestError as e:
            print(f"Error: cannot reach server at {url} ({e})")
            sys.exit(1)

        for i in range(n):
            p = prompt if prompt else SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            payload = {
                "prompt": p,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            t0 = time.perf_counter()
            resp = await client.post(f"{url}/inference", json=payload)
            elapsed = time.perf_counter() - t0

            if resp.status_code != 200:
                print(f"  [{i+1:>{len(str(n))}}/{n}] ERROR {resp.status_code}: {resp.text}")
                continue

            data = resp.json()
            latencies.append(elapsed)
            text_preview = data["text"][:60]
            print(f"  [{i+1:>{len(str(n))}}/{n}] {elapsed*1000:8.2f} ms | {text_preview}...")

    return latencies


def print_report(latencies: list[float]) -> None:
    """Print latency statistics."""
    if not latencies:
        print("\nNo successful requests to report.")
        return

    ms = [l * 1000 for l in latencies]
    print("\n" + "=" * 50)
    print("  BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  Requests total : {len(ms)}")
    print(f"  Mean latency   : {statistics.mean(ms):>10.2f} ms")
    print(f"  Median latency : {statistics.median(ms):>10.2f} ms")
    print(f"  Stdev          : {statistics.stdev(ms):>10.2f} ms" if len(ms) > 1 else "")
    print(f"  Min            : {min(ms):>10.2f} ms")
    print(f"  Max            : {max(ms):>10.2f} ms")
    print(f"  p90            : {statistics.quantiles(ms, n=10)[8]:>10.2f} ms" if len(ms) >= 10 else "")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the inference engine")
    parser.add_argument("--mode", choices=["mock", "server"], default="mock",
                        help="'mock' runs the mock engine in-process; 'server' hits a running engine via HTTP")
    parser.add_argument("--n", type=int, default=10, help="Number of queries to send (default: 10)")
    parser.add_argument("--url", type=str, default="http://localhost:8080",
                        help="Engine URL for server mode (default: http://localhost:8080)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Use a fixed prompt for all requests (default: rotate through sample prompts)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per request (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    args = parser.parse_args()

    print(f"Benchmarking in '{args.mode}' mode with {args.n} requests\n")

    if args.mode == "mock":
        latencies = asyncio.run(benchmark_mock(args.n, args.prompt, args.max_tokens, args.temperature))
    else:
        latencies = asyncio.run(benchmark_server(args.n, args.prompt, args.url, args.max_tokens, args.temperature))

    print_report(latencies)


if __name__ == "__main__":
    main()
