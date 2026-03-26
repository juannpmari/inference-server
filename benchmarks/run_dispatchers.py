#!/usr/bin/env python3
"""Standalone test runner for the three dispatch modes.

Sends real requests to the running engine, collects ResponseRecords,
and prints structured JSON results for each mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Add project root to path so benchmarks package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.dispatchers.base import Prompt, ResponseRecord
from benchmarks.dispatchers.sequential import SequentialDispatcher
from benchmarks.dispatchers.concurrent import ConcurrentDispatcher
from benchmarks.dispatchers.realistic import RealisticDispatcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ENGINE_URL = "http://localhost:8080"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# ---------------------------------------------------------------------------
# Lightweight BenchmarkClient (engine target only)
# ---------------------------------------------------------------------------

class BenchmarkClient:
    """Minimal SSE streaming client for the engine /generate/stream endpoint."""

    def __init__(self, engine_url: str, timeout: float = 60.0, max_connections: int = 50):
        self.engine_url = engine_url.rstrip("/")
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=30.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=20,
            ),
        )

    async def close(self):
        await self.http.aclose()

    async def send_request(self, prompt: Prompt) -> ResponseRecord:
        url = f"{self.engine_url}/generate/stream"
        payload = {
            "prompt": prompt.text,
            "max_tokens": prompt.max_tokens,
            "stream": True,
        }
        t0 = time.monotonic()
        token_times: list[float] = []
        output_tokens = 0

        try:
            async with self.http.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = b""
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                    return ResponseRecord(
                        prompt_name=prompt.name,
                        input_tokens=prompt.input_tokens,
                        max_tokens=prompt.max_tokens,
                        ttft=0.0,
                        itl=[],
                        e2e=time.monotonic() - t0,
                        output_tokens=0,
                        http_status=resp.status_code,
                        error=f"HTTP {resp.status_code}: {body.decode(errors='replace')[:200]}",
                    )

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "token" in data and data["token"]:
                                token_times.append(time.monotonic())
                                output_tokens += 1
                        except json.JSONDecodeError:
                            pass

            t_end = time.monotonic()
            ttft = (token_times[0] - t0) if token_times else (t_end - t0)
            itl = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
            e2e = t_end - t0

            return ResponseRecord(
                prompt_name=prompt.name,
                input_tokens=prompt.input_tokens,
                max_tokens=prompt.max_tokens,
                ttft=ttft,
                itl=itl,
                e2e=e2e,
                output_tokens=output_tokens,
                http_status=200,
                error=None,
            )
        except Exception as exc:
            return ResponseRecord(
                prompt_name=prompt.name,
                input_tokens=prompt.input_tokens,
                max_tokens=prompt.max_tokens,
                ttft=0.0,
                itl=[],
                e2e=time.monotonic() - t0,
                output_tokens=0,
                http_status=0,
                error=f"{type(exc).__name__}: {exc}",
            )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(category: str, max_tokens: int, limit: int | None = None) -> list[Prompt]:
    manifest_path = PROMPTS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest[category]
    if limit:
        entries = entries[:limit]
    prompts = []
    for entry in entries:
        text = (PROMPTS_DIR / entry["file"]).read_text().strip()
        name = Path(entry["file"]).stem
        prompts.append(Prompt(
            name=name,
            text=text,
            input_tokens=entry["input_tokens"],
            max_tokens=max_tokens,
        ))
    return prompts


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_percentiles(values: list[float], with_minmax: bool = True) -> dict:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    result = {
        "p50": round(float(np.percentile(values, 50)), 6),
        "p90": round(float(np.percentile(values, 90)), 6),
        "p99": round(float(np.percentile(values, 99)), 6),
    }
    if with_minmax:
        result["min"] = round(min(values), 6)
        result["max"] = round(max(values), 6)
    return result


def select_median_run(runs: list[list[ResponseRecord]]) -> tuple[list[ResponseRecord], int]:
    medians = []
    for i, run in enumerate(runs):
        successful = [r.e2e for r in run if r.error is None]
        med = statistics.median(successful) if successful else float("inf")
        medians.append((med, i))
    medians.sort(key=lambda x: x[0])
    idx = medians[1][1]  # middle of 3
    return runs[idx], idx


def build_client_side_stats(records: list[ResponseRecord]) -> dict:
    ok = [r for r in records if r.error is None and r.http_status == 200]
    ttfts = [r.ttft for r in ok]
    all_itl = [itl for r in ok for itl in r.itl]
    e2es = [r.e2e for r in ok]
    norm = [r.e2e / (r.input_tokens + r.output_tokens)
            for r in ok if (r.input_tokens + r.output_tokens) > 0]
    return {
        "ttft": compute_percentiles(ttfts),
        "itl": compute_percentiles(all_itl),
        "e2e": compute_percentiles(e2es),
        "normalized_e2e": compute_percentiles(norm, with_minmax=False),
        "total_requests": len(records),
        "successful_requests": len(ok),
        "failed_requests": len(records) - len(ok),
    }


def fetch_metrics_sync(engine_url: str) -> dict:
    """Blocking metrics fetch (used inside async via to_thread)."""
    try:
        resp = httpx.get(f"{engine_url}/metrics_summary", timeout=5.0)
        return resp.json()
    except Exception:
        return {}


async def fetch_metrics(engine_url: str) -> dict:
    return await asyncio.to_thread(fetch_metrics_sync, engine_url)


def compute_delta(before: dict, after: dict) -> dict:
    if not before or not after:
        return {}
    try:
        request_count = (
            after.get("session", {}).get("total_requests", 0)
            - before.get("session", {}).get("total_requests", 0)
        )
        return {
            "request_count": request_count,
            "latency": after.get("latency", {}),
            "throughput": after.get("throughput", {}),
            "kv_cache": {
                "hit_rate": after.get("kv_cache", {}).get("hit_rate", 0.0),
                "l1_utilization_ratio": after.get("kv_cache", {}).get("l1_utilization_ratio", 0.0),
                "eviction_count": (
                    after.get("kv_cache", {}).get("eviction_count", 0)
                    - before.get("kv_cache", {}).get("eviction_count", 0)
                ),
            },
            "errors": {
                "total_errors": (
                    after.get("errors", {}).get("total_errors", 0)
                    - before.get("errors", {}).get("total_errors", 0)
                ),
                "error_rate": after.get("errors", {}).get("error_rate", 0.0),
            },
            "queue": after.get("queue", {}),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Run protocol: warmup + 3 measurement runs, select median
# ---------------------------------------------------------------------------

async def run_mode(dispatcher, mode_name: str, num_runs: int = 3) -> dict:
    """Run warmup + measurement runs, return structured results."""
    logger.info("=== %s: warmup ===", mode_name)
    await dispatcher.warmup(n=2)

    logger.info("=== %s: %d measurement runs ===", mode_name, num_runs)
    all_runs: list[list[ResponseRecord]] = []
    deltas: list[dict] = []

    for i in range(num_runs):
        before = await fetch_metrics(ENGINE_URL)
        logger.info("  run %d/%d ...", i + 1, num_runs)
        records = await dispatcher.run()
        await asyncio.sleep(0.5)  # let metrics settle
        after = await fetch_metrics(ENGINE_URL)
        delta = compute_delta(before, after)
        all_runs.append(records)
        deltas.append(delta)
        if i < num_runs - 1:
            logger.info("  cooldown 5s ...")
            await asyncio.sleep(5)

    median_records, median_idx = select_median_run(all_runs)
    logger.info("  median run: #%d", median_idx + 1)

    return {
        "mode": mode_name,
        "median_run_index": median_idx,
        "client_side": build_client_side_stats(median_records),
        "server_side_delta": deltas[median_idx],
        "per_request": [
            {
                "prompt_name": r.prompt_name,
                "input_tokens": r.input_tokens,
                "max_tokens": r.max_tokens,
                "output_tokens": r.output_tokens,
                "ttft": round(r.ttft, 6),
                "itl_count": len(r.itl),
                "itl_mean": round(statistics.mean(r.itl), 6) if r.itl else 0.0,
                "e2e": round(r.e2e, 6),
                "http_status": r.http_status,
                "error": r.error,
            }
            for r in median_records
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def restart_engine():
    """Restart the engine container and wait until it can actually generate."""
    logger.info("Restarting engine ...")
    proc = await asyncio.create_subprocess_exec(
        "docker", "compose", "restart", "engine",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    # Wait until the engine can serve a real generate request
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as c:
        for _ in range(60):
            await asyncio.sleep(3)
            try:
                resp = await c.post(
                    f"{ENGINE_URL}/generate/stream",
                    json={"prompt": "test", "max_tokens": 1, "stream": True},
                )
                if resp.status_code == 200:
                    logger.info("Engine ready (generate works).")
                    await asyncio.sleep(2)  # extra settle time
                    return
            except Exception:
                pass
    raise TimeoutError("Engine not ready after restart")


async def main():
    # Use a small subset of prompts for this test run (5 short prompts)
    prompts_short = load_prompts("short", max_tokens=32, limit=5)
    prompts_all = load_prompts("short", max_tokens=32, limit=10) + \
                  load_prompts("long", max_tokens=64, limit=10)

    client = BenchmarkClient(ENGINE_URL)
    results = {}

    try:
        # 1) Sequential
        await restart_engine()
        seq = SequentialDispatcher(client, prompts_short, config={})
        results["sequential"] = await run_mode(seq, "sequential")

        # 2) Concurrent (N=4)
        await restart_engine()
        conc = ConcurrentDispatcher(client, prompts_short, config={}, concurrency=4)
        results["concurrent"] = await run_mode(conc, "concurrent-4")

        # 3) Realistic (rps=2, duration=10s for quick test)
        await restart_engine()
        real = RealisticDispatcher(
            client, prompts_all, config={},
            rps=2.0, duration_seconds=10, drain_timeout_seconds=30.0, seed=42,
        )
        results["realistic"] = await run_mode(real, "realistic-2rps-10s")

    finally:
        await client.close()

    # Write results
    output_path = Path(__file__).resolve().parent / "test_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", output_path)

    # Print to stdout
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
