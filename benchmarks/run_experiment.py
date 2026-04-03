#!/usr/bin/env python3
"""Generic experiment runner.

Reads an experiment YAML, runs each condition using the specified
dispatcher, and writes structured results to JSON.

Usage:
    python -m benchmarks.run_experiment experiments/latency_composition.yaml
    python -m benchmarks.run_experiment experiments/latency_composition.yaml --engine-url http://localhost:8080
    python -m benchmarks.run_experiment experiments/latency_composition.yaml --output results/latency.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
import yaml

from benchmarks.dispatchers.base import Prompt, ResponseRecord
from benchmarks.dispatchers.sequential import SequentialDispatcher
from benchmarks.dispatchers.concurrent import ConcurrentDispatcher
from benchmarks.dispatchers.realistic import RealisticDispatcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BENCHMARKS_DIR / "prompts"

# ---------------------------------------------------------------------------
# Client (reused from run_dispatchers.py)
# ---------------------------------------------------------------------------

class BenchmarkClient:
    """Minimal SSE streaming client for the engine /generate/stream endpoint."""

    def __init__(self, engine_url: str, max_connections: int = 50):
        self.engine_url = engine_url.rstrip("/")
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=30.0),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=20),
        )

    async def close(self):
        await self.http.aclose()

    async def send_request(self, prompt: Prompt, max_retries: int = 5) -> ResponseRecord:
        url = f"{self.engine_url}/generate/stream"
        payload = {"prompt": prompt.text, "max_tokens": prompt.max_tokens, "stream": True}

        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            token_times: list[float] = []
            output_tokens = 0

            try:
                async with self.http.stream("POST", url, json=payload) as resp:
                    if resp.status_code == 429:
                        body = b""
                        async for chunk in resp.aiter_bytes():
                            body += chunk
                        if attempt < max_retries:
                            backoff = min(2 ** attempt, 30)
                            logger.warning("429 on '%s' (attempt %d/%d), retrying in %ds",
                                           prompt.name, attempt + 1, max_retries + 1, backoff)
                            await asyncio.sleep(backoff)
                            continue
                        return ResponseRecord(
                            prompt_name=prompt.name, input_tokens=prompt.input_tokens,
                            max_tokens=prompt.max_tokens, ttft=0.0, itl=[],
                            e2e=time.monotonic() - t0, output_tokens=0,
                            http_status=429,
                            error=f"HTTP 429 after {max_retries + 1} attempts: {body.decode(errors='replace')[:200]}",
                        )

                    if resp.status_code != 200:
                        body = b""
                        async for chunk in resp.aiter_bytes():
                            body += chunk
                        return ResponseRecord(
                            prompt_name=prompt.name, input_tokens=prompt.input_tokens,
                            max_tokens=prompt.max_tokens, ttft=0.0, itl=[],
                            e2e=time.monotonic() - t0, output_tokens=0,
                            http_status=resp.status_code,
                            error=f"HTTP {resp.status_code}: {body.decode(errors='replace')[:200]}",
                        )

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
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
                return ResponseRecord(
                    prompt_name=prompt.name, input_tokens=prompt.input_tokens,
                    max_tokens=prompt.max_tokens, ttft=ttft, itl=itl,
                    e2e=t_end - t0, output_tokens=output_tokens,
                    http_status=200, error=None,
                )
            except Exception as exc:
                return ResponseRecord(
                    prompt_name=prompt.name, input_tokens=prompt.input_tokens,
                    max_tokens=prompt.max_tokens, ttft=0.0, itl=[],
                    e2e=time.monotonic() - t0, output_tokens=0,
                    http_status=0, error=f"{type(exc).__name__}: {exc}",
                )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(pool: str, max_tokens: int, limit: int | None = None, concat: int = 1) -> list[Prompt]:
    """Load prompts from a pool, optionally concatenating multiple prompts.

    Args:
        pool: Pool name from manifest (e.g. "short", "long").
        max_tokens: max_tokens to set on each Prompt.
        limit: Max number of final prompts to produce.
        concat: Number of pool prompts to concatenate into each final prompt.
              e.g. concat=3 with 32-token short prompts produces ~96-token prompts.
    """
    manifest = json.loads((PROMPTS_DIR / "manifest.json").read_text())
    entries = manifest[pool]

    # Load all raw prompts from the pool
    raw: list[tuple[str, str, int]] = []  # (name, text, input_tokens)
    for entry in entries:
        text = (PROMPTS_DIR / entry["file"]).read_text().strip()
        name = Path(entry["file"]).stem
        raw.append((name, text, entry["input_tokens"]))

    prompts = []
    # Build concatenated prompts, cycling through the pool if concat > pool size
    num_prompts = limit if limit else max(1, len(raw) // concat) if concat <= len(raw) else 1
    for p_idx in range(num_prompts):
        parts_text = []
        parts_tokens = 0
        parts_names = []
        for c in range(concat):
            idx = (p_idx * concat + c) % len(raw)
            name, text, tok = raw[idx]
            parts_text.append(text)
            parts_tokens += tok
            parts_names.append(name)
        prompts.append(Prompt(
            name="_".join(parts_names),
            text=" ".join(parts_text),
            input_tokens=parts_tokens,
            max_tokens=max_tokens,
        ))

    if limit:
        prompts = prompts[:limit]
    return prompts


# ---------------------------------------------------------------------------
# Dispatcher factory
# ---------------------------------------------------------------------------

def make_dispatcher(client: BenchmarkClient, prompts: list[Prompt], dispatch_cfg: dict):
    mode = dispatch_cfg["mode"]
    if mode == "sequential":
        return SequentialDispatcher(client, prompts, config={}, delay=dispatch_cfg.get("delay", 0.0))
    elif mode == "concurrent":
        return ConcurrentDispatcher(client, prompts, config={}, concurrency=dispatch_cfg.get("concurrency", 4))
    elif mode == "realistic":
        return RealisticDispatcher(
            client, prompts, config={},
            rps=dispatch_cfg.get("rps", 1.0),
            duration_seconds=dispatch_cfg.get("duration_seconds", 60),
            drain_timeout_seconds=dispatch_cfg.get("drain_timeout_seconds", 30.0),
            seed=dispatch_cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown dispatch mode: {mode}")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

async def fetch_metrics(engine_url: str) -> dict:
    try:
        resp = await asyncio.to_thread(lambda: httpx.get(f"{engine_url}/metrics_summary", timeout=5.0).json())
        return resp
    except Exception:
        return {}


def compute_delta(before: dict, after: dict) -> dict:
    if not before or not after:
        return {}
    try:
        return {
            "request_count": after.get("session", {}).get("total_requests", 0) - before.get("session", {}).get("total_requests", 0),
            "latency": after.get("latency", {}),
            "throughput": after.get("throughput", {}),
            "kv_cache": {
                "hit_rate": after.get("kv_cache", {}).get("hit_rate", 0.0),
                "l1_utilization_ratio": after.get("kv_cache", {}).get("l1_utilization_ratio", 0.0),
                "eviction_count": after.get("kv_cache", {}).get("eviction_count", 0) - before.get("kv_cache", {}).get("eviction_count", 0),
            },
            "errors": {
                "total_errors": after.get("errors", {}).get("total_errors", 0) - before.get("errors", {}).get("total_errors", 0),
                "error_rate": after.get("errors", {}).get("error_rate", 0.0),
            },
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Run a single condition
# ---------------------------------------------------------------------------

async def run_condition(
    client: BenchmarkClient,
    engine_url: str,
    condition_name: str,
    condition_cfg: dict,
    dispatch_cfg: dict,
    protocol: dict,
) -> dict:
    """Run one condition (e.g. 'siso') through warmup + N measurement runs."""
    prompts = load_prompts(
        pool=condition_cfg["prompt_pool"],
        max_tokens=condition_cfg["max_tokens"],
        limit=protocol.get("prompts_per_condition"),
        concat=condition_cfg.get("concat", 1),
    )
    dispatcher = make_dispatcher(client, prompts, dispatch_cfg)

    warmup_n = protocol.get("warmup_requests", 3)
    num_runs = protocol.get("runs", 3)
    cooldown = protocol.get("cooldown_seconds", 5)

    logger.info("--- Condition '%s': warmup (%d requests) ---", condition_name, warmup_n)
    await dispatcher.warmup(n=warmup_n)

    all_runs: list[list[ResponseRecord]] = []
    deltas: list[dict] = []

    for i in range(num_runs):
        before = await fetch_metrics(engine_url)
        logger.info("  [%s] run %d/%d ...", condition_name, i + 1, num_runs)
        records = await dispatcher.run()
        await asyncio.sleep(0.5)
        after = await fetch_metrics(engine_url)
        all_runs.append(records)
        deltas.append(compute_delta(before, after))
        if i < num_runs - 1:
            logger.info("  cooldown %ds ...", cooldown)
            await asyncio.sleep(cooldown)

    # Select median run
    medians = []
    for idx, run in enumerate(all_runs):
        successful = [r.e2e for r in run if r.error is None]
        med = statistics.median(successful) if successful else float("inf")
        medians.append((med, idx))
    medians.sort(key=lambda x: x[0])
    median_idx = medians[len(medians) // 2][1]
    median_records = all_runs[median_idx]

    logger.info("  [%s] median run: #%d", condition_name, median_idx + 1)

    ok = [r for r in median_records if r.error is None and r.http_status == 200]
    ttfts = [r.ttft for r in ok]
    e2es = [r.e2e for r in ok]
    all_itl = [itl for r in ok for itl in r.itl]
    decode_times = [r.e2e - r.ttft for r in ok]

    def percentiles(vals):
        if not vals:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
        return {
            "mean": round(float(np.mean(vals)), 6),
            "p50": round(float(np.percentile(vals, 50)), 6),
            "p90": round(float(np.percentile(vals, 90)), 6),
            "p99": round(float(np.percentile(vals, 99)), 6),
        }

    return {
        "condition": condition_name,
        "prompt_pool": condition_cfg["prompt_pool"],
        "max_tokens": condition_cfg["max_tokens"],
        "total_requests": len(median_records),
        "successful_requests": len(ok),
        "failed_requests": len(median_records) - len(ok),
        "stats": {
            "ttft": percentiles(ttfts),
            "decode_time": percentiles(decode_times),
            "itl": percentiles(all_itl),
            "e2e": percentiles(e2es),
        },
        "server_side_delta": deltas[median_idx],
        "per_request": [
            {
                "prompt_name": r.prompt_name,
                "input_tokens": r.input_tokens,
                "max_tokens": r.max_tokens,
                "output_tokens": r.output_tokens,
                "ttft": round(r.ttft, 6),
                "decode_time": round(r.e2e - r.ttft, 6),
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
# Engine restart
# ---------------------------------------------------------------------------

async def restart_engine(engine_url: str):
    logger.info("Restarting engine ...")
    proc = await asyncio.create_subprocess_exec(
        "docker", "compose", "restart", "engine",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as c:
        for _ in range(60):
            await asyncio.sleep(3)
            try:
                resp = await c.post(
                    f"{engine_url}/generate/stream",
                    json={"prompt": "test", "max_tokens": 1, "stream": True},
                )
                if resp.status_code == 200:
                    logger.info("Engine ready.")
                    await asyncio.sleep(2)
                    return
            except Exception:
                pass
    raise TimeoutError("Engine not ready after restart")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_experiment(experiment_path: str, engine_url: str, output_path: str | None, restart: bool):
    with open(experiment_path) as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    conditions = exp["conditions"]
    dispatch_cfg = exp["dispatch"]
    protocol = exp.get("protocol", {})

    logger.info("Experiment: %s", exp["name"])
    logger.info("Conditions: %s", list(conditions.keys()))
    logger.info("Dispatch: %s", dispatch_cfg["mode"])

    client = BenchmarkClient(engine_url)
    results = {
        "experiment": exp["name"],
        "description": exp.get("description", ""),
        "dispatch_mode": dispatch_cfg["mode"],
        "conditions": {},
    }

    try:
        for cond_name, cond_cfg in conditions.items():
            if restart:
                await restart_engine(engine_url)
            result = await run_condition(client, engine_url, cond_name, cond_cfg, dispatch_cfg, protocol)
            results["conditions"][cond_name] = result
    finally:
        await client.close()

    # Determine output path
    if output_path is None:
        output_path = str(BENCHMARKS_DIR / "results" / f"{exp['name']}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", output_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run a benchmark experiment from a YAML definition.")
    parser.add_argument("experiment", help="Path to experiment YAML file")
    parser.add_argument("--engine-url", default="http://localhost:8080", help="Engine URL (default: %(default)s)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: benchmarks/results/<experiment_name>.json)")
    parser.add_argument("--restart", action="store_true", help="Restart engine between conditions")
    args = parser.parse_args()

    asyncio.run(run_experiment(args.experiment, args.engine_url, args.output, args.restart))


if __name__ == "__main__":
    main()
