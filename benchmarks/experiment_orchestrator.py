#!/usr/bin/env python3
"""Generic experiment orchestrator.

Reads an experiment YAML, runs each condition using the specified
dispatcher, and writes structured results to JSON.

Usage:
    python -m benchmarks.experiment_orchestrator experiments/latency_composition.yaml
    python -m benchmarks.experiment_orchestrator experiments/latency_composition.yaml --engine-url http://localhost:8080
    python -m benchmarks.experiment_orchestrator experiments/latency_composition.yaml --output results/latency.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import statistics
import subprocess
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

    def __init__(self, engine_url: str, max_connections: int = 100):
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

TOKENS_PER_PROMPT = 32


def load_prompts(input_length: int, max_tokens: int, limit: int | None = None, repeat: int = 1) -> list[Prompt]:
    """Load prompts from data/, concatenating to reach the desired input length.

    Args:
        input_length: Desired input length in tokens (must be divisible by 32).
        max_tokens: max_tokens to set on each Prompt.
        limit: Max number of unique prompts to produce.
        repeat: How many times to repeat the prompt list. Useful for prefix
                caching experiments where cache hits require identical prompts.
    """
    if input_length % TOKENS_PER_PROMPT != 0:
        raise ValueError(f"input_length ({input_length}) must be divisible by {TOKENS_PER_PROMPT}")
    concat = input_length // TOKENS_PER_PROMPT

    # Load all 32-token prompts from data/
    data_dir = PROMPTS_DIR / "data"
    raw: list[tuple[str, str]] = []  # (name, text)
    for path in sorted(data_dir.glob("*.txt")):
        raw.append((path.stem, path.read_text().strip()))

    if not raw:
        raise FileNotFoundError(f"No prompt files found in {data_dir}")

    # Build concatenated prompts, cycling through files if needed
    num_prompts = limit if limit else max(1, len(raw) // concat) if concat <= len(raw) else 1
    prompts = []
    for p_idx in range(num_prompts):
        parts_text = []
        parts_names = []
        for c in range(concat):
            idx = (p_idx * concat + c) % len(raw)
            name, text = raw[idx]
            parts_text.append(text)
            parts_names.append(name)
        prompts.append(Prompt(
            name="_".join(parts_names),
            text=" ".join(parts_text),
            input_tokens=input_length,
            max_tokens=max_tokens,
        ))

    if limit:
        prompts = prompts[:limit]
    return prompts * repeat


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
# Helpers
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BENCHMARKS_DIR),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _build_ordered_conditions(exp: dict) -> list[tuple[str, dict, dict | None]]:
    """Extract ordered (name, cfg, group_env) tuples from experiment config."""
    if "condition_groups" in exp:
        ordered: list[tuple[str, dict, dict | None]] = []
        for group in exp["condition_groups"]:
            group_env = group.get("engine_env")
            for cond_name, cond_cfg in group["conditions"].items():
                ordered.append((cond_name, cond_cfg, group_env))
        return ordered
    return [(name, cfg, None) for name, cfg in exp["conditions"].items()]


VALID_DISPATCH_MODES = {"sequential", "concurrent", "realistic"}


def validate_yaml(cfg: dict) -> None:
    """Validate experiment YAML config upfront. Raises ValueError on problems."""
    if "experiment" not in cfg:
        raise ValueError("Missing top-level 'experiment' key")

    exp = cfg["experiment"]

    if "name" not in exp:
        raise ValueError("Missing 'experiment.name'")

    if "dispatch" not in exp:
        raise ValueError("Missing 'experiment.dispatch'")

    mode = exp["dispatch"].get("mode")
    if mode not in VALID_DISPATCH_MODES:
        raise ValueError(f"Invalid dispatch mode '{mode}', must be one of {VALID_DISPATCH_MODES}")

    has_conditions = "conditions" in exp
    has_groups = "condition_groups" in exp
    if not has_conditions and not has_groups:
        raise ValueError("Must specify either 'conditions' or 'condition_groups'")
    if has_conditions and has_groups:
        raise ValueError("Cannot specify both 'conditions' and 'condition_groups'")

    ordered = _build_ordered_conditions(exp)
    for cond_name, cond_cfg, _ in ordered:
        if "input_length" not in cond_cfg:
            raise ValueError(f"Condition '{cond_name}': missing 'input_length'")
        if "max_tokens" not in cond_cfg:
            raise ValueError(f"Condition '{cond_name}': missing 'max_tokens'")
        il = cond_cfg["input_length"]
        if il % TOKENS_PER_PROMPT != 0:
            raise ValueError(
                f"Condition '{cond_name}': input_length ({il}) must be divisible by {TOKENS_PER_PROMPT}"
            )


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
        input_length=condition_cfg["input_length"],
        max_tokens=condition_cfg["max_tokens"],
        limit=protocol.get("prompts_per_condition"),
        repeat=protocol.get("repeat", 1),
    )
    effective_dispatch = {**dispatch_cfg, **condition_cfg.get("dispatch", {})}
    dispatcher = make_dispatcher(client, prompts, effective_dispatch)

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

    # Client-side aggregate throughput
    total_output_tokens = sum(r.output_tokens for r in ok)
    wall_clock = max(r.e2e for r in ok) if ok else 0.0
    client_throughput = {
        "total_output_tokens": total_output_tokens,
        "wall_clock_seconds": round(wall_clock, 6),
        "output_tokens_per_second": round(total_output_tokens / wall_clock, 2) if wall_clock > 0 else 0.0,
    }

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
        "input_length": condition_cfg["input_length"],
        "max_tokens": condition_cfg["max_tokens"],
        "dispatch": effective_dispatch,
        "total_requests": len(median_records),
        "successful_requests": len(ok),
        "failed_requests": len(median_records) - len(ok),
        "client_side_throughput": client_throughput,
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
# .env file helpers
# ---------------------------------------------------------------------------

def _update_env_file(env_path: Path, overrides: dict[str, str]):
    """Read .env, update/add keys from overrides, write back."""
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in overrides:
                new_lines.append(f"{key}={overrides[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)
    for key, val in overrides.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")
    env_path.write_text("\n".join(new_lines) + "\n")


# ---------------------------------------------------------------------------
# Engine restart
# ---------------------------------------------------------------------------

async def restart_engine(engine_url: str, engine_env: dict[str, str] | None = None):
    project_root = BENCHMARKS_DIR.parent
    if engine_env:
        logger.info("Updating .env with: %s", engine_env)
        _update_env_file(project_root / ".env", engine_env)
        logger.info("Recreating engine container ...")
        cmd = ("docker", "compose", "up", "-d", "--force-recreate", "engine")
    else:
        logger.info("Restarting engine ...")
        cmd = ("docker", "compose", "restart", "engine")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=str(project_root),
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
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(experiment_path: str, restart: bool):
    """Validate YAML and print experiment plan without executing."""
    with open(experiment_path) as f:
        cfg = yaml.safe_load(f)
    validate_yaml(cfg)

    exp = cfg["experiment"]
    dispatch_cfg = exp["dispatch"]
    protocol = exp.get("protocol", {})
    ordered = _build_ordered_conditions(exp)
    use_groups = "condition_groups" in exp

    print(f"Experiment : {exp['name']}")
    print(f"Description: {exp.get('description', '')}")
    print(f"Dispatch   : {dispatch_cfg['mode']}")
    print(f"Conditions : {len(ordered)}")
    print()

    runs = protocol.get("runs", 3)
    warmup = protocol.get("warmup_requests", 3)
    prompts = protocol.get("prompts_per_condition", "all")
    timeout = protocol.get("timeout_per_condition", 600)
    print(f"Protocol: runs={runs}, warmup={warmup}, prompts_per_condition={prompts}, timeout={timeout}s")
    print()

    print("Conditions:")
    last_env: dict | None = None
    for cond_name, cond_cfg, group_env in ordered:
        overrides = cond_cfg.get("dispatch", {})
        override_str = f"  dispatch overrides: {overrides}" if overrides else ""
        print(f"  - {cond_name}: input_length={cond_cfg['input_length']}, "
              f"max_tokens={cond_cfg['max_tokens']}{override_str}")
        if use_groups and group_env is not None and group_env != last_env:
            print(f"    ^ engine restart with env: {group_env}")
            last_env = group_env
        elif restart:
            print(f"    ^ engine restart")
    print()
    print("Dry run complete — no requests were sent.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_experiment(
    experiment_path: str,
    engine_url: str,
    output_path: str | None,
    restart: bool,
    resume: bool = False,
    only: str | None = None,
    skip: str | None = None,
) -> dict:
    start_time = datetime.datetime.now(datetime.timezone.utc)

    with open(experiment_path) as f:
        cfg = yaml.safe_load(f)

    validate_yaml(cfg)

    exp = cfg["experiment"]
    dispatch_cfg = exp["dispatch"]
    protocol = exp.get("protocol", {})
    use_groups = "condition_groups" in exp
    timeout = protocol.get("timeout_per_condition", 600)

    ordered_conditions = _build_ordered_conditions(exp)
    all_names = {name for name, _, _ in ordered_conditions}

    # --- Condition filtering (--only / --skip) ---
    if only:
        only_set = set(only.split(","))
        unknown = only_set - all_names
        if unknown:
            raise ValueError(f"--only references unknown conditions: {unknown}")
        ordered_conditions = [(n, c, e) for n, c, e in ordered_conditions if n in only_set]
    if skip:
        skip_set = set(skip.split(","))
        unknown = skip_set - all_names
        if unknown:
            raise ValueError(f"--skip references unknown conditions: {unknown}")
        ordered_conditions = [(n, c, e) for n, c, e in ordered_conditions if n not in skip_set]

    logger.info("Experiment: %s", exp["name"])
    logger.info("Conditions: %s", [c[0] for c in ordered_conditions])
    logger.info("Dispatch: %s", dispatch_cfg["mode"])

    client = BenchmarkClient(engine_url)
    results: dict = {
        "experiment": exp["name"],
        "description": exp.get("description", ""),
        "dispatch_mode": dispatch_cfg["mode"],
        "dispatch_config": dispatch_cfg,
        "metadata": {
            "timestamp": start_time.isoformat(),
            "git_commit": _get_git_commit(),
            "experiment_yaml": cfg,
            "engine_url": engine_url,
        },
        "conditions": {},
    }

    # Resolve output path early so intermediate saves work.
    if output_path is None:
        output_dir = BENCHMARKS_DIR / "results" / dispatch_cfg["mode"] / exp["name"]
        out = str(output_dir / f"{exp['name']}.json")
    else:
        out = output_path

    # --- Incremental results: load existing output if present ---
    if Path(out).exists():
        try:
            existing = json.loads(Path(out).read_text())
            results["conditions"] = existing.get("conditions", {})
            logger.info("Loaded %d existing conditions from %s",
                        len(results["conditions"]), out)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Could not load existing results from %s, starting fresh", out)

    # --- Resumability: skip already-completed conditions ---
    if resume:
        completed = set(results["conditions"].keys())
        before_count = len(ordered_conditions)
        ordered_conditions = [
            (n, c, e) for n, c, e in ordered_conditions if n not in completed
        ]
        skipped = before_count - len(ordered_conditions)
        if skipped:
            logger.info("Resuming: skipped %d already-completed conditions", skipped)
        if not ordered_conditions:
            logger.info("All conditions already completed — nothing to run")
            results["metadata"]["total_duration_seconds"] = 0.0
            return results

    env_file_path = BENCHMARKS_DIR.parent / ".env"
    original_env = env_file_path.read_text() if env_file_path.exists() else None
    try:
        last_env: dict | None = None
        for cond_name, cond_cfg, group_env in ordered_conditions:
            if use_groups and group_env is not None and group_env != last_env:
                await restart_engine(engine_url, engine_env=group_env)
                last_env = group_env
            elif restart:
                await restart_engine(engine_url)

            # --- Per-condition timeout ---
            try:
                result = await asyncio.wait_for(
                    run_condition(client, engine_url, cond_name, cond_cfg, dispatch_cfg, protocol),
                    timeout=timeout,
                )
                results["conditions"][cond_name] = result
            except asyncio.TimeoutError:
                logger.warning("Condition '%s' timed out after %ds — skipping", cond_name, timeout)
                results["conditions"][cond_name] = {
                    "condition": cond_name,
                    "error": f"Timed out after {timeout} seconds",
                    "input_length": cond_cfg["input_length"],
                    "max_tokens": cond_cfg["max_tokens"],
                }

            # Write intermediate results after each condition so completed
            # work is preserved if a later condition fails.
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(results, indent=2))
            logger.info("Intermediate results saved to %s", out)
    finally:
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
        results["metadata"]["total_duration_seconds"] = round(elapsed, 2)
        # Final write with duration included
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(results, indent=2))

        if original_env is not None and use_groups:
            env_file_path.write_text(original_env)
        await client.close()

    logger.info("Final results written to %s", out)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run a benchmark experiment from a YAML definition.")
    parser.add_argument("experiment", help="Path to experiment YAML file")
    parser.add_argument("--engine-url", default="http://localhost:8080", help="Engine URL (default: %(default)s)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: benchmarks/results/<experiment_name>.json)")
    parser.add_argument("--restart", action="store_true", help="Restart engine between conditions")
    parser.add_argument("--resume", action="store_true", help="Skip conditions already present in the output file")
    parser.add_argument("--only", default=None, help="Comma-separated condition names to run (others skipped)")
    parser.add_argument("--skip", default=None, help="Comma-separated condition names to skip")
    parser.add_argument("--dry-run", action="store_true", help="Validate YAML and print plan without executing")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.experiment, args.restart)
    else:
        asyncio.run(run_experiment(
            args.experiment, args.engine_url, args.output, args.restart,
            resume=args.resume, only=args.only, skip=args.skip,
        ))


if __name__ == "__main__":
    main()
