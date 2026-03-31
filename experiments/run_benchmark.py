"""
APC vs LMCache benchmark runner.

Runs a concurrency sweep (1 → 32 concurrent requests) against a vLLM server,
collecting cache hit ratio, prefill time (TTFT), and GPU memory utilization.

Must be run twice — once per scenario:
  1. APC-only:  vLLM with --enable-prefix-caching (no LMCache)
  2. LMCache:   vLLM with LMCache connector enabled

Usage:
    # Generate prompts first
    python experiments/generate_prompts.py

    # Run APC-only scenario (start vLLM with APC, no LMCache)
    python experiments/run_benchmark.py --scenario apc --url http://localhost:8080

    # Run LMCache scenario (restart vLLM with LMCache enabled)
    python experiments/run_benchmark.py --scenario lmcache --url http://localhost:8080

    # Plot combined results
    python experiments/run_benchmark.py --plot-only

Outputs:
    experiments/results_apc.json
    experiments/results_lmcache.json
    experiments/apc_vs_lmcache.png
"""

import argparse
import asyncio
import json
import time
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import aiohttp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACTIVE_GROUPS = [1, 2, 3, 4, 5, 6, 7, 8]  # number of prefix groups active per level
CONCURRENCY = 8              # fixed concurrency for all levels
MAX_TOKENS = 128
WARMUP_REQUESTS = 4          # prime the cache before measuring
REQUESTS_PER_LEVEL = 80      # total requests sent at each concurrency level
GPU_SAMPLE_INTERVAL = 0.5    # seconds between GPU memory samples
NUM_RUNS = 3                 # repeat each level this many times and average


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt_id: int
    ttft_ms: float            # time to first token (≈ prefill time)
    total_ms: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class LevelResult:
    concurrency: int
    requests: list[RequestResult] = field(default_factory=list)
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    cache_hit_rate: float = 0.0
    gpu_memory_pct: float = 0.0
    gpu_memory_samples: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU memory sampling
# ---------------------------------------------------------------------------

async def sample_gpu_memory(interval: float, samples: list[float], stop_event: asyncio.Event):
    """Continuously sample GPU memory utilization via nvidia-smi."""
    while not stop_event.is_set():
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                # Take the first GPU line
                line = stdout.decode().strip().split("\n")[0]
                used, total = (float(x.strip()) for x in line.split(","))
                samples.append(used / total * 100.0)
        except FileNotFoundError:
            # nvidia-smi not available — fall back to vLLM metrics later
            pass
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Metrics scraping (Prometheus endpoint)
# ---------------------------------------------------------------------------

async def scrape_vllm_metrics(session: aiohttp.ClientSession, base_url: str) -> dict:
    """Scrape vLLM's /metrics endpoint, return key metric values.

    vLLM 0.11 uses:
      vllm:kv_cache_usage_perc          (gauge, 0-1)
      vllm:prefix_cache_hits_total      (counter, tokens)
      vllm:prefix_cache_queries_total   (counter, tokens)
    """
    metrics = {}
    try:
        async with session.get(f"{base_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            text = await resp.text()
            for line in text.split("\n"):
                if line.startswith("#"):
                    continue
                # vLLM 0.11 metric names (colon-separated)
                if "kv_cache_usage_perc" in line and "cache_config" not in line:
                    metrics["gpu_cache_pct"] = float(line.split()[-1]) * 100
                if "prefix_cache_hits_total" in line:
                    metrics["prefix_cache_hits"] = float(line.split()[-1])
                if "prefix_cache_queries_total" in line:
                    metrics["prefix_cache_queries"] = float(line.split()[-1])
    except Exception:
        pass
    # Compute hit rate from counters
    if "prefix_cache_hits" in metrics and "prefix_cache_queries" in metrics:
        q = metrics["prefix_cache_queries"]
        metrics["prefix_cache_hit_rate"] = metrics["prefix_cache_hits"] / q if q > 0 else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Model name discovery
# ---------------------------------------------------------------------------

async def discover_model_name(base_url: str) -> str:
    """Query /v1/models to get the served model name."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            name = data["data"][0]["id"]
            print(f"  Detected model: {name}")
            return name


# ---------------------------------------------------------------------------
# Request sender (streaming, to measure TTFT)
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict],
    prompt_id: int,
    max_tokens: int,
) -> RequestResult:
    """Send one chat completion request (streaming) and measure TTFT."""
    payload = {
        "model": send_request._model_name,  # set at startup via /v1/models
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t_start = time.perf_counter()
    t_first_token = None
    completion_tokens = 0

    timeout = aiohttp.ClientTimeout(total=120)
    async with session.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                completion_tokens += 1  # approximate: one chunk ≈ one token

    t_end = time.perf_counter()
    if t_first_token is None:
        t_first_token = t_end  # no tokens generated

    return RequestResult(
        prompt_id=prompt_id,
        ttft_ms=(t_first_token - t_start) * 1000,
        total_ms=(t_end - t_start) * 1000,
        prompt_tokens=0,  # filled from usage if available
        completion_tokens=completion_tokens,
    )


# ---------------------------------------------------------------------------
# Concurrency-level runner
# ---------------------------------------------------------------------------

async def run_level(
    base_url: str,
    prompts: list[dict],
    concurrency: int,
    total_requests: int,
    num_groups: int = None,
) -> LevelResult:
    """Send `total_requests` at the given concurrency level and collect metrics.

    If num_groups is set, only use prompts from groups 0..num_groups-1.
    """
    if num_groups is not None:
        prompts = [p for p in prompts if p.get("group", 0) < num_groups]
    result = LevelResult(concurrency=concurrency)

    gpu_samples: list[float] = []
    stop_gpu = asyncio.Event()

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Start GPU sampling
        gpu_task = asyncio.create_task(
            sample_gpu_memory(GPU_SAMPLE_INTERVAL, gpu_samples, stop_gpu)
        )

        # Warmup: prime the prefix cache by sending one request per unique
        # prefix group.  This ensures all prefixes have been seen at least
        # once so the measurement phase captures steady-state hit rates.
        seen_groups = set()
        warmup_prompts = []
        for p in prompts:
            g = p.get("group")
            if g not in seen_groups:
                seen_groups.add(g)
                warmup_prompts.append(p)
        print(f"  [c={concurrency}] warming up ({len(warmup_prompts)} requests, "
              f"1 per prefix group)...")
        for p in warmup_prompts:
            await send_request(session, base_url, p["messages"], p["id"], MAX_TOKENS)

        # Scrape metrics baseline after warmup
        metrics_before = await scrape_vllm_metrics(session, base_url)

        # Send requests in closed-loop batches of exactly `concurrency`.
        # Each batch fires `concurrency` requests simultaneously, waits for
        # all to finish, then fires the next batch.  This makes the access
        # pattern deterministic (no scheduling variance).
        print(f"  [c={concurrency}] sending {total_requests} requests...")
        results = []
        req_idx = 0
        while req_idx < total_requests:
            batch_size = min(concurrency, total_requests - req_idx)
            batch_tasks = []
            for j in range(batch_size):
                p = prompts[(req_idx + j) % len(prompts)]
                batch_tasks.append(
                    send_request(session, base_url, p["messages"], p["id"], MAX_TOKENS)
                )
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            req_idx += batch_size

        # Scrape metrics after
        metrics_after = await scrape_vllm_metrics(session, base_url)

        stop_gpu.set()
        await gpu_task

    # Collect successful results
    for r in results:
        if isinstance(r, RequestResult):
            result.requests.append(r)

    if result.requests:
        ttfts = [r.ttft_ms for r in result.requests]
        ttfts_sorted = sorted(ttfts)
        result.avg_ttft_ms = sum(ttfts) / len(ttfts)
        result.p50_ttft_ms = ttfts_sorted[len(ttfts_sorted) // 2]
        result.p99_ttft_ms = ttfts_sorted[int(len(ttfts_sorted) * 0.99)]

    # Compute delta hit rate for this level only (not cumulative)
    hits_before = metrics_before.get("prefix_cache_hits", 0)
    hits_after = metrics_after.get("prefix_cache_hits", 0)
    queries_before = metrics_before.get("prefix_cache_queries", 0)
    queries_after = metrics_after.get("prefix_cache_queries", 0)
    delta_queries = queries_after - queries_before
    delta_hits = hits_after - hits_before
    result.cache_hit_rate = delta_hits / delta_queries if delta_queries > 0 else 0.0
    result.gpu_memory_pct = metrics_after.get("gpu_cache_pct", 0.0)
    result.gpu_memory_samples = gpu_samples

    errors = sum(1 for r in results if isinstance(r, Exception))
    ok = len(result.requests)
    print(
        f"  [c={concurrency}] done — {ok} ok, {errors} errors | "
        f"avg TTFT={result.avg_ttft_ms:.1f}ms | "
        f"cache hit={result.cache_hit_rate:.2%} | "
        f"gpu cache={result.gpu_memory_pct:.1f}%"
    )
    return result


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

async def run_benchmark(base_url: str, prompts: list[dict], scenario: str, output_dir: Path):
    """Sweep over active prefix groups at fixed concurrency, repeated NUM_RUNS times."""
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario.upper()}")
    print(f"  Server:   {base_url}")
    print(f"  Groups:   {ACTIVE_GROUPS}")
    print(f"  Conc:     {CONCURRENCY}")
    print(f"  Runs:     {NUM_RUNS}")
    print(f"{'='*60}\n")

    send_request._model_name = await discover_model_name(base_url)

    run_results: list[list[LevelResult]] = [[] for _ in ACTIVE_GROUPS]

    for run_id in range(NUM_RUNS):
        print(f"\n── Run {run_id + 1}/{NUM_RUNS} ──")
        for li, ng in enumerate(ACTIVE_GROUPS):
            lr = await run_level(
                base_url, prompts, CONCURRENCY, REQUESTS_PER_LEVEL,
                num_groups=ng,
            )
            run_results[li].append(lr)

    levels_out = []
    for li, ng in enumerate(ACTIVE_GROUPS):
        runs = run_results[li]
        ttfts = [r.avg_ttft_ms for r in runs]
        hits = [r.cache_hit_rate for r in runs]
        gpus = [r.gpu_memory_pct for r in runs]
        mean_ttft = sum(ttfts) / len(ttfts)
        mean_hit = sum(hits) / len(hits)
        # Peak GPU memory: max of nvidia-smi samples per run, averaged across runs
        peak_gpus = []
        for r in runs:
            if r.gpu_memory_samples:
                peak_gpus.append(max(r.gpu_memory_samples))
            else:
                peak_gpus.append(r.gpu_memory_pct)
        peak_gpu_mean = sum(peak_gpus) / len(peak_gpus) if peak_gpus else 0.0
        levels_out.append({
            "concurrency": ng,  # x-axis = active groups
            "avg_ttft_ms": mean_ttft,
            "std_ttft_ms": (sum((t - mean_ttft)**2 for t in ttfts) / len(ttfts)) ** 0.5,
            "cache_hit_rate": mean_hit,
            "std_hit_rate": (sum((h - mean_hit)**2 for h in hits) / len(hits)) ** 0.5,
            "gpu_memory_pct": sum(gpus) / len(gpus),
            "peak_gpu_memory_pct": peak_gpu_mean,
            "per_run_ttft": ttfts,
            "per_run_hit_rate": hits,
        })

    out = {
        "scenario": scenario,
        "concurrency_levels": ACTIVE_GROUPS,
        "concurrency": CONCURRENCY,
        "requests_per_level": REQUESTS_PER_LEVEL,
        "max_tokens": MAX_TOKENS,
        "num_runs": NUM_RUNS,
        "levels": levels_out,
    }
    out_path = output_dir / f"results_{scenario}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_results(output_dir: Path) -> tuple[dict | None, dict | None]:
    apc_path = output_dir / "results_apc.json"
    lmc_path = output_dir / "results_lmcache.json"
    apc = json.loads(apc_path.read_text()) if apc_path.exists() else None
    lmc = json.loads(lmc_path.read_text()) if lmc_path.exists() else None
    return apc, lmc


def plot_results(output_dir: Path):
    """Generate the dual-panel plot from saved results."""
    apc, lmc = load_results(output_dir)
    if not apc or not lmc:
        missing = []
        if not apc:
            missing.append("results_apc.json")
        if not lmc:
            missing.append("results_lmcache.json")
        print(f"Missing result files: {', '.join(missing)}")
        print("Run both scenarios first.")
        return

    def extract(data):
        levels = data["levels"]
        conc = [l["concurrency"] for l in levels]
        hit  = [l["cache_hit_rate"] * 100 for l in levels]  # as percentage
        ttft = [l["avg_ttft_ms"] for l in levels]
        gpu  = [l["gpu_memory_pct"] for l in levels]
        peak_gpu = [l.get("peak_gpu_memory_pct", l.get("gpu_memory_pct", 0)) for l in levels]
        # Error bars (std dev) — present when num_runs > 1
        hit_err = [l.get("std_hit_rate", 0) * 100 for l in levels]
        ttft_err = [l.get("std_ttft_ms", 0) for l in levels]
        return conc, hit, ttft, gpu, peak_gpu, hit_err, ttft_err

    conc_a, hit_a, ttft_a, gpu_a, peak_gpu_a, hit_err_a, ttft_err_a = extract(apc)
    conc_l, hit_l, ttft_l, gpu_l, peak_gpu_l, hit_err_l, ttft_err_l = extract(lmc)

    # ── Style constants ──
    APC_COLOR = "#e74c3c"
    LMC_COLOR = "#2ecc71"
    APC_LABEL = "vLLM APC (GPU-only, memory-constrained)"
    LMC_LABEL = "LMCache (GPU + CPU-backed cache)"
    BG_COLOR = "#fafafa"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.12},
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_top, ax_bot):
        ax.set_facecolor(BG_COLOR)
        ax.grid(True, axis="y", color="white", linewidth=1.2, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)

    # ── Top panel: cache hit rate ──
    has_errbars = any(e > 0 for e in hit_err_a + hit_err_l)

    # Error band (±1 std) when multi-run data is available
    if has_errbars:
        hit_a_lo = [h - e for h, e in zip(hit_a, hit_err_a)]
        hit_a_hi = [h + e for h, e in zip(hit_a, hit_err_a)]
        hit_l_lo = [h - e for h, e in zip(hit_l, hit_err_l)]
        hit_l_hi = [h + e for h, e in zip(hit_l, hit_err_l)]
        ax_top.fill_between(conc_a, hit_a_lo, hit_a_hi, alpha=0.15, color=APC_COLOR, zorder=2)
        ax_top.fill_between(conc_l, hit_l_lo, hit_l_hi, alpha=0.12, color=LMC_COLOR, zorder=2)
    else:
        ax_top.fill_between(conc_l, hit_l, alpha=0.12, color=LMC_COLOR, zorder=2)
        ax_top.fill_between(conc_a, hit_a, alpha=0.10, color=APC_COLOR, zorder=2)

    ax_top.plot(conc_l, hit_l, "s-", color=LMC_COLOR, linewidth=2.5, markersize=6,
                label=LMC_LABEL, zorder=3)
    ax_top.plot(conc_a, hit_a, "o-", color=APC_COLOR, linewidth=2.5, markersize=6,
                label=APC_LABEL, zorder=3)

    ax_top.set_ylabel("Cache Hit Rate (%)", fontsize=11, fontweight="bold")
    ax_top.set_ylim(-5, 110)
    ax_top.set_yticks([0, 25, 50, 75, 100])
    ax_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))

    # Annotate the values at rightmost point
    ax_top.annotate(f"{hit_l[-1]:.0f}%", xy=(conc_l[-1], hit_l[-1]),
                    xytext=(10, -2), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=LMC_COLOR)
    ax_top.annotate(f"{hit_a[-1]:.0f}%", xy=(conc_a[-1], hit_a[-1]),
                    xytext=(10, 2), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=APC_COLOR)

    # ── Top panel: peak GPU memory overlay (secondary y-axis) ──
    ax_top_gpu = ax_top.twinx()
    ax_top_gpu.fill_between(conc_a, peak_gpu_a, alpha=0.18, color="#7986cb", label="Peak GPU Mem (APC)", zorder=1)
    ax_top_gpu.fill_between(conc_l, peak_gpu_l, alpha=0.15, color="#b39ddb", label="Peak GPU Mem (LMCache)", zorder=1)
    ax_top_gpu.set_ylim(0, 100)
    ax_top_gpu.set_ylabel("Peak GPU Memory (%)", fontsize=9, color="#666666")
    ax_top_gpu.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))
    ax_top_gpu.tick_params(axis="y", colors="#999999")
    for spine in ax_top_gpu.spines.values():
        spine.set_visible(False)

    # Combined legend from both axes
    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_top_gpu.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9, frameon=True,
                  fancybox=True, shadow=False, framealpha=0.9, edgecolor="#dddddd")

    fig.suptitle("Prefix Caching Under Concurrency Pressure", fontsize=14,
                 fontweight="bold", y=0.97)
    n_runs = apc.get("num_runs", 1)
    n_groups = apc.get("metadata", {}).get("num_prefix_groups", "?")
    runs_note = f"  |  mean of {n_runs} runs" if n_runs > 1 else ""
    ax_top.set_title(f"Qwen2-1.5B-Instruct  |  8 RAG prefix groups (~2 048 tok each)  |  RTX 4050 6 GB{runs_note}",
                     fontsize=9, color="#888888", pad=4)

    # ── Bottom panel: prefill time (TTFT) ──
    if has_errbars:
        ttft_a_lo = [t - e for t, e in zip(ttft_a, ttft_err_a)]
        ttft_a_hi = [t + e for t, e in zip(ttft_a, ttft_err_a)]
        ttft_l_lo = [t - e for t, e in zip(ttft_l, ttft_err_l)]
        ttft_l_hi = [t + e for t, e in zip(ttft_l, ttft_err_l)]
        ax_bot.fill_between(conc_a, ttft_a_lo, ttft_a_hi, alpha=0.15, color=APC_COLOR, zorder=2)
        ax_bot.fill_between(conc_l, ttft_l_lo, ttft_l_hi, alpha=0.12, color=LMC_COLOR, zorder=2)
    else:
        ax_bot.fill_between(conc_a, ttft_a, alpha=0.10, color=APC_COLOR, zorder=2)
        ax_bot.fill_between(conc_l, ttft_l, alpha=0.10, color=LMC_COLOR, zorder=2)

    ax_bot.plot(conc_a, ttft_a, "o-", color=APC_COLOR, linewidth=2.5, markersize=6,
                label=APC_LABEL, zorder=3)
    ax_bot.plot(conc_l, ttft_l, "s-", color=LMC_COLOR, linewidth=2.5, markersize=6,
                label=LMC_LABEL, zorder=3)

    ax_bot.set_ylabel("Avg Prefill Time / TTFT (ms)", fontsize=11, fontweight="bold")
    ax_bot.set_xlabel("Active Prefix Groups (concurrency fixed at 8)", fontsize=11, fontweight="bold")

    # Gap annotation at peak concurrency
    gap_ms = ttft_a[-1] - ttft_l[-1]
    if gap_ms > 0:
        mid_y = (ttft_a[-1] + ttft_l[-1]) / 2
        # Bracket arrow
        ax_bot.annotate(
            "", xy=(conc_a[-1] + 0.6, ttft_l[-1]),
            xytext=(conc_a[-1] + 0.6, ttft_a[-1]),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.8),
            zorder=5,
        )
        ax_bot.text(
            conc_a[-1] + 1.6, mid_y, f"{gap_ms:.0f} ms\n({ttft_a[-1]/ttft_l[-1]:.1f}x)",
            fontsize=11, fontweight="bold", va="center", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            zorder=5,
        )

    # ── Bottom panel: peak GPU memory overlay (secondary y-axis) ──
    ax_bot_gpu = ax_bot.twinx()
    ax_bot_gpu.fill_between(conc_a, peak_gpu_a, alpha=0.18, color="#7986cb", label="Peak GPU Mem (APC)", zorder=1)
    ax_bot_gpu.fill_between(conc_l, peak_gpu_l, alpha=0.15, color="#b39ddb", label="Peak GPU Mem (LMCache)", zorder=1)
    ax_bot_gpu.set_ylim(0, 100)
    ax_bot_gpu.set_ylabel("Peak GPU Memory (%)", fontsize=9, color="#666666")
    ax_bot_gpu.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))
    ax_bot_gpu.tick_params(axis="y", colors="#999999")
    for spine in ax_bot_gpu.spines.values():
        spine.set_visible(False)

    # Combined legend from both axes
    h1, l1 = ax_bot.get_legend_handles_labels()
    h2, l2 = ax_bot_gpu.get_legend_handles_labels()
    ax_bot.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, frameon=True,
                  fancybox=True, shadow=False, framealpha=0.9, edgecolor="#dddddd")

    # X-axis
    ax_bot.set_xticks(conc_a)
    ax_bot.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax_bot.set_xlim(conc_a[0] - 0.5, conc_a[-1] + 4)

    out_path = output_dir / "apc_vs_lmcache.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global ACTIVE_GROUPS, CONCURRENCY, REQUESTS_PER_LEVEL, NUM_RUNS

    parser = argparse.ArgumentParser(description="APC vs LMCache benchmark")
    parser.add_argument(
        "--scenario", choices=["apc", "lmcache"], default=None,
        help="Which scenario to run (omit for --plot-only)",
    )
    parser.add_argument(
        "--url", default="http://localhost:8080",
        help="vLLM server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--prompts", default=None,
        help="Path to prompts.json (default: experiments/prompts.json)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for results (default: experiments/)",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip benchmark, just regenerate the plot from existing results",
    )
    parser.add_argument(
        "--requests-per-level", type=int, default=None,
        help=f"Requests per concurrency level (default: {REQUESTS_PER_LEVEL})",
    )
    parser.add_argument(
        "--levels", type=str, default=None,
        help="Comma-separated active group counts (default: 1,2,3,4,5,6,7,8)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help=f"Fixed concurrency for all levels (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help=f"Number of runs to average (default: {NUM_RUNS})",
    )
    args = parser.parse_args()

    exp_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    if args.levels:
        ACTIVE_GROUPS = [int(x) for x in args.levels.split(",")]

    if args.concurrency is not None:
        CONCURRENCY = args.concurrency

    if args.requests_per_level is not None:
        REQUESTS_PER_LEVEL = args.requests_per_level

    if args.runs is not None:
        NUM_RUNS = args.runs

    if args.plot_only:
        plot_results(exp_dir)
        return

    if not args.scenario:
        parser.error("--scenario is required unless using --plot-only")

    # Load prompts
    prompts_path = Path(args.prompts) if args.prompts else exp_dir / "prompts.json"
    if not prompts_path.exists():
        print(f"Prompts not found at {prompts_path}")
        print("Run:  python experiments/generate_prompts.py")
        return
    prompts = json.loads(prompts_path.read_text())["prompts"]

    # Run benchmark
    asyncio.run(run_benchmark(args.url, prompts, args.scenario, exp_dir))

    # Auto-plot if both results exist
    apc_exists = (exp_dir / "results_apc.json").exists()
    lmc_exists = (exp_dir / "results_lmcache.json").exists()
    if apc_exists and lmc_exists:
        print("\nBoth result files present — generating plot...")
        plot_results(exp_dir)


if __name__ == "__main__":
    main()
