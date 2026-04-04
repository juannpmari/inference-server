#!/usr/bin/env python3
"""Plot output token throughput vs concurrency level.

Reads the output of experiment_orchestrator.py for the throughput_vs_concurrency
experiment and produces a line plot showing how aggregate output token
throughput scales with the number of concurrent requests.

Usage:
    python -m benchmarks.plotting.plot_throughput_vs_concurrency
    python -m benchmarks.plotting.plot_throughput_vs_concurrency --source server
    python -m benchmarks.plotting.plot_throughput_vs_concurrency --input results/throughput.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
_STYLE = "seaborn-v0_8-whitegrid"
try:
    plt.style.use(_STYLE)
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

COLORS = {
    "client_line": "#4C72B0",
    "server_line": "#C44E52",
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

VALID_STATS = ("mean", "p50", "p90", "p99")


def plot_throughput_vs_concurrency(
    data: dict, out_dir: str, source: str = "client",
    stat: str = "mean", skip_x: set[int] | None = None,
) -> str:
    conditions = data["conditions"]

    concurrency_levels = []
    throughputs = []
    latencies = []

    for cond_name, cond in conditions.items():
        concurrency = cond["dispatch"]["concurrency"]
        if skip_x and concurrency in skip_x:
            continue
        if source == "server":
            tput = (cond.get("server_side_delta", {})
                       .get("throughput", {})
                       .get("output_tokens_per_second", 0.0))
        else:
            tput = cond["client_side_throughput"]["output_tokens_per_second"]
        latency_ms = cond["stats"]["e2e"][stat] * 1000
        concurrency_levels.append(concurrency)
        throughputs.append(tput)
        latencies.append(latency_ms)

    # Sort by concurrency
    order = np.argsort(concurrency_levels)
    concurrency_levels = np.array(concurrency_levels)[order]
    throughputs = np.array(throughputs)[order]
    latencies = np.array(latencies)[order]

    color_key = f"{source}_line"
    color = COLORS.get(color_key, COLORS["client_line"])

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(concurrency_levels, throughputs, "o-", color=color, lw=2,
             markersize=7, zorder=3, label=f"Output throughput ({source})")

    for x, y in zip(concurrency_levels, throughputs):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=color)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Concurrency Level")
    ax1.set_ylabel("Output Tokens/s", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(concurrency_levels)
    ax1.set_xticklabels([str(int(c)) for c in concurrency_levels])

    # Secondary y-axis: E2E latency
    lat_color = COLORS.get("server_line" if source == "client" else "client_line", "#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(concurrency_levels, latencies, "s--", color=lat_color, lw=1.5,
             markersize=5, zorder=3, alpha=0.8, label=f"E2E latency {stat} ({source})")

    for x, y in zip(concurrency_levels, latencies):
        ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, -12), ha="center", fontsize=8, color=lat_color, alpha=0.8)

    ax2.set_ylabel(f"E2E Latency — {stat} (ms)", color=lat_color)
    ax2.tick_params(axis="y", labelcolor=lat_color)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(f"Output Token Throughput vs Concurrency Level ({stat})")

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"throughput_vs_concurrency_{source}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    parser = argparse.ArgumentParser(description="Plot output token throughput vs concurrency level")
    parser.add_argument("--input", default=None, help="Path to experiment results JSON")
    parser.add_argument("--output-dir", default=None, help="Directory for output PNGs")
    parser.add_argument("--dispatch-mode", default="concurrent", help="Dispatch mode subfolder (default: %(default)s)")
    parser.add_argument("--source", default="client", choices=("client", "server"),
                        help="Throughput source: client-side or server-side (default: client)")
    parser.add_argument("--stat", default="mean", choices=VALID_STATS,
                        help="E2E latency statistic for secondary axis (default: mean)")
    parser.add_argument("--skip-x", default=None,
                        help="Comma-separated concurrency values to exclude (e.g. 2048,4096)")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results" / args.dispatch_mode / "throughput_vs_concurrency"
    input_path = args.input or str(results_dir / "throughput_vs_concurrency.json")
    output_dir = args.output_dir or str(results_dir)

    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    skip_x = set(int(v) for v in args.skip_x.split(",")) if args.skip_x else None
    fpath = plot_throughput_vs_concurrency(data, output_dir, source=args.source, stat=args.stat, skip_x=skip_x)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
