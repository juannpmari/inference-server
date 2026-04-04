#!/usr/bin/env python3
"""Plot TTFT percentiles vs concurrency level.

Reads the output of experiment_orchestrator.py for the ttft_vs_concurrency
experiment and produces a multi-line plot showing how mean, p50, p90,
and p99 TTFT scale with the number of concurrent requests.

Usage:
    python -m benchmarks.plotting.plot_ttft_vs_concurrency
    python -m benchmarks.plotting.plot_ttft_vs_concurrency --no-scatter
    python -m benchmarks.plotting.plot_ttft_vs_concurrency --input results/ttft.json
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
    "mean":    "#4C72B0",   # blue
    "p50":     "#55A868",   # green
    "p90":     "#DD8452",   # orange
    "p99":     "#C44E52",   # red
    "scatter": "#AAAAAA",   # light gray
}

STATS = [
    ("mean", "Mean",  "o-"),
    ("p50",  "P50",   "s-"),
    ("p90",  "P90",   "^-"),
    ("p99",  "P99",   "D-"),
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

VALID_STATS = ("mean", "p50", "p90", "p99")


def plot_ttft_vs_concurrency(
    data: dict, out_dir: str, scatter: bool = True,
    stat: str | None = None, skip_x: set[int] | None = None,
) -> str:
    conditions = data["conditions"]

    # If a single stat is requested, only plot that one
    active_stats = [(s, l, m) for s, l, m in STATS if s == stat] if stat else STATS

    concurrency_levels = []
    stat_values = {key: [] for key, _, _ in active_stats}
    scatter_data: dict[int, list[float]] = {}

    for cond_name, cond in conditions.items():
        conc = cond["dispatch"]["concurrency"]
        if skip_x and conc in skip_x:
            continue
        concurrency_levels.append(conc)
        for key, _, _ in active_stats:
            stat_values[key].append(cond["stats"]["ttft"][key])
        # Collect per-request TTFT for scatter
        ok = [r for r in cond["per_request"]
              if r["error"] is None and r["http_status"] == 200]
        scatter_data[conc] = [r["ttft"] for r in ok]

    # Sort by concurrency
    order = np.argsort(concurrency_levels)
    concurrency_levels = np.array(concurrency_levels)[order]
    for key in stat_values:
        stat_values[key] = np.array(stat_values[key])[order] * 1000  # s -> ms

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter layer
    if scatter:
        rng = np.random.RandomState(42)
        for conc in concurrency_levels:
            ttfts_ms = np.array(scatter_data[conc]) * 1000
            jitter = rng.uniform(-0.5, 0.5, size=len(ttfts_ms))
            ax.scatter(conc + jitter, ttfts_ms, s=20, alpha=0.2,
                       color=COLORS["scatter"], edgecolors="none", zorder=1)

    # Stat lines
    for key, label, marker_style in active_stats:
        ax.plot(concurrency_levels, stat_values[key], marker_style,
                color=COLORS[key], lw=2, markersize=7, zorder=3,
                label=f"TTFT {label}")

    # Annotate the top stat line
    annotate_key = active_stats[-1][0]
    for x, y in zip(concurrency_levels, stat_values[annotate_key]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9,
                    color=COLORS[annotate_key])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrency Level")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs Concurrency Level")
    ax.set_xticks(concurrency_levels)
    ax.set_xticklabels([str(int(c)) for c in concurrency_levels])
    ax.legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, "ttft_vs_concurrency.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    parser = argparse.ArgumentParser(
        description="Plot TTFT percentiles vs concurrency level")
    parser.add_argument("--input", default=None,
                        help="Path to experiment results JSON")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output PNGs")
    parser.add_argument("--dispatch-mode", default="concurrent",
                        help="Dispatch mode subfolder (default: %(default)s)")
    parser.add_argument("--no-scatter", action="store_true",
                        help="Hide individual request TTFT scatter points")
    parser.add_argument("--stat", default=None, choices=VALID_STATS,
                        help="Plot only this statistic (default: all four)")
    parser.add_argument("--skip-x", default=None,
                        help="Comma-separated concurrency values to exclude (e.g. 2048,4096)")
    args = parser.parse_args()

    results_dir = (Path(__file__).resolve().parent.parent
                   / "results" / args.dispatch_mode / "ttft_vs_concurrency")
    input_path = args.input or str(results_dir / "ttft_vs_concurrency.json")
    output_dir = args.output_dir or str(results_dir)

    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    skip_x = set(int(v) for v in args.skip_x.split(",")) if args.skip_x else None
    fpath = plot_ttft_vs_concurrency(data, output_dir,
                                     scatter=not args.no_scatter,
                                     stat=args.stat, skip_x=skip_x)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
