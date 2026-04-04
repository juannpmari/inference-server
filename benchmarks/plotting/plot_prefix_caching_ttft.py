#!/usr/bin/env python3
"""Plot prefix caching impact on TTFT.

Reads the output of experiment_orchestrator.py for the prefix_caching_ttft experiment
and produces a dual-line plot comparing TTFT with and without prefix caching
across input lengths.

Usage:
    python -m benchmarks.plotting.plot_prefix_caching_ttft
    python -m benchmarks.plotting.plot_prefix_caching_ttft --stat p50
    python -m benchmarks.plotting.plot_prefix_caching_ttft --input results/prefix_caching_ttft.json
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
    "cache_off_line": "#C44E52",
    "cache_off_scatter": "#E8A0A0",
    "cache_on_line": "#4C72B0",
    "cache_on_scatter": "#A0C0E0",
}

VALID_STATS = ("mean", "p50", "p90", "p99")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_prefix_caching_ttft(data: dict, out_dir: str, stat: str = "mean", scatter: bool = True) -> str:
    conditions = data["conditions"]

    series: dict[str, list[tuple[float, float, list[float]]]] = {
        "nocache": [],
        "cache": [],
    }

    for cond_name, cond in conditions.items():
        per_req = cond["per_request"]
        ok = [r for r in per_req if r["error"] is None and r["http_status"] == 200]
        if not ok:
            continue
        avg_input = np.mean([r["input_tokens"] for r in ok])
        stat_val = cond["stats"]["ttft"][stat]
        ttfts = [r["ttft"] for r in ok]

        if cond_name.startswith("nocache_"):
            series["nocache"].append((avg_input, stat_val, ttfts))
        elif cond_name.startswith("cache_"):
            series["cache"].append((avg_input, stat_val, ttfts))

    for key in series:
        series[key].sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(9, 6))

    plot_configs = [
        ("Cache OFF", "nocache", "cache_off", -2),
        ("Cache ON", "cache", "cache_on", 2),
    ]

    for label, series_key, color_prefix, jitter_offset in plot_configs:
        if not series[series_key]:
            continue
        tokens = [s[0] for s in series[series_key]]
        vals = [s[1] for s in series[series_key]]
        per_req = [s[2] for s in series[series_key]]

        # Scatter individual requests
        if scatter:
            for tok, reqs in zip(tokens, per_req):
                xs = np.full(len(reqs), tok) + jitter_offset
                jitter = np.random.default_rng(42).uniform(-1, 1, size=len(reqs))
                ax.scatter(xs + jitter, np.array(reqs) * 1000, s=20, alpha=0.3,
                           color=COLORS[f"{color_prefix}_scatter"], edgecolors="none", zorder=2)

        # Line for the chosen stat
        ax.plot(tokens, np.array(vals) * 1000, "o-", color=COLORS[f"{color_prefix}_line"],
                lw=2, markersize=6, zorder=3, label=f"{label} ({stat.upper()})")

    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"Prefix Caching Impact on TTFT ({stat.upper()})")
    ax.legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"prefix_caching_ttft_{stat}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    parser = argparse.ArgumentParser(description="Plot prefix caching impact on TTFT")
    parser.add_argument("--input", default=None, help="Path to experiment results JSON")
    parser.add_argument("--output-dir", default=None, help="Directory for output PNGs")
    parser.add_argument("--dispatch-mode", default="sequential", help="Dispatch mode subfolder (default: %(default)s)")
    parser.add_argument("--stat", default="mean", choices=VALID_STATS,
                        help="Which aggregation to plot (default: mean)")
    parser.add_argument("--no-scatter", action="store_true",
                        help="Hide individual request points, show only the aggregated lines")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results" / args.dispatch_mode / "prefix_caching_ttft"
    input_path = args.input or str(results_dir / "prefix_caching_ttft.json")
    output_dir = args.output_dir or str(results_dir)

    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    fpath = plot_prefix_caching_ttft(data, output_dir, stat=args.stat, scatter=not args.no_scatter)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
