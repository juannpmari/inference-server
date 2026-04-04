#!/usr/bin/env python3
"""Plot latency composition (prefill vs decode) as a stacked barchart.

Reads the output of run_experiment.py for the latency_composition experiment
and produces a stacked bar chart with one bar per condition (SISO, SILO, LISO, LILO).

Usage:
    python -m benchmarks.plotting.plot_latency_composition
    python -m benchmarks.plotting.plot_latency_composition --stat p50
    python -m benchmarks.plotting.plot_latency_composition --stat p90 --input results/latency_composition.json
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
    "prefill": "#4C72B0",
    "decode": "#DD8452",
}

CONDITION_LABELS = {
    "siso": "SISO\n(32, 32)",
    "silo": "SILO\n(32, 256)",
    "liso": "LISO\n(256, 32)",
    "lilo": "LILO\n(256, 256)",
}

VALID_STATS = ("mean", "p50", "p90", "p99")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_latency_composition(data: dict, out_dir: str, stat: str = "mean") -> str:
    conditions = data["conditions"]

    labels = []
    prefill_vals = []
    decode_vals = []

    for cond_name, cond in conditions.items():
        stats = cond["stats"]
        labels.append(CONDITION_LABELS.get(cond_name, cond_name))
        prefill_vals.append(stats["ttft"][stat])
        decode_vals.append(stats["decode_time"][stat])

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2), 6))
    ax.bar(x, prefill_vals, width, label="Prefill (TTFT)", color=COLORS["prefill"])
    ax.bar(x, decode_vals, width, bottom=prefill_vals, label="Decode", color=COLORS["decode"])

    ax.set_xlabel("Condition")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Latency Decomposition: Prefill vs Decode ({stat.upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()

    for i in range(len(labels)):
        total = prefill_vals[i] + decode_vals[i]
        # Label inside the prefill segment
        ax.text(x[i], prefill_vals[i] / 2, f"{prefill_vals[i]*1000:.1f}ms",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        # Label inside the decode segment
        ax.text(x[i], prefill_vals[i] + decode_vals[i] / 2, f"{decode_vals[i]*1000:.1f}ms",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        # Total label on top
        ax.annotate(f"{total*1000:.1f}ms", xy=(x[i], total), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"latency_composition_{stat}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    parser = argparse.ArgumentParser(description="Plot latency composition stacked barchart")
    parser.add_argument("--input", default=None, help="Path to experiment results JSON")
    parser.add_argument("--output-dir", default=None, help="Directory for output PNGs")
    parser.add_argument("--dispatch-mode", default="sequential", help="Dispatch mode subfolder (default: %(default)s)")
    parser.add_argument("--stat", default="mean", choices=VALID_STATS,
                        help="Which aggregation to plot (default: mean)")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results" / args.dispatch_mode / "latency_composition"
    if args.input is None:
        args.input = str(results_dir / "latency_composition.json")
    if args.output_dir is None:
        args.output_dir = str(results_dir)

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    fpath = plot_latency_composition(data, args.output_dir, stat=args.stat)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
