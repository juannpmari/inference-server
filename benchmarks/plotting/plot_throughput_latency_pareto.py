#!/usr/bin/env python3
"""Plot throughput vs latency Pareto frontier across concurrency levels.

Reads the output of experiment_orchestrator.py for the throughput_latency_pareto
experiment and produces a scatter plot with the Pareto frontier highlighted.
Each point represents a concurrency level; Pareto-optimal points are connected
and labeled.

Usage:
    python -m benchmarks.plotting.plot_throughput_latency_pareto
    python -m benchmarks.plotting.plot_throughput_latency_pareto --stat p90
    python -m benchmarks.plotting.plot_throughput_latency_pareto --input results/pareto.json --source server
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
    "scatter": "#A8B6CC",
    "pareto_line": "#C44E52",
    "pareto_marker": "#C44E52",
    "label": "#2D3436",
}

# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def pareto_frontier(latencies: np.ndarray, throughputs: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-optimal points.

    A point is Pareto-optimal if no other point has both lower latency AND
    higher throughput.  We sort by latency ascending and sweep: a point is
    on the frontier if its throughput exceeds the running maximum.
    """
    order = np.argsort(latencies)
    pareto_indices = []
    max_throughput = -np.inf
    for idx in order:
        if throughputs[idx] > max_throughput:
            pareto_indices.append(idx)
            max_throughput = throughputs[idx]
    return np.array(pareto_indices)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_throughput_latency_pareto(
    data: dict, out_dir: str, stat: str = "mean", source: str = "client",
    skip_x: set[int] | None = None,
) -> str:
    conditions = data["conditions"]

    concurrencies = []
    latencies = []
    throughputs = []

    for cond_name, cond in conditions.items():
        concurrency = cond["dispatch"]["concurrency"]
        if skip_x and concurrency in skip_x:
            continue
        latency_s = cond["stats"]["e2e"][stat]
        latency_ms = latency_s * 1000

        if source == "server":
            tput = (cond.get("server_side_delta", {})
                       .get("throughput", {})
                       .get("output_tokens_per_second", 0.0))
        else:
            tput = cond["client_side_throughput"]["output_tokens_per_second"]

        concurrencies.append(concurrency)
        latencies.append(latency_ms)
        throughputs.append(tput)

    concurrencies = np.array(concurrencies)
    latencies = np.array(latencies)
    throughputs = np.array(throughputs)

    # Pareto frontier
    pareto_idx = pareto_frontier(latencies, throughputs)
    # Sort pareto points by latency for line drawing
    pareto_order = np.argsort(latencies[pareto_idx])
    pareto_idx = pareto_idx[pareto_order]

    fig, ax = plt.subplots(figsize=(9, 6))

    # All points (non-pareto) as faded scatter
    non_pareto_mask = np.ones(len(latencies), dtype=bool)
    non_pareto_mask[pareto_idx] = False
    ax.scatter(latencies[non_pareto_mask], throughputs[non_pareto_mask],
               color=COLORS["scatter"], s=80, zorder=2, edgecolors="white",
               linewidths=0.5, label="Dominated")

    # Pareto frontier line + markers
    ax.plot(latencies[pareto_idx], throughputs[pareto_idx], "o-",
            color=COLORS["pareto_line"], lw=2.5, markersize=9, zorder=4,
            markeredgecolor="white", markeredgewidth=1, label="Pareto frontier")

    # Label every point with its concurrency level
    for i in range(len(concurrencies)):
        is_pareto = i in pareto_idx
        weight = "bold" if is_pareto else "normal"
        color = COLORS["pareto_line"] if is_pareto else COLORS["label"]
        ax.annotate(
            f"N={concurrencies[i]}",
            (latencies[i], throughputs[i]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight=weight,
            color=color,
            zorder=5,
        )

    ax.set_xlabel(f"End-to-End Latency — {stat} (ms)")
    ax.set_ylabel("Output Throughput (tok/s)")
    ax.set_title(f"Throughput vs Latency Pareto Frontier ({source})")
    ax.legend(loc="best")

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"throughput_latency_pareto_{source}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    parser = argparse.ArgumentParser(
        description="Plot throughput vs latency Pareto frontier"
    )
    parser.add_argument("--input", default=None, help="Path to experiment results JSON")
    parser.add_argument("--output-dir", default=None, help="Directory for output PNGs")
    parser.add_argument("--dispatch-mode", default="concurrent",
                        help="Dispatch mode subfolder (default: %(default)s)")
    parser.add_argument("--stat", default="mean", choices=("mean", "p50", "p90", "p99"),
                        help="E2E latency statistic to use (default: mean)")
    parser.add_argument("--source", default="client", choices=("client", "server"),
                        help="Throughput source: client-side or server-side (default: client)")
    parser.add_argument("--skip-x", default=None,
                        help="Comma-separated concurrency values to exclude (e.g. 2048,4096)")
    args = parser.parse_args()

    results_dir = (Path(__file__).resolve().parent.parent
                   / "results" / args.dispatch_mode / "throughput_latency_pareto")
    input_path = args.input or str(results_dir / "throughput_latency_pareto.json")
    output_dir = args.output_dir or str(results_dir)

    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    skip_x = set(int(v) for v in args.skip_x.split(",")) if args.skip_x else None
    fpath = plot_throughput_latency_pareto(data, output_dir, stat=args.stat, source=args.source, skip_x=skip_x)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
