#!/usr/bin/env python3
"""Plot decode time vs output token length.

Reads the output of experiment_orchestrator.py for the decode_time_vs_output_length
experiment and produces a line plot showing how decode time scales with output size.

Usage:
    python -m benchmarks.plotting.plot_decode_time_vs_output_length
    python -m benchmarks.plotting.plot_decode_time_vs_output_length --stat p50
    python -m benchmarks.plotting.plot_decode_time_vs_output_length --input results/decode_time_vs_output_length.json
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
    "line": "#4C72B0",
    "scatter": "#55A868",
    "fit": "#C44E52",
}

VALID_STATS = ("mean", "p50", "p90", "p99")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_decode_time_vs_output_length(data: dict, out_dir: str, stat: str = "mean", scatter: bool = True) -> str:
    conditions = data["conditions"]

    output_tokens = []
    decode_vals = []
    decode_per_request: list[list[float]] = []

    for cond_name, cond in conditions.items():
        per_req = cond["per_request"]
        ok = [r for r in per_req if r["error"] is None and r["http_status"] == 200]
        if not ok:
            continue
        avg_output = np.mean([r["output_tokens"] for r in ok])
        output_tokens.append(avg_output)
        decode_vals.append(cond["stats"]["decode_time"][stat])
        decode_per_request.append([r["decode_time"] for r in ok])

    # Sort by output tokens
    order = np.argsort(output_tokens)
    output_tokens = np.array(output_tokens)[order]
    decode_vals = np.array(decode_vals)[order]
    decode_per_request = [decode_per_request[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter individual requests
    if scatter:
        for i, (tok, reqs) in enumerate(zip(output_tokens, decode_per_request)):
            xs = np.full(len(reqs), tok)
            jitter = np.random.default_rng(42).uniform(-1.5, 1.5, size=len(reqs))
            ax.scatter(xs + jitter, np.array(reqs) * 1000, s=25, alpha=0.35,
                       color=COLORS["scatter"], edgecolors="none", zorder=2)

    # Line for the chosen stat
    ax.plot(output_tokens, decode_vals * 1000, "o-", color=COLORS["line"], lw=2,
            markersize=7, zorder=3, label=f"Decode Time ({stat.upper()})")

    # Linear fit
    if len(output_tokens) >= 2:
        coeffs = np.polyfit(output_tokens, decode_vals * 1000, 1)
        x_fit = np.linspace(output_tokens.min(), output_tokens.max(), 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color=COLORS["fit"], lw=1.5,
                label=f"Linear fit ({coeffs[0]:.3f} ms/tok)", zorder=2)

    ax.set_xlabel("Output Tokens")
    ax.set_ylabel("Decode Time (ms)")
    ax.set_title(f"Decode Time vs Output Length ({stat.upper()})")
    ax.legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"decode_time_vs_output_length_{stat}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    parser = argparse.ArgumentParser(description="Plot decode time vs output length")
    parser.add_argument("--input", default=None, help="Path to experiment results JSON")
    parser.add_argument("--output-dir", default=None, help="Directory for output PNGs")
    parser.add_argument("--dispatch-mode", default="sequential", help="Dispatch mode subfolder (default: %(default)s)")
    parser.add_argument("--stat", default="mean", choices=VALID_STATS,
                        help="Which aggregation to plot (default: mean)")
    parser.add_argument("--no-scatter", action="store_true",
                        help="Hide individual request points, show only the aggregated line")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results" / args.dispatch_mode / "decode_time_vs_output_length"
    input_path = args.input or str(results_dir / "decode_time_vs_output_length.json")
    output_dir = args.output_dir or str(results_dir)

    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    fpath = plot_decode_time_vs_output_length(data, output_dir, stat=args.stat, scatter=not args.no_scatter)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
