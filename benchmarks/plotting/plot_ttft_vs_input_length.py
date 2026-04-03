#!/usr/bin/env python3
"""Plot TTFT vs input token length.

Reads the output of run_experiment.py for the ttft_vs_input_length experiment
and produces a line plot with error bands showing how TTFT scales with input size.

Usage:
    python -m benchmarks.plotting.plot_ttft_vs_input_length
    python -m benchmarks.plotting.plot_ttft_vs_input_length --stat p50
    python -m benchmarks.plotting.plot_ttft_vs_input_length --input results/ttft_vs_input_length.json
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

def plot_ttft_vs_input_length(data: dict, out_dir: str, stat: str = "mean") -> str:
    conditions = data["conditions"]

    input_tokens = []
    ttft_vals = []
    ttft_per_request: list[list[float]] = []

    for cond_name, cond in conditions.items():
        # Use the actual input_tokens from per_request data (more accurate than condition name)
        per_req = cond["per_request"]
        ok = [r for r in per_req if r["error"] is None and r["http_status"] == 200]
        if not ok:
            continue
        avg_input = np.mean([r["input_tokens"] for r in ok])
        input_tokens.append(avg_input)
        ttft_vals.append(cond["stats"]["ttft"][stat])
        ttft_per_request.append([r["ttft"] for r in ok])

    # Sort by input tokens
    order = np.argsort(input_tokens)
    input_tokens = np.array(input_tokens)[order]
    ttft_vals = np.array(ttft_vals)[order]
    ttft_per_request = [ttft_per_request[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter individual requests
    for i, (tok, reqs) in enumerate(zip(input_tokens, ttft_per_request)):
        xs = np.full(len(reqs), tok)
        jitter = np.random.default_rng(42).uniform(-1.5, 1.5, size=len(reqs))
        ax.scatter(xs + jitter, np.array(reqs) * 1000, s=25, alpha=0.35,
                   color=COLORS["scatter"], edgecolors="none", zorder=2)

    # Line for the chosen stat
    ax.plot(input_tokens, ttft_vals * 1000, "o-", color=COLORS["line"], lw=2,
            markersize=7, zorder=3, label=f"TTFT ({stat.upper()})")

    # Linear fit
    if len(input_tokens) >= 2:
        coeffs = np.polyfit(input_tokens, ttft_vals * 1000, 1)
        x_fit = np.linspace(input_tokens.min(), input_tokens.max(), 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color=COLORS["fit"], lw=1.5,
                label=f"Linear fit ({coeffs[0]:.3f} ms/tok)", zorder=2)

    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"TTFT vs Input Length ({stat.upper()})")
    ax.legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"ttft_vs_input_length_{stat}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    default_input = str(Path(__file__).resolve().parent.parent / "results" / "ttft_vs_input_length.json")
    default_output = str(Path(__file__).resolve().parent.parent.parent / "resources" / "ttft_vs_input_length")

    parser = argparse.ArgumentParser(description="Plot TTFT vs input length")
    parser.add_argument("--input", default=default_input, help="Path to experiment results JSON")
    parser.add_argument("--output-dir", default=default_output, help="Directory for output PNGs")
    parser.add_argument("--stat", default="mean", choices=VALID_STATS,
                        help="Which aggregation to plot (default: mean)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    fpath = plot_ttft_vs_input_length(data, args.output_dir, stat=args.stat)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
