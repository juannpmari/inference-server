#!/usr/bin/env python3
"""Generate sequential benchmark plots from test_results.json.

Produces four PNG charts that characterise single-stream inference
behaviour: latency decomposition, prefill scaling, cache impact, and
decode linearity.

Usage:
    python plot_sequential.py
    python plot_sequential.py --input path/to/results.json --output-dir path/to/out/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
_STYLE = "seaborn-v0_8-whitegrid"
try:
    plt.style.use(_STYLE)
except OSError:
    # Fallback for older matplotlib without the seaborn-v0_8 aliases
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # default style is fine

COLORS = {
    "prefill": "#4C72B0",
    "decode": "#DD8452",
    "scatter": "#55A868",
    "fit": "#C44E52",
    "ref": "#8172B3",
    "bar_cmap": plt.cm.Set2,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_group(name: str) -> str:
    """Extract the prefix before the first underscore+digits (e.g. 'short')."""
    m = re.match(r"^([a-zA-Z]+)", name)
    return m.group(1) if m else name


def _load_data(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if "sequential" not in data:
        print(f"ERROR: key 'sequential' not found in {path}", file=sys.stderr)
        sys.exit(1)
    return data["sequential"]


def _server_metric_nonzero(seq: Dict[str, Any], *keys: str) -> float:
    """Walk into server_side_delta by keys; return value if > 0 else 0."""
    node = seq.get("server_side_delta", {})
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k, 0.0)
        else:
            return 0.0
    return float(node) if node else 0.0


def _group_requests(per_request: List[Dict]) -> Dict[str, List[Dict]]:
    """Group per_request entries by prompt prefix, ordered by mean input_tokens."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in per_request:
        groups[_prompt_group(r["prompt_name"])].append(r)
    # sort groups by average input_tokens ascending
    sorted_groups = dict(
        sorted(groups.items(), key=lambda kv: np.mean([r["input_tokens"] for r in kv[1]]))
    )
    return sorted_groups

# ---------------------------------------------------------------------------
# S1 -- Latency Decomposition (Stacked Bar)
# ---------------------------------------------------------------------------

def plot_s1(seq: Dict[str, Any], out_dir: str) -> str:
    per_request = seq["per_request"]
    groups = _group_requests(per_request)

    server_ttft = _server_metric_nonzero(seq, "latency", "ttft", "avg")
    server_e2e = _server_metric_nonzero(seq, "latency", "e2e", "avg")

    labels: List[str] = []
    prefill_vals: List[float] = []
    decode_vals: List[float] = []

    for group_name, reqs in groups.items():
        avg_input = np.mean([r["input_tokens"] for r in reqs])
        labels.append(f"{group_name}\n(~{int(avg_input)} tok)")

        if server_ttft > 0:
            pf = server_ttft
        else:
            pf = float(np.mean([r["ttft"] for r in reqs]))

        if server_e2e > 0:
            dc = server_e2e - server_ttft
        else:
            dc = float(np.mean([r["e2e"] - r["ttft"] for r in reqs]))

        prefill_vals.append(pf)
        decode_vals.append(dc)

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.8), 5))
    bars_pf = ax.bar(x, prefill_vals, width, label="Prefill (TTFT)", color=COLORS["prefill"])
    bars_dc = ax.bar(x, decode_vals, width, bottom=prefill_vals, label="Decode", color=COLORS["decode"])

    ax.set_xlabel("Prompt Workload")
    ax.set_ylabel("Time (s)")
    ax.set_title("S1 -- Latency Decomposition (Sequential)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # annotate total on top
    for i in range(len(labels)):
        total = prefill_vals[i] + decode_vals[i]
        ax.annotate(
            f"{total:.3f}s",
            xy=(x[i], total),
            ha="center", va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fpath = os.path.join(out_dir, "plot_s1_latency_decomposition.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath

# ---------------------------------------------------------------------------
# S2 -- Prefill Scaling: TTFT vs Input Token Count
# ---------------------------------------------------------------------------

def plot_s2(seq: Dict[str, Any], out_dir: str) -> str:
    per_request = seq["per_request"]
    input_tokens = np.array([r["input_tokens"] for r in per_request], dtype=float)
    ttfts = np.array([r["ttft"] for r in per_request], dtype=float)

    unique_inputs = np.unique(input_tokens)
    single_bucket = len(unique_inputs) == 1

    fig, ax = plt.subplots(figsize=(7, 5))

    if single_bucket:
        # Strip / swarm-style plot
        token_val = unique_inputs[0]
        jitter = np.random.default_rng(42).uniform(-0.3, 0.3, size=len(ttfts))
        ax.scatter(
            np.zeros_like(ttfts) + jitter,
            ttfts * 1000,
            s=60, alpha=0.7, color=COLORS["scatter"], edgecolors="k", linewidths=0.5,
        )
        ax.axhline(np.mean(ttfts) * 1000, color=COLORS["fit"], ls="--", lw=1.5, label=f"Mean = {np.mean(ttfts)*1000:.1f} ms")
        ax.set_xticks([0])
        ax.set_xticklabels([f"{int(token_val)} tokens"])
        ax.set_xlabel("Input Token Count")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("S2 -- Prefill Scaling: TTFT vs Input Tokens (Sequential)")
        ax.annotate(
            f"All requests have {int(token_val)} input tokens.\n"
            "Trend line requires varied input sizes.",
            xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top",
            fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
        )
        ax.legend()
    else:
        ax.scatter(input_tokens, ttfts, s=60, alpha=0.7, color=COLORS["scatter"],
                   edgecolors="k", linewidths=0.5, label="Requests")
        # linear fit on two smallest unique input sizes
        sorted_uniq = np.sort(unique_inputs)
        ref_mask = np.isin(input_tokens, sorted_uniq[:2])
        if ref_mask.sum() >= 2:
            coeffs_ref = np.polyfit(input_tokens[ref_mask], ttfts[ref_mask], 1)
            x_line = np.linspace(input_tokens.min(), input_tokens.max(), 200)
            ax.plot(x_line, np.polyval(coeffs_ref, x_line), ls="--", color=COLORS["ref"],
                    lw=1.5, label="Linear ref (smallest sizes)")

        # overall trend
        if len(input_tokens) >= 2:
            coeffs = np.polyfit(input_tokens, ttfts, 1)
            x_line = np.linspace(input_tokens.min(), input_tokens.max(), 200)
            ax.plot(x_line, np.polyval(coeffs, x_line), color=COLORS["fit"], lw=2,
                    label=f"Linear fit (slope={coeffs[0]*1000:.3f} ms/tok)")

        ax.set_xlabel("Input Token Count")
        ax.set_ylabel("TTFT (s)")
        ax.set_title("S2 -- Prefill Scaling: TTFT vs Input Tokens (Sequential)")
        ax.legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, "plot_s2_prefill_scaling.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath

# ---------------------------------------------------------------------------
# S3 -- Cache Impact on TTFT
# ---------------------------------------------------------------------------

def plot_s3(seq: Dict[str, Any], out_dir: str) -> str:
    per_request = seq["per_request"]
    kv = seq.get("server_side_delta", {}).get("kv_cache", {})
    hit_rate = kv.get("hit_rate", 0.0)
    l1_util = kv.get("l1_utilization_ratio", 0.0)

    # If we had cache_on / cache_off scenarios we would group by them.
    # With a single run we show per-request TTFT coloured by prompt group.
    groups = _group_requests(per_request)

    fig, ax = plt.subplots(figsize=(max(7, len(per_request) * 0.7), 5))

    bar_labels: List[str] = []
    bar_ttfts: List[float] = []
    bar_colors: List[Any] = []
    cmap = COLORS["bar_cmap"]
    group_names = list(groups.keys())
    color_map = {g: cmap(i / max(len(group_names) - 1, 1)) for i, g in enumerate(group_names)}

    for group_name, reqs in groups.items():
        for r in reqs:
            bar_labels.append(r["prompt_name"])
            bar_ttfts.append(r["ttft"])
            bar_colors.append(color_map[group_name])

    x = np.arange(len(bar_labels))
    bars = ax.bar(x, bar_ttfts, color=bar_colors, edgecolor="k", linewidth=0.4)

    ax.set_xlabel("Request")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("S3 -- Cache Impact on TTFT (Sequential)")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)

    # legend for groups
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color_map[g], edgecolor="k", label=g) for g in group_names]
    ax.legend(handles=legend_handles, title="Prompt group")

    # annotate cache stats
    annotation = (
        f"KV cache hit rate: {hit_rate:.1%}\n"
        f"L1 utilization: {l1_util:.1%}"
    )
    ax.annotate(
        annotation,
        xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"),
    )

    plt.tight_layout()
    fpath = os.path.join(out_dir, "plot_s3_cache_impact.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath

# ---------------------------------------------------------------------------
# S4 -- Decode Linearity: Decode Time vs Output Token Count
# ---------------------------------------------------------------------------

def plot_s4(seq: Dict[str, Any], out_dir: str) -> str:
    per_request = seq["per_request"]
    output_tokens = np.array([r["output_tokens"] for r in per_request], dtype=float)
    decode_times = np.array([r["e2e"] - r["ttft"] for r in per_request], dtype=float)

    unique_outputs = np.unique(output_tokens)
    single_bucket = len(unique_outputs) == 1

    fig, ax = plt.subplots(figsize=(7, 5))

    if single_bucket:
        token_val = unique_outputs[0]
        jitter = np.random.default_rng(42).uniform(-0.3, 0.3, size=len(decode_times))
        ax.scatter(
            np.zeros_like(decode_times) + jitter,
            decode_times * 1000,
            s=60, alpha=0.7, color=COLORS["scatter"], edgecolors="k", linewidths=0.5,
        )
        mean_dt = np.mean(decode_times)
        est_itl = mean_dt / token_val * 1000 if token_val > 0 else 0
        ax.axhline(mean_dt * 1000, color=COLORS["fit"], ls="--", lw=1.5,
                    label=f"Mean = {mean_dt*1000:.1f} ms  (est ITL = {est_itl:.2f} ms/tok)")
        ax.set_xticks([0])
        ax.set_xticklabels([f"{int(token_val)} tokens"])
        ax.set_xlabel("Output Token Count")
        ax.set_ylabel("Decode Time (ms)")
        ax.set_title("S4 -- Decode Linearity: Decode Time vs Output Tokens (Sequential)")
        ax.annotate(
            f"All requests have {int(token_val)} output tokens.\n"
            "Linear fit requires varied output sizes.",
            xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top",
            fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
        )
        ax.legend(fontsize=8)
    else:
        ax.scatter(output_tokens, decode_times, s=60, alpha=0.7, color=COLORS["scatter"],
                   edgecolors="k", linewidths=0.5, label="Requests")
        if len(output_tokens) >= 2:
            coeffs = np.polyfit(output_tokens, decode_times, 1)
            x_line = np.linspace(output_tokens.min(), output_tokens.max(), 200)
            slope_ms = coeffs[0] * 1000
            ax.plot(x_line, np.polyval(coeffs, x_line), color=COLORS["fit"], lw=2,
                    label=f"Linear fit (slope = {slope_ms:.3f} ms/tok)")
            ax.annotate(
                f"Estimated inter-token latency: {slope_ms:.3f} ms",
                xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
            )
        ax.set_xlabel("Output Token Count")
        ax.set_ylabel("Decode Time (s)")
        ax.set_title("S4 -- Decode Linearity: Decode Time vs Output Tokens (Sequential)")
        ax.legend()

    plt.tight_layout()
    fpath = os.path.join(out_dir, "plot_s4_decode_linearity.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    default_input = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_results.json",
    )
    default_output = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "resources", "sequential",
    )

    parser = argparse.ArgumentParser(description="Plot sequential benchmark results")
    parser.add_argument(
        "--input", default=default_input,
        help="Path to test_results.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", default=default_output,
        help="Directory for output PNGs (default: %(default)s)",
    )
    args = parser.parse_args()

    seq = _load_data(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    saved: List[str] = []
    saved.append(plot_s1(seq, args.output_dir))
    saved.append(plot_s2(seq, args.output_dir))
    saved.append(plot_s3(seq, args.output_dir))
    saved.append(plot_s4(seq, args.output_dir))

    print(f"Saved {len(saved)} plots:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
