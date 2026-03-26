#!/usr/bin/env python3
"""Generate plots for realistic benchmark results.

Reads test_results.json and produces 4 PNG plots characterising
throughput, latency distribution, error/queue pressure, and cache
behaviour under realistic (Poisson-arrival) workloads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

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
    # Fallback for older matplotlib versions
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

COLOR_PRIMARY = "#2563eb"
COLOR_SECONDARY = "#f97316"
COLOR_SHORT = "#6366f1"
COLOR_LONG = "#ec4899"
COLOR_BAR = "#94a3b8"
COLOR_ERROR_BAR = "#ef4444"
COLOR_QUEUE_LINE = "#2563eb"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_arrival_rate(mode: str) -> float:
    """Extract the arrival rate (rps) from a mode string like 'realistic-2rps-10s'."""
    m = re.search(r"(\d+(?:\.\d+)?)rps", mode)
    if m:
        return float(m.group(1))
    # Fallback: try bare number after 'realistic-'
    m = re.search(r"realistic[_-](\d+(?:\.\d+)?)", mode)
    if m:
        return float(m.group(1))
    return 0.0


def _prompt_type(name: str) -> str:
    """Return 'short' or 'long' based on prompt_name prefix."""
    if name.startswith("short"):
        return "short"
    if name.startswith("long"):
        return "long"
    return "other"


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a list of floats."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _load_realistic_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of realistic benchmark entries.

    Supports two layouts:
      1. Single entry under data["realistic"]
      2. Multiple entries under data["realistic"] being a list, or
         multiple top-level keys matching 'realistic*'.
    """
    entries: list[dict[str, Any]] = []

    if "realistic" in data:
        val = data["realistic"]
        if isinstance(val, list):
            entries.extend(val)
        elif isinstance(val, dict):
            if "mode" in val:
                entries.append(val)
            else:
                # Nested dict keyed by sub-name
                for v in val.values():
                    if isinstance(v, dict) and "mode" in v:
                        entries.append(v)

    # Also pick up any other top-level keys starting with 'realistic'
    for key, val in data.items():
        if key == "realistic":
            continue
        if key.startswith("realistic") and isinstance(val, dict) and "mode" in val:
            entries.append(val)

    return entries


# ---------------------------------------------------------------------------
# Plot R1 -- Throughput & TTFT vs Arrival Rate
# ---------------------------------------------------------------------------

def plot_r1(entries: list[dict], output_dir: str) -> str:
    """Throughput (output tok/s) and TTFT p95 vs arrival rate."""
    rates: list[float] = []
    throughputs: list[float] = []
    ttft_p95s: list[float] = []

    for entry in entries:
        rate = _parse_arrival_rate(entry["mode"])
        rates.append(rate)

        # Throughput
        tp = entry.get("server_side_delta", {}).get("throughput", {}).get("output_tokens_per_second", 0.0)
        if tp == 0.0:
            per_req = entry.get("per_request", [])
            total_out = sum(r.get("output_tokens", 0) for r in per_req)
            max_e2e = max((r.get("e2e", 0) for r in per_req), default=1.0)
            tp = total_out / max_e2e if max_e2e > 0 else 0.0
        throughputs.append(tp)

        # TTFT p95
        ttft_p95 = entry.get("server_side_delta", {}).get("latency", {}).get("ttft", {}).get("p95", 0.0)
        if ttft_p95 == 0.0:
            per_req = entry.get("per_request", [])
            ttft_vals = [r["ttft"] for r in per_req if "ttft" in r]
            ttft_p95 = _percentile(ttft_vals, 95) if ttft_vals else 0.0
        ttft_p95s.append(ttft_p95)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    single_rate = len(rates) == 1

    if single_rate:
        # Annotated bar chart for single data point
        x_pos = [0, 1]
        labels = ["Throughput\n(tok/s)", "TTFT p95\n(ms)"]
        values = [throughputs[0], ttft_p95s[0] * 1000]  # convert TTFT to ms
        colors = [COLOR_PRIMARY, COLOR_SECONDARY]

        bars = ax1.bar(x_pos, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("Value")
        ax1.set_title(f"Throughput & TTFT p95  |  {rates[0]:.0f} rps", fontsize=13, fontweight="bold")

        for bar, val in zip(bars, values):
            ax1.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
    else:
        # Multi-rate: dual-axis line chart
        order = np.argsort(rates)
        sorted_rates = [rates[i] for i in order]
        sorted_tp = [throughputs[i] for i in order]
        sorted_ttft = [t * 1000 for i in order for t in [ttft_p95s[i]]]

        ln1 = ax1.plot(sorted_rates, sorted_tp, "o-", color=COLOR_PRIMARY, linewidth=2, markersize=7, label="Throughput (tok/s)")
        ax1.set_xlabel("Arrival Rate (rps)", fontsize=11)
        ax1.set_ylabel("Output Tokens / s", color=COLOR_PRIMARY, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=COLOR_PRIMARY)

        ax2 = ax1.twinx()
        ln2 = ax2.plot(sorted_rates, sorted_ttft, "s--", color=COLOR_SECONDARY, linewidth=2, markersize=7, label="TTFT p95 (ms)")
        ax2.set_ylabel("TTFT p95 (ms)", color=COLOR_SECONDARY, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=COLOR_SECONDARY)

        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", framealpha=0.9)
        ax1.set_title("Throughput & TTFT p95 vs Arrival Rate", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_r1_throughput_ttft.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot R2 -- TTFT Distribution per Arrival Rate
# ---------------------------------------------------------------------------

def plot_r2(entries: list[dict], output_dir: str) -> str:
    """Box plot of per-request TTFT, coloured by prompt type."""
    fig, ax = plt.subplots(figsize=(9, 5))

    single_rate = len(entries) == 1

    if single_rate:
        entry = entries[0]
        rate = _parse_arrival_rate(entry["mode"])
        per_req = entry.get("per_request", [])

        types = [_prompt_type(r["prompt_name"]) for r in per_req]
        ttfts = [r["ttft"] * 1000 for r in per_req]  # ms
        unique_types = sorted(set(types))
        type_data = {t: [] for t in unique_types}
        for t, v in zip(types, ttfts):
            type_data[t].append(v)

        positions = list(range(len(unique_types)))
        bp_data = [type_data[t] for t in unique_types]
        bp = ax.boxplot(
            bp_data, positions=positions, widths=0.4, patch_artist=True,
            showfliers=False, medianprops=dict(color="black", linewidth=1.5),
        )
        type_colors = {"short": COLOR_SHORT, "long": COLOR_LONG, "other": COLOR_BAR}
        for patch, t in zip(bp["boxes"], unique_types):
            patch.set_facecolor(type_colors.get(t, COLOR_BAR))
            patch.set_alpha(0.5)

        # Overlay individual points with jitter
        rng = np.random.default_rng(42)
        for idx, t in enumerate(unique_types):
            vals = np.array(type_data[t])
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                idx + jitter, vals, color=type_colors.get(t, COLOR_BAR),
                edgecolor="white", linewidth=0.5, s=40, zorder=5, label=t,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{t}\n({len(type_data[t])} reqs)" for t in unique_types])
        ax.set_xlabel(f"Prompt Type  |  {rate:.0f} rps", fontsize=11)

        # Annotate p99/p50 ratio
        all_ttfts = ttfts
        p50 = _percentile(all_ttfts, 50)
        p99 = _percentile(all_ttfts, 99)
        ratio = p99 / p50 if p50 > 0 else float("inf")
        ax.annotate(
            f"p99/p50 = {ratio:.2f}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )
    else:
        # Multiple rates: one box per rate
        rates_sorted = sorted(
            enumerate(entries), key=lambda x: _parse_arrival_rate(x[1]["mode"])
        )
        bp_data = []
        labels = []
        all_types_sets: list[set] = []
        for _, entry in rates_sorted:
            per_req = entry.get("per_request", [])
            ttfts = [r["ttft"] * 1000 for r in per_req]
            bp_data.append(ttfts)
            rate = _parse_arrival_rate(entry["mode"])
            labels.append(f"{rate:.0f} rps")
            all_types_sets.append({_prompt_type(r["prompt_name"]) for r in per_req})

        positions = list(range(len(bp_data)))
        bp = ax.boxplot(
            bp_data, positions=positions, widths=0.4, patch_artist=True,
            showfliers=False, medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR_PRIMARY)
            patch.set_alpha(0.4)

        # Overlay points coloured by type
        rng = np.random.default_rng(42)
        type_colors = {"short": COLOR_SHORT, "long": COLOR_LONG, "other": COLOR_BAR}
        plotted_labels: set[str] = set()
        for pos, (_, entry) in zip(positions, rates_sorted):
            per_req = entry.get("per_request", [])
            for r in per_req:
                t = _prompt_type(r["prompt_name"])
                jit = rng.uniform(-0.12, 0.12)
                lbl = t if t not in plotted_labels else None
                ax.scatter(
                    pos + jit, r["ttft"] * 1000,
                    color=type_colors.get(t, COLOR_BAR),
                    edgecolor="white", linewidth=0.5, s=40, zorder=5, label=lbl,
                )
                if lbl:
                    plotted_labels.add(t)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Arrival Rate", fontsize=11)

        # p99/p50 annotation for the whole dataset
        all_ttfts = [r["ttft"] * 1000 for e in entries for r in e.get("per_request", [])]
        p50 = _percentile(all_ttfts, 50)
        p99 = _percentile(all_ttfts, 99)
        ratio = p99 / p50 if p50 > 0 else float("inf")
        ax.annotate(
            f"p99/p50 = {ratio:.2f}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )

    ax.set_ylabel("TTFT (ms)", fontsize=11)
    ax.set_title("TTFT Distribution by Prompt Type", fontsize=13, fontweight="bold")

    # De-duplicate legend entries
    handles, lbls = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen[l] = h
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="upper left", framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_r2_ttft_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot R3 -- Error Rate & Queue Pressure vs Arrival Rate
# ---------------------------------------------------------------------------

def plot_r3(entries: list[dict], output_dir: str) -> str:
    """Error rate (%) and max queue depth vs arrival rate."""
    rates: list[float] = []
    error_rates: list[float] = []
    max_depths: list[int] = []
    annotations: list[str] = []

    for entry in entries:
        rate = _parse_arrival_rate(entry["mode"])
        rates.append(rate)
        err = entry.get("server_side_delta", {}).get("errors", {}).get("error_rate", 0.0)
        error_rates.append(err * 100 if err < 1.0 else err)  # ensure %
        depth = entry.get("server_side_delta", {}).get("queue", {}).get("max_depth", 0)
        max_depths.append(depth)
        cs = entry.get("client_side", {})
        annotations.append(f"{cs.get('failed_requests', 0)}/{cs.get('total_requests', 0)}")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    single_rate = len(rates) == 1
    all_zero = all(e == 0.0 for e in error_rates) and all(d == 0 for d in max_depths)

    if single_rate:
        x_pos = [0, 1]
        bar_labels = ["Error Rate (%)", "Max Queue Depth"]
        values = [error_rates[0], float(max_depths[0])]
        colors = [COLOR_ERROR_BAR, COLOR_QUEUE_LINE]

        bars = ax1.bar(x_pos, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bar_labels)
        ax1.set_ylabel("Value")
        ax1.set_title(f"Error Rate & Queue Depth  |  {rates[0]:.0f} rps", fontsize=13, fontweight="bold")

        for bar, val in zip(bars, values):
            ax1.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        # Failed / total annotation
        ax1.annotate(
            f"failed / total = {annotations[0]}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )
    else:
        order = np.argsort(rates)
        sorted_rates = [rates[i] for i in order]
        sorted_err = [error_rates[i] for i in order]
        sorted_depth = [max_depths[i] for i in order]
        sorted_ann = [annotations[i] for i in order]

        x = np.arange(len(sorted_rates))
        width = 0.35

        bars = ax1.bar(x, sorted_err, width, color=COLOR_ERROR_BAR, alpha=0.8, label="Error Rate (%)")
        ax1.set_xlabel("Arrival Rate (rps)", fontsize=11)
        ax1.set_ylabel("Error Rate (%)", color=COLOR_ERROR_BAR, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=COLOR_ERROR_BAR)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{r:.0f}" for r in sorted_rates])

        ax2 = ax1.twinx()
        ax2.plot(x, sorted_depth, "D-", color=COLOR_QUEUE_LINE, linewidth=2, markersize=7, label="Max Queue Depth")
        ax2.set_ylabel("Max Queue Depth", color=COLOR_QUEUE_LINE, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=COLOR_QUEUE_LINE)

        # Annotate failed/total on each bar
        for xi, ann in zip(x, sorted_ann):
            ax1.annotate(
                ann, xy=(xi, 0), xytext=(0, -18), textcoords="offset points",
                ha="center", fontsize=8, color="gray",
            )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)
        ax1.set_title("Error Rate & Queue Pressure vs Arrival Rate", fontsize=13, fontweight="bold")

    if all_zero:
        ax1.annotate(
            "No errors or queue pressure observed",
            xy=(0.5, 0.5), xycoords="axes fraction",
            ha="center", va="center", fontsize=11, fontstyle="italic", color="gray",
        )
        # Ensure y-axis shows a small range so bars/annotations are visible
        ax1.set_ylim(-0.5, 2.0)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_r3_error_queue.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot R4 -- Cache Hit Rate vs TTFT Reduction
# ---------------------------------------------------------------------------

def plot_r4(entries: list[dict], output_dir: str) -> str:
    """Cache hit rate vs TTFT; falls back to per-request TTFT by type."""
    # Check if we have multiple distinct cache configs
    hit_rates = [
        e.get("server_side_delta", {}).get("kv_cache", {}).get("hit_rate", 0.0)
        for e in entries
    ]
    unique_hrs = set(hit_rates)
    multiple_configs = len(unique_hrs) > 1

    fig, ax = plt.subplots(figsize=(8, 5))
    type_colors = {"short": COLOR_SHORT, "long": COLOR_LONG, "other": COLOR_BAR}

    if multiple_configs:
        # Scatter: x = hit_rate, y = avg TTFT
        plotted_labels: set[str] = set()
        for entry in entries:
            hr = entry.get("server_side_delta", {}).get("kv_cache", {}).get("hit_rate", 0.0)
            per_req = entry.get("per_request", [])
            # Group by type
            type_data: dict[str, list[float]] = {}
            for r in per_req:
                t = _prompt_type(r["prompt_name"])
                type_data.setdefault(t, []).append(r["ttft"] * 1000)
            for t, vals in type_data.items():
                lbl = t if t not in plotted_labels else None
                ax.scatter(
                    hr * 100, np.mean(vals),
                    color=type_colors.get(t, COLOR_BAR), s=80,
                    edgecolor="white", linewidth=0.8, zorder=5, label=lbl,
                )
                if lbl:
                    plotted_labels.add(t)

        ax.set_xlabel("KV Cache Hit Rate (%)", fontsize=11)
        ax.set_ylabel("Avg TTFT (ms)", fontsize=11)
        ax.set_title("Cache Hit Rate vs TTFT", fontsize=13, fontweight="bold")
        ax.legend(title="Prompt Type", framealpha=0.9)
    else:
        # Single cache config -- grouped bar of per-request TTFT by type
        entry = entries[0]
        per_req = entry.get("per_request", [])
        kv = entry.get("server_side_delta", {}).get("kv_cache", {})
        hr = kv.get("hit_rate", 0.0)
        l1_util = kv.get("l1_utilization_ratio", 0.0)

        type_data: dict[str, list[float]] = {}
        for r in per_req:
            t = _prompt_type(r["prompt_name"])
            type_data.setdefault(t, []).append(r["ttft"] * 1000)

        unique_types = sorted(type_data.keys())
        x = np.arange(len(unique_types))
        means = [np.mean(type_data[t]) for t in unique_types]
        stds = [np.std(type_data[t]) for t in unique_types]
        colors = [type_colors.get(t, COLOR_BAR) for t in unique_types]

        bars = ax.bar(x, means, yerr=stds, width=0.45, color=colors, alpha=0.75,
                      edgecolor="white", linewidth=1.2, capsize=4)

        # Overlay individual points
        rng = np.random.default_rng(42)
        for idx, t in enumerate(unique_types):
            vals = np.array(type_data[t])
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(idx + jitter, vals, color=colors[idx], edgecolor="white",
                       linewidth=0.5, s=30, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n({len(type_data[t])} reqs)" for t in unique_types])
        ax.set_xlabel("Prompt Type", fontsize=11)
        ax.set_ylabel("TTFT (ms)", fontsize=11)
        ax.set_title("Per-Request TTFT by Prompt Type", fontsize=13, fontweight="bold")

        # Annotate bar values
        for bar, val in zip(bars, means):
            ax.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        # Annotate cache stats
        ax.annotate(
            f"Cache hit rate: {hr * 100:.1f}%\nL1 utilization: {l1_util * 100:.1f}%",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "plot_r4_cache_ttft.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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
        "resources", "realistic",
    )

    parser = argparse.ArgumentParser(
        description="Generate realistic benchmark plots from test_results.json",
    )
    parser.add_argument(
        "--input", default=default_input,
        help="Path to test_results.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", default=default_output,
        help="Directory to save PNG plots (default: %(default)s)",
    )
    args = parser.parse_args()

    # Load data
    with open(args.input, "r") as f:
        data = json.load(f)

    entries = _load_realistic_entries(data)
    if not entries:
        print("ERROR: No 'realistic' entries found in input file.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    saved: list[str] = []
    saved.append(plot_r1(entries, args.output_dir))
    saved.append(plot_r2(entries, args.output_dir))
    saved.append(plot_r3(entries, args.output_dir))
    saved.append(plot_r4(entries, args.output_dir))

    print(f"Saved {len(saved)} plots to {args.output_dir}/")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
