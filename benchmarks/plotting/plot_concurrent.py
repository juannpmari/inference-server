#!/usr/bin/env python3
"""Generate concurrency benchmark plots from test_results.json.

Produces 4 PNG plots:
  C1 - Throughput Scaling Curve
  C2 - TTFT Tail Inflation (Percentile Fan)
  C3 - Error Rate & Queue Saturation
  C4 - Throughput vs Latency Pareto Frontier
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
SEQ_COLOR = "#888888"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_concurrency(mode_str: str) -> int:
    """Extract concurrency level N from a mode string like 'concurrent-4'."""
    m = re.search(r"concurrent[_-](\d+)", mode_str)
    if m:
        return int(m.group(1))
    return 0


def _compute_throughput(run: dict) -> float:
    """Compute output tokens/s from per-request data (total tokens / max e2e)."""
    server_tput = run.get("server_side_delta", {}).get("throughput", {}).get(
        "output_tokens_per_second", 0.0
    )
    if server_tput and server_tput > 0:
        return server_tput

    per_req = run.get("per_request", [])
    if not per_req:
        return 0.0
    total_output = sum(r.get("output_tokens", 0) for r in per_req)
    max_e2e = max(r.get("e2e", 0) for r in per_req)
    if max_e2e <= 0:
        return 0.0
    return total_output / max_e2e


def _collect_concurrent_runs(data: dict) -> list[dict]:
    """Gather all concurrent run entries from the JSON data.

    Supports several layouts:
      1. data["concurrent"] is a single run dict  (current format)
      2. data["concurrent"] is a list of run dicts
      3. Top-level keys like "concurrent-4", "concurrent-8", etc.
    """
    runs: list[dict] = []

    conc = data.get("concurrent")
    if conc is not None:
        if isinstance(conc, list):
            runs.extend(conc)
        elif isinstance(conc, dict):
            # Could be a single run or a mapping of runs
            if "mode" in conc:
                runs.append(conc)
            else:
                # Possibly {"concurrent-4": {...}, "concurrent-8": {...}}
                for v in conc.values():
                    if isinstance(v, dict) and "mode" in v:
                        runs.append(v)

    # Also check top-level keys matching "concurrent-*"
    for key, val in data.items():
        if key == "concurrent":
            continue
        if re.match(r"concurrent[_-]\d+", key) and isinstance(val, dict):
            runs.append(val)

    # Deduplicate by mode string
    seen = set()
    unique: list[dict] = []
    for r in runs:
        mode = r.get("mode", "")
        if mode not in seen:
            seen.add(mode)
            unique.append(r)

    # Sort by concurrency level
    unique.sort(key=lambda r: _parse_concurrency(r.get("mode", "")))
    return unique


def _get_sequential(data: dict) -> dict | None:
    return data.get("sequential")


# ---------------------------------------------------------------------------
# Plot C1 — Throughput Scaling Curve
# ---------------------------------------------------------------------------

def plot_c1(runs: list[dict], seq: dict | None, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [_parse_concurrency(r["mode"]) for r in runs]
    tputs = [_compute_throughput(r) for r in runs]

    single = len(ns) == 1

    if single:
        # Bar chart: one bar per prompt workload + aggregate
        run = runs[0]
        n = ns[0]
        per_req = run.get("per_request", [])

        # Group by prompt workload (prompt_name prefix)
        workloads: dict[str, list[dict]] = {}
        for r in per_req:
            name = r.get("prompt_name", "unknown")
            workloads.setdefault(name, []).append(r)

        labels = []
        values = []
        for wl_name, reqs in sorted(workloads.items()):
            total_tok = sum(r.get("output_tokens", 0) for r in reqs)
            max_e2e = max(r.get("e2e", 0) for r in reqs) if reqs else 1
            labels.append(wl_name)
            values.append(total_tok / max_e2e if max_e2e > 0 else 0)

        # Add aggregate bar
        labels.append(f"Aggregate\n(N={n})")
        values.append(tputs[0])

        x_pos = range(len(labels))
        bars = ax.bar(x_pos, values, color=COLORS[: len(labels)], edgecolor="white")

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        # Sequential baseline line
        if seq:
            seq_tput = _compute_throughput(seq)
            ax.axhline(
                seq_tput,
                color=SEQ_COLOR,
                linestyle="--",
                linewidth=1.5,
                label=f"Sequential baseline ({seq_tput:.1f} tok/s)",
            )
            ax.legend(fontsize=9)

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Output Tokens / s")
        ax.set_title(f"C1 — Throughput at Concurrency N={n}")
    else:
        # Multi-concurrency line chart
        ax.plot(ns, tputs, "o-", color=COLORS[0], linewidth=2, markersize=7, label="Concurrent")
        for n, t in zip(ns, tputs):
            ax.annotate(f"{t:.1f}", (n, t), textcoords="offset points", xytext=(5, 5), fontsize=8)

        if seq:
            seq_tput = _compute_throughput(seq)
            ax.axhline(seq_tput, color=SEQ_COLOR, linestyle="--", linewidth=1.5,
                        label=f"Sequential (N=1): {seq_tput:.1f} tok/s")

        ax.set_xlabel("Concurrency Level (N)")
        ax.set_ylabel("Output Tokens / s")
        ax.set_title("C1 — Throughput Scaling Curve")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_c1_throughput_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot C2 — TTFT Tail Inflation (Percentile Fan)
# ---------------------------------------------------------------------------

def plot_c2(runs: list[dict], seq: dict | None, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [_parse_concurrency(r["mode"]) for r in runs]

    single = len(ns) == 1

    if single:
        run = runs[0]
        n = ns[0]
        cs = run["client_side"]["ttft"]
        percentiles = ["p50", "p90", "p99"]
        vals = [cs[p] * 1000 for p in percentiles]  # ms

        x_pos = range(len(percentiles))
        colors_bar = [COLORS[0], COLORS[1], COLORS[3]]
        bars = ax.bar(x_pos, vals, color=colors_bar, edgecolor="white", width=0.5)

        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.1f} ms",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        # p99/p50 ratio annotation
        ratio = cs["p99"] / cs["p50"] if cs["p50"] > 0 else float("inf")
        ax.annotate(
            f"p99/p50 = {ratio:.2f}x",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#cccccc"),
        )

        # Sequential baseline markers
        if seq:
            seq_cs = seq["client_side"]["ttft"]
            for i, p in enumerate(percentiles):
                ax.plot(
                    i,
                    seq_cs[p] * 1000,
                    marker="D",
                    color=SEQ_COLOR,
                    markersize=8,
                    zorder=5,
                    label=f"Seq {p}" if i == 0 else None,
                )
                ax.annotate(
                    f"{seq_cs[p]*1000:.1f}",
                    (i, seq_cs[p] * 1000),
                    textcoords="offset points",
                    xytext=(12, 0),
                    fontsize=8,
                    color=SEQ_COLOR,
                )
            ax.legend(["Sequential"], fontsize=9)

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(percentiles, fontsize=10)
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(f"C2 — TTFT Percentiles at Concurrency N={n}")
    else:
        p50s = [r["client_side"]["ttft"]["p50"] * 1000 for r in runs]
        p90s = [r["client_side"]["ttft"]["p90"] * 1000 for r in runs]
        p99s = [r["client_side"]["ttft"]["p99"] * 1000 for r in runs]

        ax.fill_between(ns, p50s, p99s, alpha=0.2, color=COLORS[0], label="p50–p99 band")
        ax.fill_between(ns, p50s, p90s, alpha=0.3, color=COLORS[0], label="p50–p90 band")
        ax.plot(ns, p50s, "o-", color=COLORS[0], label="p50")
        ax.plot(ns, p90s, "s--", color=COLORS[1], label="p90")
        ax.plot(ns, p99s, "^:", color=COLORS[3], label="p99")

        if seq:
            seq_cs = seq["client_side"]["ttft"]
            ax.axhline(seq_cs["p99"] * 1000, color=SEQ_COLOR, linestyle=":", linewidth=1,
                        label=f"Seq p99 ({seq_cs['p99']*1000:.1f} ms)")

        ax.set_xlabel("Concurrency Level (N)")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("C2 — TTFT Tail Inflation (Percentile Fan)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_c2_ttft_tail.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot C3 — Error Rate & Queue Saturation
# ---------------------------------------------------------------------------

def plot_c3(runs: list[dict], seq: dict | None, output_dir: str) -> str:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ns = [_parse_concurrency(r["mode"]) for r in runs]

    error_rates = [
        r.get("server_side_delta", {}).get("errors", {}).get("error_rate", 0.0)
        for r in runs
    ]
    max_depths = [
        r.get("server_side_delta", {}).get("queue", {}).get("max_depth", 0)
        for r in runs
    ]

    single = len(ns) == 1

    if single:
        n = ns[0]
        labels = ["Error Rate (%)", "Max Queue Depth"]
        values = [error_rates[0], max_depths[0]]
        colors_bar = [COLORS[3], COLORS[0]]

        bars = ax1.bar(range(len(labels)), values, color=colors_bar, edgecolor="white", width=0.45)
        for bar, val in zip(bars, values):
            display = f"{val:.2f}%" if "Error" in labels[bars.index(bar)] else str(int(val))
            # safer indexing
            ax1.annotate(
                f"{val}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_ylabel("Value")
        ax1.set_title(f"C3 — Error Rate & Queue Depth at N={n}")

        # If all zeros, add note
        if all(v == 0 for v in values):
            ax1.annotate(
                "All values are 0 — no errors, no queue saturation",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                fontsize=11,
                color="#666666",
                style="italic",
            )
            ax1.set_ylim(0, 1)
    else:
        # Dual-axis chart
        x = ns
        width = 0.35 if len(ns) > 1 else 0.6

        bars = ax1.bar(x, error_rates, width=width, color=COLORS[3], alpha=0.8, label="Error Rate (%)")
        ax1.set_xlabel("Concurrency Level (N)")
        ax1.set_ylabel("Error Rate (%)", color=COLORS[3])
        ax1.tick_params(axis="y", labelcolor=COLORS[3])
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        ax2 = ax1.twinx()
        ax2.plot(x, max_depths, "s-", color=COLORS[0], linewidth=2, markersize=7, label="Max Queue Depth")
        ax2.set_ylabel("Max Queue Depth", color=COLORS[0])
        ax2.tick_params(axis="y", labelcolor=COLORS[0])

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

        ax1.set_title("C3 — Error Rate & Queue Saturation vs Concurrency")

        # Handle all-zero case
        if all(v == 0 for v in error_rates) and all(v == 0 for v in max_depths):
            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_c3_error_queue.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot C4 — Throughput vs Latency Pareto Frontier
# ---------------------------------------------------------------------------

def plot_c4(runs: list[dict], seq: dict | None, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))

    single = len(runs) == 1

    for i, run in enumerate(runs):
        n = _parse_concurrency(run["mode"])
        ttft_p99 = run["client_side"]["ttft"]["p99"]
        tput = _compute_throughput(run)
        color = COLORS[i % len(COLORS)]
        ax.scatter(
            ttft_p99,
            tput,
            s=120,
            color=color,
            edgecolors="white",
            linewidth=1.5,
            zorder=5,
            label=f"N={n}",
        )
        offset_x = 8
        offset_y = 8
        ax.annotate(
            f"N={n}\n{tput:.1f} tok/s\nTTFT p99={ttft_p99*1000:.0f} ms",
            (ttft_p99, tput),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8),
        )

    # Sequential baseline point
    if seq:
        seq_ttft_p99 = seq["client_side"]["ttft"]["p99"]
        seq_tput = _compute_throughput(seq)
        ax.scatter(
            seq_ttft_p99,
            seq_tput,
            s=120,
            color=SEQ_COLOR,
            marker="D",
            edgecolors="white",
            linewidth=1.5,
            zorder=5,
            label="Sequential (N=1)",
        )
        ax.annotate(
            f"Seq\n{seq_tput:.1f} tok/s\np99={seq_ttft_p99*1000:.0f} ms",
            (seq_ttft_p99, seq_tput),
            textcoords="offset points",
            xytext=(-60, -20),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8),
        )

    # Draw Pareto frontier line if multiple points
    all_points = []
    for run in runs:
        all_points.append(
            (run["client_side"]["ttft"]["p99"], _compute_throughput(run))
        )
    if seq:
        all_points.append(
            (seq["client_side"]["ttft"]["p99"], _compute_throughput(seq))
        )

    if len(all_points) >= 2:
        # Sort by x (ttft ascending), draw frontier through non-dominated points
        all_points.sort(key=lambda p: p[0])
        frontier_x = [all_points[0][0]]
        frontier_y = [all_points[0][1]]
        max_tput = all_points[0][1]
        for x, y in all_points[1:]:
            if y >= max_tput:
                frontier_x.append(x)
                frontier_y.append(y)
                max_tput = y
        if len(frontier_x) >= 2:
            ax.plot(frontier_x, frontier_y, "--", color="#aaaaaa", linewidth=1, alpha=0.7)

    ax.set_xlabel("TTFT p99 (s) — lower is better")
    ax.set_ylabel("Output Tokens / s — higher is better")
    ax.set_title("C4 — Throughput vs Latency Pareto Frontier")
    ax.legend(fontsize=9)

    # Add "better" direction arrow
    ax.annotate(
        "",
        xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
        xytext=(ax.get_xlim()[1], ax.get_ylim()[0]),
        arrowprops=dict(arrowstyle="->", color="#cccccc", lw=1.5),
    )
    ax.annotate(
        "better",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=9,
        color="#999999",
        style="italic",
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "plot_c4_pareto.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_input = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "test_results.json",
    )
    default_output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "resources",
        "concurrent",
    )

    parser = argparse.ArgumentParser(
        description="Generate concurrency benchmark plots from test_results.json"
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help="Path to test_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help="Directory for output PNG files",
    )
    args = parser.parse_args()

    # Load data
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    runs = _collect_concurrent_runs(data)
    if not runs:
        print("ERROR: No concurrent run data found in the JSON.", file=sys.stderr)
        sys.exit(1)

    seq = _get_sequential(data)

    print(f"Found {len(runs)} concurrent run(s): "
          f"{[r.get('mode','?') for r in runs]}")
    if seq:
        print("Sequential baseline found.")

    saved: list[str] = []
    saved.append(plot_c1(runs, seq, output_dir))
    saved.append(plot_c2(runs, seq, output_dir))
    saved.append(plot_c3(runs, seq, output_dir))
    saved.append(plot_c4(runs, seq, output_dir))

    print(f"\nSaved {len(saved)} plots:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
