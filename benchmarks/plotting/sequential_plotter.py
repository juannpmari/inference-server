#!/usr/bin/env python3
"""Unified plotter for sequential (single-request) experiments.

Consolidates ttft_vs_input_length, decode_time_vs_output_length,
latency_composition, and prefix_caching_ttft into a single class with
one method per plot type.

Usage:
    python -m benchmarks.plotting.sequential_plotter ttft-input
    python -m benchmarks.plotting.sequential_plotter decode --stat p90
    python -m benchmarks.plotting.sequential_plotter latency --stat p50
    python -m benchmarks.plotting.sequential_plotter prefix-cache --no-scatter
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

VALID_STATS = ("mean", "p50", "p90", "p99")

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


_apply_style()


class SequentialPlotter:
    """Plots that visualise metrics from sequential (single-request) experiments.

    Each public ``plot_*`` method accepts the loaded experiment dict, an
    output directory, and keyword arguments for filtering/stats.  All methods
    return the path to the saved PNG.

    To add a new plot type:
      1. Add a ``plot_<name>`` method following the same signature convention.
      2. Register it in ``PLOT_REGISTRY`` with its experiment folder name.
      3. Add the YAML name mapping in ``YAML_PLOT_MAP``.
    """

    # Maps CLI subcommand name -> (method_name, experiment_folder)
    PLOT_REGISTRY: dict[str, tuple[str, str]] = {
        "ttft-input":    ("plot_ttft_vs_input_length",        "input_length_sweep"),
        "decode":        ("plot_decode_time_vs_output_length", "decode_time_vs_output_length"),
        "latency":       ("plot_latency_composition",          "latency_composition"),
        "prefix-cache":  ("plot_prefix_caching_ttft",          "input_length_sweep"),
    }

    # Maps YAML ``plots:`` key -> method name (used by the orchestrator)
    YAML_PLOT_MAP: dict[str, str] = {
        "ttft_vs_input_length":        "plot_ttft_vs_input_length",
        "decode_time_vs_output_length": "plot_decode_time_vs_output_length",
        "latency_composition":          "plot_latency_composition",
        "prefix_caching_ttft":          "plot_prefix_caching_ttft",
    }

    # ------------------------------------------------------------------
    # Plot: TTFT vs input length
    # ------------------------------------------------------------------

    TTFT_INPUT_COLORS = {
        "line": "#4C72B0",
        "scatter": "#55A868",
        "fit": "#C44E52",
    }

    def plot_ttft_vs_input_length(
        self,
        data: dict,
        out_dir: str,
        *,
        stat: str = "mean",
        scatter: bool = True,
    ) -> str:
        conditions = data["conditions"]

        input_tokens: list[float] = []
        ttft_vals: list[float] = []
        ttft_per_request: list[list[float]] = []

        # If the data contains both cache/nocache groups (input_length_sweep),
        # only plot the nocache conditions for the single-series view.
        has_groups = any(k.startswith("nocache_") for k in conditions) and any(
            k.startswith("cache_") for k in conditions
        )

        for cond_name, cond in conditions.items():
            if has_groups and not cond_name.startswith("nocache_"):
                continue
            per_req = cond["per_request"]
            ok = [r for r in per_req if r["error"] is None and r["http_status"] == 200]
            if not ok:
                continue
            avg_input = np.mean([r["input_tokens"] for r in ok])
            input_tokens.append(avg_input)
            ttft_vals.append(cond["stats"]["ttft"][stat])
            ttft_per_request.append([r["ttft"] for r in ok])

        order = np.argsort(input_tokens)
        input_tokens = np.array(input_tokens)[order]
        ttft_vals = np.array(ttft_vals)[order]
        ttft_per_request = [ttft_per_request[i] for i in order]

        fig, ax = plt.subplots(figsize=(9, 6))

        if scatter:
            for tok, reqs in zip(input_tokens, ttft_per_request):
                xs = np.full(len(reqs), tok)
                jitter = np.random.default_rng(42).uniform(-1.5, 1.5, size=len(reqs))
                ax.scatter(xs + jitter, np.array(reqs) * 1000, s=25, alpha=0.35,
                           color=self.TTFT_INPUT_COLORS["scatter"], edgecolors="none",
                           zorder=2)

        ax.plot(input_tokens, ttft_vals * 1000, "o-",
                color=self.TTFT_INPUT_COLORS["line"], lw=2, markersize=7,
                zorder=3, label=f"TTFT ({stat.upper()})")

        if len(input_tokens) >= 2:
            coeffs = np.polyfit(input_tokens, ttft_vals * 1000, 1)
            x_fit = np.linspace(input_tokens.min(), input_tokens.max(), 200)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), "--",
                    color=self.TTFT_INPUT_COLORS["fit"], lw=1.5,
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

    # ------------------------------------------------------------------
    # Plot: decode time vs output length
    # ------------------------------------------------------------------

    DECODE_COLORS = {
        "line": "#4C72B0",
        "scatter": "#55A868",
        "fit": "#C44E52",
    }

    def plot_decode_time_vs_output_length(
        self,
        data: dict,
        out_dir: str,
        *,
        stat: str = "mean",
        scatter: bool = True,
    ) -> str:
        conditions = data["conditions"]

        output_tokens: list[float] = []
        decode_vals: list[float] = []
        decode_per_request: list[list[float]] = []

        for cond in conditions.values():
            per_req = cond["per_request"]
            ok = [r for r in per_req if r["error"] is None and r["http_status"] == 200]
            if not ok:
                continue
            avg_output = np.mean([r["output_tokens"] for r in ok])
            output_tokens.append(avg_output)
            decode_vals.append(cond["stats"]["decode_time"][stat])
            decode_per_request.append([r["decode_time"] for r in ok])

        order = np.argsort(output_tokens)
        output_tokens = np.array(output_tokens)[order]
        decode_vals = np.array(decode_vals)[order]
        decode_per_request = [decode_per_request[i] for i in order]

        fig, ax = plt.subplots(figsize=(9, 6))

        if scatter:
            for tok, reqs in zip(output_tokens, decode_per_request):
                xs = np.full(len(reqs), tok)
                jitter = np.random.default_rng(42).uniform(-1.5, 1.5, size=len(reqs))
                ax.scatter(xs + jitter, np.array(reqs) * 1000, s=25, alpha=0.35,
                           color=self.DECODE_COLORS["scatter"], edgecolors="none",
                           zorder=2)

        ax.plot(output_tokens, decode_vals * 1000, "o-",
                color=self.DECODE_COLORS["line"], lw=2, markersize=7,
                zorder=3, label=f"Decode Time ({stat.upper()})")

        if len(output_tokens) >= 2:
            coeffs = np.polyfit(output_tokens, decode_vals * 1000, 1)
            x_fit = np.linspace(output_tokens.min(), output_tokens.max(), 200)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), "--",
                    color=self.DECODE_COLORS["fit"], lw=1.5,
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

    # ------------------------------------------------------------------
    # Plot: latency composition (stacked bar)
    # ------------------------------------------------------------------

    LATENCY_COLORS = {
        "prefill": "#4C72B0",
        "decode": "#DD8452",
    }

    CONDITION_LABELS = {
        "siso": "SISO\n(32, 32)",
        "silo": "SILO\n(32, 256)",
        "liso": "LISO\n(256, 32)",
        "lilo": "LILO\n(256, 256)",
    }

    def plot_latency_composition(
        self,
        data: dict,
        out_dir: str,
        *,
        stat: str = "mean",
    ) -> str:
        conditions = data["conditions"]

        labels: list[str] = []
        prefill_vals: list[float] = []
        decode_vals: list[float] = []

        for cond_name, cond in conditions.items():
            stats = cond["stats"]
            labels.append(self.CONDITION_LABELS.get(cond_name, cond_name))
            prefill_vals.append(stats["ttft"][stat])
            decode_vals.append(stats["decode_time"][stat])

        x = np.arange(len(labels))
        width = 0.5

        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2), 6))
        ax.bar(x, prefill_vals, width, label="Prefill (TTFT)",
               color=self.LATENCY_COLORS["prefill"])
        ax.bar(x, decode_vals, width, bottom=prefill_vals, label="Decode",
               color=self.LATENCY_COLORS["decode"])

        ax.set_xlabel("Condition")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Latency Decomposition: Prefill vs Decode ({stat.upper()})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend()

        for i in range(len(labels)):
            total = prefill_vals[i] + decode_vals[i]
            ax.text(x[i], prefill_vals[i] / 2, f"{prefill_vals[i]*1000:.1f}ms",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")
            ax.text(x[i], prefill_vals[i] + decode_vals[i] / 2,
                    f"{decode_vals[i]*1000:.1f}ms",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")
            ax.annotate(f"{total*1000:.1f}ms", xy=(x[i], total),
                        ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        fpath = os.path.join(out_dir, f"latency_composition_{stat}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fpath

    # ------------------------------------------------------------------
    # Plot: prefix caching TTFT comparison
    # ------------------------------------------------------------------

    PREFIX_CACHE_COLORS = {
        "cache_off_line": "#C44E52",
        "cache_off_scatter": "#E8A0A0",
        "cache_on_line": "#4C72B0",
        "cache_on_scatter": "#A0C0E0",
    }

    def plot_prefix_caching_ttft(
        self,
        data: dict,
        out_dir: str,
        *,
        stat: str = "mean",
        scatter: bool = True,
    ) -> str:
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

            if scatter:
                for tok, reqs in zip(tokens, per_req):
                    xs = np.full(len(reqs), tok) + jitter_offset
                    jitter = np.random.default_rng(42).uniform(-1, 1, size=len(reqs))
                    ax.scatter(xs + jitter, np.array(reqs) * 1000, s=20, alpha=0.3,
                               color=self.PREFIX_CACHE_COLORS[f"{color_prefix}_scatter"],
                               edgecolors="none", zorder=2)

            ax.plot(tokens, np.array(vals) * 1000, "o-",
                    color=self.PREFIX_CACHE_COLORS[f"{color_prefix}_line"],
                    lw=2, markersize=6, zorder=3,
                    label=f"{label} ({stat.upper()})")

        ax.set_xlabel("Input Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(f"Prefix Caching Impact on TTFT ({stat.upper()})")
        ax.legend()

        plt.tight_layout()
        fpath = os.path.join(out_dir, f"prefix_caching_ttft_{stat}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fpath

    # ------------------------------------------------------------------
    # Orchestrator integration
    # ------------------------------------------------------------------

    @classmethod
    def generate_plots(
        cls,
        data: dict,
        out_dir: str,
        plot_configs: dict[str, dict],
    ) -> list[str]:
        """Generate all plots listed in the experiment YAML ``plots:`` section.

        Args:
            data: Full experiment results dict (as written to results JSON).
            out_dir: Directory where PNGs will be saved.
            plot_configs: The ``plots`` mapping from the experiment YAML.
                          Keys are YAML plot names, values may contain
                          per-plot overrides (``stat``, ``scatter``).

        Returns:
            List of saved file paths.
        """
        saved: list[str] = []
        plotter = cls()

        for yaml_name, plot_cfg in plot_configs.items():
            method_name = cls.YAML_PLOT_MAP.get(yaml_name)
            if method_name is None:
                import logging
                logging.getLogger(__name__).warning(
                    "Unknown plot type '%s' in YAML — skipping", yaml_name,
                )
                continue

            plot_cfg = plot_cfg or {}
            kwargs: dict = {}

            if "stat" in plot_cfg:
                kwargs["stat"] = plot_cfg["stat"]

            if "scatter" in plot_cfg:
                kwargs["scatter"] = plot_cfg["scatter"]

            method = getattr(plotter, method_name)
            fpath = method(data, out_dir, **kwargs)
            saved.append(fpath)

        return saved

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @classmethod
    def cli(cls) -> None:
        shared = argparse.ArgumentParser(add_help=False)
        shared.add_argument("--input", default=None, help="Path to experiment results JSON")
        shared.add_argument("--output-dir", default=None, help="Directory for output PNGs")
        shared.add_argument("--dispatch-mode", default="sequential",
                            help="Dispatch mode subfolder (default: %(default)s)")
        shared.add_argument("--stat", default="mean", choices=VALID_STATS,
                            help="Latency statistic to use (default: mean)")

        parser = argparse.ArgumentParser(
            description="Sequential experiment plotter",
        )
        sub = parser.add_subparsers(dest="plot", required=True)

        ttft_input = sub.add_parser("ttft-input", parents=[shared],
                                    help="TTFT vs input token length")
        ttft_input.add_argument("--no-scatter", action="store_true",
                                help="Hide individual request scatter points")

        decode = sub.add_parser("decode", parents=[shared],
                                help="Decode time vs output token length")
        decode.add_argument("--no-scatter", action="store_true",
                            help="Hide individual request scatter points")

        sub.add_parser("latency", parents=[shared],
                       help="Latency composition stacked barchart")

        prefix = sub.add_parser("prefix-cache", parents=[shared],
                                help="Prefix caching TTFT comparison")
        prefix.add_argument("--no-scatter", action="store_true",
                            help="Hide individual request scatter points")

        args = parser.parse_args()

        method_name, experiment_folder = cls.PLOT_REGISTRY[args.plot]
        results_dir = (Path(__file__).resolve().parent.parent
                       / "results" / args.dispatch_mode / experiment_folder)
        input_path = args.input or str(results_dir / f"{experiment_folder}.json")
        output_dir = args.output_dir or str(results_dir)

        with open(input_path) as f:
            data = json.load(f)

        os.makedirs(output_dir, exist_ok=True)

        plotter = cls()
        method = getattr(plotter, method_name)

        kwargs: dict = {"stat": args.stat}
        if args.plot in ("ttft-input", "decode", "prefix-cache"):
            kwargs["scatter"] = not getattr(args, "no_scatter", False)

        fpath = method(data, output_dir, **kwargs)
        print(f"Saved: {fpath}")


if __name__ == "__main__":
    SequentialPlotter.cli()
