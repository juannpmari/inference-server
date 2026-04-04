#!/usr/bin/env python3
"""Unified plotter for realistic (arrival-rate sweep) experiments.

Consolidates throughput_vs_arrival_rate, queue_depth_vs_arrival_rate, and
ttft_vs_arrival_rate into a single class with one method per plot type.

Usage:
    python -m benchmarks.plotting.realistic_plotter throughput
    python -m benchmarks.plotting.realistic_plotter queue-depth
    python -m benchmarks.plotting.realistic_plotter ttft --stat p90
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
import seaborn as sns

VALID_STATS = ("mean", "p50", "p90", "p99")

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", context="talk", palette="muted", font_scale=0.9)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
})


class RealisticPlotter:
    """Plots that visualise metrics across arrival-rate levels.

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
        "throughput":  ("plot_throughput_vs_arrival_rate", "arrival_rate_sweep"),
        "queue-depth": ("plot_queue_depth_vs_arrival_rate", "arrival_rate_sweep"),
        "ttft":        ("plot_ttft_vs_arrival_rate", "arrival_rate_sweep"),
    }

    # Maps YAML ``plots:`` key -> method name (used by the orchestrator)
    YAML_PLOT_MAP: dict[str, str] = {
        "throughput_vs_arrival_rate":  "plot_throughput_vs_arrival_rate",
        "queue_depth_vs_arrival_rate": "plot_queue_depth_vs_arrival_rate",
        "ttft_vs_arrival_rate":        "plot_ttft_vs_arrival_rate",
    }

    # ------------------------------------------------------------------
    # Common data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_grouped_series(
        data: dict,
        cache_filter: str | None = None,
    ) -> dict[str, list[tuple[float, dict]]]:
        """Split conditions into cache_disabled / cache_enabled groups.

        Returns a dict with two keys, each mapping to a list of
        ``(rps, cond_data)`` tuples sorted by RPS.  Conditions whose name
        starts with ``nocache_`` are placed in ``cache_disabled``; those
        starting with ``cache_`` go into ``cache_enabled``.

        Args:
            cache_filter: If ``"cache"``, only return cache_enabled series.
                          If ``"no_cache"``, only return cache_disabled series.
                          If *None*, return both.
        """
        groups: dict[str, list[tuple[float, dict]]] = {
            "cache_disabled": [],
            "cache_enabled": [],
        }

        for cond_name, cond_data in data["conditions"].items():
            rps = cond_data["dispatch"]["rps"]
            if cond_name.startswith("nocache_"):
                groups["cache_disabled"].append((rps, cond_data))
            elif cond_name.startswith("cache_"):
                groups["cache_enabled"].append((rps, cond_data))
            else:
                # Default to cache_disabled if prefix is ambiguous
                groups["cache_disabled"].append((rps, cond_data))

        for key in groups:
            groups[key].sort(key=lambda pair: pair[0])

        if cache_filter == "cache":
            groups["cache_disabled"] = []
        elif cache_filter == "no_cache":
            groups["cache_enabled"] = []

        return groups

    # ------------------------------------------------------------------
    # Plot: throughput vs arrival rate
    # ------------------------------------------------------------------

    def plot_throughput_vs_arrival_rate(
        self,
        data: dict,
        out_dir: str,
        *,
        source: str = "client",
        cache_filter: str | None = None,
    ) -> str:
        groups = self._extract_grouped_series(data, cache_filter=cache_filter)

        fig, ax = plt.subplots(figsize=(9, 6))

        palette = sns.color_palette("muted")
        series_styles = {
            "cache_disabled": {"color": palette[3], "label": "Cache disabled"},
            "cache_enabled":  {"color": palette[0], "label": "Cache enabled"},
        }

        all_rps: list[float] = []
        all_tput: list[float] = []

        for group_key, pairs in groups.items():
            if not pairs:
                continue
            style = series_styles[group_key]
            rps_vals = [rps for rps, _ in pairs]
            tput_vals = []
            for _, cond in pairs:
                if source == "server":
                    tput = (cond.get("server_side_delta", {})
                                .get("throughput", {})
                                .get("output_tokens_per_second", 0.0))
                else:
                    tput = cond["client_side_throughput"]["output_tokens_per_second"]
                tput_vals.append(tput)

            ax.plot(rps_vals, tput_vals, "o-", color=style["color"], lw=2.5,
                    markersize=8, zorder=3, label=style["label"],
                    markeredgecolor="white", markeredgewidth=1.2)

            for x, y in zip(rps_vals, tput_vals):
                ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=8,
                            color=style["color"], fontweight="bold")

            all_rps.extend(rps_vals)
            all_tput.extend(tput_vals)

        # Dashed diagonal reference line (ideal linear scaling)
        if all_rps:
            rps_range = np.linspace(min(all_rps), max(all_rps), 50)
            min_rps = min(all_rps)
            min_tput = min(all_tput) if all_tput else 1.0
            ideal = rps_range * (min_tput / min_rps) if min_rps > 0 else rps_range
            ax.plot(rps_range, ideal, "--", color="#aaaaaa", lw=1.5, alpha=0.7,
                    label="Ideal linear scaling", zorder=1)

        ax.set_xlabel("Arrival Rate (RPS)")
        ax.set_ylabel("Output Tokens/s")
        ax.set_title(f"Output Token Throughput vs Arrival Rate ({source})",
                      fontweight="bold", pad=15)
        ax.legend(loc="best", framealpha=0.9, edgecolor="#cccccc")
        sns.despine(ax=ax, left=True, bottom=True)

        plt.tight_layout()
        fpath = os.path.join(out_dir, f"throughput_vs_arrival_rate_{source}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fpath

    # ------------------------------------------------------------------
    # Plot: queue depth vs arrival rate
    # ------------------------------------------------------------------

    def plot_queue_depth_vs_arrival_rate(
        self,
        data: dict,
        out_dir: str,
        *,
        cache_filter: str | None = None,
    ) -> str:
        groups = self._extract_grouped_series(data, cache_filter=cache_filter)

        fig, ax = plt.subplots(figsize=(9, 6))

        palette = sns.color_palette("muted")
        series_styles = {
            "cache_disabled": {"color": palette[3], "label": "Cache disabled"},
            "cache_enabled":  {"color": palette[0], "label": "Cache enabled"},
        }

        for group_key, pairs in groups.items():
            if not pairs:
                continue
            style = series_styles[group_key]
            rps_vals = [rps for rps, _ in pairs]
            depth_vals = []
            for _, cond in pairs:
                depth = (cond.get("server_side_delta", {})
                             .get("queue", {})
                             .get("max_depth", 0))
                depth_vals.append(depth)

            ax.plot(rps_vals, depth_vals, "o-", color=style["color"], lw=2.5,
                    markersize=8, zorder=3, label=style["label"],
                    markeredgecolor="white", markeredgewidth=1.2)

            for x, y in zip(rps_vals, depth_vals):
                ax.annotate(f"{y}", (x, y), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=8,
                            color=style["color"], fontweight="bold")

        ax.set_xlabel("Arrival Rate (RPS)")
        ax.set_ylabel("Max Queue Depth")
        ax.set_title("Max Queue Depth vs Arrival Rate",
                      fontweight="bold", pad=15)
        ax.legend(loc="best", framealpha=0.9, edgecolor="#cccccc")
        sns.despine(ax=ax, left=True, bottom=True)

        plt.tight_layout()
        fpath = os.path.join(out_dir, "queue_depth_vs_arrival_rate.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fpath

    # ------------------------------------------------------------------
    # Plot: TTFT vs arrival rate (percentile breakdown)
    # ------------------------------------------------------------------

    # 6-line style matrix: group x percentile
    @staticmethod
    def _ttft_styles():
        palette = sns.color_palette("muted")
        red, blue = palette[3], palette[0]
        red_light = sns.light_palette(red, n_colors=4)
        blue_light = sns.light_palette(blue, n_colors=4)
        return {
            "cache_disabled": {
                "p50": {"color": red,           "linestyle": "-",  "label": "No cache p50"},
                "p90": {"color": red_light[2],  "linestyle": "--", "label": "No cache p90"},
                "p99": {"color": red_light[1],  "linestyle": ":",  "label": "No cache p99"},
            },
            "cache_enabled": {
                "p50": {"color": blue,           "linestyle": "-",  "label": "Cache p50"},
                "p90": {"color": blue_light[2],  "linestyle": "--", "label": "Cache p90"},
                "p99": {"color": blue_light[1],  "linestyle": ":",  "label": "Cache p99"},
            },
        }

    def plot_ttft_vs_arrival_rate(
        self,
        data: dict,
        out_dir: str,
        *,
        stat: str | None = None,
        cache_filter: str | None = None,
    ) -> str:
        groups = self._extract_grouped_series(data, cache_filter=cache_filter)

        percentiles = [stat] if stat else ["p50", "p90", "p99"]

        fig, ax = plt.subplots(figsize=(9, 6))

        ttft_styles = self._ttft_styles()

        for group_key, pairs in groups.items():
            if not pairs:
                continue
            rps_vals = [rps for rps, _ in pairs]

            for pct in percentiles:
                style = ttft_styles[group_key][pct]
                ttft_vals = []
                saturated_mask = []

                for _, cond in pairs:
                    ttft_vals.append(cond["stats"]["ttft"][pct] * 1000)
                    saturated_mask.append(cond.get("saturation_detected", False))

                ax.plot(rps_vals, ttft_vals, marker="o", color=style["color"],
                        linestyle=style["linestyle"], lw=2.5, markersize=7,
                        zorder=3, label=style["label"],
                        markeredgecolor="white", markeredgewidth=1.0)

                # Mark saturated points with a star
                for i, (x, y, sat) in enumerate(
                    zip(rps_vals, ttft_vals, saturated_mask)
                ):
                    if sat:
                        ax.plot(x, y, marker="*", color=style["color"],
                                markersize=14, zorder=4)

        ax.set_xlabel("Arrival Rate (RPS)")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("TTFT vs Arrival Rate", fontweight="bold", pad=15)
        ax.legend(loc="best", fontsize=8, framealpha=0.9, edgecolor="#cccccc")
        sns.despine(ax=ax, left=True, bottom=True)

        plt.tight_layout()
        fpath = os.path.join(out_dir, "ttft_vs_arrival_rate.png")
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
                          per-plot overrides (``stat``, ``source``).

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

            if "source" in plot_cfg:
                kwargs["source"] = plot_cfg["source"]

            if "cache_filter" in plot_cfg:
                kwargs["cache_filter"] = plot_cfg["cache_filter"]

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
        shared.add_argument("--input", default=None,
                            help="Path to experiment results JSON")
        shared.add_argument("--output-dir", default=None,
                            help="Directory for output PNGs")
        shared.add_argument("--dispatch-mode", default="realistic",
                            help="Dispatch mode subfolder (default: %(default)s)")
        shared.add_argument("--cache-filter", default=None,
                            choices=("cache", "no_cache"),
                            help="Plot only cache-enabled or cache-disabled series")

        parser = argparse.ArgumentParser(
            description="Realistic (arrival-rate sweep) experiment plotter",
        )
        sub = parser.add_subparsers(dest="plot", required=True)

        throughput_parser = sub.add_parser(
            "throughput", parents=[shared],
            help="Output token throughput vs arrival rate",
        )
        throughput_parser.add_argument(
            "--source", default="client", choices=("client", "server"),
            help="Throughput source: client-side or server-side (default: client)",
        )

        sub.add_parser(
            "queue-depth", parents=[shared],
            help="Max queue depth vs arrival rate",
        )

        ttft_parser = sub.add_parser(
            "ttft", parents=[shared],
            help="TTFT percentiles vs arrival rate",
        )
        ttft_parser.add_argument(
            "--stat", default=None, choices=VALID_STATS,
            help="Single TTFT percentile to plot (default: all of p50/p90/p99)",
        )

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

        kwargs: dict = {}
        if args.plot == "throughput":
            kwargs["source"] = args.source
        elif args.plot == "ttft":
            kwargs["stat"] = getattr(args, "stat", None)
        if args.cache_filter:
            kwargs["cache_filter"] = args.cache_filter

        fpath = method(data, output_dir, **kwargs)
        print(f"Saved: {fpath}")


if __name__ == "__main__":
    RealisticPlotter.cli()
