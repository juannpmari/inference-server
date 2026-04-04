#!/usr/bin/env python3
"""Unified plotter for concurrency-sweep experiments.

Consolidates throughput_vs_concurrency, throughput_latency_pareto, and
ttft_vs_concurrency into a single class with one method per plot type.

Usage:
    python -m benchmarks.plotting.concurrency_plotter throughput
    python -m benchmarks.plotting.concurrency_plotter pareto --stat p90
    python -m benchmarks.plotting.concurrency_plotter ttft --no-scatter --skip-x 2048,4096
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


class ConcurrencyPlotter:
    """Plots that visualise metrics across concurrency levels.

    Each public ``plot_*`` method accepts the loaded experiment dict, an
    output directory, and keyword arguments for filtering/stats.  All methods
    return the path to the saved PNG.

    To add a new plot type:
      1. Add a ``plot_<name>`` method following the same signature convention.
      2. Register it in ``PLOT_REGISTRY`` with its experiment folder name.
      3. Add the YAML name mapping in ``YAML_PLOT_MAP``.
      4. Override ``_add_extra_args`` if the plot needs CLI flags beyond the
         common ones.
    """

    # Maps CLI subcommand name -> (method_name, experiment_folder)
    PLOT_REGISTRY: dict[str, tuple[str, str]] = {
        "throughput": ("plot_throughput_vs_concurrency", "concurrency_sweep"),
        "pareto":     ("plot_throughput_latency_pareto", "concurrency_sweep"),
        "ttft":       ("plot_ttft_vs_concurrency", "concurrency_sweep"),
    }

    # Maps YAML ``plots:`` key -> method name (used by the orchestrator)
    YAML_PLOT_MAP: dict[str, str] = {
        "throughput_vs_concurrency": "plot_throughput_vs_concurrency",
        "throughput_vs_latency":    "plot_throughput_latency_pareto",
        "ttft_vs_concurrency":      "plot_ttft_vs_concurrency",
    }

    # ------------------------------------------------------------------
    # Common data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_throughput(cond: dict, source: str) -> float:
        if source == "server":
            return (cond.get("server_side_delta", {})
                        .get("throughput", {})
                        .get("output_tokens_per_second", 0.0))
        return cond["client_side_throughput"]["output_tokens_per_second"]

    @staticmethod
    def _sorted_by_concurrency(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        order = np.argsort(arrays[0])
        return tuple(np.asarray(a)[order] for a in arrays)

    # ------------------------------------------------------------------
    # Plot: throughput vs concurrency
    # ------------------------------------------------------------------

    THROUGHPUT_COLORS = {
        "client_line": "#4C72B0",
        "server_line": "#C44E52",
    }

    def plot_throughput_vs_concurrency(
        self,
        data: dict,
        out_dir: str,
        *,
        source: str = "client",
        stat: str = "mean",
        skip_x: set[int] | None = None,
    ) -> str:
        conditions = data["conditions"]

        concurrency_levels: list[int] = []
        throughputs: list[float] = []
        latencies: list[float] = []

        for cond in conditions.values():
            concurrency = cond["dispatch"]["concurrency"]
            if skip_x and concurrency in skip_x:
                continue
            concurrency_levels.append(concurrency)
            throughputs.append(self._get_throughput(cond, source))
            latencies.append(cond["stats"]["e2e"][stat] * 1000)

        concurrency_levels, throughputs, latencies = self._sorted_by_concurrency(
            concurrency_levels, throughputs, latencies,
        )

        color = self.THROUGHPUT_COLORS.get(f"{source}_line", "#4C72B0")

        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.plot(concurrency_levels, throughputs, "o-", color=color, lw=2,
                 markersize=7, zorder=3, label=f"Output throughput ({source})")
        for x, y in zip(concurrency_levels, throughputs):
            ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, color=color)

        ax1.set_xscale("log", base=2)
        ax1.set_xlabel("Concurrency Level")
        ax1.set_ylabel("Output Tokens/s", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xticks(concurrency_levels)
        ax1.set_xticklabels([str(int(c)) for c in concurrency_levels])

        lat_color = self.THROUGHPUT_COLORS.get(
            "server_line" if source == "client" else "client_line", "#C44E52")
        ax2 = ax1.twinx()
        ax2.plot(concurrency_levels, latencies, "s--", color=lat_color, lw=1.5,
                 markersize=5, zorder=3, alpha=0.8,
                 label=f"E2E latency {stat} ({source})")
        for x, y in zip(concurrency_levels, latencies):
            ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, -12), ha="center", fontsize=8,
                         color=lat_color, alpha=0.8)

        ax2.set_ylabel(f"E2E Latency \u2014 {stat} (ms)", color=lat_color)
        ax2.tick_params(axis="y", labelcolor=lat_color)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.set_title(f"Output Token Throughput vs Concurrency Level ({stat})")

        plt.tight_layout()
        fpath = os.path.join(out_dir, f"throughput_vs_concurrency_{source}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fpath

    # ------------------------------------------------------------------
    # Plot: throughput-latency Pareto frontier
    # ------------------------------------------------------------------

    PARETO_COLORS = {
        "scatter": "#A8B6CC",
        "pareto_line": "#C44E52",
        "pareto_marker": "#C44E52",
        "label": "#2D3436",
    }

    @staticmethod
    def _pareto_frontier(latencies: np.ndarray, throughputs: np.ndarray) -> np.ndarray:
        order = np.argsort(latencies)
        pareto_indices: list[int] = []
        max_throughput = -np.inf
        for idx in order:
            if throughputs[idx] > max_throughput:
                pareto_indices.append(idx)
                max_throughput = throughputs[idx]
        return np.array(pareto_indices)

    def plot_throughput_latency_pareto(
        self,
        data: dict,
        out_dir: str,
        *,
        stat: str = "mean",
        source: str = "client",
        skip_x: set[int] | None = None,
    ) -> str:
        conditions = data["conditions"]

        concurrencies: list[int] = []
        latencies: list[float] = []
        throughputs: list[float] = []

        for cond in conditions.values():
            concurrency = cond["dispatch"]["concurrency"]
            if skip_x and concurrency in skip_x:
                continue
            concurrencies.append(concurrency)
            latencies.append(cond["stats"]["e2e"][stat] * 1000)
            throughputs.append(self._get_throughput(cond, source))

        concurrencies = np.array(concurrencies)
        latencies = np.array(latencies)
        throughputs = np.array(throughputs)

        pareto_idx = self._pareto_frontier(latencies, throughputs)
        pareto_idx = pareto_idx[np.argsort(latencies[pareto_idx])]

        fig, ax = plt.subplots(figsize=(9, 6))

        non_pareto_mask = np.ones(len(latencies), dtype=bool)
        non_pareto_mask[pareto_idx] = False
        ax.scatter(latencies[non_pareto_mask], throughputs[non_pareto_mask],
                   color=self.PARETO_COLORS["scatter"], s=80, zorder=2,
                   edgecolors="white", linewidths=0.5, label="Dominated")

        ax.plot(latencies[pareto_idx], throughputs[pareto_idx], "o-",
                color=self.PARETO_COLORS["pareto_line"], lw=2.5, markersize=9,
                zorder=4, markeredgecolor="white", markeredgewidth=1,
                label="Pareto frontier")

        for i in range(len(concurrencies)):
            is_pareto = i in pareto_idx
            ax.annotate(
                f"N={concurrencies[i]}",
                (latencies[i], throughputs[i]),
                textcoords="offset points", xytext=(8, 8), fontsize=9,
                fontweight="bold" if is_pareto else "normal",
                color=self.PARETO_COLORS["pareto_line"] if is_pareto
                      else self.PARETO_COLORS["label"],
                zorder=5,
            )

        ax.set_xlabel(f"End-to-End Latency \u2014 {stat} (ms)")
        ax.set_ylabel("Output Throughput (tok/s)")
        ax.set_title(f"Throughput vs Latency Pareto Frontier ({source})")
        ax.legend(loc="best")

        plt.tight_layout()
        fpath = os.path.join(out_dir, f"throughput_latency_pareto_{source}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fpath

    # ------------------------------------------------------------------
    # Plot: TTFT vs concurrency
    # ------------------------------------------------------------------

    TTFT_COLORS = {
        "mean":    "#4C72B0",
        "p50":     "#55A868",
        "p90":     "#DD8452",
        "p99":     "#C44E52",
        "scatter": "#AAAAAA",
    }

    TTFT_STAT_STYLES = [
        ("mean", "Mean",  "o-"),
        ("p50",  "P50",   "s-"),
        ("p90",  "P90",   "^-"),
        ("p99",  "P99",   "D-"),
    ]

    def plot_ttft_vs_concurrency(
        self,
        data: dict,
        out_dir: str,
        *,
        scatter: bool = True,
        stat: str | None = None,
        skip_x: set[int] | None = None,
    ) -> str:
        conditions = data["conditions"]

        active_stats = ([(s, l, m) for s, l, m in self.TTFT_STAT_STYLES if s == stat]
                        if stat else self.TTFT_STAT_STYLES)

        concurrency_levels: list[int] = []
        stat_values: dict[str, list[float]] = {key: [] for key, _, _ in active_stats}
        scatter_data: dict[int, list[float]] = {}

        for cond in conditions.values():
            conc = cond["dispatch"]["concurrency"]
            if skip_x and conc in skip_x:
                continue
            concurrency_levels.append(conc)
            for key, _, _ in active_stats:
                stat_values[key].append(cond["stats"]["ttft"][key])
            ok = [r for r in cond["per_request"]
                  if r["error"] is None and r["http_status"] == 200]
            scatter_data[conc] = [r["ttft"] for r in ok]

        concurrency_levels = np.array(concurrency_levels)
        order = np.argsort(concurrency_levels)
        concurrency_levels = concurrency_levels[order]
        for key in stat_values:
            stat_values[key] = np.array(stat_values[key])[order] * 1000

        fig, ax = plt.subplots(figsize=(9, 6))

        if scatter:
            rng = np.random.RandomState(42)
            for conc in concurrency_levels:
                ttfts_ms = np.array(scatter_data[conc]) * 1000
                jitter = rng.uniform(-0.5, 0.5, size=len(ttfts_ms))
                ax.scatter(conc + jitter, ttfts_ms, s=20, alpha=0.2,
                           color=self.TTFT_COLORS["scatter"], edgecolors="none",
                           zorder=1)

        for key, label, marker_style in active_stats:
            ax.plot(concurrency_levels, stat_values[key], marker_style,
                    color=self.TTFT_COLORS[key], lw=2, markersize=7, zorder=3,
                    label=f"TTFT {label}")

        annotate_key = active_stats[-1][0]
        for x, y in zip(concurrency_levels, stat_values[annotate_key]):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9,
                        color=self.TTFT_COLORS[annotate_key])

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrency Level")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("TTFT vs Concurrency Level")
        ax.set_xticks(concurrency_levels)
        ax.set_xticklabels([str(int(c)) for c in concurrency_levels])
        ax.legend()

        plt.tight_layout()
        fpath = os.path.join(out_dir, "ttft_vs_concurrency.png")
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
                          per-plot overrides (``stat``, ``source``, ``skip_x``,
                          ``scatter``).

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

            skip_raw = plot_cfg.get("skip_x")
            if skip_raw:
                kwargs["skip_x"] = set(int(v) for v in str(skip_raw).split(","))

            if "stat" in plot_cfg:
                kwargs["stat"] = plot_cfg["stat"]

            if "source" in plot_cfg:
                kwargs["source"] = plot_cfg["source"]

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
        shared.add_argument("--dispatch-mode", default="concurrent",
                            help="Dispatch mode subfolder (default: %(default)s)")
        shared.add_argument("--stat", default=None, choices=VALID_STATS,
                            help="Latency statistic to use (default depends on plot type)")
        shared.add_argument("--source", default="client", choices=("client", "server"),
                            help="Throughput source: client-side or server-side (default: client)")
        shared.add_argument("--skip-x", default=None,
                            help="Comma-separated concurrency values to exclude (e.g. 2048,4096)")

        parser = argparse.ArgumentParser(
            description="Concurrency-sweep experiment plotter",
        )
        sub = parser.add_subparsers(dest="plot", required=True)

        sub.add_parser("throughput", parents=[shared],
                       help="Output token throughput vs concurrency")
        sub.add_parser("pareto", parents=[shared],
                       help="Throughput-latency Pareto frontier")

        ttft_parser = sub.add_parser("ttft", parents=[shared],
                                     help="TTFT percentiles vs concurrency")
        ttft_parser.add_argument("--no-scatter", action="store_true",
                                 help="Hide individual request TTFT scatter points")

        args = parser.parse_args()

        method_name, experiment_folder = cls.PLOT_REGISTRY[args.plot]
        results_dir = (Path(__file__).resolve().parent.parent
                       / "results" / args.dispatch_mode / experiment_folder)
        input_path = args.input or str(results_dir / f"{experiment_folder}.json")
        output_dir = args.output_dir or str(results_dir)

        with open(input_path) as f:
            data = json.load(f)

        os.makedirs(output_dir, exist_ok=True)
        skip_x = set(int(v) for v in args.skip_x.split(",")) if args.skip_x else None

        plotter = cls()
        method = getattr(plotter, method_name)

        # Build kwargs based on plot type
        kwargs: dict = {"skip_x": skip_x}
        if args.plot == "throughput":
            kwargs["source"] = args.source
            kwargs["stat"] = args.stat or "mean"
        elif args.plot == "pareto":
            kwargs["source"] = args.source
            kwargs["stat"] = args.stat or "mean"
        elif args.plot == "ttft":
            kwargs["stat"] = args.stat  # None means all four
            kwargs["scatter"] = not getattr(args, "no_scatter", False)

        fpath = method(data, output_dir, **kwargs)
        print(f"Saved: {fpath}")


if __name__ == "__main__":
    ConcurrencyPlotter.cli()
