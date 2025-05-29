#!/usr/bin/env python3
"""
Comprehensive Analysis and Visualization Tool for Graph Licensing Optimization

This script provides extensive analysis and visualization capabilities for
benchmark results from the graph licensing optimization project.

Features:
- Performance analysis (runtime, cost, efficiency)
- Algorithm comparison across different metrics
- Scalability analysis
- Quality metrics visualization
- Statistical analysis and correlations
- Interactive dashboard-like reports

Usage:
    python analysis_visualizer.py [options]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for consistent, professional plots
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


class AnalysisVisualizer:
    """Comprehensive analysis and visualization tool for benchmark results."""

    def __init__(self, results_path: str = "results/benchmark"):
        """Initialize the analyzer with benchmark results.

        Args:
            results_path: Path to the benchmark results directory
        """
        self.results_path = Path(results_path)
        self.df = None
        self.summary = None
        self.output_dir = Path("results/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes for consistent visualization - more contrasting colors
        self.algorithm_colors = {
            "Greedy": "#00AA00",  # Green
            "Genetic": "#FF4500",  # Red-Orange
            "SimulatedAnnealing": "#0066CC",  # Blue
            "TabuSearch": "#FFD700",  # Yellow
            "ILP": "#8A2BE2",  # Purple
            "Naive": "#DC143C",  # Red
            "DominatingSet": "#FF8C00",  # Orange
            "Randomized": "#708090",  # Gray
        }

    def load_data(self) -> bool:
        """Load benchmark data from CSV and JSON files.

        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Load CSV results
            csv_path = self.results_path / "benchmark_results.csv"
            if not csv_path.exists():
                print(f"Error: Benchmark results not found at {csv_path}")
                return False

            self.df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.df)} benchmark results")

            # Load summary JSON if available
            json_path = self.results_path / "benchmark_summary.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    self.summary = json.load(f)
                print("Loaded benchmark summary")

            # Process data
            self._process_data()
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _process_data(self):
        """Process and clean the loaded data."""
        if self.df is None:
            return

        # Extract graph type from test_name
        self.df["graph_type"] = self.df["test_name"].str.split("_").str[0]

        # Add efficiency metrics
        self.df["cost_efficiency"] = 1 / self.df["total_cost"]
        self.df["time_efficiency"] = 1 / self.df["runtime_seconds"]
        self.df["overall_efficiency"] = (self.df["cost_efficiency"] * self.df["time_efficiency"]) ** 0.5

        # Add performance categories
        self.df["runtime_category"] = pd.cut(
            self.df["runtime_seconds"],
            bins=[0, 0.001, 0.01, 0.1, 1, float("inf")],
            labels=["Very Fast", "Fast", "Medium", "Slow", "Very Slow"],
        )

        self.df["cost_category"] = pd.qcut(self.df["total_cost"], q=3, labels=["Low Cost", "Medium Cost", "High Cost"])

        # Add solution quality metrics
        self.df["group_utilization"] = self.df["total_group_members"] / (
            self.df["n_group_licenses"] * self.df["group_size_limit"]
        )
        self.df["solo_ratio"] = self.df["n_solo_licenses"] / self.df["n_nodes"]
        self.df["group_ratio"] = self.df["n_group_licenses"] / self.df["n_nodes"]

    def create_overview_dashboard(self):
        """Create individual plots saved as separate PNG files."""
        if self.df is None:
            print("No data loaded")
            return

        print("Creating individual analysis plots...")

        # 1. Algorithm Performance Overview
        self._create_algorithm_performance_plot()

        # 2. Runtime vs Cost Scatter
        self._create_runtime_vs_cost_plot()

        # 3. Scalability Analysis
        self._create_scalability_plot()

        # 4. Solution Quality Distribution
        self._create_solution_quality_plot()

        # 5. Runtime Distribution
        self._create_runtime_distribution_plot()

        # 6. Cost Distribution
        self._create_cost_distribution_plot()

        # 7. Correlation Heatmap
        self._create_correlation_heatmap_plot()

        # 8. Cost Efficiency
        self._create_cost_efficiency_plot()

        # 9. Time Efficiency
        self._create_time_efficiency_plot()

        # 10. Overall Efficiency
        self._create_overall_efficiency_plot()

        print("All individual plots created successfully!")

    def _create_algorithm_performance_plot(self):
        """Create algorithm performance overview plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Group by algorithm and calculate metrics
        algo_stats = (
            self.df.groupby("algorithm")
            .agg({"total_cost": ["mean", "std"], "runtime_seconds": ["mean", "std"]})
            .round(4)
        )

        # Flatten column names
        algo_stats.columns = ["_".join(col).strip() for col in algo_stats.columns.values]

        # Create bar plot for average cost
        x_pos = np.arange(len(algo_stats))
        bars = ax.bar(x_pos, algo_stats["total_cost_mean"], yerr=algo_stats["total_cost_std"], capsize=5, alpha=0.7)

        # Color bars by algorithm
        for i, (algo, bar) in enumerate(zip(algo_stats.index, bars)):
            bar.set_color(self.algorithm_colors.get(algo, "#777777"))

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Average Total Cost")
        ax.set_title("Algorithm Performance Overview (Cost)", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algo_stats.index, rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "algorithm_performance.png", bbox_inches="tight")
        plt.close()

    def _create_runtime_vs_cost_plot(self):
        """Create runtime vs cost scatter plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        for algo in self.df["algorithm"].unique():
            algo_data = self.df[self.df["algorithm"] == algo]
            ax.scatter(
                algo_data["runtime_seconds"],
                algo_data["total_cost"],
                alpha=0.6,
                label=algo,
                s=60,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax.set_xlabel("Runtime (seconds)")
        ax.set_ylabel("Total Cost")
        ax.set_title("Runtime vs Cost Trade-off", fontsize=14, fontweight="bold")
        ax.set_xscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "runtime_vs_cost.png", bbox_inches="tight")
        plt.close()

    def _create_scalability_plot(self):
        """Create scalability analysis plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        for algo in self.df["algorithm"].unique():
            algo_data = self.df[self.df["algorithm"] == algo]
            # Group by number of nodes and calculate mean runtime
            scalability = algo_data.groupby("n_nodes")["runtime_seconds"].mean().sort_index()

            ax.plot(
                scalability.index,
                scalability.values,
                marker="o",
                label=algo,
                linewidth=2,
                markersize=8,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Average Runtime (seconds)")
        ax.set_title("Algorithm Scalability", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "scalability_analysis.png", bbox_inches="tight")
        plt.close()

    def _create_solution_quality_plot(self):
        """Create solution quality distribution plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        valid_solutions = self.df[self.df["is_valid_solution"]]

        ax.hist(valid_solutions["group_utilization"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax.set_xlabel("Group License Utilization")
        ax.set_ylabel("Frequency")
        ax.set_title("Solution Quality Distribution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_util = valid_solutions["group_utilization"].mean()
        ax.axvline(mean_util, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_util:.2f}")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "solution_quality.png", bbox_inches="tight")
        plt.close()

    def _create_runtime_distribution_plot(self):
        """Create runtime distribution plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create boxplot data
        runtime_data = []
        labels = []
        colors = []

        for algo in sorted(self.df["algorithm"].unique()):
            algo_data = self.df[self.df["algorithm"] == algo]["runtime_seconds"]
            runtime_data.append(algo_data.values)
            labels.append(algo)
            colors.append(self.algorithm_colors.get(algo, "#777777"))

        bp = ax.boxplot(runtime_data, labels=labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_yscale("log")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("Runtime Distribution by Algorithm", fontsize=14, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "runtime_distribution.png", bbox_inches="tight")
        plt.close()

    def _create_cost_distribution_plot(self):
        """Create cost distribution plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create boxplot data
        cost_data = []
        labels = []
        colors = []

        for algo in sorted(self.df["algorithm"].unique()):
            algo_data = self.df[self.df["algorithm"] == algo]["total_cost"]
            cost_data.append(algo_data.values)
            labels.append(algo)
            colors.append(self.algorithm_colors.get(algo, "#777777"))

        bp = ax.boxplot(cost_data, labels=labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Total Cost")
        ax.set_title("Cost Distribution by Algorithm", fontsize=14, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "cost_distribution.png", bbox_inches="tight")
        plt.close()

    def _create_correlation_heatmap_plot(self):
        """Create correlation heatmap plot."""
        fig, ax = plt.subplots(figsize=(12, 10))

        numerical_cols = [
            "n_nodes",
            "n_edges",
            "runtime_seconds",
            "total_cost",
            "n_solo_licenses",
            "n_group_licenses",
            "avg_group_size",
            "group_utilization",
            "edge_to_node_ratio",
        ]

        corr_matrix = self.df[numerical_cols].corr()

        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True, ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png", bbox_inches="tight")
        plt.close()

    def _create_cost_efficiency_plot(self):
        """Create cost efficiency comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        efficiency_data = self.df.groupby("algorithm")["cost_efficiency"].mean().sort_values(ascending=False)

        bars = ax.bar(range(len(efficiency_data)), efficiency_data.values, alpha=0.8)

        # Color bars by algorithm
        for i, (algo, bar) in enumerate(zip(efficiency_data.index, bars)):
            bar.set_color(self.algorithm_colors.get(algo, "#777777"))

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Cost Efficiency (1/cost)")
        ax.set_title("Cost Efficiency Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(efficiency_data)))
        ax.set_xticklabels(efficiency_data.index, rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(efficiency_data.values):
            ax.text(i, v + max(efficiency_data.values) * 0.01, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.output_dir / "cost_efficiency.png", bbox_inches="tight")
        plt.close()

    def _create_time_efficiency_plot(self):
        """Create time efficiency comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        efficiency_data = self.df.groupby("algorithm")["time_efficiency"].mean().sort_values(ascending=False)

        bars = ax.bar(range(len(efficiency_data)), efficiency_data.values, alpha=0.8)

        # Color bars by algorithm
        for i, (algo, bar) in enumerate(zip(efficiency_data.index, bars)):
            bar.set_color(self.algorithm_colors.get(algo, "#777777"))

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Time Efficiency (1/runtime)")
        ax.set_title("Time Efficiency Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(efficiency_data)))
        ax.set_xticklabels(efficiency_data.index, rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(efficiency_data.values):
            ax.text(i, v + max(efficiency_data.values) * 0.01, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.output_dir / "time_efficiency.png", bbox_inches="tight")
        plt.close()

    def _create_overall_efficiency_plot(self):
        """Create overall efficiency comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        efficiency_data = self.df.groupby("algorithm")["overall_efficiency"].mean().sort_values(ascending=False)

        bars = ax.bar(range(len(efficiency_data)), efficiency_data.values, alpha=0.8)

        # Color bars by algorithm
        for i, (algo, bar) in enumerate(zip(efficiency_data.index, bars)):
            bar.set_color(self.algorithm_colors.get(algo, "#777777"))

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Overall Efficiency")
        ax.set_title("Overall Efficiency Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(efficiency_data)))
        ax.set_xticklabels(efficiency_data.index, rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(efficiency_data.values):
            ax.text(i, v + max(efficiency_data.values) * 0.01, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.output_dir / "overall_efficiency.png", bbox_inches="tight")
        plt.close()

    def _plot_algorithm_performance_overview(self, ax):
        """Plot algorithm performance overview."""
        # Group by algorithm and calculate metrics
        algo_stats = (
            self.df.groupby("algorithm")
            .agg({"total_cost": ["mean", "std"], "runtime_seconds": ["mean", "std"], "success": "mean"})
            .round(4)
        )

        # Flatten column names
        algo_stats.columns = ["_".join(col).strip() for col in algo_stats.columns.values]

        # Create bar plot for average cost
        x_pos = np.arange(len(algo_stats))
        bars = ax.bar(x_pos, algo_stats["total_cost_mean"], yerr=algo_stats["total_cost_std"], capsize=5, alpha=0.7)

        # Color bars by algorithm
        for i, (algo, bar) in enumerate(zip(algo_stats.index, bars)):
            bar.set_color(self.algorithm_colors.get(algo, "#777777"))

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Average Total Cost")
        ax.set_title("Algorithm Performance Overview (Cost)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algo_stats.index, rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_runtime_vs_cost_scatter(self, ax):
        """Plot runtime vs cost scatter with algorithm distinction."""
        for algo in self.df["algorithm"].unique():
            algo_data = self.df[self.df["algorithm"] == algo]
            ax.scatter(
                algo_data["runtime_seconds"],
                algo_data["total_cost"],
                alpha=0.6,
                label=algo,
                s=60,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax.set_xlabel("Runtime (seconds)")
        ax.set_ylabel("Total Cost")
        ax.set_title("Runtime vs Cost Trade-off")
        ax.set_xscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_graph_type_analysis(self, ax):
        """Plot analysis by graph type."""
        graph_stats = self.df.groupby("graph_type")["total_cost"].agg(["mean", "std", "count"])

        x_pos = np.arange(len(graph_stats))
        bars = ax.bar(x_pos, graph_stats["mean"], yerr=graph_stats["std"], capsize=5, alpha=0.7)

        # Color bars by graph type
        for i, (graph_type, bar) in enumerate(zip(graph_stats.index, bars)):
            bar.set_color(self.graph_colors.get(graph_type, "#777777"))

        ax.set_xlabel("Graph Type")
        ax.set_ylabel("Average Total Cost")
        ax.set_title("Performance by Graph Type")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(graph_stats.index, rotation=45)
        ax.grid(True, alpha=0.3)

        # Add count annotations
        for i, count in enumerate(graph_stats["count"]):
            ax.annotate(
                f"n={count}",
                (i, graph_stats["mean"].iloc[i] + graph_stats["std"].iloc[i]),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _plot_scalability_analysis(self, ax):
        """Plot scalability analysis showing how algorithms scale with graph size."""
        for algo in self.df["algorithm"].unique():
            algo_data = self.df[self.df["algorithm"] == algo]
            # Group by number of nodes and calculate mean runtime
            scalability = algo_data.groupby("n_nodes")["runtime_seconds"].mean().sort_index()

            ax.plot(
                scalability.index,
                scalability.values,
                marker="o",
                label=algo,
                linewidth=2,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Average Runtime (seconds)")
        ax.set_title("Algorithm Scalability")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_success_rates(self, ax):
        """Plot success rates by algorithm."""
        success_rates = self.df.groupby("algorithm")["success"].mean().sort_values(ascending=True)

        bars = ax.barh(range(len(success_rates)), success_rates.values)

        # Color bars
        for i, (algo, bar) in enumerate(zip(success_rates.index, bars)):
            bar.set_color(self.algorithm_colors.get(algo, "#777777"))

        ax.set_yticks(range(len(success_rates)))
        ax.set_yticklabels(success_rates.index)
        ax.set_xlabel("Success Rate")
        ax.set_title("Algorithm Success Rates")
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Add percentage labels
        for i, rate in enumerate(success_rates.values):
            ax.text(rate + 0.01, i, f"{rate:.1%}", va="center")

    def _plot_solution_quality_distribution(self, ax):
        """Plot solution quality distribution."""
        valid_solutions = self.df[self.df["is_valid_solution"] == True]

        ax.hist(valid_solutions["group_utilization"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax.set_xlabel("Group License Utilization")
        ax.set_ylabel("Frequency")
        ax.set_title("Solution Quality Distribution")
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_util = valid_solutions["group_utilization"].mean()
        ax.axvline(mean_util, color="red", linestyle="--", label=f"Mean: {mean_util:.2f}")
        ax.legend()

    def _plot_runtime_distribution(self, ax):
        """Plot runtime distribution by algorithm."""
        self.df.boxplot(column="runtime_seconds", by="algorithm", ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("Runtime Distribution by Algorithm")
        plt.setp(ax.get_xticklabels(), rotation=45)

    def _plot_cost_distribution(self, ax):
        """Plot cost distribution by algorithm."""
        self.df.boxplot(column="total_cost", by="algorithm", ax=ax)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Total Cost")
        ax.set_title("Cost Distribution by Algorithm")
        plt.setp(ax.get_xticklabels(), rotation=45)

    def _plot_correlation_heatmap(self, ax):
        """Plot correlation heatmap of numerical variables."""
        numerical_cols = [
            "n_nodes",
            "n_edges",
            "runtime_seconds",
            "total_cost",
            "n_solo_licenses",
            "n_group_licenses",
            "avg_group_size",
            "group_utilization",
            "edge_to_node_ratio",
        ]

        corr_matrix = self.df[numerical_cols].corr()

        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True, ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Matrix")

    def _plot_algorithm_efficiency_comparison(self, ax):
        """Plot algorithm efficiency comparison."""
        efficiency_data = self.df.groupby("algorithm")[
            ["cost_efficiency", "time_efficiency", "overall_efficiency"]
        ].mean()

        x = np.arange(len(efficiency_data))
        width = 0.25

        bars1 = ax.bar(x - width, efficiency_data["cost_efficiency"], width, label="Cost Efficiency", alpha=0.8)
        bars2 = ax.bar(x, efficiency_data["time_efficiency"], width, label="Time Efficiency", alpha=0.8)
        bars3 = ax.bar(x + width, efficiency_data["overall_efficiency"], width, label="Overall Efficiency", alpha=0.8)

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Efficiency (1/metric)")
        ax.set_title("Algorithm Efficiency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(efficiency_data.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_detailed_algorithm_comparison(self):
        """Create detailed comparison plots for each algorithm."""
        if self.df is None:
            print("No data loaded")
            return

        algorithms = self.df["algorithm"].unique()
        n_algorithms = len(algorithms)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Detailed Algorithm Comparison", fontsize=16, fontweight="bold")

        # 1. Performance vs Graph Size
        ax1 = axes[0, 0]
        for algo in algorithms:
            algo_data = self.df[self.df["algorithm"] == algo]
            perf_by_size = algo_data.groupby("n_nodes").agg({"total_cost": "mean", "runtime_seconds": "mean"})

            ax1.plot(
                perf_by_size.index,
                perf_by_size["total_cost"],
                marker="o",
                label=algo,
                linewidth=2,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax1.set_xlabel("Graph Size (nodes)")
        ax1.set_ylabel("Average Cost")
        ax1.set_title("Cost vs Graph Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Runtime vs Graph Size
        ax2 = axes[0, 1]
        for algo in algorithms:
            algo_data = self.df[self.df["algorithm"] == algo]
            perf_by_size = algo_data.groupby("n_nodes").agg({"total_cost": "mean", "runtime_seconds": "mean"})

            ax2.plot(
                perf_by_size.index,
                perf_by_size["runtime_seconds"],
                marker="s",
                label=algo,
                linewidth=2,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax2.set_xlabel("Graph Size (nodes)")
        ax2.set_ylabel("Average Runtime (seconds)")
        ax2.set_title("Runtime vs Graph Size")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Performance by Graph Type
        ax3 = axes[1, 0]
        graph_algo_pivot = self.df.pivot_table(
            values="total_cost", index="graph_type", columns="algorithm", aggfunc="mean"
        )

        graph_algo_pivot.plot(kind="bar", ax=ax3, width=0.8)
        ax3.set_xlabel("Graph Type")
        ax3.set_ylabel("Average Cost")
        ax3.set_title("Performance by Graph Type")
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3)

        # 4. Solution Quality Metrics
        ax4 = axes[1, 1]
        quality_metrics = self.df.groupby("algorithm").agg(
            {"solo_ratio": "mean", "group_ratio": "mean", "group_utilization": "mean"}
        )

        x = np.arange(len(quality_metrics))
        width = 0.25

        ax4.bar(x - width, quality_metrics["solo_ratio"], width, label="Solo Ratio", alpha=0.8)
        ax4.bar(x, quality_metrics["group_ratio"], width, label="Group Ratio", alpha=0.8)
        ax4.bar(x + width, quality_metrics["group_utilization"], width, label="Group Utilization", alpha=0.8)

        ax4.set_xlabel("Algorithm")
        ax4.set_ylabel("Ratio")
        ax4.set_title("Solution Quality Metrics")
        ax4.set_xticks(x)
        ax4.set_xticklabels(quality_metrics.index, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "detailed_algorithm_comparison.png", bbox_inches="tight")
        plt.show()

    def create_graph_type_analysis(self):
        """Create comprehensive analysis by graph type."""
        if self.df is None:
            print("No data loaded")
            return

        graph_types = self.df["graph_type"].unique()
        n_types = len(graph_types)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Graph Type Analysis", fontsize=16, fontweight="bold")

        # 1. Cost comparison by graph type
        ax1 = axes[0, 0]
        self.df.boxplot(column="total_cost", by="graph_type", ax=ax1)
        ax1.set_xlabel("Graph Type")
        ax1.set_ylabel("Total Cost")
        ax1.set_title("Cost Distribution by Graph Type")
        plt.setp(ax1.get_xticklabels(), rotation=45)

        # 2. Runtime comparison by graph type
        ax2 = axes[0, 1]
        self.df.boxplot(column="runtime_seconds", by="graph_type", ax=ax2)
        ax2.set_yscale("log")
        ax2.set_xlabel("Graph Type")
        ax2.set_ylabel("Runtime (seconds)")
        ax2.set_title("Runtime Distribution by Graph Type")
        plt.setp(ax2.get_xticklabels(), rotation=45)

        # 3. Graph properties analysis
        ax3 = axes[1, 0]
        graph_props = self.df.groupby("graph_type").agg(
            {"n_nodes": "mean", "n_edges": "mean", "edge_to_node_ratio": "mean"}
        )

        x = np.arange(len(graph_props))
        ax3_twin = ax3.twinx()

        bars1 = ax3.bar(x - 0.2, graph_props["n_nodes"], 0.4, label="Avg Nodes", alpha=0.7, color="skyblue")
        bars2 = ax3.bar(x + 0.2, graph_props["n_edges"], 0.4, label="Avg Edges", alpha=0.7, color="lightcoral")

        line = ax3_twin.plot(x, graph_props["edge_to_node_ratio"], "ro-", label="Edge-to-Node Ratio", linewidth=2)

        ax3.set_xlabel("Graph Type")
        ax3.set_ylabel("Count")
        ax3_twin.set_ylabel("Edge-to-Node Ratio")
        ax3.set_title("Graph Properties by Type")
        ax3.set_xticks(x)
        ax3.set_xticklabels(graph_props.index, rotation=45)

        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # 4. Success rates by graph type
        ax4 = axes[1, 1]
        success_by_type = self.df.groupby(["graph_type", "algorithm"])["success"].mean().unstack()

        success_by_type.plot(kind="bar", ax=ax4, width=0.8, stacked=False)
        ax4.set_xlabel("Graph Type")
        ax4.set_ylabel("Success Rate")
        ax4.set_title("Success Rates by Graph Type and Algorithm")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graph_type_analysis.png", bbox_inches="tight")
        plt.show()

    def create_scalability_analysis(self):
        """Create detailed scalability analysis."""
        if self.df is None:
            print("No data loaded")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Scalability Analysis", fontsize=16, fontweight="bold")

        # 1. Runtime scaling
        ax1 = axes[0, 0]
        for algo in self.df["algorithm"].unique():
            algo_data = self.df[self.df["algorithm"] == algo]
            scaling = algo_data.groupby("n_nodes")["runtime_seconds"].agg(["mean", "std"])

            ax1.errorbar(
                scaling.index,
                scaling["mean"],
                yerr=scaling["std"],
                marker="o",
                label=algo,
                capsize=5,
                linewidth=2,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax1.set_xlabel("Number of Nodes")
        ax1.set_ylabel("Runtime (seconds)")
        ax1.set_title("Runtime Scalability")
        ax1.set_yscale("log")
        ax1.set_xscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Cost scaling
        ax2 = axes[0, 1]
        for algo in self.df["algorithm"].unique():
            algo_data = self.df[self.df["algorithm"] == algo]
            scaling = algo_data.groupby("n_nodes")["total_cost"].agg(["mean", "std"])

            ax2.errorbar(
                scaling.index,
                scaling["mean"],
                yerr=scaling["std"],
                marker="s",
                label=algo,
                capsize=5,
                linewidth=2,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax2.set_xlabel("Number of Nodes")
        ax2.set_ylabel("Total Cost")
        ax2.set_title("Cost Scalability")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Efficiency scaling
        ax3 = axes[1, 0]
        for algo in self.df["algorithm"].unique():
            algo_data = self.df[self.df["algorithm"] == algo]
            scaling = algo_data.groupby("n_nodes")["overall_efficiency"].mean()

            ax3.plot(
                scaling.index,
                scaling.values,
                marker="^",
                label=algo,
                linewidth=2,
                color=self.algorithm_colors.get(algo, "#777777"),
            )

        ax3.set_xlabel("Number of Nodes")
        ax3.set_ylabel("Overall Efficiency")
        ax3.set_title("Efficiency Scalability")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Complexity analysis
        ax4 = axes[1, 1]

        # Calculate time complexity indicators
        complexity_data = []
        for algo in self.df["algorithm"].unique():
            algo_data = self.df[self.df["algorithm"] == algo]
            sizes = sorted(algo_data["n_nodes"].unique())
            runtimes = [algo_data[algo_data["n_nodes"] == size]["runtime_seconds"].mean() for size in sizes]

            if len(sizes) > 2:
                # Fit different complexity curves
                log_sizes = np.log(sizes)
                log_runtimes = np.log(runtimes)

                # Linear fit in log space (power law)
                slope, intercept, r_value, _, _ = stats.linregress(log_sizes, log_runtimes)

                complexity_data.append({"Algorithm": algo, "Complexity_Exponent": slope, "R_squared": r_value**2})

        if complexity_data:
            complexity_df = pd.DataFrame(complexity_data)

            bars = ax4.bar(range(len(complexity_df)), complexity_df["Complexity_Exponent"])

            # Color bars by algorithm
            for i, (_, row) in enumerate(complexity_df.iterrows()):
                bars[i].set_color(self.algorithm_colors.get(row["Algorithm"], "#777777"))

            ax4.set_xlabel("Algorithm")
            ax4.set_ylabel("Complexity Exponent")
            ax4.set_title("Estimated Time Complexity")
            ax4.set_xticks(range(len(complexity_df)))
            ax4.set_xticklabels(complexity_df["Algorithm"], rotation=45)
            ax4.grid(True, alpha=0.3)

            # Add R² annotations
            for i, r2 in enumerate(complexity_df["R_squared"]):
                ax4.annotate(
                    f"R²={r2:.2f}",
                    (i, complexity_df["Complexity_Exponent"].iloc[i] + 0.1),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(self.output_dir / "scalability_analysis.png", bbox_inches="tight")
        plt.show()

    def create_statistical_analysis(self):
        """Create statistical analysis and hypothesis testing."""
        if self.df is None:
            print("No data loaded")
            return

        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS REPORT")
        print("=" * 60)

        # 1. Descriptive statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 40)

        desc_stats = (
            self.df.groupby("algorithm")
            .agg(
                {
                    "total_cost": ["mean", "std", "min", "max", "median"],
                    "runtime_seconds": ["mean", "std", "min", "max", "median"],
                    "success": ["mean", "count"],
                }
            )
            .round(4)
        )

        print(desc_stats)

        # 2. Algorithm comparison tests
        print("\n2. ALGORITHM COMPARISON TESTS")
        print("-" * 40)

        algorithms = self.df["algorithm"].unique()

        # Pairwise t-tests for cost
        print("\nPairwise t-tests for Total Cost:")
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i + 1 :]:
                data1 = self.df[self.df["algorithm"] == algo1]["total_cost"]
                data2 = self.df[self.df["algorithm"] == algo2]["total_cost"]

                if len(data1) > 1 and len(data2) > 1:
                    statistic, p_value = stats.ttest_ind(data1, data2)
                    significance = (
                        "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    )
                    print(f"{algo1} vs {algo2}: t={statistic:.3f}, p={p_value:.4f} {significance}")

        # ANOVA for overall comparison
        print("\nANOVA Test for Cost Differences:")
        cost_groups = [self.df[self.df["algorithm"] == algo]["total_cost"].values for algo in algorithms]
        cost_groups = [group for group in cost_groups if len(group) > 0]

        if len(cost_groups) > 1:
            f_stat, p_value = stats.f_oneway(*cost_groups)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.4f} {significance}")

        # 3. Correlation analysis
        print("\n3. CORRELATION ANALYSIS")
        print("-" * 40)

        numerical_cols = [
            "n_nodes",
            "n_edges",
            "runtime_seconds",
            "total_cost",
            "edge_to_node_ratio",
            "group_utilization",
        ]

        correlations = self.df[numerical_cols].corr()["total_cost"].sort_values(key=abs, ascending=False)

        print("Correlations with Total Cost:")
        for var, corr in correlations.items():
            if var != "total_cost":
                significance = ""
                if abs(corr) > 0.7:
                    significance = "***"
                elif abs(corr) > 0.5:
                    significance = "**"
                elif abs(corr) > 0.3:
                    significance = "*"
                print(f"{var}: {corr:.3f} {significance}")

        # 4. Performance rankings
        print("\n4. ALGORITHM PERFORMANCE RANKINGS")
        print("-" * 40)

        # Rank by different criteria
        rankings = {}

        # Cost ranking (lower is better)
        cost_ranking = self.df.groupby("algorithm")["total_cost"].mean().sort_values()
        rankings["Cost"] = list(cost_ranking.index)

        # Runtime ranking (lower is better)
        runtime_ranking = self.df.groupby("algorithm")["runtime_seconds"].mean().sort_values()
        rankings["Runtime"] = list(runtime_ranking.index)

        # Success rate ranking (higher is better)
        success_ranking = self.df.groupby("algorithm")["success"].mean().sort_values(ascending=False)
        rankings["Success Rate"] = list(success_ranking.index)

        # Overall efficiency ranking (higher is better)
        efficiency_ranking = self.df.groupby("algorithm")["overall_efficiency"].mean().sort_values(ascending=False)
        rankings["Efficiency"] = list(efficiency_ranking.index)

        ranking_df = pd.DataFrame(rankings)
        ranking_df.index = range(1, len(ranking_df) + 1)
        print(ranking_df)

        # 5. Save statistical report
        with open(self.output_dir / "statistical_analysis_report.txt", "w") as f:
            f.write("GRAPH LICENSING OPTIMIZATION - STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. DESCRIPTIVE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(str(desc_stats) + "\n\n")

            f.write("4. ALGORITHM PERFORMANCE RANKINGS\n")
            f.write("-" * 40 + "\n")
            f.write(str(ranking_df) + "\n\n")

            f.write("5. CORRELATION MATRIX\n")
            f.write("-" * 40 + "\n")
            f.write(str(correlations) + "\n")

        print(f"\nStatistical analysis report saved to: {self.output_dir / 'statistical_analysis_report.txt'}")

    def create_custom_analysis(self, custom_query: str):
        """Create custom analysis based on user query."""
        if self.df is None:
            print("No data loaded")
            return

        print(f"\nCustom Analysis: {custom_query}")
        print("-" * 50)

        # Simple query processing
        query_lower = custom_query.lower()

        if "best" in query_lower and "algorithm" in query_lower:
            best_cost = self.df.groupby("algorithm")["total_cost"].mean().idxmin()
            best_runtime = self.df.groupby("algorithm")["runtime_seconds"].mean().idxmin()
            best_success = self.df.groupby("algorithm")["success"].mean().idxmax()

            print(f"Best algorithm by cost: {best_cost}")
            print(f"Best algorithm by runtime: {best_runtime}")
            print(f"Best algorithm by success rate: {best_success}")

        elif "worst" in query_lower and "graph" in query_lower:
            worst_graph = self.df.groupby("graph_type")["total_cost"].mean().idxmax()
            print(f"Most challenging graph type: {worst_graph}")

        elif "scalability" in query_lower:
            print("Scalability ranking (by runtime growth):")
            for algo in self.df["algorithm"].unique():
                algo_data = self.df[self.df["algorithm"] == algo]
                sizes = sorted(algo_data["n_nodes"].unique())
                if len(sizes) > 1:
                    runtime_growth = (
                        algo_data[algo_data["n_nodes"] == max(sizes)]["runtime_seconds"].mean()
                        / algo_data[algo_data["n_nodes"] == min(sizes)]["runtime_seconds"].mean()
                    )
                    print(f"{algo}: {runtime_growth:.2f}x growth")

        else:
            print("Available custom queries:")
            print("- 'best algorithm'")
            print("- 'worst graph type'")
            print("- 'scalability comparison'")

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report with all visualizations."""
        if not self.load_data():
            return

        print("Generating comprehensive analysis report...")
        print("This may take a few minutes...")

        # Create all visualizations
        print("1/5 Creating overview dashboard...")
        self.create_overview_dashboard()

        print("2/5 Creating detailed algorithm comparison...")
        self.create_detailed_algorithm_comparison()

        print("3/5 Creating graph type analysis...")
        self.create_graph_type_analysis()

        print("4/5 Creating scalability analysis...")
        self.create_scalability_analysis()

        print("5/5 Creating statistical analysis...")
        self.create_statistical_analysis()

        # Create summary report
        self._create_summary_report()

        print(f"\nComprehensive analysis completed!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Generated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  - {file.name}")
        for file in sorted(self.output_dir.glob("*.txt")):
            print(f"  - {file.name}")

    def _create_summary_report(self):
        """Create a text summary of key findings."""
        if self.df is None:
            return

        # Calculate key metrics
        best_algo_cost = self.df.groupby("algorithm")["total_cost"].mean().idxmin()
        best_algo_runtime = self.df.groupby("algorithm")["runtime_seconds"].mean().idxmin()
        most_challenging_graph = self.df.groupby("graph_type")["total_cost"].mean().idxmax()

        total_tests = len(self.df)
        success_rate = self.df["success"].mean()

        with open(self.output_dir / "executive_summary.txt", "w") as f:
            f.write("GRAPH LICENSING OPTIMIZATION - EXECUTIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write("KEY FINDINGS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Total tests conducted: {total_tests}\n")
            f.write(f"• Overall success rate: {success_rate:.1%}\n")
            f.write(f"• Best algorithm (cost): {best_algo_cost}\n")
            f.write(f"• Fastest algorithm: {best_algo_runtime}\n")
            f.write(f"• Most challenging graph type: {most_challenging_graph}\n\n")

            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")

            # Cost-performance trade-off analysis
            algo_stats = (
                self.df.groupby("algorithm")
                .agg({"total_cost": "mean", "runtime_seconds": "mean", "success": "mean"})
                .round(3)
            )

            f.write("Algorithm Recommendations:\n")
            for algo, stats in algo_stats.iterrows():
                cost_rank = (algo_stats["total_cost"] <= stats["total_cost"]).sum()
                runtime_rank = (algo_stats["runtime_seconds"] <= stats["runtime_seconds"]).sum()

                if cost_rank <= 2 and runtime_rank <= 2:
                    f.write(f"• {algo}: Excellent balance of cost and speed\n")
                elif cost_rank <= 2:
                    f.write(f"• {algo}: Best for cost optimization\n")
                elif runtime_rank <= 2:
                    f.write(f"• {algo}: Best for time-critical applications\n")

            f.write(f"\nFor detailed analysis, see the generated visualization files.\n")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Analysis and Visualization Tool for Graph Licensing Optimization"
    )

    parser.add_argument(
        "--results-path",
        default="results/benchmark",
        help="Path to benchmark results directory (default: results/benchmark)",
    )

    parser.add_argument(
        "--output-dir",
        default="results/analysis",
        help="Output directory for analysis results (default: results/analysis)",
    )

    parser.add_argument(
        "--analysis-type",
        choices=["overview", "algorithms", "graphs", "scalability", "statistical", "all"],
        default="all",
        help="Type of analysis to perform (default: all)",
    )

    parser.add_argument(
        "--custom-query", help="Custom analysis query (e.g., 'best algorithm', 'scalability comparison')"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = AnalysisVisualizer(args.results_path)
    analyzer.output_dir = Path(args.output_dir)
    analyzer.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if not analyzer.load_data():
        sys.exit(1)

    # Perform analysis based on type
    if args.custom_query:
        analyzer.create_custom_analysis(args.custom_query)
    elif args.analysis_type == "overview":
        analyzer.create_overview_dashboard()
    elif args.analysis_type == "algorithms":
        analyzer.create_detailed_algorithm_comparison()
    elif args.analysis_type == "graphs":
        analyzer.create_graph_type_analysis()
    elif args.analysis_type == "scalability":
        analyzer.create_scalability_analysis()
    elif args.analysis_type == "statistical":
        analyzer.create_statistical_analysis()
    elif args.analysis_type == "all":
        analyzer.generate_comprehensive_report()

    print(f"\nAnalysis completed! Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
