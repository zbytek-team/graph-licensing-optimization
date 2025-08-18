import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("results/plots")


def plot_metric_summary(df: pd.DataFrame, metric: str) -> Path:
    """Plot average metric value per algorithm as a bar chart.

    Returns the path to the saved figure.
    """
    grouped = df.groupby("algorithm")[metric].mean().sort_values()
    ax = grouped.plot(kind="bar")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Average {metric.replace('_', ' ')} by Algorithm")
    fig = ax.get_figure()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{metric}_by_algorithm.png"
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    return output_file


def plot_algorithm_comparison(df: pd.DataFrame, metric: str) -> list[Path]:
    """Plot metric versus graph size for each algorithm and graph type.

    Returns a list of paths to saved figures.
    """
    saved_paths: list[Path] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for graph_type, group in df.groupby("graph_type"):
        pivot = group.pivot_table(
            index="nodes", columns="algorithm", values=metric, aggfunc="mean"
        )
        ax = pivot.plot(marker="o")
        ax.set_xlabel("Graph size (nodes)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison ({graph_type})")
        ax.legend(title="Algorithm")
        fig = ax.get_figure()
        output_file = OUTPUT_DIR / f"{graph_type}_{metric}_comparison.png"
        fig.tight_layout()
        fig.savefig(output_file)
        plt.close(fig)
        saved_paths.append(output_file)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from benchmark CSV results."
    )
    parser.add_argument("csv_path", help="Path to the benchmark CSV file")
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        choices=["cost", "execution_time", "groups_count"],
        required=True,
        help="Metric to plot. Can be used multiple times.",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    for metric in args.metrics:
        plot_metric_summary(df, metric)
        plot_algorithm_comparison(df, metric)


if __name__ == "__main__":
    main()
