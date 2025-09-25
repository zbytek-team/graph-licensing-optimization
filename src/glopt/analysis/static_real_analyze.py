from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glopt.analysis.commons import (
    apply_plot_style,
    compute_pareto_front,
    describe_numeric,
    ensure_dir,
    expand_license_counts,
    load_dataset,
    normalize_cost_columns,
    pivot_complete_blocks,
    run_friedman_nemenyi,
    save_table,
    write_text,
)

REQUIRED_DATA_FILES = ["duolingo_df.csv", "roman_df.csv", "timeout_df.csv"]


def _candidate_data_dirs() -> list[Path]:
    candidates: list[Path] = []
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    env_specific = os.environ.get("BENCHMARK_REAL_DATA_DIR")
    if env_specific:
        candidates.append(Path(env_specific))

    env_root = os.environ.get("GLOPT_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "benchmark_real")

    candidates.extend(
        [
            Path("/data/benchmark_real"),
            Path("/data/analysis/processed/benchmark_real"),
            Path("/data/glopt/analysis/processed/benchmark_real"),
            BASE_DIR.parent / "glopt" / "analysis" / "processed" / "benchmark_real",
            BASE_DIR / "data" / "benchmark_real",
        ]
    )

    unique: list[Path] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _resolve_data_dir() -> Path:
    for candidate in _candidate_data_dirs():
        if all((candidate / name).exists() for name in REQUIRED_DATA_FILES):
            return candidate
    searched = "\n".join(str(path) for path in _candidate_data_dirs())
    raise FileNotFoundError(f"Nie można odnaleźć danych benchmark_real. Sprawdzono następujące katalogi:\n{searched}")


DATA_DIR = _resolve_data_dir()
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ensure_dir(BASE_DIR / "results" / "benchmark_real")
FIG_DIR = ensure_dir(RESULTS_DIR / "figures")
TAB_DIR = ensure_dir(RESULTS_DIR / "tables")
REPORTS_DIR = ensure_dir(RESULTS_DIR / "reports")


EXCLUDED = {"ILPSolver"}

METRIC_DISPLAY = {
    "time_s": "czas [s]",
    "total_cost": "łączny koszt",
    "cost_per_node": "koszt na węzeł",
}


def _axis_label(metric: str, aggregate: str | None = None) -> str:
    base = METRIC_DISPLAY.get(metric, metric.replace("_", " "))
    if aggregate == "mean":
        return f"Średni {base}"
    return base.capitalize()


def _filter_algorithms(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["algorithm"].isin(EXCLUDED)].copy()


def _plot_metric_vs_nodes(df: pd.DataFrame, value: str, title: str, filename: str) -> None:
    aggregated = df.groupby(["algorithm", "n_nodes"], as_index=False).agg(mean=(value, "mean"))
    fig, ax = plt.subplots()
    sns.lineplot(
        data=aggregated,
        x="n_nodes",
        y="mean",
        hue="algorithm",
        marker="o",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Liczba węzłów")
    ax.set_ylabel(_axis_label(value, "mean"))
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Algorytm")
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def _plot_metric_by_graph(df: pd.DataFrame, value: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots()
    sns.barplot(
        data=df,
        x="graph",
        y=value,
        hue="algorithm",
        estimator=np.mean,
        errorbar="se",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Typ grafu")
    ax.set_ylabel(_axis_label(value, "mean"))
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Algorytm")
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def _plot_duo_vs_roman(df: pd.DataFrame, value: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots()
    sns.barplot(
        data=df,
        x="graph",
        y=value,
        hue="dataset",
        estimator=np.mean,
        errorbar="se",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Typ grafu")
    ax.set_ylabel(_axis_label(value, "mean"))
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Zbiór danych")
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def main() -> None:
    apply_plot_style()

    duolingo = expand_license_counts(load_dataset(DATA_DIR / "duolingo_df.csv"))
    duolingo, _ = normalize_cost_columns(duolingo)
    roman = expand_license_counts(load_dataset(DATA_DIR / "roman_df.csv"))
    roman, _ = normalize_cost_columns(roman)
    timeout = expand_license_counts(load_dataset(DATA_DIR / "timeout_df.csv"))
    timeout, _ = normalize_cost_columns(timeout)

    duolingo = _filter_algorithms(duolingo)
    roman = _filter_algorithms(roman)

    duolingo["dataset"] = "duolingo_super"
    roman["dataset"] = "roman_domination"
    combined = pd.concat([duolingo, roman], ignore_index=True)

    numeric_cols = [
        "time_s",
        "total_cost",
        "cost_per_node",
        "license_group",
        "license_individual",
    ]

    duo_overall = describe_numeric(duolingo, numeric_cols)
    save_table(duo_overall, TAB_DIR / "duolingo_overall_stats.csv")

    duo_algo = describe_numeric(duolingo, numeric_cols, ["algorithm"])
    save_table(duo_algo, TAB_DIR / "duolingo_algorithm_stats.csv")

    duo_graph = describe_numeric(
        duolingo,
        ["time_s", "total_cost", "cost_per_node"],
        ["algorithm", "graph"],
    )
    save_table(duo_graph, TAB_DIR / "duolingo_algorithm_graph_stats.csv")

    roman_overall = describe_numeric(roman, numeric_cols)
    save_table(roman_overall, TAB_DIR / "roman_overall_stats.csv")

    roman_algo = describe_numeric(roman, numeric_cols, ["algorithm"])
    save_table(roman_algo, TAB_DIR / "roman_algorithm_stats.csv")

    roman_graph = describe_numeric(
        roman,
        ["time_s", "total_cost", "cost_per_node"],
        ["algorithm", "graph"],
    )
    save_table(roman_graph, TAB_DIR / "roman_algorithm_graph_stats.csv")

    timeout_counts_agg = timeout.groupby("algorithm").size().rename("count")
    timeout_counts_agg.to_csv(TAB_DIR / "timeouts_by_algorithm.csv")

    timeout_counts_graph = timeout.groupby(["algorithm", "graph"]).size().rename("count")
    timeout_counts_graph.to_csv(TAB_DIR / "timeouts_by_algorithm_graph.csv")

    pareto_cols = [
        "algorithm",
        "graph",
        "n_nodes",
        "total_cost",
        "time_s",
        "cost_per_node",
    ]
    pareto_df = compute_pareto_front(duolingo[pareto_cols], "total_cost", "time_s")
    pareto_df.to_csv(TAB_DIR / "duolingo_pareto_cost_time.csv", index=False)

    id_cols = ["graph", "n_nodes", "license_config", "rep", "sample"]
    pivot_time = pivot_complete_blocks(duolingo, id_cols, "algorithm", "time_s")
    pivot_cost = pivot_complete_blocks(duolingo, id_cols, "algorithm", "cost_per_node")

    friedman_lines: list[str] = []
    time_result = run_friedman_nemenyi(pivot_time)
    if time_result:
        save_table(
            time_result.mean_ranks.to_frame("mean_rank"),
            TAB_DIR / "friedman_time_mean_ranks.csv",
        )
        if time_result.nemenyi is not None:
            save_table(time_result.nemenyi, TAB_DIR / "nemenyi_time_pvalues.csv")
        friedman_lines.append(f"Friedman test (time_s): statistic={time_result.statistic:.3f}, p-value={time_result.pvalue:.4g}")
    else:
        friedman_lines.append("Friedman test (time_s): insufficient paired samples")

    cost_result = run_friedman_nemenyi(pivot_cost)
    if cost_result:
        save_table(
            cost_result.mean_ranks.to_frame("mean_rank"),
            TAB_DIR / "friedman_cost_per_node_mean_ranks.csv",
        )
        if cost_result.nemenyi is not None:
            save_table(
                cost_result.nemenyi,
                TAB_DIR / "nemenyi_cost_per_node_pvalues.csv",
            )
        friedman_lines.append(f"Friedman test (cost_per_node): statistic={cost_result.statistic:.3f}, p-value={cost_result.pvalue:.4g}")
    else:
        friedman_lines.append("Friedman test (cost_per_node): insufficient paired samples")

    _plot_metric_vs_nodes(
        duolingo,
        "time_s",
        "Benchmark real: czas vs liczba węzłów",
        "time_vs_nodes.pdf",
    )
    _plot_metric_vs_nodes(
        duolingo,
        "cost_per_node",
        "Benchmark real: koszt/węzeł vs liczba węzłów",
        "cost_per_node_vs_nodes.pdf",
    )
    _plot_metric_by_graph(
        duolingo,
        "time_s",
        "Benchmark real: czasy według grafu",
        "time_by_graph.pdf",
    )

    pareto_fig, ax = plt.subplots()
    sns.scatterplot(
        data=duolingo,
        x="total_cost",
        y="time_s",
        hue="algorithm",
        alpha=0.4,
        ax=ax,
    )
    sns.lineplot(
        data=pareto_df,
        x="total_cost",
        y="time_s",
        color="black",
        label="Pareto",
        ax=ax,
    )
    ax.set_title("Benchmark real: Pareto koszt-czas")
    ax.set_xlabel("Łączny koszt")
    ax.set_ylabel("Czas [s]")
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Algorytm")
    plt.tight_layout()
    pareto_fig.savefig(FIG_DIR / "pareto_cost_time.pdf")
    plt.close(pareto_fig)

    comparison = (
        combined.groupby(["dataset", "algorithm"])
        .agg(
            time_mean=("time_s", "mean"),
            cost_mean=("total_cost", "mean"),
            cpn_mean=("cost_per_node", "mean"),
        )
        .reset_index()
    )
    save_table(comparison, TAB_DIR / "duo_vs_roman_algorithm_means.csv")

    by_graph = combined.groupby(["dataset", "graph"]).agg(time_mean=("time_s", "mean"), cpn_mean=("cost_per_node", "mean")).reset_index()
    save_table(by_graph, TAB_DIR / "duo_vs_roman_graph_means.csv")
    _plot_duo_vs_roman(
        combined,
        "time_s",
        "Porównanie czasów (real)",
        "duo_vs_roman_time_by_graph.pdf",
    )
    _plot_duo_vs_roman(
        combined,
        "cost_per_node",
        "Porównanie kosztu/węzeł (real)",
        "duo_vs_roman_cost_per_node_by_graph.pdf",
    )

    lines: list[str] = []
    lines.append("# Benchmark real - podsumowanie")
    lines.append("Wykluczono ILPSolver ze wszystkich statystyk.")
    lines.append("")
    lines.append(f"Duolingo - średni czas: {duo_overall.loc[duo_overall['metric'] == 'time_s', 'mean'].iloc[0]:.2f} s")
    lines.append(f"Roman - średni czas: {roman_overall.loc[roman_overall['metric'] == 'time_s', 'mean'].iloc[0]:.2f} s")
    lines.append("")
    lines.extend(friedman_lines)
    write_text(REPORTS_DIR / "summary.md", lines)


if __name__ == "__main__":
    main()
