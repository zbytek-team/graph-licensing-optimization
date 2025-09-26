from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glopt.analysis.commons import (
    ALGORITHM_CANONICAL_ORDER,
    SAMPLE_IDENTIFIER_COLUMNS,
    algorithm_display_name,
    algorithm_palette,
    apply_algorithm_labels,
    apply_plot_style,
    compute_pareto_front,
    describe_numeric,
    ensure_dir,
    expand_license_counts,
    filter_complete_samples,
    load_dataset,
    normalize_cost_columns,
    number_to_polish_words,
    pivot_complete_blocks,
    run_friedman_nemenyi,
    save_table,
    write_text,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "benchmark"
RESULTS_DIR = ensure_dir(BASE_DIR / "results" / "benchmark")
FIG_DIR = ensure_dir(RESULTS_DIR / "figures")
TAB_DIR = ensure_dir(RESULTS_DIR / "tables")
REPORTS_DIR = ensure_dir(RESULTS_DIR / "reports")

METRIC_DISPLAY = {
    "time_s": "czas [s]",
    "total_cost": "łączny koszt",
    "cost_per_node": "koszt na węzeł",
}

GRAPH_LABELS = {
    "random": "losowy",
    "small_world": "małoświatowy",
    "scale_free": "bezskalowy",
}

DATASET_LABELS = {
    "duolingo_super": "Duolingo Super",
    "roman_domination": "Dominowanie rzymskie",
}

ALGORITHM_ORDER_DISPLAY = [algorithm_display_name(name) for name in ALGORITHM_CANONICAL_ORDER]
SAMPLE_ID_COLS = list(SAMPLE_IDENTIFIER_COLUMNS)


def _axis_label(metric: str, aggregate: str | None = None) -> str:
    base = METRIC_DISPLAY.get(metric, metric.replace("_", " "))
    if aggregate == "mean":
        return f"Średni {base}"
    return base.capitalize()


def _mean_gap(df: pd.DataFrame, value: str, group: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="algorithm", columns=group, values=value, aggfunc="mean")
    pivot["range"] = pivot.max(axis=1) - pivot.min(axis=1)
    pivot["spread_ratio"] = pivot.max(axis=1) / pivot.min(axis=1)
    return pivot.sort_values("range", ascending=False)


def _plot_metric_vs_nodes(df: pd.DataFrame, value: str, title: str, filename: str) -> None:
    filtered_df, _ = filter_complete_samples(
        df,
        SAMPLE_ID_COLS,
        value_cols=[value],
        warn_label=title,
    )
    aggregated = filtered_df.groupby(["algorithm", "n_nodes"], as_index=False).agg(mean=(value, "mean"))
    if aggregated.empty:
        warnings.warn(
            f"Brak danych do wykresu '{title}' po odfiltrowaniu niekompletnych próbek.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    algo_values = aggregated["algorithm"].unique().tolist()
    palette = algorithm_palette(algo_values)
    order = [name for name in ALGORITHM_ORDER_DISPLAY if name in algo_values]
    fig, ax = plt.subplots()
    sns.lineplot(
        data=aggregated,
        x="n_nodes",
        y="mean",
        hue="algorithm",
        hue_order=order if order else None,
        marker="o",
        palette=palette,
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


def _plot_metric_by_graph(
    df: pd.DataFrame,
    value: str,
    title_prefix: str,
    filename_prefix: str,
    ymax: float | None = None,
) -> None:
    for graph_key, graph_label in GRAPH_LABELS.items():
        subset = df[df["graph_key"] == graph_key]
        if subset.empty:
            continue
        filtered_subset, _ = filter_complete_samples(
            subset,
            SAMPLE_ID_COLS,
            value_cols=[value],
            warn_label=f"{title_prefix} ({graph_label})",
        )
        if filtered_subset.empty:
            continue
        algo_values = filtered_subset["algorithm"].unique().tolist()
        palette = algorithm_palette(algo_values)
        order = [name for name in ALGORITHM_ORDER_DISPLAY if name in algo_values]
        fig, ax = plt.subplots()
        sns.barplot(
            data=filtered_subset,
            x="algorithm",
            y=value,
            hue="algorithm",
            order=order if order else None,
            hue_order=order if order else None,
            palette=palette,
            legend=False,
            estimator=np.mean,
            errorbar="se",
            ax=ax,
        )
        ax.set_title(f"{title_prefix} - {graph_label}")
        ax.set_xlabel("Algorytm")
        ax.set_ylabel(_axis_label(value, "mean"))
        ax.set_ylim(bottom=0)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"{filename_prefix}_{graph_key}.pdf")
        plt.close(fig)


def _plot_duo_vs_roman(df: pd.DataFrame, value: str, title: str, filename: str) -> None:
    filtered_parts: list[pd.DataFrame] = []
    for dataset_name, dataset_frame in df.groupby("dataset"):
        filtered_dataset, _ = filter_complete_samples(
            dataset_frame,
            SAMPLE_ID_COLS,
            value_cols=[value],
            warn_label=f"{title} ({dataset_name})",
        )
        if filtered_dataset.empty:
            continue
        filtered_parts.append(filtered_dataset)

    if not filtered_parts:
        warnings.warn(
            f"Brak danych do wykresu '{title}' po odfiltrowaniu niekompletnych próbek.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    plot_df = pd.concat(filtered_parts, ignore_index=True)
    fig, ax = plt.subplots()
    order = [label for label in GRAPH_LABELS.values() if label in plot_df["graph"].unique()]
    sns.barplot(
        data=plot_df,
        x="graph",
        y=value,
        hue="dataset",
        estimator=np.mean,
        errorbar="se",
        order=order if order else None,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Typ grafu")
    ax.set_ylabel(_axis_label(value, "mean"))
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=15)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Zbiór danych")
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def _plot_license_mix(summary: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots()
    indices = np.arange(len(summary))
    width = 0.6
    ax.bar(
        indices,
        summary["group_share"],
        width,
        label="Licencje grupowe",
        color="#4477aa",
    )
    ax.bar(
        indices,
        summary["individual_share"],
        width,
        bottom=summary["group_share"],
        label="Licencje indywidualne",
        color="#ee6677",
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(summary["dataset"], rotation=20)
    ax.set_ylabel("Udział licencji")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def main() -> None:
    apply_plot_style()

    duolingo_raw = expand_license_counts(load_dataset(DATA_DIR / "duolingo_df.csv"))
    duolingo, duolingo_unit_costs = normalize_cost_columns(duolingo_raw, attach_group_multiplier=True)
    duolingo = apply_algorithm_labels(duolingo)
    duolingo["graph_key"] = duolingo["graph"]
    duolingo["graph"] = duolingo["graph_key"].map(GRAPH_LABELS).fillna(duolingo["graph_key"])
    duolingo["dataset"] = DATASET_LABELS["duolingo_super"]

    roman_raw = expand_license_counts(load_dataset(DATA_DIR / "roman_df.csv"))
    roman, roman_unit_costs = normalize_cost_columns(roman_raw, attach_group_multiplier=True)
    roman = apply_algorithm_labels(roman)
    roman["graph_key"] = roman["graph"]
    roman["graph"] = roman["graph_key"].map(GRAPH_LABELS).fillna(roman["graph_key"])
    roman["dataset"] = DATASET_LABELS["roman_domination"]

    timeout_raw = expand_license_counts(load_dataset(DATA_DIR / "timeout_df.csv"))
    combined_unit_costs = {**duolingo_unit_costs, **roman_unit_costs}
    timeout, _ = normalize_cost_columns(
        timeout_raw,
        unit_costs=combined_unit_costs,
        attach_group_multiplier=True,
    )
    timeout = apply_algorithm_labels(timeout)
    timeout["graph_key"] = timeout["graph"]
    timeout["graph"] = timeout["graph_key"].map(GRAPH_LABELS).fillna(timeout["graph_key"])

    combined = pd.concat([duolingo, roman], ignore_index=True)

    numeric_cols = [
        "time_s",
        "total_cost",
        "cost_per_node",
        "license_group",
        "license_individual",
    ]
    if "license_group_multiplier" in duolingo.columns:
        numeric_cols.append("license_group_multiplier")

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

    timeout_counts_algo = timeout.groupby("algorithm").size().rename("count").sort_values(ascending=False)
    timeout_counts_algo.to_csv(TAB_DIR / "timeouts_by_algorithm.csv")

    timeout_counts_graph = timeout.groupby(["algorithm", "graph"]).size().rename("count")
    timeout_counts_graph.to_csv(TAB_DIR / "timeouts_by_algorithm_graph.csv")

    timeout_license = timeout.groupby(["algorithm", "license_config"]).size().rename("count")
    timeout_license.to_csv(TAB_DIR / "timeouts_by_algorithm_license.csv")

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

    id_cols = list(SAMPLE_ID_COLS)
    pivot_time = pivot_complete_blocks(duolingo, id_cols, "algorithm", "time_s")
    pivot_cost = pivot_complete_blocks(duolingo, id_cols, "algorithm", "cost_per_node")

    friedman_reports: list[str] = []
    time_result = run_friedman_nemenyi(pivot_time)
    if time_result:
        save_table(
            time_result.mean_ranks.to_frame("mean_rank"),
            TAB_DIR / "friedman_time_mean_ranks.csv",
        )
        if time_result.nemenyi is not None:
            save_table(time_result.nemenyi, TAB_DIR / "nemenyi_time_pvalues.csv")
        friedman_reports.append(f"Friedman test (time_s): statistic={time_result.statistic:.3f}, p-value={time_result.pvalue:.4g}")
    else:
        friedman_reports.append("Friedman test (time_s): insufficient paired samples")

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
        friedman_reports.append(f"Friedman test (cost_per_node): statistic={cost_result.statistic:.3f}, p-value={cost_result.pvalue:.4g}")
    else:
        friedman_reports.append("Friedman test (cost_per_node): insufficient paired samples")

    # Visualisations
    _plot_metric_vs_nodes(
        duolingo,
        "time_s",
        "Duolingo: czas vs liczba węzłów",
        "duolingo_time_vs_nodes.pdf",
    )
    _plot_metric_vs_nodes(
        duolingo,
        "total_cost",
        "Duolingo: koszt vs liczba węzłów",
        "duolingo_cost_vs_nodes.pdf",
    )
    _plot_metric_vs_nodes(
        duolingo,
        "cost_per_node",
        "Duolingo: koszt/węzeł vs liczba węzłów",
        "duolingo_cost_per_node_vs_nodes.pdf",
    )
    _plot_metric_by_graph(
        duolingo,
        "time_s",
        "Duolingo: czasy według typu grafu",
        "duolingo_time_by_graph",
        ymax=15.0,
    )
    _plot_metric_by_graph(
        duolingo,
        "cost_per_node",
        "Duolingo: koszt/węzeł według grafu",
        "duolingo_cost_per_node_by_graph",
    )

    pareto_fig, ax = plt.subplots()
    pareto_algos = duolingo["algorithm"].unique().tolist()
    sns.scatterplot(
        data=duolingo,
        x="total_cost",
        y="time_s",
        hue="algorithm",
        palette=algorithm_palette(pareto_algos),
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
    ax.set_title("Duolingo: Pareto koszt-czas")
    ax.set_xlabel("Łączny koszt")
    ax.set_ylabel("Czas [s]")
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Algorytm")
    plt.tight_layout()
    pareto_fig.savefig(FIG_DIR / "duolingo_pareto_cost_time.pdf")
    plt.close(pareto_fig)

    duo_license_totals = duolingo[["license_group", "license_individual", "license_other"]].sum()
    roman_license_totals = roman[["license_group", "license_individual", "license_other"]].sum()
    license_summary = pd.DataFrame(
        {
            "dataset": [
                DATASET_LABELS["duolingo_super"],
                DATASET_LABELS["roman_domination"],
            ],
            "group": [
                duo_license_totals["license_group"],
                roman_license_totals["license_group"],
            ],
            "individual": [
                duo_license_totals["license_individual"],
                roman_license_totals["license_individual"],
            ],
            "other": [
                duo_license_totals["license_other"],
                roman_license_totals["license_other"],
            ],
        }
    )
    for column in ["group", "individual", "other"]:
        license_summary[column] = license_summary[column].round().astype(int)
    license_summary["total"] = license_summary[["group", "individual", "other"]].sum(axis=1)
    license_summary["group_share"] = license_summary["group"] / license_summary["total"]
    license_summary["individual_share"] = license_summary["individual"] / license_summary["total"]
    save_table(license_summary, TAB_DIR / "license_mix_duo_vs_roman.csv")
    _plot_license_mix(license_summary, "license_mix_duo_vs_roman.pdf")

    license_summary_words = license_summary[["dataset", "group", "individual", "other"]].copy()
    license_summary_words["group_slowo"] = license_summary_words["group"].apply(number_to_polish_words)
    license_summary_words["individual_slowo"] = license_summary_words["individual"].apply(number_to_polish_words)
    license_summary_words["other_slowo"] = license_summary_words["other"].apply(number_to_polish_words)
    save_table(license_summary_words, TAB_DIR / "license_mix_duo_vs_roman_words.csv")
    license_text_lines = ["# Udział licencji (słownie)", ""]
    for _, row in license_summary_words.iterrows():
        license_text_lines.append(f"- {row['dataset']}: {row['group_slowo']} grupowych, {row['individual_slowo']} indywidualnych, {row['other_slowo']} pozostałych")
    write_text(REPORTS_DIR / "license_mix_duo_vs_roman.md", license_text_lines)

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
        "Porównanie czasów według grafu",
        "duo_vs_roman_time_by_graph.pdf",
    )
    _plot_duo_vs_roman(
        combined,
        "cost_per_node",
        "Porównanie kosztu/węzeł według grafu",
        "duo_vs_roman_cost_per_node_by_graph.pdf",
    )

    # Summary report
    lines: list[str] = []
    lines.append("# Benchmark - podsumowanie")
    lines.append("")
    duo_label = DATASET_LABELS["duolingo_super"]
    roman_label = DATASET_LABELS["roman_domination"]
    lines.append(f"{duo_label} - średni czas: {duo_overall.loc[duo_overall['metric'] == 'time_s', 'mean'].iloc[0]:.2f} s")
    lines.append(f"{duo_label} - średni koszt/węzeł: {duo_overall.loc[duo_overall['metric'] == 'cost_per_node', 'mean'].iloc[0]:.3f}")
    lines.append(f"{roman_label} - średni czas: {roman_overall.loc[roman_overall['metric'] == 'time_s', 'mean'].iloc[0]:.2f} s")
    lines.append(f"{roman_label} - średni koszt/węzeł: {roman_overall.loc[roman_overall['metric'] == 'cost_per_node', 'mean'].iloc[0]:.3f}")
    lines.append("")
    lines.extend(friedman_reports)
    lines.append("")
    lines.append(f"Największe rozpiętości średniego czasu ({duo_label}):")
    time_gap = _mean_gap(duolingo, "time_s", "graph")["range"].head(5)
    for alg, value in time_gap.items():
        lines.append(f"- {alg}: {value:.2f} s")
    lines.append("")
    lines.append("Rozkład udziału licencji: license_mix_duo_vs_roman.csv oraz license_mix_duo_vs_roman_words.csv")
    write_text(REPORTS_DIR / "summary.md", lines)


if __name__ == "__main__":
    main()
