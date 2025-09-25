from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glopt.analysis.commons import (
    ALGORITHM_CANONICAL_ORDER,
    algorithm_display_name,
    algorithm_palette,
    apply_algorithm_labels,
    apply_plot_style,
    describe_numeric,
    ensure_dir,
    expand_license_counts,
    load_dataset,
    normalize_cost_columns,
    save_table,
    write_text,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "extensions"
RESULTS_DIR = ensure_dir(BASE_DIR / "results" / "extensions")
FIG_DIR = ensure_dir(RESULTS_DIR / "figures")
TAB_DIR = ensure_dir(RESULTS_DIR / "tables")
REPORTS_DIR = ensure_dir(RESULTS_DIR / "reports")

LICENSE_FAMILY = {
    "duolingo_p_2": "duolingo",
    "duolingo_p_4": "duolingo",
    "duolingo_p_5": "duolingo",
    "roman_p_3": "roman",
    "roman_p_4": "roman",
    "roman_p_5": "roman",
    "netflix": "netflix",
    "spotify": "spotify",
}

TARGET_ALGOS = ["AntColonyOptimization", "GeneticAlgorithm", "TabuSearch"]

TARGET_ALGOS_DISPLAY = [algorithm_display_name(name) for name in TARGET_ALGOS]
ALGORITHM_ORDER_DISPLAY = [algorithm_display_name(name) for name in ALGORITHM_CANONICAL_ORDER]

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


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["license_family"] = result["license_config"].map(LICENSE_FAMILY).fillna("other")
    result["variant"] = result["license_config"].str.replace(".*p_", "p_", regex=True)
    result["variant"] = result.apply(
        lambda row: row["license_config"] if row["variant"] == row["license_config"] else row["variant"],
        axis=1,
    )
    return result


def _plot_metric_by_license(
    df: pd.DataFrame,
    metric: str,
    filename: str,
    title: str,
    focus_algos: list[str] | None = None,
) -> None:
    subset = df if not focus_algos else df[df["algorithm"].isin(focus_algos)]
    if subset.empty:
        return
    fig, ax = plt.subplots()
    algo_values = subset["algorithm"].unique().tolist()
    palette = algorithm_palette(algo_values)
    order = [name for name in ALGORITHM_ORDER_DISPLAY if name in algo_values]
    sns.barplot(
        data=subset,
        x="license_config",
        y=metric,
        hue="algorithm",
        hue_order=order if order else None,
        palette=palette,
        estimator=np.mean,
        errorbar="se",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Konfiguracja licencji")
    ax.set_ylabel(_axis_label(metric, "mean"))
    ax.tick_params(axis="x", rotation=25)
    if metric == "time_s" and not focus_algos:
        ax.set_ylim(0, 7.5)
    else:
        ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Algorytm")
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def _plot_family_comparison(df: pd.DataFrame, family: str, metric: str, filename: str, title: str) -> None:
    subset = df[df["license_family"] == family]
    if subset.empty:
        return
    fig, ax = plt.subplots()
    sns.barplot(
        data=subset,
        x="license_config",
        y=metric,
        hue="source",
        estimator=np.mean,
        errorbar="se",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Konfiguracja licencji")
    ax.set_ylabel(_axis_label(metric, "mean"))
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Źródło danych")
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def main() -> None:
    apply_plot_style()

    extensions_raw = expand_license_counts(load_dataset(DATA_DIR / "df.csv"))
    timeout_raw = expand_license_counts(load_dataset(DATA_DIR / "timeout_df.csv"))

    baseline_raw = pd.concat([extensions_raw, timeout_raw], ignore_index=True)
    _, unit_costs = normalize_cost_columns(baseline_raw, attach_group_multiplier=True)

    extensions_norm, _ = normalize_cost_columns(extensions_raw, unit_costs=unit_costs, attach_group_multiplier=True)
    timeout_norm, _ = normalize_cost_columns(timeout_raw, unit_costs=unit_costs, attach_group_multiplier=True)

    extensions = apply_algorithm_labels(add_metadata(extensions_norm))
    timeout = apply_algorithm_labels(add_metadata(timeout_norm))

    metrics = [
        "time_s",
        "total_cost",
        "cost_per_node",
        "license_group",
        "license_individual",
    ]

    overall_stats = describe_numeric(extensions, metrics, ["license_config"])
    save_table(overall_stats, TAB_DIR / "overall_stats_by_license.csv")

    algo_stats = describe_numeric(extensions, metrics, ["license_config", "algorithm"])
    save_table(algo_stats, TAB_DIR / "algo_stats_by_license.csv")

    family_stats = describe_numeric(
        extensions,
        ["time_s", "total_cost", "cost_per_node"],
        ["license_family", "algorithm"],
    )
    save_table(family_stats, TAB_DIR / "family_stats.csv")

    license_mix = extensions.groupby("license_config")[["license_group", "license_individual", "license_other"]].sum()
    license_mix["total"] = license_mix.sum(axis=1)
    license_mix["group_share"] = license_mix["license_group"] / license_mix["total"]
    license_mix["individual_share"] = license_mix["license_individual"] / license_mix["total"]
    save_table(license_mix, TAB_DIR / "license_mix_summary.csv")

    timeout_counts = timeout.groupby(["license_config", "algorithm"]).size().rename("count").sort_values(ascending=False)
    timeout_counts.to_csv(TAB_DIR / "timeouts_by_license_algorithm.csv")
    timeout_graph = timeout.groupby(["license_config", "algorithm", "graph"]).size().rename("count")
    timeout_graph.to_csv(TAB_DIR / "timeouts_by_license_algorithm_graph.csv")

    _plot_metric_by_license(
        extensions,
        "time_s",
        "time_by_license_all_algos.pdf",
        "Czas vs konfiguracja licencji",
    )
    _plot_metric_by_license(
        extensions,
        "cost_per_node",
        "cost_per_node_by_license_targets.pdf",
        "Koszt/węzeł dla kluczowych algorytmów",
        TARGET_ALGOS_DISPLAY,
    )

    # Porównanie z benchmarkiem na wspólnych n_nodes
    bench_duo_raw = expand_license_counts(load_dataset(BASE_DIR / "data" / "benchmark" / "duolingo_df.csv"))
    bench_duo_norm, _ = normalize_cost_columns(bench_duo_raw, attach_group_multiplier=True)
    bench_duo = apply_algorithm_labels(bench_duo_norm)
    bench_roman_raw = expand_license_counts(load_dataset(BASE_DIR / "data" / "benchmark" / "roman_df.csv"))
    bench_roman_norm, _ = normalize_cost_columns(bench_roman_raw, attach_group_multiplier=True)
    bench_roman = apply_algorithm_labels(bench_roman_norm)

    target_nodes = sorted(set(extensions["n_nodes"].unique()) & set(bench_duo["n_nodes"].unique()))
    bench_duo = bench_duo[bench_duo["n_nodes"].isin(target_nodes)].copy()
    bench_roman = bench_roman[bench_roman["n_nodes"].isin(target_nodes)].copy()

    bench_duo["source"] = "benchmark_duolingo_super"
    bench_roman["source"] = "benchmark_roman_domination"

    ext_duo = extensions[extensions["license_family"] == "duolingo"].copy()
    ext_duo["source"] = ext_duo["license_config"]

    ext_roman = extensions[extensions["license_family"] == "roman"].copy()
    ext_roman["source"] = ext_roman["license_config"]

    duo_compare = pd.concat([bench_duo, ext_duo], ignore_index=True)
    roman_compare = pd.concat([bench_roman, ext_roman], ignore_index=True)

    duo_stats = describe_numeric(
        duo_compare,
        ["time_s", "total_cost", "cost_per_node"],
        ["source", "algorithm"],
    )
    roman_stats = describe_numeric(
        roman_compare,
        ["time_s", "total_cost", "cost_per_node"],
        ["source", "algorithm"],
    )
    save_table(duo_stats, TAB_DIR / "duolingo_family_comparison.csv")
    save_table(roman_stats, TAB_DIR / "roman_family_comparison.csv")

    _plot_family_comparison(
        duo_compare,
        "duolingo",
        "cost_per_node",
        "duolingo_cost_per_node_comparison.pdf",
        "Duolingo: koszt/węzeł",
    )
    _plot_family_comparison(
        duo_compare,
        "duolingo",
        "time_s",
        "duolingo_time_comparison.pdf",
        "Duolingo: czas",
    )
    _plot_family_comparison(
        roman_compare,
        "roman",
        "cost_per_node",
        "roman_cost_per_node_comparison.pdf",
        "Roman: koszt/węzeł",
    )
    _plot_family_comparison(
        roman_compare,
        "roman",
        "time_s",
        "roman_time_comparison.pdf",
        "Roman: czas",
    )

    summary_lines: list[str] = []
    summary_lines.append("# Extensions - podsumowanie")
    summary_lines.append("")
    summary_lines.append("Konfiguracje duolingo (p_2/p_4/p_5) redukują koszt/węzeł względem benchmarku:")
    duo_cpn = duo_stats[(duo_stats["metric"] == "cost_per_node") & (duo_stats["source"] != "benchmark_duolingo_super")]
    duo_base = duo_stats[(duo_stats["metric"] == "cost_per_node") & (duo_stats["source"] == "benchmark_duolingo_super")].groupby("algorithm")["mean"].mean()
    for _, row in duo_cpn.sort_values("mean").head(5).iterrows():
        baseline = duo_base.get(row["algorithm"], np.nan)
        baseline_txt = f"{baseline:.2f}" if not np.isnan(baseline) else "n/a"
        summary_lines.append(f"- {row['source']} / {row['algorithm']}: średnia {row['mean']:.2f} vs benchmark {baseline_txt}")
    summary_lines.append("")
    summary_lines.append("Konfiguracje roman (p_3/p_4/p_5) względem benchmarku:")
    roman_cpn = roman_stats[(roman_stats["metric"] == "cost_per_node") & (roman_stats["source"] != "benchmark_roman_domination")]
    base_roman_med = roman_stats[(roman_stats["metric"] == "cost_per_node") & (roman_stats["source"] == "benchmark_roman_domination")].groupby("algorithm")["mean"].mean()
    for _, row in roman_cpn.sort_values("mean").head(5).iterrows():
        baseline = base_roman_med.get(row["algorithm"], np.nan)
        baseline_txt = f"{baseline:.2f}" if not np.isnan(baseline) else "n/a"
        summary_lines.append(f"- {row['source']} / {row['algorithm']}: średnia {row['mean']:.2f} vs benchmark {baseline_txt}")
    summary_lines.append("")
    summary_lines.append("Szczegółowe statystyki i wykresy zapisano w katalogach tables/ oraz figures/")
    write_text(REPORTS_DIR / "summary.md", summary_lines)


if __name__ == "__main__":
    main()
