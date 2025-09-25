from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glopt.analysis.commons import (
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
DATA_DIR = BASE_DIR / "data" / "dynamic"
RESULTS_DIR = ensure_dir(BASE_DIR / "results" / "dynamic")
FIG_DIR = ensure_dir(RESULTS_DIR / "figures")
TAB_DIR = ensure_dir(RESULTS_DIR / "tables")
REPORTS_DIR = ensure_dir(RESULTS_DIR / "reports")
TARGET_ALGOS = ["GeneticAlgorithm", "AntColonyOptimization", "TabuSearch"]

TARGET_ALGOS_DISPLAY = [algorithm_display_name(name) for name in TARGET_ALGOS]

METHOD_LABELS = {
    "low": "low",
    "med": "med",
    "high": "high",
    "pref_pref": "pref_pref",
    "pref_triadic": "pref_triadic",
    "rand_rewire": "rand_rewire",
}

SOURCE_LABELS = {
    "real": "Dane rzeczywiste",
    "synthetic": "Dane syntetyczne",
}

WARM_LABELS = {True: "ciepły start", False: "zimny start"}
WARM_LABEL_WARM = WARM_LABELS[True]
WARM_LABEL_COLD = WARM_LABELS[False]

METRIC_DISPLAY = {
    "time_s": "czas [s]",
    "total_cost": "łączny koszt",
    "cost_per_node": "koszt na węzeł",
}


def _slugify_label(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(label))
    ascii_label = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_label.replace(" ", "_").lower()


def _axis_label(metric: str, aggregate: str | None = None) -> str:
    base = METRIC_DISPLAY.get(metric, metric.replace("_", " "))
    if aggregate == "mean":
        return f"Średni {base}"
    return base.capitalize()


def extract_variant(run_id: str) -> str:
    if not isinstance(run_id, str):
        return "unknown"
    match = re.search(r"dynamic_(.+?)_dynamic", run_id)
    if match:
        return match.group(1)
    return "unknown"


def assign_source(variant: str) -> str:
    return "synthetic" if variant in {"low", "med", "high"} else "real"


def add_labels(df: pd.DataFrame, source_hint: str | None = None) -> pd.DataFrame:
    result = df.copy()
    result["variant"] = result["run_id"].apply(extract_variant)
    if source_hint:
        result["source"] = source_hint
    else:
        result["source"] = result["variant"].apply(assign_source)
    result["warm_label"] = result["warm_start"].map(WARM_LABELS)
    result["step"] = result.get("step", 0).fillna(0).astype(int)
    return result


def _plot_metric_vs_nodes(df: pd.DataFrame, source: str, value: str, filename: str) -> None:
    subset = df[df["source"] == source]
    if subset.empty:
        return
    aggregated = subset.groupby(["algorithm", "n_nodes"], as_index=False).agg(mean=(value, "mean"))
    fig, ax = plt.subplots()
    algo_values = aggregated["algorithm"].unique().tolist()
    palette = algorithm_palette(algo_values)
    sns.lineplot(
        data=aggregated,
        x="n_nodes",
        y="mean",
        hue="algorithm",
        palette=palette,
        marker="o",
        ax=ax,
    )
    source_title = SOURCE_LABELS.get(source, source)
    ax.set_title(f"{source_title} - średni {METRIC_DISPLAY.get(value, value)} względem liczby węzłów")
    ax.set_xlabel("Liczba węzłów")
    ax.set_ylabel(_axis_label(value, "mean"))
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Algorytm")
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def _plot_metric_over_steps(
    df: pd.DataFrame,
    source: str,
    algorithm: str,
    metric: str,
    filename_prefix: str,
) -> None:
    subset = df[(df["source"] == source) & (df["algorithm"] == algorithm)]
    if subset.empty:
        return
    aggregated = subset.groupby(["variant", "step", "warm_label"], as_index=False).agg(mean=(metric, "mean"))
    if aggregated.empty:
        return
    source_title = SOURCE_LABELS.get(source, source)
    metric_title = METRIC_DISPLAY.get(metric, metric)
    for variant, frame in aggregated.groupby("variant"):
        if frame.empty:
            continue
        fig, ax = plt.subplots()
        sns.lineplot(data=frame, x="step", y="mean", hue="warm_label", marker="o", ax=ax)
        ax.set_title(f"{source_title} - {algorithm} ({variant}): {metric_title} w krokach")
        ax.set_xlabel("Krok symulacji")
        ax.set_ylabel(_axis_label(metric, "mean"))
        ax.set_ylim(bottom=0)
        mean_value = frame["mean"].mean()
        if mean_value and np.isfinite(mean_value):
            upper = max(mean_value * 2.0, frame["mean"].max())
            if upper > 0:
                ax.set_ylim(0, upper)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Tryb startu")
        else:
            legend = ax.get_legend()
            if legend:
                legend.set_visible(False)
        safe_variant = str(variant).replace("/", "_").replace(" ", "_")
        fig.savefig(FIG_DIR / f"{filename_prefix}_{safe_variant}.pdf")
        plt.close(fig)


def compute_warm_cold_delta(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = df.groupby(
        ["source", "variant", "algorithm", "step", "warm_label"],
        as_index=False,
    ).agg(mean=(metric, "mean"))
    pivot = grouped.pivot_table(
        index=["source", "variant", "algorithm", "step"],
        columns="warm_label",
        values="mean",
    )
    if WARM_LABEL_WARM not in pivot.columns or WARM_LABEL_COLD not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.rename(columns={WARM_LABEL_WARM: "mean_cieply", WARM_LABEL_COLD: "mean_zimny"})
    pivot = pivot.dropna(subset=["mean_cieply", "mean_zimny"], how="any")
    pivot["delta"] = pivot["mean_cieply"] - pivot["mean_zimny"]
    return pivot.reset_index()


def main() -> None:
    apply_plot_style()

    real_raw = expand_license_counts(load_dataset(DATA_DIR / "real_df.csv"))
    synthetic_raw = expand_license_counts(load_dataset(DATA_DIR / "synthetic_df.csv"))
    timeout_raw = expand_license_counts(load_dataset(DATA_DIR / "timeout_df.csv"))

    baseline_df = pd.concat([real_raw, synthetic_raw], ignore_index=True)
    _, unit_costs = normalize_cost_columns(baseline_df, attach_group_multiplier=True)

    real_norm, _ = normalize_cost_columns(real_raw, unit_costs=unit_costs, attach_group_multiplier=True)
    synthetic_norm, _ = normalize_cost_columns(synthetic_raw, unit_costs=unit_costs, attach_group_multiplier=True)
    timeout_norm, _ = normalize_cost_columns(timeout_raw, unit_costs=unit_costs, attach_group_multiplier=True)

    real_df = apply_algorithm_labels(add_labels(real_norm, "real"))
    synthetic_df = apply_algorithm_labels(add_labels(synthetic_norm, "synthetic"))
    timeout_df = apply_algorithm_labels(add_labels(timeout_norm))

    combined = pd.concat([real_df, synthetic_df], ignore_index=True)
    combined["method"] = combined["variant"].map(METHOD_LABELS).fillna(combined["variant"].where(combined["variant"].notna(), "unknown"))

    metrics = ["time_s", "total_cost", "cost_per_node"]
    real_overall = describe_numeric(real_df, metrics)
    synthetic_overall = describe_numeric(synthetic_df, metrics)
    save_table(real_overall, TAB_DIR / "real_overall_stats.csv")
    save_table(synthetic_overall, TAB_DIR / "synthetic_overall_stats.csv")

    real_algo = describe_numeric(real_df, metrics, ["algorithm", "variant", "warm_label"])
    synthetic_algo = describe_numeric(synthetic_df, metrics, ["algorithm", "variant", "warm_label"])
    save_table(real_algo, TAB_DIR / "real_algorithm_variant_warm_stats.csv")
    save_table(synthetic_algo, TAB_DIR / "synthetic_algorithm_variant_warm_stats.csv")

    real_mutation_means = (
        real_df.groupby("variant", dropna=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            mean_cost_per_node=("cost_per_node", "mean"),
            mean_time_s=("time_s", "mean"),
        )
        .reset_index()
        .rename(columns={"variant": "mutation_method"})
    )
    save_table(
        real_mutation_means.set_index("mutation_method"),
        TAB_DIR / "real_mutation_means.csv",
    )

    combined_algo = describe_numeric(combined, metrics, ["source", "algorithm", "variant", "warm_label"])
    save_table(combined_algo, TAB_DIR / "combined_algorithm_variant_warm_stats.csv")
    combined_variant = describe_numeric(combined, metrics, ["source", "variant"])
    save_table(combined_variant, TAB_DIR / "combined_variant_stats.csv")

    method_whitelist = set(METHOD_LABELS.values())
    method_summary = (
        combined[combined["method"].isin(method_whitelist)]
        .groupby(["algorithm", "method"], dropna=False, as_index=False)
        .agg(
            koszt_per_node=("cost_per_node", "mean"),
            sredni_czas=("time_s", "mean"),
        )
        .sort_values(["algorithm", "method"])
    )
    if not method_summary.empty:
        export_table = method_summary.rename(
            columns={
                "algorithm": "algorytm",
                "method": "metoda mutacji",
                "koszt_per_node": "koszt per node",
                "sredni_czas": "sredni czas",
            }
        )
        save_table(
            export_table.set_index(["algorytm", "metoda mutacji"]),
            TAB_DIR / "algorithm_method_mean_cost_time.csv",
        )

    timeout_counts = timeout_df.groupby(["source", "variant", "algorithm"]).size().rename("count").sort_values(ascending=False)
    timeout_counts.to_csv(TAB_DIR / "timeouts_by_source_variant_algorithm.csv")
    timeout_graph = timeout_df.groupby(["source", "variant", "algorithm", "graph"]).size().rename("count")
    timeout_graph.to_csv(TAB_DIR / "timeouts_by_source_variant_algorithm_graph.csv")

    for algo in TARGET_ALGOS_DISPLAY:
        safe_name = _slugify_label(algo)
        _plot_metric_over_steps(
            combined,
            "real",
            algo,
            "total_cost",
            f"real_{safe_name}_cost_over_steps",
        )
        _plot_metric_over_steps(
            combined,
            "real",
            algo,
            "time_s",
            f"real_{safe_name}_time_over_steps",
        )
        _plot_metric_over_steps(
            combined,
            "synthetic",
            algo,
            "total_cost",
            f"synthetic_{safe_name}_cost_over_steps",
        )
        _plot_metric_over_steps(
            combined,
            "synthetic",
            algo,
            "time_s",
            f"synthetic_{safe_name}_time_over_steps",
        )

    greedy_display = algorithm_display_name("GreedyAlgorithm")
    greedy_safe = _slugify_label(greedy_display)
    _plot_metric_over_steps(
        combined,
        "real",
        greedy_display,
        "total_cost",
        f"real_{greedy_safe}_cost_over_steps",
    )
    _plot_metric_over_steps(
        combined,
        "real",
        greedy_display,
        "time_s",
        f"real_{greedy_safe}_time_over_steps",
    )
    _plot_metric_over_steps(
        combined,
        "synthetic",
        greedy_display,
        "total_cost",
        f"synthetic_{greedy_safe}_cost_over_steps",
    )
    _plot_metric_over_steps(
        combined,
        "synthetic",
        greedy_display,
        "time_s",
        f"synthetic_{greedy_safe}_time_over_steps",
    )

    delta_cost = compute_warm_cold_delta(combined, "total_cost")
    delta_time = compute_warm_cold_delta(combined, "time_s")
    if not delta_cost.empty:
        save_table(delta_cost, TAB_DIR / "warm_vs_cold_delta_cost.csv")
    if not delta_time.empty:
        save_table(delta_time, TAB_DIR / "warm_vs_cold_delta_time.csv")

    summary_lines: list[str] = []
    summary_lines.append("# Dynamiczne mutacje - podsumowanie")
    summary_lines.append("")

    real_time_target = real_algo[(real_algo["metric"] == "time_s") & (real_algo["algorithm"].isin(TARGET_ALGOS_DISPLAY))].sort_values("mean")
    if not real_time_target.empty:
        summary_lines.append("Dane rzeczywiste - czasy dla algorytmów docelowych:")
        for _, row in real_time_target.head(6).iterrows():
            summary_lines.append(f"- {row['algorithm']} ({row['variant']}, {row['warm_label']}): średnia {row['mean']:.2f} s")
        summary_lines.append("")

    synthetic_cpn = synthetic_algo[(synthetic_algo["metric"] == "cost_per_node") & (synthetic_algo["algorithm"].isin(TARGET_ALGOS_DISPLAY))].sort_values("mean")
    if not synthetic_cpn.empty:
        summary_lines.append("Dane syntetyczne - koszt/węzeł dla algorytmów docelowych:")
        for _, row in synthetic_cpn.head(6).iterrows():
            summary_lines.append(f"- {row['algorithm']} ({row['variant']}, {row['warm_label']}): średnia {row['mean']:.2f}")
        summary_lines.append("")

    summary_lines.append("Ciepły vs zimny start - koszt: ujemna delta oznacza przewagę ciepłego startu")
    if delta_cost.empty:
        summary_lines.append("- Brak pełnych par ciepły/zimny start dla wskazanych algorytmów w danych")
    else:
        key_deltas = delta_cost.groupby(["source", "algorithm"]).agg(mean_delta=("delta", "mean")).reset_index()
        for _, row in key_deltas.sort_values("mean_delta").head(6).iterrows():
            source_title = SOURCE_LABELS.get(row["source"], row["source"])
            summary_lines.append(f"- {source_title} {row['algorithm']}: średnia różnica {row['mean_delta']:.2f}")
    summary_lines.append("")
    summary_lines.append("Statystyki szczegółowe zapisano w katalogu tables/")
    write_text(REPORTS_DIR / "summary.md", summary_lines)


if __name__ == "__main__":
    main()
