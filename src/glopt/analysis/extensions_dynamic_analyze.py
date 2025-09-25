from __future__ import annotations

import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glopt.analysis.commons import (
    ALGORITHM_CANONICAL_ORDER,
    algorithm_display_name,
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

MUTATION_VARIANT_GROUPS = {
    "low": ["spotify", "duolingo_p_2"],
    "med": ["duolingo_p_4"],
    "high": ["netflix", "duolingo_p_5"],
    "pref_pref": ["roman_p_3"],
    "pref_triadic": ["roman_p_4"],
    "rand_rewire": ["roman_p_5"],
}

MUTATION_VARIANT_ORDER = list(MUTATION_VARIANT_GROUPS.keys())

MUTATION_VARIANT_DISPLAY = {variant: f"{variant} ({', '.join(groups)})" for variant, groups in MUTATION_VARIANT_GROUPS.items()}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "extensions_dynamic"
RESULTS_DIR = ensure_dir(BASE_DIR / "results" / "extensions_dynamic")
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

LICENSE_MUTATION_VARIANT: dict[str, str] = {}
for variant, configs in MUTATION_VARIANT_GROUPS.items():
    for config in configs:
        LICENSE_MUTATION_VARIANT[config] = variant

TARGET_ALGOS = ["AntColonyOptimization", "GeneticAlgorithm", "TabuSearch"]

TARGET_ALGOS_DISPLAY = [algorithm_display_name(name) for name in TARGET_ALGOS]
ALGORITHM_ORDER_DISPLAY = [algorithm_display_name(name) for name in ALGORITHM_CANONICAL_ORDER]

WARM_LABELS = {True: "ciepły start", False: "zimny start"}
WARM_LABEL_WARM = WARM_LABELS[True]
WARM_LABEL_COLD = WARM_LABELS[False]

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


def _slugify_label(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(label))
    ascii_label = normalized.encode("ascii", "ignore").decode("ascii")
    safe = []
    for char in ascii_label.lower():
        if char.isalnum():
            safe.append(char)
        else:
            safe.append("_")
    slug = "".join(safe)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["license_family"] = out["license_config"].map(LICENSE_FAMILY).fillna("other")
    out["warm_label"] = out["warm_start"].map(WARM_LABELS)
    if "step" in out.columns:
        out["step"] = out["step"].fillna(0).astype(int)
    out["variant"] = out["license_config"].where(
        ~out["license_config"].str.contains("p_"),
        out["license_config"].str.extract(r"(p_\d+)", expand=False),
    )
    out["mutation_variant"] = out["license_config"].map(LICENSE_MUTATION_VARIANT).fillna("unknown")
    return out


def _plot_steps_by_license(df: pd.DataFrame, metric: str, title: str, filename_prefix: str) -> None:
    aggregated = df.groupby(["license_config", "warm_label", "step"], as_index=False).agg(mean=(metric, "mean"))
    if aggregated.empty:
        return
    for license_config, frame in aggregated.groupby("license_config"):
        if frame.empty:
            continue
        fig, ax = plt.subplots()
        sns.lineplot(data=frame, x="step", y="mean", hue="warm_label", marker="o", ax=ax)
        ax.set_title(f"{title} - {license_config}")
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
        safe_license = str(license_config).replace("/", "_").replace(" ", "_")
        fig.savefig(FIG_DIR / f"{filename_prefix}_{safe_license}.pdf")
        plt.close(fig)


def _plot_steps_by_algorithm(df: pd.DataFrame, algorithm: str, metric: str, filename_prefix: str) -> None:
    subset = df[df["algorithm"] == algorithm]
    aggregated = subset.groupby(["license_config", "warm_label", "step"], as_index=False).agg(mean=(metric, "mean"))
    if aggregated.empty:
        return
    metric_title = METRIC_DISPLAY.get(metric, metric)
    for license_config, frame in aggregated.groupby("license_config"):
        if frame.empty:
            continue
        fig, ax = plt.subplots()
        sns.lineplot(data=frame, x="step", y="mean", hue="warm_label", marker="o", ax=ax)
        ax.set_title(f"{algorithm} - {metric_title} ({license_config})")
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
        safe_license = str(license_config).replace("/", "_").replace(" ", "_")
        fig.savefig(FIG_DIR / f"{filename_prefix}_{safe_license}.pdf")
        plt.close(fig)


def _plot_metric_vs_nodes(df: pd.DataFrame, metric: str, title: str, filename_prefix: str) -> None:
    aggregated = df.groupby(["license_config", "warm_label", "n_nodes"], as_index=False).agg(mean=(metric, "mean"))
    if aggregated.empty:
        return
    for license_config, frame in aggregated.groupby("license_config"):
        if frame.empty:
            continue
        fig, ax = plt.subplots()
        sns.lineplot(
            data=frame,
            x="n_nodes",
            y="mean",
            hue="warm_label",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"{title} - {license_config}")
        ax.set_xlabel("Liczba węzłów")
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
        safe_license = str(license_config).replace("/", "_").replace(" ", "_")
        fig.savefig(FIG_DIR / f"{filename_prefix}_{safe_license}.pdf")
        plt.close(fig)


def _plot_license_mix(license_mix: pd.DataFrame, filename: str) -> None:
    if license_mix.empty:
        return
    data = license_mix.reset_index().rename(columns={"index": "license_config"})
    fig, ax = plt.subplots()
    indices = np.arange(len(data))
    width = 0.6
    ax.bar(
        indices,
        data["group_share"],
        width,
        label="Licencje grupowe",
        color="#4477aa",
    )
    ax.bar(
        indices,
        data["individual_share"],
        width,
        bottom=data["group_share"],
        label="Licencje indywidualne",
        color="#ee6677",
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(data["license_config"], rotation=25)
    ax.set_ylabel("Udział licencji")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def compute_warm_cold_delta(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = df.groupby(
        [
            "license_family",
            "license_config",
            "algorithm",
            "step",
            "warm_label",
        ],
        as_index=False,
    ).agg(mean=(metric, "mean"))
    if grouped.empty:
        return pd.DataFrame()
    pivot = grouped.pivot_table(
        index=["license_family", "license_config", "algorithm", "step"],
        columns="warm_label",
        values="mean",
    )
    if WARM_LABEL_WARM not in pivot.columns or WARM_LABEL_COLD not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.rename(columns={WARM_LABEL_WARM: "mean_cieply", WARM_LABEL_COLD: "mean_zimny"})
    pivot = pivot.dropna(subset=["mean_cieply", "mean_zimny"], how="any")
    pivot["delta"] = pivot["mean_cieply"] - pivot["mean_zimny"]
    return pivot.reset_index()


def _plot_genetic_mutation_steps(df: pd.DataFrame) -> None:
    genetic_label = algorithm_display_name("GeneticAlgorithm")
    subset = df[(df["algorithm"] == genetic_label) & df["mutation_variant"].notna()]
    if subset.empty:
        return

    metric_titles = {
        "time_s": "Średni czas względem kroku",
        "cost_per_node": "Średni koszt/węzeł względem kroku",
    }

    order_map = {variant: idx for idx, variant in enumerate(MUTATION_VARIANT_ORDER)}

    for metric, title in metric_titles.items():
        aggregated = subset.groupby(["mutation_variant", "step", "warm_label"], as_index=False).agg(mean=(metric, "mean"))
        if aggregated.empty:
            continue
        variants = sorted(
            [v for v in aggregated["mutation_variant"].unique() if v and v != "unknown"],
            key=lambda v: order_map.get(v, len(order_map) + 1),
        )
        for variant in variants:
            frame = aggregated[aggregated["mutation_variant"] == variant]
            if frame.empty:
                continue
            fig, ax = plt.subplots()
            sns.lineplot(
                data=frame,
                x="step",
                y="mean",
                hue="warm_label",
                marker="o",
                ax=ax,
            )
            display_variant = MUTATION_VARIANT_DISPLAY.get(variant, variant)
            ax.set_title(f"{genetic_label} - {title} ({display_variant})")
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
            safe_variant = _slugify_label(display_variant)
            fig.savefig(FIG_DIR / f"genetic_{metric}_steps_{safe_variant}.pdf")
            plt.close(fig)


def main() -> None:
    apply_plot_style()

    df_raw = expand_license_counts(load_dataset(DATA_DIR / "df.csv"))
    timeout_raw = expand_license_counts(load_dataset(DATA_DIR / "timeout_df.csv"))

    baseline_raw = pd.concat([df_raw, timeout_raw], ignore_index=True)
    _, unit_costs = normalize_cost_columns(baseline_raw, attach_group_multiplier=True)

    df_norm, _ = normalize_cost_columns(df_raw, unit_costs=unit_costs, attach_group_multiplier=True)
    timeout_norm, _ = normalize_cost_columns(timeout_raw, unit_costs=unit_costs, attach_group_multiplier=True)

    df = apply_algorithm_labels(add_metadata(df_norm))
    timeout_df = apply_algorithm_labels(add_metadata(timeout_norm))

    metrics = [
        "time_s",
        "total_cost",
        "cost_per_node",
        "license_group",
        "license_individual",
    ]

    overall = describe_numeric(df, metrics, ["license_config"])
    save_table(overall, TAB_DIR / "overall_by_license.csv")

    overall_family = describe_numeric(df, ["time_s", "total_cost", "cost_per_node"], ["license_family"])
    save_table(overall_family, TAB_DIR / "overall_by_family.csv")

    algo_warm = describe_numeric(
        df,
        ["time_s", "total_cost", "cost_per_node"],
        ["license_config", "algorithm", "warm_label"],
    )
    save_table(algo_warm, TAB_DIR / "algo_by_license_warm.csv")

    step_stats = describe_numeric(
        df,
        ["time_s", "total_cost", "cost_per_node"],
        ["license_config", "step", "warm_label"],
    )
    save_table(step_stats, TAB_DIR / "step_stats.csv")

    timeout_counts = timeout_df.groupby(["license_config", "algorithm", "warm_label"]).size().rename("count").sort_values(ascending=False)
    timeout_counts.to_csv(TAB_DIR / "timeouts_by_license_algorithm_warm.csv")

    license_mix = df.groupby("license_config")[["license_group", "license_individual", "license_other"]].sum()
    license_mix["total"] = license_mix.sum(axis=1)
    license_mix["group_share"] = license_mix["license_group"] / license_mix["total"]
    license_mix["individual_share"] = license_mix["license_individual"] / license_mix["total"]
    save_table(license_mix, TAB_DIR / "license_mix.csv")
    _plot_license_mix(license_mix, "license_mix.pdf")

    _plot_steps_by_license(
        df,
        "cost_per_node",
        "Koszt na węzeł w kolejnych krokach",
        "cost_per_node_steps_by_license",
    )
    _plot_steps_by_license(
        df,
        "time_s",
        "Czas obliczeń w kolejnych krokach",
        "time_steps_by_license",
    )

    metric_titles = {
        "total_cost": "Łączny koszt względem liczby węzłów",
        "cost_per_node": "Koszt na węzeł względem liczby węzłów",
        "time_s": "Czas obliczeń względem liczby węzłów",
    }
    for metric, title in metric_titles.items():
        _plot_metric_vs_nodes(df, metric, title, f"{metric}_vs_nodes")

    for algo in TARGET_ALGOS_DISPLAY:
        safe_name = _slugify_label(algo)
        _plot_steps_by_algorithm(df, algo, "total_cost", f"{safe_name}_cost_steps")
        _plot_steps_by_algorithm(df, algo, "time_s", f"{safe_name}_time_steps")
        _plot_steps_by_algorithm(df, algo, "cost_per_node", f"{safe_name}_cost_per_node_steps")

    delta_cost = compute_warm_cold_delta(df, "total_cost")
    delta_time = compute_warm_cold_delta(df, "time_s")
    delta_cpn = compute_warm_cold_delta(df, "cost_per_node")
    if not delta_cost.empty:
        save_table(delta_cost, TAB_DIR / "warm_vs_cold_delta_cost.csv")
    if not delta_time.empty:
        save_table(delta_time, TAB_DIR / "warm_vs_cold_delta_time.csv")
    if not delta_cpn.empty:
        save_table(delta_cpn, TAB_DIR / "warm_vs_cold_delta_cpn.csv")

    _plot_genetic_mutation_steps(df)

    summary_lines: list[str] = []
    summary_lines.append("# Extensions dynamic - podsumowanie")
    summary_lines.append("")
    total_rows = len(df)
    summary_lines.append(f"Łącznie rekordów: {total_rows}")
    best_cpn = algo_warm[(algo_warm["metric"] == "cost_per_node") & (algo_warm["warm_label"] == WARM_LABEL_WARM)].sort_values("mean").head(5)
    if not best_cpn.empty:
        summary_lines.append("Top5 najniższych kosztów/węzeł (ciepły start):")
        for _, row in best_cpn.iterrows():
            summary_lines.append(f"- {row['license_config']} / {row['algorithm']}: średnia {row['mean']:.2f}")
    if not delta_cpn.empty:
        summary_lines.append("")
        summary_lines.append("Różnice ciepły vs zimny start (koszt na węzeł):")
        mean_delta = delta_cpn.groupby(["license_config", "algorithm"])["delta"].mean().reset_index().sort_values("delta")
        for _, row in mean_delta.head(5).iterrows():
            summary_lines.append(f"- {row['license_config']} / {row['algorithm']}: średnia delta {row['delta']:.3f}")
    else:
        summary_lines.append("")
        summary_lines.append("Brak pełnych par ciepły/zimny start dla algorytmów w danych (koszt na węzeł).")
    summary_lines.append("")
    summary_lines.append("Pełne statystyki oraz wykresy zapisano w katalogach tables/ i figures/")
    write_text(REPORTS_DIR / "summary.md", summary_lines)


if __name__ == "__main__":
    main()
