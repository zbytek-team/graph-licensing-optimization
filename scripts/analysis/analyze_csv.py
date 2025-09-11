from __future__ import annotations

import argparse
from pathlib import Path

from .commons import ensure_dir, load_rows
from .ilp_boundary import plot_ilp_boundary
from .leaderboard import write_leaderboards
from .plots_compare_configs import plot_compare_configs
from .plots_cost_time import plot_cost_vs_n, plot_time_vs_n
from .plots_density import plot_density_vs_time
from .plots_heatmap import plot_cost_heatmap
from .plots_license_mix import plot_license_mix
from .plots_pareto import plot_pareto
from .plots_profiles import plot_performance_profiles
from .stats_tests import write_stats_reports
from .summary_pandas import write_pandas_summaries
from .tables_aggregates import write_aggregates
from .time_scaling import write_time_scaling


def analyze_static(csv_path: Path, out_dir: Path) -> None:
    rows = load_rows(csv_path)
    title = csv_path.stem
    # Per (license, graph)
    combos: set[tuple[str, str]] = set()
    for r in rows:
        lic = str(r.get("license_config", ""))
        gname = str(r.get("graph", ""))
        if lic and gname:
            combos.add((lic, gname))
    for lic, gname in sorted(combos):
        sub = [r for r in rows if str(r.get("license_config", "")) == lic and str(r.get("graph", "")) == gname]
        d = out_dir / lic / gname
        ensure_dir(d)
        plot_cost_vs_n(sub, title=f"{title} — {lic} — {gname}", out_path=d / f"{gname}_cost_vs_n")
        plot_time_vs_n(sub, title=f"{title} — {lic} — {gname} time vs n", out_path=d / f"{gname}_time_vs_n")
        plot_pareto(sub, title=f"{title} — {lic} — {gname} Pareto", out_path=d / f"{gname}_pareto_cost_time")
        plot_density_vs_time(sub, title=f"{title} — {lic} — {gname} density vs time", out_path=d / f"{gname}_density_vs_time")
        plot_performance_profiles(sub, title_prefix=f"{title} — {lic} — {gname}", out_dir=d)
    # Overall
    all_dir = out_dir / "all"
    ensure_dir(all_dir)
    plot_cost_heatmap(rows, title=f"{title} — cost heatmap", out_path=all_dir / "heatmap_cost")
    # Global mix (all configs together)
    plot_license_mix(rows, title=f"{title} — license mix by algorithm (all configs)", out_path=all_dir / "license_mix_all_configs")
    # Per-config license mix breakdowns
    configs = sorted({str(r.get("license_config", "")) for r in rows if r.get("license_config")})
    for cfg in configs:
        sub_cfg = [r for r in rows if str(r.get("license_config", "")) == cfg]
        out_cfg = out_dir / cfg / "all"
        ensure_dir(out_cfg)
        plot_license_mix(sub_cfg, title=f"{title} — license mix by algorithm — {cfg}", out_path=out_cfg / f"license_mix_{cfg}")
    write_aggregates(rows, out_path=all_dir / "aggregates.csv")
    write_pandas_summaries(csv_path, all_dir)
    # Statistical tests and time scaling summaries
    write_stats_reports(rows, all_dir)
    write_time_scaling(rows, all_dir)
    write_leaderboards(rows, all_dir)
    plot_ilp_boundary(rows, all_dir)
    # Focus: duolingo_super vs roman_domination
    cfgs = {str(r.get("license_config", "")) for r in rows}
    graphs = sorted({str(r.get("graph", "")) for r in rows if r.get("graph")})
    if {"duolingo_super", "roman_domination"}.issubset(cfgs):
        out_cmp = out_dir / "compare_duo_roman"
        ensure_dir(out_cmp)
        # Global (all graphs)
        plot_compare_configs(rows, "duolingo_super", "roman_domination", title, out_cmp)
        # Per selected graphs (e.g., random / facebook_ego)
        for g in graphs:
            sub_out = out_cmp / "by_graph" / g
            ensure_dir(sub_out)
            plot_compare_configs(
                rows,
                "duolingo_super",
                "roman_domination",
                title,
                sub_out,
                graphs_filter=[g],
                tag=g,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to input CSV")
    ap.add_argument("--out", required=True, help="output directory for results")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # Decide static vs dynamic by presence of 'step' column quickly
    # Here both requested analyses are static
    analyze_static(csv_path, out_dir)


if __name__ == "__main__":
    main()
