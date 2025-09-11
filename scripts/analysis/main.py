from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from .commons import ensure_dir, load_rows
from .dynamic_warmcold import plot_dynamic_warm_cold
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


def analyze_benchmark(csv_path: Path, rows: list[dict[str, object]]) -> None:
    run_dir = csv_path.parent.parent
    title = csv_path.stem
    # per (license, graph)
    combos = set()
    for r in rows:
        lic = str(r.get("license_config", ""))
        gname = str(r.get("graph", ""))
        if lic and gname:
            combos.add((lic, gname))
    for lic, gname in sorted(combos):
        sub = [r for r in rows if str(r.get("license_config", "")) == lic and str(r.get("graph", "")) == gname]
        out_dir = run_dir / "analysis" / lic / gname
        ensure_dir(out_dir)
        plot_cost_vs_n(sub, title=f"{title} — {lic} — {gname}", out_path=out_dir / f"{gname}_cost_vs_n")
        plot_time_vs_n(
            sub,
            title=f"{title} — {lic} — {gname} time vs n",
            out_path=out_dir / f"{gname}_time_vs_n",
        )
        plot_pareto(
            sub,
            title=f"{title} — {lic} — {gname} Pareto",
            out_path=out_dir / f"{gname}_pareto_cost_time",
        )
        plot_density_vs_time(
            sub,
            title=f"{title} — {lic} — {gname} density vs time",
            out_path=out_dir / f"{gname}_density_vs_time",
        )
        plot_performance_profiles(sub, title_prefix=f"{title} — {lic} — {gname}", out_dir=out_dir)
    # overall heatmap and license mix
    out_dir_all = run_dir / "analysis" / "all"
    ensure_dir(out_dir_all)
    plot_cost_heatmap(rows, title=f"{title} — cost heatmap", out_path=out_dir_all / "heatmap_cost")
    # Global mix (all configs together)
    plot_license_mix(rows, title=f"{title} — license mix by algorithm (all configs)", out_path=out_dir_all / "license_mix_all_configs")
    # Per-config license mix breakdowns
    configs = sorted({str(r.get("license_config", "")) for r in rows if r.get("license_config")})
    for cfg in configs:
        sub_cfg = [r for r in rows if str(r.get("license_config", "")) == cfg]
        out_cfg = run_dir / "analysis" / cfg / "all"
        ensure_dir(out_cfg)
        plot_license_mix(sub_cfg, title=f"{title} — license mix by algorithm — {cfg}", out_path=out_cfg / f"license_mix_{cfg}")
    write_aggregates(rows, out_path=out_dir_all / "aggregates.csv")
    # pandas-based summaries if available
    write_pandas_summaries(csv_path, out_dir_all)
    # Statistical tests and time scaling summaries
    write_stats_reports(rows, out_dir_all)
    write_time_scaling(rows, out_dir_all)
    # Focus on duolingo_super vs roman_domination comparisons
    cfgs = {str(r.get("license_config", "")) for r in rows}
    if {"duolingo_super", "roman_domination"}.issubset(cfgs):
        out_cmp = run_dir / "analysis" / "compare_duo_roman"
        ensure_dir(out_cmp)
        plot_compare_configs(rows, "duolingo_super", "roman_domination", title, out_cmp)


def analyze_dynamic(csv_path: Path, rows: list[dict[str, object]]) -> None:
    run_dir = csv_path.parent.parent
    title = csv_path.stem
    # split by (graph, license)
    combos = set()
    for r in rows:
        g = str(r.get("graph", ""))
        lic = str(r.get("license_config", ""))
        combos.add((g, lic))
    for g, lic in sorted(combos):
        sub = [r for r in rows if str(r.get("graph", "")) == g and str(r.get("license_config", "")) == lic]
        out_dir = run_dir / "analysis" / lic / g
        plot_dynamic_warm_cold(sub, title_prefix=f"{title} — {lic} — {g}", out_dir=out_dir)


def main() -> None:
    runs_dir = Path("runs")
    csvs = sorted(runs_dir.glob("*/csv/*.csv"))

    if csvs:
        for csv_path in csvs:
            rows = load_rows(csv_path)
            run_dir = csv_path.parent.parent
            name = run_dir.name
            print(f"analyzing {csv_path}")
            if name.endswith("_benchmark") or csv_path.stem.endswith("_benchmark") or name.endswith("_benchmark_real"):
                analyze_benchmark(csv_path, rows)
            elif name.endswith("_dynamic") or name.endswith("_dynamic_real"):
                analyze_dynamic(csv_path, rows)
            else:
                analyze_benchmark(csv_path, rows)
    else:
        print("no CSVs under runs/*/csv — checking filtered_results.zip")
        zip_path = Path("filtered_results.zip")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as z:
                rows: list[dict[str, object]] = []
                members = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not members:
                    print("filtered_results.zip contains no CSV files")
                    return
                # Read and enrich rows from all CSV files inside the zip
                for name in members:
                    print(f"loading {name}")
                    with z.open(name) as f:
                        reader = csv.DictReader(line.decode("utf-8", "ignore") for line in f)
                        for r in reader:
                            rows.append(r)
                # Write a convenience combined CSV under runs/filtered_zip/csv
                out_base = runs_dir / "filtered_zip"
                csv_dir = out_base / "csv"
                ensure_dir(csv_dir)
                out_csv = csv_dir / "filtered_combined.csv"
                if rows:
                    with out_csv.open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(rows)
                    print(f"wrote combined CSV: {out_csv}")
                # Enrich derived columns (density/avg_degree)
                rows = load_rows(out_csv)
                analyze_benchmark(out_csv, rows)
        else:
            print("no data to analyze")


if __name__ == "__main__":
    main()
