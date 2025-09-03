from __future__ import annotations

from pathlib import Path

from .commons import ensure_dir, load_rows
from .dynamic_warmcold import plot_dynamic_warm_cold
from .plots_cost_time import plot_cost_vs_n, plot_time_vs_n
from .plots_density import plot_density_vs_time
from .plots_heatmap import plot_cost_heatmap
from .plots_license_mix import plot_license_mix
from .plots_pareto import plot_pareto
from .plots_profiles import plot_performance_profiles
from .tables_aggregates import write_aggregates


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
    plot_license_mix(rows, title=f"{title} — license mix by algorithm", out_path=out_dir_all / "license_mix")
    write_aggregates(rows, out_path=out_dir_all / "aggregates.csv")


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
    if not csvs:
        print("no CSVs found under runs/*/csv")
        return
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
            # default: try static analyses
            analyze_benchmark(csv_path, rows)


if __name__ == "__main__":
    main()
