from __future__ import annotations

import argparse
from pathlib import Path

from .commons import ensure_dir


def find_latest_run(runs_dir: Path) -> Path | None:
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs)[-1]


def emit_report(run_dir: Path, out_path: Path) -> None:
    title = run_dir.name
    analysis = run_dir / "analysis"
    ensure_dir(out_path.parent)

    # Collect figures and tables
    figs = sorted(analysis.rglob("*.png"))
    aggs = sorted(analysis.rglob("aggregates.csv"))
    pandas_summ = sorted(analysis.rglob("summary_by_algo_graph_n.csv"))
    pivots = sorted(analysis.rglob("pivot_cost_per_node.csv"))

    lines: list[str] = []
    lines.append(f"# Report -- {title}\n")
    lines.append("This report summarizes experiments and analysis outputs.\n")
    lines.append("## Contents\n")
    lines.append("- Figures\n- Tables\n- Notes\n")

    lines.append("## Figures\n")
    for p in figs:
        rel = p.relative_to(out_path.parent)
        lines.append(f"- {rel.as_posix()}")
    if not figs:
        lines.append("- (no figures found)")
    lines.append("")

    lines.append("## Tables\n")
    for p in aggs + pandas_summ + pivots:
        rel = p.relative_to(out_path.parent)
        lines.append(f"- {rel.as_posix()}")
    if not (aggs or pandas_summ or pivots):
        lines.append("- (no tables found)")
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- Cost vs n and time vs n curves help discuss scalability (Ch. 6.3).")
    lines.append("- Pareto front and performance profiles formalize trade-offs (Ch. 6.1).")
    lines.append("- Density/time scatter and aggregates relate structure to difficulty (Ch. 6.3).\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", dest="run", default="", help="run id under runs/ (default: latest)")
    args, _ = parser.parse_known_args()

    runs_dir = Path("runs")
    if args.run:
        run_dir = runs_dir / args.run
    else:
        run_dir = find_latest_run(runs_dir) or (runs_dir / "filtered_zip")

    if not run_dir.exists():
        raise SystemExit(f"run directory not found: {run_dir}")

    out_dir = Path("docs") / "reports" / run_dir.name
    ensure_dir(out_dir)
    out_path = out_dir / "report.md"
    emit_report(run_dir, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
