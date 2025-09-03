from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, pstdev
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF, group_cost_by_n


def plot_cost_vs_n(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    series = group_cost_by_n(rows)
    plt.figure(figsize=(8, 5))
    for alg, pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if xs and ys:
            plt.plot(xs, ys, marker="o", label=alg)
    plt.xlabel("n_nodes")
    plt.ylabel("total_cost")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_time_vs_n(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    series_t: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            alg = str(r["algorithm"])
            n = int(float(r.get("n_nodes", 0)))
            t = float(r.get("time_ms", 0.0)) + 1e-9
        except Exception:
            continue
        series_t[alg][n].append(t)
    plt.figure(figsize=(8, 5))
    for alg, dn in series_t.items():
        xs = sorted(dn.keys())
        means = [mean(dn[n]) for n in xs]
        cis = [1.96 * ((pstdev(dn[n]) if len(dn[n]) <= 1 else stdev(dn[n])) / (len(dn[n]) ** 0.5)) if len(dn[n]) > 1 else 0.0 for n in xs]
        if xs:
            plt.plot(xs, means, marker="o", label=alg)
            lower = [m - ci for m, ci in zip(means, cis)]
            upper = [m + ci for m, ci in zip(means, cis)]
            plt.fill_between(xs, lower, upper, alpha=0.15)
    plt.yscale("log")
    plt.xlabel("n_nodes")
    plt.ylabel("time_ms (log)")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()

