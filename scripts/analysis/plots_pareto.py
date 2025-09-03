from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF


def plot_pareto(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    pts = []
    for r in rows:
        try:
            alg = str(r["algorithm"])
            c = float(r["total_cost"]) ; t = float(r["time_ms"]) ; d = float(r.get("density", 0.0))
        except Exception:  # robust parsing
            continue
        pts.append((t, c, alg, d))
    pareto = []
    for i, (ti, ci, *_ ) in enumerate(pts):
        dominated = False
        for j, (tj, cj, *_ ) in enumerate(pts):
            if j != i and tj <= ti and cj <= ci and (tj < ti or cj < ci):
                dominated = True
                break
        if not dominated:
            pareto.append((ti, ci))
    plt.figure(figsize=(6.5, 5))
    colors = {}
    import itertools
    palette = itertools.cycle([f"C{k}" for k in range(10)])
    for t, c, alg, _ in pts:
        if alg not in colors:
            colors[alg] = next(palette)
        plt.scatter(t, c, s=18, alpha=0.8, c=colors[alg], label=alg if alg not in plt.gca().get_legend_handles_labels()[1] else "")
    if pareto:
        xs = [p[0] for p in pareto] ; ys = [p[1] for p in pareto]
        order = sorted(range(len(pareto)), key=lambda k: xs[k])
        xs = [xs[k] for k in order] ; ys = [ys[k] for k in order]
        plt.plot(xs, ys, color="k", linewidth=2, alpha=0.6)
    plt.xlabel("time_ms") ; plt.ylabel("total_cost") ; plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout() ; plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
