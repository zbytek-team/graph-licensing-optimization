from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .commons import GENERATE_PDF, ensure_dir


def plot_density_vs_time(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    # Group by algorithm to avoid O(n) scatter calls
    by_alg: dict[str, tuple[list[float], list[float]]] = {}
    for r in rows:
        try:
            alg = str(r.get("algorithm", ""))
            d = float(r.get("density", 0.0))
            t = float(r.get("time_ms", 0.0))
        except Exception:  # robust parsing
            continue
        if not alg:
            continue
        xs, ys = by_alg.setdefault(alg, ([], []))
        xs.append(d)
        ys.append(t)
    if not by_alg:
        return
    plt.figure(figsize=(6.5, 5))
    for alg, (xs, ys) in sorted(by_alg.items()):
        plt.scatter(xs, ys, s=18, alpha=0.8, label=alg)
    plt.yscale("log")
    plt.xlabel("density")
    plt.ylabel("time_ms (log)")
    plt.title(title)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
