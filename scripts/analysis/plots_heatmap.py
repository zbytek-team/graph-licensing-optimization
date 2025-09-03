from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .commons import GENERATE_PDF, ensure_dir


def plot_cost_heatmap(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    dens = []
    deg = []
    cost = []
    for r in rows:
        try:
            d = float(r.get("density", 0.0))
            g = float(r.get("avg_degree", 0.0))
            c = float(r.get("total_cost", 0.0))
        except Exception:  # robust parsing of possibly incomplete rows
            continue
        dens.append(d)
        deg.append(g)
        cost.append(c)
    if not dens:
        return
    dens = np.array(dens)
    deg = np.array(deg)
    cost = np.array(cost)
    dbins = np.linspace(float(dens.min()), float(dens.max()) + 1e-12, 10)
    gbins = np.linspace(float(deg.min()), float(deg.max()) + 1e-12, 10)
    H = np.full((len(dbins) - 1, len(gbins) - 1), np.nan)
    for i in range(len(dbins) - 1):
        for j in range(len(gbins) - 1):
            mask = (dens >= dbins[i]) & (dens < dbins[i + 1]) & (deg >= gbins[j]) & (deg < gbins[j + 1])
            if mask.any():
                H[i, j] = float(np.mean(cost[mask]))
    plt.figure(figsize=(6.5, 5))
    im = plt.imshow(H, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, label="mean cost")
    plt.xlabel("avg_degree bins")
    plt.ylabel("density bins")
    plt.title(title)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
