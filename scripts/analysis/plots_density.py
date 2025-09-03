from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF


def plot_density_vs_time(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 5))
    colors = {}
    import itertools
    palette = itertools.cycle([f"C{k}" for k in range(10)])
    for r in rows:
        try:
            alg = str(r['algorithm']) ; d = float(r.get('density', 0.0)) ; t = float(r.get('time_ms', 0.0))
        except Exception:
            continue
        if alg not in colors:
            colors[alg] = next(palette)
        plt.scatter(d, t, s=18, alpha=0.8, c=colors[alg], label=alg if alg not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.yscale('log')
    plt.xlabel('density'); plt.ylabel('time_ms (log)'); plt.title(title)
    ensure_dir(out_path.parent)
    plt.tight_layout(); plt.savefig(out_path.with_suffix('.png'), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()

