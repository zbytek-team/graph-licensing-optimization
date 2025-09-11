from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .commons import GENERATE_PDF, ensure_dir


def plot_ilp_boundary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Heatmap timeout rate for ILP over (n, density) bins + outcome scatter."""
    ensure_dir(out_dir)
    ilp = [r for r in rows if str(r.get("algorithm", "")) in {"ILPSolver", "ilp"}]
    if not ilp:
        return
    data: list[tuple[int, float, bool]] = []
    for r in ilp:
        try:
            n = int(float(r.get("n_nodes", 0)))
            d = float(r.get("density", 0.0))
        except Exception:
            continue
        note = str(r.get("notes", ""))
        timeout = note == "timeout"
        data.append((n, d, timeout))
    if not data:
        return

    ns = np.array([n for n, _d, _t in data], dtype=float)
    ds = np.array([_d for _n, _d, _t in data], dtype=float)
    ts = np.array([1.0 if t else 0.0 for _n, _d, t in data], dtype=float)

    # Bins
    n_bins = np.unique(np.percentile(ns, [0, 33, 66, 100])).astype(float)
    if len(n_bins) < 4:
        n_bins = np.linspace(float(ns.min()), float(ns.max()) + 1e-12, 4)
    d_bins = np.linspace(float(ds.min()), float(ds.max()) + 1e-12, 6)

    H = np.zeros((len(n_bins) - 1, len(d_bins) - 1), dtype=float)
    C = np.zeros_like(H)
    for i in range(len(n_bins) - 1):
        for j in range(len(d_bins) - 1):
            mask = (ns >= n_bins[i]) & (ns < n_bins[i + 1]) & (ds >= d_bins[j]) & (ds < d_bins[j + 1])
            cnt = float(mask.sum())
            if cnt > 0:
                H[i, j] = float(ts[mask].mean())
                C[i, j] = cnt
            else:
                H[i, j] = np.nan

    plt.figure(figsize=(7, 4.6))
    im = plt.imshow(H, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    plt.colorbar(im, label="ILP timeout rate")
    plt.xlabel("density bins")
    plt.ylabel("n bins")
    plt.title("ILP timeout boundary (higher = more timeouts)")
    plt.tight_layout()
    out = out_dir / "ilp_timeout_boundary.png"
    plt.savefig(out, dpi=220)
    if GENERATE_PDF:
        plt.savefig(out.with_suffix(".pdf"))
    plt.close()

    # Outcome scatter
    plt.figure(figsize=(7, 4.6))
    ok = ts < 0.5
    to = ts >= 0.5
    plt.scatter(ds[ok], ns[ok], s=16, alpha=0.6, c="tab:blue", label="success")
    plt.scatter(ds[to], ns[to], s=16, alpha=0.6, c="tab:red", label="timeout")
    plt.xlabel("density")
    plt.ylabel("n_nodes")
    plt.title("ILP outcomes by (density, n)")
    plt.legend()
    plt.tight_layout()
    out2 = out_dir / "ilp_outcomes_scatter.png"
    plt.savefig(out2, dpi=220)
    if GENERATE_PDF:
        plt.savefig(out2.with_suffix(".pdf"))
    plt.close()
