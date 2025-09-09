from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from .commons import GENERATE_PDF, ensure_dir


def _aggregate(rows: List[dict[str, Any]], metric: str, graphs_filter: List[str] | None = None) -> Dict[str, float]:
    # mean over (graph, n) per algorithm
    per_alg: Dict[str, List[float]] = defaultdict(list)
    by_key: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            alg = str(r["algorithm"]) if r.get("algorithm") else ""
            g = str(r.get("graph", ""))
            n = int(float(r.get("n_nodes", 0)))
            v = float(r.get(metric, 0.0))
        except Exception:
            continue
        if not alg or not g or n <= 0:
            continue
        if graphs_filter and g not in graphs_filter:
            continue
        by_key[(alg, g, n)][metric].append(v)
    for (alg, _g, _n), dm in by_key.items():
        vals = dm.get(metric, [])
        if vals:
            per_alg[alg].append(mean(vals))
    return {alg: mean(vs) for alg, vs in per_alg.items() if vs}


def plot_compare_configs(
    rows: List[dict[str, Any]],
    cfg_a: str,
    cfg_b: str,
    title_prefix: str,
    out_dir: Path,
    graphs_filter: List[str] | None = None,
    tag: str | None = None,
) -> None:
    ensure_dir(out_dir)
    A = [r for r in rows if str(r.get("license_config", "")) == cfg_a]
    B = [r for r in rows if str(r.get("license_config", "")) == cfg_b]
    if not A or not B:
        return
    for metric, ylabel, fname in [
        ("cost_per_node", "cost per node", "compare_cost_per_node"),
        ("time_ms", "time [ms] (mean over inst)", "compare_time_ms"),
    ]:
        aggA = _aggregate(A, metric, graphs_filter=graphs_filter)
        aggB = _aggregate(B, metric, graphs_filter=graphs_filter)
        algs = sorted(set(aggA.keys()) | set(aggB.keys()))
        if not algs:
            continue
        xa = [aggA.get(a, float("nan")) for a in algs]
        xb = [aggB.get(a, float("nan")) for a in algs]
        x = range(len(algs))
        width = 0.38
        plt.figure(figsize=(max(6.5, 0.6 * len(algs)), 5))
        plt.bar([i - width / 2 for i in x], xa, width=width, label=cfg_a)
        plt.bar([i + width / 2 for i in x], xb, width=width, label=cfg_b)
        plt.xticks(list(x), algs, rotation=30, ha="right")
        plt.ylabel(ylabel)
        extra = f" — {', '.join(graphs_filter)}" if graphs_filter else ""
        plt.title(f"{title_prefix} — {ylabel}: {cfg_a} vs {cfg_b}{extra}")
        plt.legend()
        suffix = ("_" + tag) if tag else ""
        out = out_dir / f"{fname}_{cfg_a}_vs_{cfg_b}{suffix}"
        plt.tight_layout()
        plt.savefig(out.with_suffix(".png"), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix(".pdf"))
        plt.close()
