from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF


def plot_performance_profiles(rows: list[dict[str, Any]], title_prefix: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    # Instances based on (graph, n_nodes) if present; fallback to index
    insts = sorted({(str(r.get('graph','')), int(float(r.get('n_nodes', 0)))) for r in rows if r.get('n_nodes')})
    # Cost profiles
    perf_cost: dict[str, list[float]] = defaultdict(list)
    for g, n in insts:
        per_alg = defaultdict(list)
        for r in rows:
            try:
                if str(r.get('graph','')) == g and int(float(r.get('n_nodes', 0))) == n:
                    per_alg[str(r['algorithm'])].append(float(r['total_cost']))
            except Exception:
                pass
        means = {alg: (mean(vs) if vs else float('inf')) for alg, vs in per_alg.items()}
        best = min((v for v in means.values() if v > 0), default=None)
        if not best:
            continue
        for alg, v in means.items():
            if v > 0:
                perf_cost[alg].append(v / best)
    if perf_cost:
        taus = [1.0 + i * 0.05 for i in range(0, 61)]
        plt.figure(figsize=(6.5, 5))
        for alg, ratios in sorted(perf_cost.items()):
            ratios = sorted(ratios)
            ys = []
            for tau in taus:
                c = sum(1 for r in ratios if r <= tau)
                ys.append(c / len(ratios) if ratios else 0.0)
            plt.plot(taus, ys, label=alg)
        plt.xlabel('tau') ; plt.ylabel('fraction of instances')
        plt.title(f"{title_prefix} Performance profile (cost)")
        plt.legend(ncol=2, fontsize=8)
        out = out_dir / f"perf_profile_cost"
        plt.tight_layout(); plt.savefig(out.with_suffix('.png'), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix('.pdf'))
        plt.close()

    # Time profiles
    perf_time: dict[str, list[float]] = defaultdict(list)
    for g, n in insts:
        per_alg = defaultdict(list)
        for r in rows:
            try:
                if str(r.get('graph','')) == g and int(float(r.get('n_nodes', 0))) == n:
                    per_alg[str(r['algorithm'])].append(float(r['time_ms']))
            except Exception:
                pass
        means = {alg: (mean(vs) if vs else float('inf')) for alg, vs in per_alg.items()}
        best = min((v for v in means.values() if v > 0), default=None)
        if not best:
            continue
        for alg, v in means.items():
            if v > 0:
                perf_time[alg].append(v / best)
    if perf_time:
        taus = [1.0 + i * 0.1 for i in range(0, 61)]
        plt.figure(figsize=(6.5, 5))
        for alg, ratios in sorted(perf_time.items()):
            ratios = sorted(ratios)
            ys = []
            for tau in taus:
                c = sum(1 for r in ratios if r <= tau)
                ys.append(c / len(ratios) if ratios else 0.0)
            plt.plot(taus, ys, label=alg)
        plt.xlabel('tau') ; plt.ylabel('fraction of instances')
        plt.title(f"{title_prefix} Performance profile (time)")
        plt.legend(ncol=2, fontsize=8)
        out = out_dir / f"perf_profile_time"
        plt.tight_layout(); plt.savefig(out.with_suffix('.png'), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix('.pdf'))
        plt.close()

