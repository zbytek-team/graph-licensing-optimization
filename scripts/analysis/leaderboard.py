from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from .commons import GENERATE_PDF, ensure_dir


def _bin_label(n: int) -> str:
    if n <= 200:
        return "n<=200"
    if n <= 1000:
        return "200<n<=1000"
    return "n>1000"


def _winners_by_bin(rows: List[dict[str, Any]], metric: str) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Return {(bin, graph): {algorithm: wins}} using min metric per instance.
    Instances are (license_config, graph, n_nodes); winner is alg with lowest mean metric.
    """
    # aggregate per (alg, lic, graph, n) mean metric
    acc: Dict[Tuple[str, str, str, int, str], List[float]] = defaultdict(list)
    for r in rows:
        try:
            alg = str(r["algorithm"]) if r.get("algorithm") else ""
            lic = str(r.get("license_config", ""))
            g = str(r.get("graph", ""))
            n = int(float(r.get("n_nodes", 0)))
            v = float(r.get(metric, 0.0))
        except Exception:
            continue
        if not alg or not g or n <= 0:
            continue
        acc[(alg, lic, g, n, metric)].append(v)
    means: Dict[Tuple[str, str, str, int], Dict[str, float]] = defaultdict(dict)
    for (alg, lic, g, n, _m), vs in acc.items():
        means[(lic, g, n)][alg] = sum(vs) / len(vs)

    # choose winner per (lic,g,n)
    wins: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (lic, g, n), alg2val in means.items():
        if not alg2val:
            continue
        winner = min(alg2val.items(), key=lambda kv: kv[1])[0]
        b = _bin_label(n)
        wins[(b, g)][winner] += 1
    return wins


def write_leaderboards(rows: List[dict[str, Any]], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for metric, fname in [("cost_per_node", "leaderboard_cost"), ("time_ms", "leaderboard_time")]:
        wins = _winners_by_bin(rows, metric)
        # write CSV and bar plots (overall across graphs)
        # accumulate overall wins per algorithm across bins
        overall: Dict[str, int] = defaultdict(int)
        out_csv = out_dir / f"{fname}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            f.write("bin,graph,algorithm,wins\n")
            for (b, g), m in sorted(wins.items()):
                for alg, c in sorted(m.items(), key=lambda kv: (-kv[1], kv[0])):
                    overall[alg] += c
                    f.write(f"{b},{g},{alg},{c}\n")

        if overall:
            algs = [k for k, _ in sorted(overall.items(), key=lambda kv: (-kv[1], kv[0]))]
            vals = [overall[a] for a in algs]
            plt.figure(figsize=(max(6.5, 0.5 * len(algs)), 4.5))
            plt.bar(range(len(algs)), vals)
            plt.xticks(range(len(algs)), algs, rotation=30, ha="right")
            ylabel = "wins (per-instance minima aggregated across bins)"
            title = f"Leaderboard {metric}"
            plt.ylabel(ylabel)
            plt.title(title)
            plt.tight_layout()
            out_png = out_dir / f"{fname}.png"
            plt.savefig(out_png, dpi=220)
            if GENERATE_PDF:
                plt.savefig(out_dir / f"{fname}.pdf")
            plt.close()

