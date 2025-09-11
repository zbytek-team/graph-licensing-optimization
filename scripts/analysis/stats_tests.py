from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from .commons import ensure_dir

InstanceKey = tuple[str, int, str]  # (graph, n_nodes, license_config)


def _collect_instances(rows: list[dict[str, object]]) -> tuple[dict[InstanceKey, dict[str, float]], dict[InstanceKey, dict[str, float]], list[str]]:
    """Aggregate rows into instance -> algorithm -> mean cost/time.

    Returns (cost_map, time_map, algorithms_sorted).
    Only numeric, finite values are considered.
    """
    # Aggregate per instance and algorithm
    acc_cost: dict[tuple[InstanceKey, str], list[float]] = defaultdict(list)
    acc_time: dict[tuple[InstanceKey, str], list[float]] = defaultdict(list)
    algs: set[str] = set()
    for r in rows:
        try:
            g = str(r.get("graph", ""))
            n = int(float(str(r.get("n_nodes", 0))))
            lic = str(r.get("license_config", ""))
            alg = str(r.get("algorithm", ""))
            c = float(str(r.get("total_cost", float("nan"))))
            t = float(str(r.get("time_ms", float("nan"))))
        except Exception:
            continue
        if not alg or not g:
            continue
        if not math.isfinite(c) or not math.isfinite(t):
            continue
        key = (g, n, lic)
        acc_cost[(key, alg)].append(c)
        acc_time[(key, alg)].append(t)
        algs.add(alg)

    # Mean per instance/algorithm
    def mean(vals: list[float]) -> float:
        return sum(vals) / max(1, len(vals))

    cost_map: dict[InstanceKey, dict[str, float]] = defaultdict(dict)
    time_map: dict[InstanceKey, dict[str, float]] = defaultdict(dict)
    for (key, alg), vals in acc_cost.items():
        cost_map[key][alg] = mean(vals)
    for (key, alg), vals in acc_time.items():
        time_map[key][alg] = mean(vals)

    algs_s = sorted(algs)
    return cost_map, time_map, algs_s


def _build_matrix(metric_map: dict[InstanceKey, dict[str, float]], algorithms: list[str]) -> tuple[list[InstanceKey], list[list[float]]]:
    """Keep only instances where all algorithms have a value; return matrix N x k."""
    insts: list[InstanceKey] = []
    mat: list[list[float]] = []
    for key, alg2val in metric_map.items():
        if all(a in alg2val for a in algorithms):
            insts.append(key)
            mat.append([alg2val[a] for a in algorithms])
    return insts, mat


def _ranks_per_row(values: list[float], lower_is_better: bool = True) -> list[float]:
    """Compute ranks with average ties. 1 = best if lower_is_better."""
    idx_vals = list(enumerate(values))
    # Sort ascending for lower is better; descending otherwise
    idx_vals.sort(key=lambda x: x[1], reverse=not lower_is_better)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(idx_vals):
        j = i
        v = idx_vals[i][1]
        while j < len(idx_vals) and idx_vals[j][1] == v:
            j += 1
        # average rank within [i+1, j]
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[idx_vals[k][0]] = avg_rank
        i = j
    return ranks


def _friedman_stats(matrix: list[list[float]], lower_is_better: bool) -> tuple[list[float], float, float, int, int]:
    """Return (avg_ranks, chi_sq, iman_davenport_F, N, k)."""
    if not matrix:
        return [], float("nan"), float("nan"), 0, 0
    N = len(matrix)
    k = len(matrix[0]) if matrix else 0
    rank_sums = [0.0] * k
    for row in matrix:
        ranks = _ranks_per_row(row, lower_is_better=lower_is_better)
        for j, r in enumerate(ranks):
            rank_sums[j] += r
    avg_ranks = [s / N for s in rank_sums]
    chi_sq = (12 * N) / (k * (k + 1)) * sum(s * s for s in avg_ranks) - 3 * N * (k + 1)
    # Iman-Davenport F approximation
    F = ((N - 1) * chi_sq) / (N * (k - 1) - chi_sq) if (N * (k - 1) - chi_sq) != 0 else float("nan")
    return avg_ranks, chi_sq, F, N, k


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, float, int]:
    """Wilcoxon signed-rank test (two-sided), normal approximation.
    Returns (z, p_approx, n_eff).
    """
    diffs = [xi - yi for xi, yi in zip(x, y)]
    # Exclude zeros
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0
    abs_d = [(abs(d), i) for i, d in enumerate(diffs)]
    abs_d.sort()
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = abs_d[i][0]
        while j < n and abs_d[j][0] == v:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[abs_d[k][1]] = avg_rank
        i = j
    W_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    W_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    W = min(W_pos, W_neg)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return 0.0, 1.0, n
    # Continuity correction
    z = (W - mu + 0.5) / sigma
    p = 2 * (1 - _normal_cdf(abs(z)))
    return z, p, n


def write_stats_reports(rows: list[dict[str, object]], out_dir: Path) -> None:
    ensure_dir(out_dir)
    cost_map, time_map, algorithms = _collect_instances(rows)

    # Build matrices over common instances across all algorithms
    insts_cost, M_cost = _build_matrix(cost_map, algorithms)
    insts_time, M_time = _build_matrix(time_map, algorithms)

    # Friedman (cost, lower better)
    avg_r_cost, chi_cost, F_cost, N_cost, k = _friedman_stats(M_cost, lower_is_better=True)
    # Friedman (time, lower better)
    avg_r_time, chi_time, F_time, N_time, _ = _friedman_stats(M_time, lower_is_better=True)

    # Save average ranks
    def _save_avg_ranks(path: Path, avg_r: list[float], chi: float, F: float, N: int, k: int) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("algorithm,avg_rank\n")
            for alg, r in zip(algorithms, avg_r):
                f.write(f"{alg},{r:.6f}\n")
        with path.with_suffix(".txt").open("w", encoding="utf-8") as g:
            g.write(f"N={N} instances, k={k} algorithms\n")
            g.write(f"Friedman chi2≈{chi:.4f}, Iman-Davenport F≈{F:.4f} (df1={k - 1}, df2={(k - 1) * (max(0, N - 1))})\n")
            g.write("Note: p-values not computed here; use these stats for reference or compute externally.\n")

    _save_avg_ranks(out_dir / "stats_friedman_cost.csv", avg_r_cost, chi_cost, F_cost, N_cost, k)
    _save_avg_ranks(out_dir / "stats_friedman_time.csv", avg_r_time, chi_time, F_time, N_time, k)

    # Pairwise Wilcoxon
    def _extract_metric(M: list[list[float]], algs: list[str]) -> dict[str, list[float]]:
        res = {}
        for j, a in enumerate(algs):
            res[a] = [row[j] for row in M]
        return res

    cost_by_alg = _extract_metric(M_cost, algorithms)
    time_by_alg = _extract_metric(M_time, algorithms)

    def _save_wilcoxon(path: Path, data: dict[str, list[float]]) -> None:
        algs = list(data.keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("alg_a,alg_b,z,p_approx,n_instances\n")
            for i in range(len(algs)):
                for j in range(i + 1, len(algs)):
                    a, b = algs[i], algs[j]
                    x, y = data[a], data[b]
                    n = min(len(x), len(y))
                    if n == 0 or len(x) != len(y):
                        continue
                    z, p, n_eff = _wilcoxon_signed_rank(x, y)
                    f.write(f"{a},{b},{z:.6f},{p:.6f},{n_eff}\n")

    _save_wilcoxon(out_dir / "stats_wilcoxon_cost.csv", cost_by_alg)
    _save_wilcoxon(out_dir / "stats_wilcoxon_time.csv", time_by_alg)
