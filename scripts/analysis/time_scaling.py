from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from .commons import ensure_dir


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for simple linear regression.
    Assumes len(xs) == len(ys) >= 2.
    """
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan"), float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    # r2
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return slope, intercept, r2


def write_time_scaling(rows: list[dict[str, object]], out_dir: Path) -> None:
    """Fit time_ms ≈ a * n^b (log-log) for each (license, graph, algorithm)."""
    ensure_dir(out_dir)
    # Group per (license, graph, algorithm) and per n_nodes take mean time
    acc: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        try:
            lic = str(r.get("license_config", ""))
            g = str(r.get("graph", ""))
            alg = str(r.get("algorithm", ""))
            n = int(float(str(r.get("n_nodes", 0))))
            t = float(str(r.get("time_ms", 0.0)))
        except Exception:
            continue
        if not (lic and g and alg) or n <= 1 or t <= 0 or not math.isfinite(t):
            continue
        acc[(lic, g, alg, n)].append(t)

    # Collapse to means per n
    by_group: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for (lic, g, alg, n), vs in acc.items():
        by_group[(lic, g, alg)][n] = sum(vs) / len(vs)

    out = out_dir / "time_scaling.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("license,graph,algorithm,k_points,slope_b,intercept_log_a,r2\n")
        for (lic, g, alg), m in sorted(by_group.items()):
            ns = sorted(m.keys())
            if len(ns) < 2:
                continue
            xs = [math.log(n, 10) for n in ns]
            ys = [math.log(max(m[n], 1e-12), 10) for n in ns]
            b, a, r2 = _linreg(xs, ys)
            f.write(f"{lic},{g},{alg},{len(ns)},{b:.6f},{a:.6f},{r2:.6f}\n")
