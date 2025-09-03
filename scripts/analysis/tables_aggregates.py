from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, pstdev
from typing import Any

from .commons import ensure_dir


def write_aggregates(rows: list[dict[str, Any]], out_path: Path) -> None:
    groups = defaultdict(list)
    for r in rows:
        try:
            key = (str(r['algorithm']), str(r.get('graph','')), int(float(r.get('n_nodes', 0))))
            groups[key].append(r)
        except Exception:
            continue
    lines = []
    for (alg, gname, n), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        costs = [] ; times = []
        for r in rs:
            try:
                costs.append(float(r['total_cost'])) ; times.append(float(r['time_ms']))
            except Exception:
                pass
        if not costs:
            continue
        m_c = mean(costs)
        s_c = pstdev(costs) if len(costs) <= 1 else stdev(costs)
        m_t = mean(times)
        s_t = pstdev(times) if len(times) <= 1 else stdev(times)
        nrep = len(costs)
        ci_c = 1.96 * (s_c / (nrep ** 0.5)) if nrep > 1 else 0.0
        ci_t = 1.96 * (s_t / (nrep ** 0.5)) if nrep > 1 else 0.0
        lines.append({
            'algorithm': alg,
            'graph': gname,
            'n_nodes': n,
            'rep': nrep,
            'cost_mean': f"{m_c:.6f}",
            'cost_std': f"{s_c:.6f}",
            'cost_ci95': f"{ci_c:.6f}",
            'time_ms_mean': f"{m_t:.6f}",
            'time_ms_std': f"{s_t:.6f}",
            'time_ms_ci95': f"{ci_t:.6f}",
        })
    if lines:
        ensure_dir(out_path.parent)
        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(lines[0].keys()))
            writer.writeheader()
            writer.writerows(lines)

