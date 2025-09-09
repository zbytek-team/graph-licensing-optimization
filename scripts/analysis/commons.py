from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

GENERATE_PDF = os.getenv("ANALYZE_PDF", "0") not in {"0", "false", "False", ""}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _enrich_row(r: dict[str, Any]) -> dict[str, Any]:
    # Derive density/avg_degree if missing and (n_nodes, n_edges) present
    try:
        n = int(float(r.get("n_nodes", 0)))
        m = int(float(r.get("n_edges", 0)))
        if "avg_degree" not in r or r.get("avg_degree") in (None, ""):
            r["avg_degree"] = (2.0 * m / n) if n > 0 else 0.0
        if "density" not in r or r.get("density") in (None, ""):
            r["density"] = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
    except Exception:
        pass
    return r


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(_enrich_row(r))
    return rows


def group_cost_by_n(rows: list[dict[str, Any]], algokey: str = "algorithm") -> dict[str, list[tuple[int, float]]]:
    data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        try:
            alg = str(r[algokey])
            n = int(float(r.get("n_nodes", 0)))
            cost = float(r.get("total_cost", 0.0))
        except Exception:
            continue
        data[alg].append((n, cost))
    for alg, pts in data.items():
        pts.sort(key=lambda x: x[0])
    return data
