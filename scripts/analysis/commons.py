from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

GENERATE_PDF = os.getenv("ANALYZE_PDF", "0") not in {"0", "false", "False", ""}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
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

