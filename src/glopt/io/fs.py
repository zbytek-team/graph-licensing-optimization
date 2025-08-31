from __future__ import annotations

import os
from typing import Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_paths(run_id: str) -> Tuple[str, str, str]:
    base = os.path.join("runs", run_id)
    graphs_dir = os.path.join(base, "graphs")
    csv_dir = os.path.join(base, "csv")
    ensure_dir(graphs_dir)
    ensure_dir(csv_dir)
    return base, graphs_dir, csv_dir
