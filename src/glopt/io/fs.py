from __future__ import annotations

import os
import pathlib


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(exist_ok=True, parents=True)


def build_paths(run_id: str) -> tuple[str, str, str]:
    base = os.path.join("runs", run_id)
    graphs_dir = os.path.join(base, "graphs")
    csv_dir = os.path.join(base, "csv")
    ensure_dir(graphs_dir)
    ensure_dir(csv_dir)
    return base, graphs_dir, csv_dir
