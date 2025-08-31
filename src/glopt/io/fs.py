from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str) -> None:
    Path(path).mkdir(exist_ok=True, parents=True)


def build_paths(run_id: str) -> tuple[str, str, str]:
    base = Path("runs") / run_id
    graphs_dir = base / "graphs"
    csv_dir = base / "csv"
    ensure_dir(str(graphs_dir))
    ensure_dir(str(csv_dir))
    return str(base), str(graphs_dir), str(csv_dir)
