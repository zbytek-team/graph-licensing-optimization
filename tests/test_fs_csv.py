from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from glopt.io import build_paths, write_csv


@dataclass
class Row:
    a: int
    b: str


def test_build_paths_and_write_csv(tmp_path: Path, monkeypatch) -> None:
    # redirect runs/ under tmp
    monkeypatch.chdir(tmp_path)
    base, graphs_dir, csv_dir = build_paths("rid")
    assert Path(base).exists() and Path(graphs_dir).exists() and Path(csv_dir).exists()
    out = write_csv(csv_dir, "rid", [Row(1, "x"), Row(2, "y")])
    p = Path(out)
    assert p.exists()
    content = p.read_text(encoding="utf-8").strip().splitlines()
    assert content[0] == "a,b" and content[1] == "1,x" and content[-1] == "2,y"

