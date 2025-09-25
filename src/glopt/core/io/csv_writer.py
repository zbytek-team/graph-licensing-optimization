import csv
import pathlib
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def write_csv(csv_dir: str, run_id: str, rows: Iterable[Any]) -> str:
    out_path = Path(csv_dir) / f"{run_id}.csv"
    it = iter(rows)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        first_row = next(it, None)
        if first_row is None:
            return str(out_path)
        d0 = asdict(first_row)
        writer = csv.DictWriter(f, fieldnames=list(d0.keys()))
        writer.writeheader()
        writer.writerow(d0)
        for r in it:
            writer.writerow(asdict(r))
    return str(out_path)


class BenchmarkCSVWriter:
    def __init__(self, output_dir: str = "runs/stats") -> None:
        self.output_dir = output_dir
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = str(Path(output_dir) / f"{timestamp}.csv")
        self.fieldnames = [
            "algorithm",
            "graph_type",
            "nodes",
            "edges",
            "graph_k",
            "graph_p",
            "graph_m",
            "license_config",
            "cost",
            "execution_time",
            "groups_count",
            "avg_degree",
            "seed",
        ]
        with Path(self.csv_path).open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
