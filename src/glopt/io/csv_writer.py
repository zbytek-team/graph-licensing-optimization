import csv
import os
import pathlib
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from typing import Any


def write_csv(csv_dir: str, run_id: str, rows: Iterable[Any]) -> str:
    out_path = os.path.join(csv_dir, f"{run_id}.csv")
    first = True
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = None
        for r in rows:
            d = asdict(r)
            if first:
                writer = csv.DictWriter(f, fieldnames=list(d.keys()))
                writer.writeheader()
                first = False
            writer.writerow(d)
    return out_path


class BenchmarkCSVWriter:
    def __init__(self, output_dir: str = "runs/stats"):
        self.output_dir = output_dir
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"{timestamp}.csv")
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
        with open(self.csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def write_result(self, result: dict[str, Any]):
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(result)

    def get_csv_path(self) -> str:
        return self.csv_path
