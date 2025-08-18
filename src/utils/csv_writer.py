import csv
import os
from datetime import datetime
from typing import Dict, Any


class BenchmarkCSVWriter:
    def __init__(
        self,
        output_dir: str = "results/stats",
        summary_path: str = "results/stats_summary.csv",
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
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

        self.summary_path = summary_path
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        self.summary_fieldnames = [
            "algorithm",
            "graph_type",
            "nodes",
            "graph_k",
            "graph_p",
            "graph_m",
            "license_config",
            "cost_mean",
            "cost_std",
            "time_mean",
            "time_std",
        ]
        if not os.path.exists(summary_path):
            with open(summary_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.summary_fieldnames)
                writer.writeheader()

    def write_result(self, result: Dict[str, Any]):
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(result)

    def write_summary(self, summary: Dict[str, Any]):
        with open(self.summary_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.summary_fieldnames)
            writer.writerow(summary)

    def get_csv_path(self) -> str:
        return self.csv_path

    def get_summary_path(self) -> str:
        return self.summary_path
