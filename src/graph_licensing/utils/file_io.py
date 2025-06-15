import csv
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from ..models.license import LicenseConfig, LicenseSolution


class FileIO:
    @staticmethod
    def save_graph(graph: "nx.Graph", filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix == ".pkl":
            with open(filepath, "wb") as f:
                pickle.dump(graph, f)
        elif filepath.suffix == ".gml":
            nx.write_gml(graph, filepath)
        elif filepath.suffix == ".graphml":
            nx.write_graphml(graph, filepath)
        elif filepath.suffix == ".json":
            data = nx.node_link_data(graph, edges="edges")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            msg = f"Unsupported file format: {filepath.suffix}"
            raise ValueError(msg)

    @staticmethod
    def save_solution(solution: "LicenseSolution", filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "licenses": solution.licenses,
        }

        if filepath.suffix == ".json":
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        elif filepath.suffix == ".pkl":
            with open(filepath, "wb") as f:
                pickle.dump(solution, f)
        else:
            msg = f"Unsupported file format: {filepath.suffix}"
            raise ValueError(msg)

    @staticmethod
    def save_config(config: "LicenseConfig", filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {"license_types": {}}

        for license_type, license_config in config.license_types.items():
            data["license_types"][license_type] = {
                "price": license_config.price,
                "min_size": license_config.min_size,
                "max_size": license_config.max_size,
            }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_config(config: "LicenseConfig", filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {"license_types": {}}

        for license_type, license_config in config.license_types.items():
            data["license_types"][license_type] = {
                "price": license_config.price,
                "min_size": license_config.min_size,
                "max_size": license_config.max_size,
            }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_solution_summary(
        solution: "LicenseSolution",
        config: "LicenseConfig",
        filepath: str | Path,
    ) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        license_counts = {}
        total_cost = solution.calculate_cost(config)
        total_licenses = 0
        total_members = 0

        for license_type, groups in solution.licenses.items():
            license_counts[f"n_{license_type}_licenses"] = len(groups)
            total_licenses += len(groups)

            type_members = sum(len(members) for members in groups.values())
            license_counts[f"n_{license_type}_members"] = type_members
            total_members += type_members

        data = {
            "total_cost": total_cost,
            "total_licenses": total_licenses,
            "total_members": total_members,
            "avg_group_size": total_members / total_licenses if total_licenses > 0 else 0,
            **license_counts,
            "is_valid_solution": solution.is_valid(solution.get_covered_nodes(), config),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def save_json(data: Any, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)


class CSVLogger:
    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] = []
        self.file_handle = None
        self.writer = None

    def __enter__(self):
        self.file_handle = open(self.filepath, "w", newline="")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()

    def write_header(self, fieldnames: list[str]) -> None:
        self.fieldnames = fieldnames
        self.writer = csv.DictWriter(self.file_handle, fieldnames=fieldnames)
        self.writer.writeheader()

    def write_row(self, row: dict[str, Any]) -> None:
        if not self.writer:
            msg = "Must call write_header() before writing rows"
            raise RuntimeError(msg)
        self.writer.writerow(row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.filepath.exists():
            return pd.DataFrame()
        return pd.read_csv(self.filepath)

    def log_results(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return

        # Collect all possible fieldnames from all results
        all_fieldnames = set()
        for result in results:
            all_fieldnames.update(result.keys())
        
        fieldnames = sorted(list(all_fieldnames))

        with self:
            self.write_header(fieldnames)
            for result in results:
                # Fill missing fields with None/empty values
                complete_result = {field: result.get(field, None) for field in fieldnames}
                self.write_row(complete_result)
