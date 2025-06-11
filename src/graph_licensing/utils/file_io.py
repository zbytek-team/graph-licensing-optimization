"""File I/O utilities for graphs and results."""

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
    """File input/output utilities for the licensing optimization project."""

    @staticmethod
    def save_graph(graph: "nx.Graph", filepath: str | Path) -> None:
        """Save a NetworkX graph to file.

        Args:
            graph: Graph to save.
            filepath: Path to save the graph.
        """
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
        """Save a licensing solution to file.

        Args:
            solution: Solution to save.
            filepath: Path to save the solution.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "solo_nodes": solution.solo_nodes,
            "group_owners": solution.group_owners,
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
        """Save a license configuration to file.

        Args:
            config: Configuration to save.
            filepath: Path to save the configuration.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "solo_price": config.solo_price,
            "group_price": config.group_price,
            "group_size": config.group_size,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_solution_summary(
        solution: "LicenseSolution",
        config: "LicenseConfig",
        filepath: str | Path,
    ) -> None:
        """Export a human-readable solution summary.

        Args:
            solution: Solution to export.
            config: License configuration.
            filepath: Path to save the summary.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        total_cost = solution.calculate_cost(config)

        with open(filepath, "w") as f:
            f.write("LICENSING OPTIMIZATION SOLUTION SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")

            f.write(f"Total Cost: ${total_cost:.2f}\\n")
            f.write(f"Solo License Price: ${config.solo_price:.2f}\\n")
            f.write(f"Group License Price: ${config.group_price:.2f}\\n")
            f.write(f"Group Size Limit: {config.group_size}\\n\\n")

            f.write(f"Solo Licenses ({len(solution.solo_nodes)}): ")
            f.write(f"${len(solution.solo_nodes) * config.solo_price:.2f}\\n")
            f.write(f"Nodes: {sorted(solution.solo_nodes)}\\n\\n")

            f.write(f"Group Licenses ({len(solution.group_owners)}): ")
            f.write(f"${len(solution.group_owners) * config.group_price:.2f}\\n")

            for owner, members in solution.group_owners.items():
                cost_per_member = config.group_price / len(members)
                f.write(f"  Group {owner}: {sorted(members)} ")
                f.write(f"(${cost_per_member:.2f} per member)\\n")

            f.write("\\n" + "=" * 50 + "\\n")

    @staticmethod
    def save_json(data: Any, filepath: str | Path) -> None:
        """Save data to JSON file.

        Args:
            data: Data to save.
            filepath: Path to save the JSON file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class CSVLogger:
    """CSV logger for experiment results."""

    def __init__(self, filepath: str | Path) -> None:
        """Initialize the CSV logger.

        Args:
            filepath: Path to the CSV file.
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._fieldnames: list[str] = []

    def log_result(self, result: dict[str, Any]) -> None:
        """Log a single result.

        Args:
            result: Dictionary containing the result data.
        """
        if not self._initialized:
            self._fieldnames = list(result.keys())
            self._write_header()
            self._initialized = True

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(result)

    def log_results(self, results: list[dict[str, Any]]) -> None:
        """Log multiple results.

        Args:
            results: List of result dictionaries.
        """
        for result in results:
            self.log_result(result)

    def _write_header(self) -> None:
        """Write the CSV header."""
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
