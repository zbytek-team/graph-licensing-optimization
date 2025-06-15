"""Common utilities for CLI scripts."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import networkx as nx

from graph_licensing.algorithms import (
    AntColonyAlgorithm,
    DominatingSetAlgorithm,
    GeneticAlgorithm,
    GreedyAlgorithm,
    ILPAlgorithm,
    NaiveAlgorithm,
    RandomizedAlgorithm,
    SimulatedAnnealingAlgorithm,
    TabuSearchAlgorithm,
)
from graph_licensing.generators.graph_generator import GraphGenerator
from graph_licensing.models.license import LicenseConfig
from graph_licensing.utils import FileIO


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("graph_licensing.log"),
        ],
    )


def get_timestamp_suffix() -> str:
    """Get timestamp suffix for unique naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_timestamped_path(base_path: str, command_name: str) -> Path:
    """Create timestamped output path."""
    timestamp = get_timestamp_suffix()
    return Path(base_path) / f"{command_name}_{timestamp}"


def get_algorithms() -> Dict[str, Any]:
    """Get dictionary of all available algorithms."""
    return {
        "ant_colony": AntColonyAlgorithm(),
        "greedy": GreedyAlgorithm(),
        "genetic": GeneticAlgorithm(),
        "simulated_annealing": SimulatedAnnealingAlgorithm(),
        "tabu_search": TabuSearchAlgorithm(),
        "ilp": ILPAlgorithm(),
        "naive": NaiveAlgorithm(),
        "dominating_set": DominatingSetAlgorithm(),
        "randomized": RandomizedAlgorithm(),
    }


def create_license_config(solo_cost: float = 1.0, group_cost: float = 2.08, group_size: int = 6) -> LicenseConfig:
    """Create license configuration with given parameters."""
    return LicenseConfig.create_flexible(
        {
            "solo": {"price": solo_cost, "min_size": 1, "max_size": 1},
            # "duo": {"price": solo_cost * 1.6, "min_size": 2, "max_size": 2},
            "family": {"price": group_cost, "min_size": 2, "max_size": group_size},
        }
    )


def create_test_graph(graph_type: str, size: int, seed: int = None, **kwargs) -> nx.Graph:
    """Create test graph with given parameters."""
    return GraphGenerator.generate_graph(graph_type=graph_type, size=size, seed=seed, **kwargs)


def calculate_solution_stats(solution, config, graph=None) -> Dict[str, Any]:
    """Calculate standard statistics for a solution."""
    total_cost = solution.calculate_cost(config)

    solo_count = sum(
        1 for license_type, groups in solution.licenses.items() for members in groups.values() if len(members) == 1
    )

    group_count = sum(
        1 for license_type, groups in solution.licenses.items() for members in groups.values() if len(members) > 1
    )

    is_valid = solution.is_valid(graph, config) if graph is not None else True

    return {
        "total_cost": total_cost,
        "solo_licenses": solo_count,
        "group_licenses": group_count,
        "valid": is_valid,
    }


def create_metadata(start_time: datetime, **kwargs) -> Dict[str, Any]:
    """Create standard metadata dictionary."""
    end_time = datetime.now()
    metadata = {
        "timestamp": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
    }
    metadata.update(kwargs)
    return metadata


def save_results(results: Dict[str, Any], output_dir: Path, prefix: str = "results") -> None:
    """Save results to JSON file."""
    results_path = output_dir / f"{prefix}.json"
    FileIO.save_json(results, results_path)
    return results_path


def print_solution_summary(algorithm: str, stats: Dict[str, Any]) -> None:
    """Print standardized solution summary."""
    status = "✓" if stats["valid"] else "✗"
    print(
        f"  {status} {algorithm}: Cost: {stats['total_cost']:.2f}, "
        f"Solo: {stats['solo_licenses']}, Groups: {stats['group_licenses']}"
    )


def print_comparison_table(results: list) -> None:
    """Print comparison table of results."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    results.sort(key=lambda x: x["total_cost"])

    for i, result in enumerate(results, 1):
        status = "✓" if result["valid"] else "✗"
        print(
            f"{i}. {result['algorithm']}: ${result['total_cost']:.2f} "
            f"(Solo: {result['solo_licenses']}, Groups: {result['group_licenses']}) {status}"
        )
