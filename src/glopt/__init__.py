from .utils.runtime import ensure_python_313

# Enforce Python 3.13 at import-time (clear error early)
ensure_python_313()

from . import algorithms
from .core import (
    Algorithm,
    LicenseGroup,
    LicenseType,
    MutationOperators,
    RunResult,
    Solution,
    SolutionBuilder,
    SolutionValidator,
    generate_graph,
    instantiate_algorithms,
    run_once,
)
from .dynamic_simulator import DynamicNetworkSimulator
from .io.csv_writer import BenchmarkCSVWriter
from .io.data_loader import RealWorldDataLoader
from .io.graph_generator import GraphGeneratorFactory
from .io.graph_visualizer import GraphVisualizer
from .license_config import LicenseConfigFactory

__all__ = [
    "Algorithm",
    "BenchmarkCSVWriter",
    "DynamicNetworkSimulator",
    "GraphGeneratorFactory",
    "GraphVisualizer",
    "LicenseConfigFactory",
    "LicenseGroup",
    "LicenseType",
    "MutationOperators",
    "RealWorldDataLoader",
    "RunResult",
    "Solution",
    "SolutionBuilder",
    "SolutionValidator",
    "algorithms",
    "generate_graph",
    "instantiate_algorithms",
    "run_once",
]
