"""Expose available algorithm classes.

Some algorithm implementations rely on optional third‑party dependencies or are
in an experimental state. Importing them unconditionally would make simply
importing :mod:`src.algorithms` fail if those dependencies are missing.  This
module therefore tries to import each algorithm individually and gracefully
falls back to ``None`` when an import cannot be performed.  The test suite and
consumers that only require a subset of algorithms can still import this module
without pulling in the optional ones.
"""

from .ilp import ILPSolver
from .greedy import GreedyAlgorithm
from .tabu_search import TabuSearch

try:  # optional
    from .simulated_annealing import SimulatedAnnealing  # type: ignore
except Exception:  # pragma: no cover - missing optional dependency
    SimulatedAnnealing = None  # type: ignore

try:  # optional
    from .genetic import GeneticAlgorithm  # type: ignore
except Exception:  # pragma: no cover - missing optional dependency
    GeneticAlgorithm = None  # type: ignore

from .ant_colony import AntColonyOptimization
from .tree_dp import TreeDynamicProgramming
from .branch_and_bound import BranchAndBound
from .naive import NaiveAlgorithm
from .dominating_set import DominatingSetAlgorithm
from .randomized import RandomizedAlgorithm

__all__ = [
    "ILPSolver",
    "GreedyAlgorithm",
    "TabuSearch",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "AntColonyOptimization",
    "TreeDynamicProgramming",
    "BranchAndBound",
    "NaiveAlgorithm",
    "DominatingSetAlgorithm",
    "RandomizedAlgorithm",
]
