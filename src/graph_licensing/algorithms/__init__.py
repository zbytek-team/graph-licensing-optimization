"""New algorithms module with enhanced structure and dynamic support."""

from .ant_colony import AntColonyAlgorithm
from .base import BaseAlgorithm
from .dominating_set import DominatingSetAlgorithm
from .genetic import GeneticAlgorithm
from .greedy import GreedyAlgorithm
from .ilp import ILPAlgorithm
from .naive import NaiveAlgorithm
from .randomized import RandomizedAlgorithm
from .simulated_annealing import SimulatedAnnealingAlgorithm
from .tabu_search import TabuSearchAlgorithm

__all__ = [
    "AntColonyAlgorithm",
    "BaseAlgorithm",
    "DominatingSetAlgorithm",
    "GeneticAlgorithm", 
    "GreedyAlgorithm",
    "ILPAlgorithm",
    "NaiveAlgorithm",
    "RandomizedAlgorithm",
    "SimulatedAnnealingAlgorithm",
    "TabuSearchAlgorithm",
]
