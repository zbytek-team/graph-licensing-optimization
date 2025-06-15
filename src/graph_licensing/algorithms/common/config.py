"""Configuration classes for algorithms."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AlgorithmConfig:
    """Base configuration for algorithms."""
    seed: Optional[int] = None
    

@dataclass
class LocalSearchConfig(AlgorithmConfig):
    """Configuration for local search algorithms."""
    max_iterations: int = 1000


@dataclass
class PopulationConfig(AlgorithmConfig):
    """Configuration for population-based algorithms."""
    population_size: int = 100
    max_generations: int = 200


@dataclass
class SimulatedAnnealingConfig(LocalSearchConfig):
    """Configuration for Simulated Annealing."""
    initial_temp: float = 100.0
    final_temp: float = 0.1
    cooling_rate: float = 0.95


@dataclass
class TabuSearchConfig(LocalSearchConfig):
    """Configuration for Tabu Search."""
    tabu_tenure: int = 7
    max_no_improvement: int = 20


@dataclass
class GeneticAlgorithmConfig(PopulationConfig):
    """Configuration for Genetic Algorithm."""
    mutation_rate: float = 0.15
    crossover_rate: float = 0.85
    elite_size: int = 5


@dataclass
class AntColonyConfig(PopulationConfig):
    """Configuration for Ant Colony Optimization."""
    alpha: float = 1.0  # pheromone importance
    beta: float = 2.0   # heuristic importance
    rho: float = 0.5    # evaporation rate
    q0: float = 0.9     # exploitation vs exploration
    initial_pheromone: float = 0.1
    
    @property
    def num_ants(self) -> int:
        """Number of ants (same as population size)."""
        return self.population_size
        
    @property
    def max_iterations(self) -> int:
        """Maximum iterations (same as max generations)."""
        return self.max_generations
