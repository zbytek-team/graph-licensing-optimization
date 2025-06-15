"""Base class for population-based algorithms."""

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional

from ..base import BaseAlgorithm
from .initialization import SolutionInitializer
from .validation import SolutionValidator

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution
    from .config import PopulationConfig


class PopulationBasedAlgorithm(BaseAlgorithm):
    """Base class for population-based algorithms like GA, ACO, etc."""

    def __init__(
        self,
        name: str,
        population_size: int = 100,
        max_generations: int = 200,
        seed: int | None = None,
    ) -> None:
        super().__init__(name)
        self.population_size = population_size
        self.max_generations = max_generations
        self.seed = seed

    def supports_warm_start(self) -> bool:
        return True

    @classmethod
    def from_config(cls, config: "PopulationConfig"):
        """Create instance from configuration object."""
        return cls(
            name=cls.__name__.replace("Algorithm", ""),
            population_size=config.population_size,
            max_generations=config.max_generations,
            seed=config.seed,
        )

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using population-based approach."""
        import random
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        # Initialize population
        population = self._initialize_population(graph, config, warm_start)

        # Initialize algorithm-specific state
        self._initialize_algorithm_state(graph, config, population)

        best_solution = None
        best_fitness = float("inf")
        stagnation_count = 0
        max_stagnation = self._get_max_stagnation()

        for generation in range(self.max_generations):
            # Evaluate population
            fitness_scores = self._evaluate_population(population, config)

            # Track best solution
            current_best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_idx]
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Check for early termination
            if stagnation_count >= max_stagnation:
                break

            # Evolve population
            population = self._evolve_population(
                population, fitness_scores, graph, config, generation
            )

            # Update algorithm-specific state
            self._update_algorithm_state(population, fitness_scores, generation)

        return best_solution if best_solution else population[0]

    def _initialize_population(
        self, graph: "nx.Graph", config: "LicenseConfig", warm_start: Optional["LicenseSolution"] = None
    ) -> List["LicenseSolution"]:
        """Initialize the population with diverse solutions."""
        return SolutionInitializer.create_diverse_population(
            graph, config, self.population_size, warm_start
        )

    def _evaluate_population(
        self, population: List["LicenseSolution"], config: "LicenseConfig"
    ) -> List[float]:
        """Evaluate fitness of all solutions in population."""
        return [
            SolutionValidator.calculate_solution_fitness(solution, config)
            for solution in population
        ]

    @abstractmethod
    def _initialize_algorithm_state(
        self, graph: "nx.Graph", config: "LicenseConfig", population: List["LicenseSolution"]
    ) -> None:
        """Initialize algorithm-specific state."""
        pass

    @abstractmethod
    def _evolve_population(
        self,
        population: List["LicenseSolution"],
        fitness_scores: List[float],
        graph: "nx.Graph",
        config: "LicenseConfig",
        generation: int,
    ) -> List["LicenseSolution"]:
        """Evolve the population to the next generation."""
        pass

    @abstractmethod
    def _update_algorithm_state(
        self, population: List["LicenseSolution"], fitness_scores: List[float], generation: int
    ) -> None:
        """Update algorithm-specific state."""
        pass

    def _get_max_stagnation(self) -> int:
        """Get maximum number of generations without improvement before stopping."""
        return max(20, self.max_generations // 10)

    def _select_parents(
        self, population: List["LicenseSolution"], fitness_scores: List[float], num_parents: int
    ) -> List["LicenseSolution"]:
        """Select parents for reproduction using tournament selection."""
        import random

        parents = []
        tournament_size = max(2, len(population) // 10)

        for _ in range(num_parents):
            # Tournament selection
            tournament_indices = random.sample(range(len(population)), tournament_size)
            winner_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
            parents.append(population[winner_idx])

        return parents

    def _get_elite_solutions(
        self, population: List["LicenseSolution"], fitness_scores: List[float], elite_size: int
    ) -> List["LicenseSolution"]:
        """Get the best solutions from the current population."""
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_size]
        return [population[i] for i in elite_indices]
