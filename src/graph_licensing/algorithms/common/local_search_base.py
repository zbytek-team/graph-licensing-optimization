"""Base class for local search algorithms."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm
from .solution_operators import SolutionOperators
from .validation import SolutionValidator

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution
    from .config import LocalSearchConfig


class LocalSearchAlgorithm(BaseAlgorithm):
    """Base class for local search algorithms like SA, Tabu Search, etc."""

    def __init__(self, name: str, max_iterations: int = 1000, seed: int | None = None) -> None:
        super().__init__(name)
        self.max_iterations = max_iterations
        self.seed = seed

    def supports_warm_start(self) -> bool:
        return True

    @classmethod
    def from_config(cls, config: "LocalSearchConfig"):
        """Create instance from configuration object."""
        return cls(
            name=cls.__name__.replace("Algorithm", ""),
            max_iterations=config.max_iterations,
            seed=config.seed
        )

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using local search."""
        import random
        from ...models.license import LicenseSolution
        from .initialization import SolutionInitializer

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        # Initialize solution
        if warm_start is not None:
            current_solution = SolutionValidator.adapt_solution_to_graph(warm_start, graph, config)
        else:
            current_solution = SolutionInitializer.create_greedy_solution(graph, config)

        current_cost = SolutionValidator.calculate_solution_fitness(current_solution, config)
        best_solution = current_solution
        best_cost = current_cost

        # Initialize algorithm-specific state
        self._initialize_algorithm_state(graph, config, current_solution)

        for iteration in range(self.max_iterations):
            # Generate neighbors
            neighbors = SolutionOperators.get_solution_neighbors(
                current_solution, graph, config, max_neighbors=self._get_neighborhood_size()
            )

            if not neighbors:
                break

            # Select next solution based on algorithm-specific criteria
            next_solution = self._select_next_solution(
                current_solution, neighbors, config, iteration
            )

            if next_solution is None:
                break

            next_cost = SolutionValidator.calculate_solution_fitness(next_solution, config)

            # Update current solution
            current_solution = next_solution
            current_cost = next_cost

            # Update best solution if improved
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            # Update algorithm-specific state
            self._update_algorithm_state(current_solution, current_cost, iteration)

            # Check termination criteria
            if self._should_terminate(iteration, current_cost, best_cost):
                break

        return best_solution

    @abstractmethod
    def _initialize_algorithm_state(
        self, graph: "nx.Graph", config: "LicenseConfig", initial_solution: "LicenseSolution"
    ) -> None:
        """Initialize algorithm-specific state."""
        pass

    @abstractmethod
    def _select_next_solution(
        self,
        current_solution: "LicenseSolution",
        neighbors: list["LicenseSolution"],
        config: "LicenseConfig",
        iteration: int,
    ) -> Optional["LicenseSolution"]:
        """Select the next solution from neighbors."""
        pass

    @abstractmethod
    def _update_algorithm_state(
        self, current_solution: "LicenseSolution", current_cost: float, iteration: int
    ) -> None:
        """Update algorithm-specific state."""
        pass

    def _should_terminate(self, iteration: int, current_cost: float, best_cost: float) -> bool:
        """Check if algorithm should terminate early."""
        return False

    def _get_neighborhood_size(self) -> int:
        """Get the size of the neighborhood to explore."""
        return 50
