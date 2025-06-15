"""Refactored Simulated Annealing algorithm using common components."""

import math
import random
from typing import TYPE_CHECKING, Optional

from ..common.local_search_base import LocalSearchAlgorithm
from ..common.validation import SolutionValidator

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution
    from ..common.config import SimulatedAnnealingConfig


class SimulatedAnnealingAlgorithm(LocalSearchAlgorithm):
    """Simulated Annealing algorithm for license optimization."""

    def __init__(
        self,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        seed: int | None = None,
    ) -> None:
        super().__init__("SimulatedAnnealing", max_iterations, seed)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        
        # Algorithm state
        self.current_temp = initial_temp

    def _initialize_algorithm_state(
        self, graph: "nx.Graph", config: "LicenseConfig", initial_solution: "LicenseSolution"
    ) -> None:
        """Initialize temperature for simulated annealing."""
        self.current_temp = self.initial_temp

    def _select_next_solution(
        self,
        current_solution: "LicenseSolution",
        neighbors: list["LicenseSolution"],
        config: "LicenseConfig",
        iteration: int,
    ) -> Optional["LicenseSolution"]:
        """Select next solution using simulated annealing acceptance criteria."""
        if not neighbors:
            return None

        # Pick a random neighbor
        candidate = random.choice(neighbors)
        candidate_cost = SolutionValidator.calculate_solution_fitness(candidate, config)
        current_cost = SolutionValidator.calculate_solution_fitness(current_solution, config)

        # Always accept if better
        if candidate_cost < current_cost:
            return candidate

        # Accept worse solutions with probability based on temperature
        if self.current_temp > 0:
            delta = candidate_cost - current_cost
            acceptance_prob = math.exp(-delta / self.current_temp)
            if random.random() < acceptance_prob:
                return candidate

        return current_solution  # Stay with current solution

    def _update_algorithm_state(
        self, current_solution: "LicenseSolution", current_cost: float, iteration: int
    ) -> None:
        """Update temperature according to cooling schedule."""
        self.current_temp *= self.cooling_rate
        
    def _should_terminate(self, iteration: int, current_cost: float, best_cost: float) -> bool:
        """Terminate when temperature is too low."""
        return self.current_temp < self.final_temp

    def _get_neighborhood_size(self) -> int:
        """Use smaller neighborhood for SA since we only pick one neighbor."""
        return 20

    @classmethod
    def from_config(cls, config: "SimulatedAnnealingConfig"):
        """Create instance from configuration object."""
        return cls(
            initial_temp=config.initial_temp,
            final_temp=config.final_temp,
            cooling_rate=config.cooling_rate,
            max_iterations=config.max_iterations,
            seed=config.seed
        )
