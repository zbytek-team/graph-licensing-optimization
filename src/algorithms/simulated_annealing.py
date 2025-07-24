from ..core.types import LicenseType, Solution, Algorithm, LicenseGroup
from .greedy import GreedyAlgorithm
from ..utils.validation import SolutionValidator
from ..utils.solution_utils import SolutionBuilder
from ..utils.mutation_operators import MutationOperators

from typing import Any, List
import random
import math
import networkx as nx


class SimulatedAnnealing(Algorithm):
    @property
    def name(self) -> str:
        return "simulated_annealing"

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        # Initialize utilities
        self.validator = SolutionValidator()

        max_iterations = kwargs.get("max_iterations", 3000)
        initial_temperature = kwargs.get("initial_temperature", 1000.0)
        cooling_rate = kwargs.get("cooling_rate", 0.999)
        min_temperature = kwargs.get("min_temperature", 0.1)

        # Warm start with greedy algorithm
        greedy_solver = GreedyAlgorithm()
        current_solution = greedy_solver.solve(graph, license_types)
        best_solution = current_solution

        temperature = initial_temperature
        accepted_moves = 0
        rejected_moves = 0
        no_neighbor_count = 0

        for iteration in range(max_iterations):
            if temperature < min_temperature:
                break

            # Generate a random neighbor
            neighbor = self._generate_neighbor(current_solution, graph, license_types)

            if neighbor is None:
                no_neighbor_count += 1
                continue

            # Calculate cost difference
            delta_cost = neighbor.total_cost - current_solution.total_cost

            # Accept or reject the neighbor
            if delta_cost < 0:  # Better solution
                current_solution = neighbor
                accepted_moves += 1
                if neighbor.total_cost < best_solution.total_cost:
                    best_solution = neighbor
            else:  # Worse solution - accept with probability
                probability = math.exp(-delta_cost / temperature)
                if random.random() < probability:
                    current_solution = neighbor
                    accepted_moves += 1
                else:
                    rejected_moves += 1

            # Cool down
            temperature *= cooling_rate

        return best_solution

    def _generate_neighbor(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Generate a random neighbor using one of several neighborhood operators"""
        return MutationOperators.apply_random_mutation(solution, graph, license_types)

    def _create_solution_from_groups(self, groups: List[LicenseGroup]) -> Solution:
        """Create a solution from a list of groups"""
        return SolutionBuilder.create_solution_from_groups(groups)

    def _is_valid_solution(self, solution: Solution, graph: nx.Graph) -> bool:
        """Check if a solution is valid"""
        return self.validator.is_valid_solution(solution, graph)
