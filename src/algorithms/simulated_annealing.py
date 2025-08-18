from __future__ import annotations

import math
import random
from typing import Any, List

import networkx as nx

from src.core import Algorithm, LicenseGroup, LicenseType, Solution
from src.utils import MutationOperators, SolutionBuilder


class SimulatedAnnealing(Algorithm):
    """Metaheuristic algorithm based on the simulated annealing technique."""

    @property
    def name(self) -> str:
        return "simulated_annealing"

    def solve(
        self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any
    ) -> Solution:
        """Run simulated annealing to approximate an optimal solution.

        Parameters can be provided through ``kwargs``:

        - ``T0`` – initial temperature (default: 100.0)
        - ``alpha`` – cooling rate (default: 0.95)
        - ``max_iter`` – maximal number of iterations (default: 1000)
        """

        if len(graph.nodes()) == 0:
            return Solution(groups=[], total_cost=0.0, covered_nodes=set())

        T0 = kwargs.get("T0", 100.0)
        alpha = kwargs.get("alpha", 0.95)
        max_iter = kwargs.get("max_iter", 1000)

        current_solution = self._random_initial_solution(graph, license_types)
        best_solution = current_solution
        temperature = T0

        for iteration in range(max_iter):
            neighbor = MutationOperators.apply_random_mutation(
                current_solution, graph, license_types
            )
            if neighbor is None:
                continue

            delta = neighbor.total_cost - current_solution.total_cost
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_solution = neighbor
                if current_solution.total_cost < best_solution.total_cost:
                    best_solution = current_solution

            temperature = T0 * (alpha ** (iteration + 1))
            if temperature < 1e-8:
                break

        return best_solution

    def _random_initial_solution(
        self, graph: nx.Graph, license_types: List[LicenseType]
    ) -> Solution:
        """Create a random initial solution covering all nodes."""

        nodes = list(graph.nodes())
        random.shuffle(nodes)
        uncovered = set(nodes)
        groups: List[LicenseGroup] = []

        while uncovered:
            owner = random.choice(list(uncovered))
            neighbors = set(graph.neighbors(owner)) | {owner}
            available = neighbors & uncovered

            compatible = [
                lt for lt in license_types if lt.min_capacity <= len(available)
            ]
            if compatible:
                license_type = random.choice(compatible)
                max_size = min(len(available), license_type.max_capacity)
                size = random.randint(license_type.min_capacity, max_size)
                members = set(random.sample(list(available), size))
            else:
                license_type = SolutionBuilder.find_cheapest_single_license(
                    license_types
                )
                members = {owner}

            groups.append(
                LicenseGroup(license_type, owner, members - {owner})
            )
            uncovered -= members

        return SolutionBuilder.create_solution_from_groups(groups)


__all__ = ["SimulatedAnnealing"]

