from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import networkx as nx

from ..core import Algorithm, LicenseType, Solution
from ..core.mutations import MutationOperators
from ..core.solution_validator import SolutionValidator
from .randomized import RandomizedAlgorithm


class GeneticAlgorithm(Algorithm):

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 40,
        elite_fraction: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.population_size = max(2, population_size)
        self.generations = max(1, generations)
        self.elite_fraction = max(0.0, min(1.0, elite_fraction))
        self.seed = seed
        self.validator = SolutionValidator()

    @property
    def name(self) -> str:
        return "genetic"

    def solve(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        **_: Any,
    ) -> Solution:
        if self.seed is not None:
            random.seed(self.seed)
        if graph.number_of_nodes() == 0:
            return Solution()

        population = self._init_population(graph, license_types)
        best = min(population, key=lambda s: s.total_cost)

        for _ in range(self.generations):
            population.sort(key=lambda s: s.total_cost)
            elite_count = max(1, int(self.elite_fraction * self.population_size))
            new_pop: list[Solution] = population[:elite_count]

            while len(new_pop) < self.population_size:
                parent = self._tournament_selection(population)
                child = self._mutate(parent, graph, license_types)
                new_pop.append(child)

            population = new_pop
            current_best = min(population, key=lambda s: s.total_cost)
            if current_best.total_cost < best.total_cost:
                best = current_best

        return best

    def _init_population(self, graph: nx.Graph, license_types: Sequence[LicenseType]) -> list[Solution]:
        rand_algo = RandomizedAlgorithm()
        return [rand_algo.solve(graph, list(license_types)) for _ in range(self.population_size)]

    def _tournament_selection(self, population: list[Solution], k: int = 3) -> Solution:
        contenders = random.sample(population, k)
        return min(contenders, key=lambda s: s.total_cost)

    def _mutate(
        self,
        solution: Solution,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Solution:
        neighbors = MutationOperators.generate_neighbors(solution, graph, license_types, k=5)
        valid_neighbors = [s for s in neighbors if self.validator.is_valid_solution(s, graph)]
        if not valid_neighbors:
            return solution
        return min(valid_neighbors, key=lambda s: s.total_cost)
