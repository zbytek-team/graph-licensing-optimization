from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from glopt.core import Algorithm, LicenseType, Solution
from glopt.core.algorithms.greedy import GreedyAlgorithm
from glopt.core.algorithms.randomized import RandomizedAlgorithm
from glopt.core.mutations import MutationOperators
from glopt.core.solution_builder import SolutionBuilder
from glopt.core.solution_validator import SolutionValidator

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx


class GeneticAlgorithm(Algorithm):
    def __init__(
        self,
        population_size: int = 30,
        num_generations: int = 40,
        elite_fraction: float = 0.2,
        crossover_rate: float = 0.6,
        seed: int | None = None,
    ) -> None:
        self.population_size = max(2, population_size)
        self.num_generations = max(1, num_generations)
        self.elite_fraction = max(0.0, min(1.0, elite_fraction))
        self.crossover_rate = max(0.0, min(1.0, crossover_rate))
        self.seed = seed
        self.validator = SolutionValidator()

    @property
    def name(self) -> str:
        return "genetic"

    def solve(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        seed = kwargs.get("seed", self.seed)
        if isinstance(seed, int):
            random.seed(seed)
        deadline = kwargs.get("deadline")
        initial: Solution | None = kwargs.get("initial_solution")
        num_generations = int(kwargs.get("num_generations", self.num_generations))
        if graph.number_of_nodes() == 0:
            return Solution()
        population = self._init_population(graph, license_types, initial)
        best = min(population, key=lambda s: s.total_cost)
        from time import perf_counter as _pc

        for _ in range(num_generations):
            if deadline is not None and _pc() >= float(deadline):
                break
            population.sort(key=lambda s: s.total_cost)
            elite_count = max(1, int(self.elite_fraction * self.population_size))
            new_pop: list[Solution] = population[:elite_count]
            while len(new_pop) < self.population_size:
                if random.random() < self.crossover_rate and len(population) >= 2:
                    p1 = self._tournament_selection(population)
                    p2 = self._tournament_selection(population)
                    if p2 is p1 and len(population) > 1:
                        p2 = random.choice(population)
                    child = self._crossover(p1, p2, graph, license_types)
                    if not self.validator.is_valid_solution(child, graph):
                        base = min([p1, p2], key=lambda s: s.total_cost)
                        child = self._mutate(base, graph, license_types)
                else:
                    parent = self._tournament_selection(population)
                    child = self._mutate(parent, graph, license_types)
                new_pop.append(child)
            population = new_pop
            current_best = min(population, key=lambda s: s.total_cost)
            if current_best.total_cost < best.total_cost:
                best = current_best
        return best

    def _init_population(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        initial: Solution | None = None,
    ) -> list[Solution]:
        pop: list[Solution] = []
        if initial is not None and self.validator.is_valid_solution(initial, graph):
            pop.append(initial)
        try:
            greedy = GreedyAlgorithm().solve(graph, list(license_types))
            pop.append(greedy)
        except Exception:
            pass
        rand_algo = RandomizedAlgorithm()
        while len(pop) < self.population_size:
            pop.append(rand_algo.solve(graph, list(license_types)))
        return pop

    def _tournament_selection(self, population: list[Solution], k: int = 3) -> Solution:
        k = max(1, min(k, len(population)))
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
            try:
                greedy = GreedyAlgorithm().solve(graph, list(license_types))
                if self.validator.is_valid_solution(greedy, graph) and greedy.total_cost <= solution.total_cost:
                    return greedy
            except Exception:
                pass
            return solution
        return min(valid_neighbors, key=lambda s: s.total_cost)

    def _crossover(
        self,
        p1: Solution,
        p2: Solution,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Solution:
        def eff(g):
            return (g.license_type.cost / max(1, g.size), -g.size)

        candidates = list(p1.groups) + list(p2.groups)
        candidates.sort(key=eff)
        used = set()
        chosen: list = []
        for g in candidates:
            if used.isdisjoint(g.all_members):
                chosen.append(g)
                used.update(g.all_members)
        uncovered = set(graph.nodes()) - used
        if uncovered:
            H = graph.subgraph(uncovered)
            filler = GreedyAlgorithm().solve(H, list(license_types))
            for fg in filler.groups:
                if set(fg.all_members).issubset(uncovered):
                    chosen.append(fg)
        child = SolutionBuilder.create_solution_from_groups(chosen)
        if not self.validator.is_valid_solution(child, graph):
            return GreedyAlgorithm().solve(graph, list(license_types))
        return child
