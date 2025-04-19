from __future__ import annotations
import random
from collections import defaultdict
import networkx as nx
from src.algorithms.base import BaseSolver, Solution
from src.utils.solution_utils import calculate_cost


class AntColonySolver(BaseSolver):
    def __init__(
        self,
        ant_count: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.3,
        iterations: int = 100,
    ):
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.pheromone: dict[int, float] = defaultdict(lambda: 1.0)

    def _build_solution(
        self,
        graph: nx.Graph,
        c_single: float,
        c_group: float,
        group_size: int,
    ) -> Solution:
        uncovered = set(graph.nodes)
        solution: Solution = {"singles": [], "groups": []}
        degree = {v: graph.degree[v] for v in graph.nodes}

        while uncovered:
            nodes = list(uncovered)
            heur = [degree[v] ** self.beta for v in nodes]
            pher = [self.pheromone[v] ** self.alpha for v in nodes]
            probs = [h * p for h, p in zip(heur, pher)]
            total = sum(probs)
            chosen = random.choice(nodes) if total == 0 else random.choices(nodes, weights=probs, k=1)[0]

            neigh = [u for u in graph.neighbors(chosen) if u in uncovered]
            neigh.sort(key=lambda v: degree[v], reverse=True)
            group = [chosen] + neigh[: group_size - 1]

            if len(group) >= 2 and c_group < len(group) * c_single:
                solution["groups"].append({"license_holder": chosen, "members": group})
                uncovered.difference_update(group)
            else:
                solution["singles"].append(chosen)
                uncovered.remove(chosen)

        return solution

    def solve(self, graph: nx.Graph, c_single: float, c_group: float, group_size: int) -> Solution:
        best_solution: Solution | None = None
        best_cost = float("inf")

        for _ in range(self.iterations):
            iteration_solutions: list[tuple[Solution, float]] = []
            for _ in range(self.ant_count):
                sol = self._build_solution(graph, c_single, c_group, group_size)
                cost = calculate_cost(sol, c_single, c_group)
                iteration_solutions.append((sol, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_solution = sol

            for v in self.pheromone:
                self.pheromone[v] *= 1.0 - self.evaporation_rate

            for sol, cost in iteration_solutions:
                if cost <= 0:
                    continue
                deposit = 1.0 / cost
                for g in sol["groups"]:
                    self.pheromone[g["license_holder"]] += deposit
                for s in sol["singles"]:
                    self.pheromone[s] += deposit * 0.5

        if best_solution is None:
            best_solution = self._build_solution(graph, c_single, c_group, group_size)

        return best_solution
