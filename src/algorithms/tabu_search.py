from src.core import LicenseType, Solution, Algorithm
from .greedy import GreedyAlgorithm
from src.validation.solution_validator import SolutionValidator
from src.utils import MutationOperators
from typing import Any, List
import networkx as nx


class TabuSearch(Algorithm):
    @property
    def name(self) -> str:
        return "tabu_search"

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        self.validator = SolutionValidator()
        max_iterations = kwargs.get("max_iterations", 1000)
        tabu_tenure = kwargs.get("tabu_tenure", 20)
        greedy_solver = GreedyAlgorithm()
        current_solution = greedy_solver.solve(graph, license_types)
        best_solution = current_solution
        tabu_list = set()
        for _ in range(max_iterations):
            neighbors = self._generate_neighbors(current_solution, graph, license_types)
            best_neighbor = None
            best_neighbor_cost = float("inf")
            for neighbor in neighbors:
                neighbor_hash = self._solution_hash(neighbor)
                if neighbor_hash not in tabu_list or neighbor.total_cost < best_solution.total_cost:
                    if neighbor.total_cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = neighbor.total_cost
            if best_neighbor is None:
                break
            current_solution = best_neighbor
            if current_solution.total_cost < best_solution.total_cost:
                best_solution = current_solution
            tabu_list.add(self._solution_hash(current_solution))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop()
        return best_solution

    def _generate_neighbors(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> List[Solution]:
        neighbors = []
        for _ in range(10):
            neighbor = MutationOperators.apply_random_mutation(solution, graph, license_types)
            if neighbor:
                neighbors.append(neighbor)
        return neighbors

    def _solution_hash(self, solution: Solution) -> str:
        groups_repr = []
        for group in sorted(solution.groups, key=lambda g: (g.owner, g.license_type.name)):
            members_str = ",".join(map(str, sorted(group.all_members)))
            groups_repr.append(f"{group.license_type.name}:{group.owner}:{members_str}")
        return "|".join(groups_repr)
