from ..core.types import LicenseType, Solution, Algorithm, LicenseGroup
from .greedy import GreedyAlgorithm
from ..utils.validation import SolutionValidator
from ..utils.solution_utils import SolutionBuilder
from ..utils.mutation_operators import MutationOperators

from typing import Any, List, Set, Tuple
import random
import networkx as nx


class TabuSearch(Algorithm):
    @property
    def name(self) -> str:
        return "tabu_search"

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        # Initialize utilities
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

    def _generate_initial_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []

        while uncovered:
            node = random.choice(list(uncovered))
            neighbors = set(graph.neighbors(node)) | {node}
            available = neighbors & uncovered

            license_type = random.choice(license_types)

            max_size = min(len(available), license_type.max_capacity)
            min_size = max(1, license_type.min_capacity)

            if max_size < min_size:
                license_type = min(license_types, key=lambda lt: lt.min_capacity)
                min_size = license_type.min_capacity
                max_size = min(len(available), license_type.max_capacity)

            if max_size >= min_size:
                group_size = random.randint(min_size, max_size)
                members = random.sample(list(available), group_size)

                additional_members = set(members) - {node}
                group = LicenseGroup(license_type, node, additional_members)
                groups.append(group)
                uncovered -= set(members)

        total_cost = sum(g.license_type.cost for g in groups)
        covered = set()
        for g in groups:
            covered.update(g.all_members)

        return Solution(groups, total_cost, covered)

    def _generate_neighbors(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> List[Solution]:
        neighbors = []

        # Generate multiple neighbors using mutation operators
        for _ in range(10):  # Generate multiple neighbors
            neighbor = MutationOperators.apply_random_mutation(solution, graph, license_types)
            if neighbor:
                neighbors.append(neighbor)

        return neighbors

    def _is_valid_solution(self, solution: Solution, graph: nx.Graph) -> bool:
        return self.validator.is_valid_solution(solution, graph)

    def _solution_hash(self, solution: Solution) -> str:
        groups_repr = []
        for group in sorted(solution.groups, key=lambda g: (g.owner, g.license_type.name)):
            members_str = ",".join(map(str, sorted(group.all_members)))
            groups_repr.append(f"{group.license_type.name}:{group.owner}:{members_str}")
        return "|".join(groups_repr)
