"""Moduł implementuje algorytm branch and bound dla dystrybucji licencji.

Wejście zwykle obejmuje obiekt `networkx.Graph` oraz konfiguracje licencji (`LicenseType`, `LicenseGroup`).
"""

from src.core import LicenseType, Solution, Algorithm, LicenseGroup
from typing import Any, List, Set, Tuple, Optional
import networkx as nx
from dataclasses import dataclass
import heapq
from itertools import combinations


@dataclass
class BranchNode:
    assignments: dict
    unassigned_nodes: Set[int]
    lower_bound: float
    upper_bound: float
    level: int

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound


class BranchAndBound(Algorithm):
    @property
    def name(self) -> str:
        return "branch_and_bound"

    def solve(
        self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any
    ) -> Solution:
        self.graph = graph
        self.license_types = license_types
        self.nodes = list(graph.nodes())
        self.best_solution = None
        self.best_cost = float("inf")

        max_iterations = kwargs.get("max_iterations", 10000)

        initial_lower_bound = self._calculate_lower_bound({}, set(self.nodes))
        initial_upper_bound = self._calculate_upper_bound_greedy()

        root = BranchNode(
            assignments={},
            unassigned_nodes=set(self.nodes),
            lower_bound=initial_lower_bound,
            upper_bound=initial_upper_bound,
            level=0,
        )

        priority_queue = [root]
        iterations = 0

        while priority_queue and iterations < max_iterations:
            current_node = heapq.heappop(priority_queue)
            iterations += 1

            if current_node.lower_bound >= self.best_cost:
                continue

            if not current_node.unassigned_nodes:
                solution = self._build_solution_from_assignments(
                    current_node.assignments
                )
                if solution and solution.total_cost < self.best_cost:
                    self.best_solution = solution
                    self.best_cost = solution.total_cost
                continue

            children = self._branch(current_node)
            for child in children:
                if child.lower_bound < self.best_cost:
                    heapq.heappush(priority_queue, child)

        if self.best_solution is None:
            return self._fallback_greedy_solution()

        return self.best_solution

    def _branch(self, node: BranchNode) -> List[BranchNode]:
        children = []

        if not node.unassigned_nodes:
            return children

        target_node = min(node.unassigned_nodes)

        possible_assignments = self._get_possible_assignments(
            target_node, node.assignments
        )

        for assignment in possible_assignments:
            owner, license_idx, members = assignment

            new_assignments = node.assignments.copy()
            new_assignments[owner] = (license_idx, members)

            new_unassigned = node.unassigned_nodes - members

            if self._is_valid_partial_assignment(new_assignments):
                lower_bound = self._calculate_lower_bound(
                    new_assignments, new_unassigned
                )
                upper_bound = self._calculate_upper_bound(
                    new_assignments, new_unassigned
                )

                child = BranchNode(
                    assignments=new_assignments,
                    unassigned_nodes=new_unassigned,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    level=node.level + 1,
                )
                children.append(child)

        return children

    def _get_possible_assignments(
        self, target_node: int, current_assignments: dict
    ) -> List[Tuple[int, int, Set[int]]]:
        assignments = []

        neighbors = set(self.graph.neighbors(target_node)) | {target_node}
        available_neighbors = {
            n
            for n in neighbors
            if n not in self._get_assigned_nodes(current_assignments)
        }

        if not available_neighbors:
            return assignments

        for license_idx, license_type in enumerate(self.license_types):
            min_cap = license_type.min_capacity
            max_cap = license_type.max_capacity

            if len(available_neighbors) < min_cap:
                continue

            max_possible = min(max_cap, len(available_neighbors))

            for size in range(min_cap, max_possible + 1):
                if target_node in available_neighbors:
                    for members_combo in combinations(available_neighbors, size):
                        if target_node in members_combo:
                            members = set(members_combo)
                            assignments.append((target_node, license_idx, members))

        return assignments

    def _get_assigned_nodes(self, assignments: dict) -> Set[int]:
        assigned = set()
        for owner, (license_idx, members) in assignments.items():
            assigned.update(members)
        return assigned

    def _is_valid_partial_assignment(self, assignments: dict) -> bool:
        assigned_nodes = set()

        for owner, (license_idx, members) in assignments.items():
            if assigned_nodes & members:
                return False
            assigned_nodes.update(members)

            license_type = self.license_types[license_idx]
            if not (
                license_type.min_capacity <= len(members) <= license_type.max_capacity
            ):
                return False

            if owner not in members:
                return False

            for member in members:
                if member != owner and not self.graph.has_edge(owner, member):
                    return False

        return True

    def _calculate_lower_bound(
        self, assignments: dict, unassigned_nodes: Set[int]
    ) -> float:
        current_cost = sum(
            self.license_types[license_idx].cost
            for owner, (license_idx, members) in assignments.items()
        )

        if not unassigned_nodes:
            return current_cost

        min_cost_per_node = min(lt.cost / lt.max_capacity for lt in self.license_types)
        remaining_cost = len(unassigned_nodes) * min_cost_per_node

        return current_cost + remaining_cost

    def _calculate_upper_bound(
        self, assignments: dict, unassigned_nodes: Set[int]
    ) -> float:
        current_cost = sum(
            self.license_types[license_idx].cost
            for owner, (license_idx, members) in assignments.items()
        )

        if not unassigned_nodes:
            return current_cost

        cheapest_license = min(self.license_types, key=lambda x: x.cost)
        remaining_cost = len(unassigned_nodes) * cheapest_license.cost

        return current_cost + remaining_cost

    def _calculate_upper_bound_greedy(self) -> float:
        uncovered = set(self.nodes)
        total_cost = 0

        while uncovered:
            best_ratio = float("inf")
            best_assignment = None

            for node in uncovered:
                neighbors = (set(self.graph.neighbors(node)) | {node}) & uncovered

                for license_type in self.license_types:
                    if len(neighbors) >= license_type.min_capacity:
                        members = set(list(neighbors)[: license_type.max_capacity])
                        ratio = license_type.cost / len(members)

                        if ratio < best_ratio:
                            best_ratio = ratio
                            best_assignment = (node, license_type, members)

            if best_assignment:
                node, license_type, members = best_assignment
                uncovered -= members
                total_cost += license_type.cost
            else:
                break

        return total_cost

    def _build_solution_from_assignments(self, assignments: dict) -> Optional[Solution]:
        if not self._covers_all_nodes(assignments):
            return None

        groups = []
        covered_nodes = set()

        for owner, (license_idx, members) in assignments.items():
            license_type = self.license_types[license_idx]
            additional_members = members - {owner}

            group = LicenseGroup(
                license_type=license_type,
                owner=owner,
                additional_members=additional_members,
            )
            groups.append(group)
            covered_nodes.update(members)

        total_cost = sum(group.license_type.cost for group in groups)
        return Solution(
            groups=groups, total_cost=total_cost, covered_nodes=covered_nodes
        )

    def _covers_all_nodes(self, assignments: dict) -> bool:
        covered = set()
        for owner, (license_idx, members) in assignments.items():
            covered.update(members)
        return covered == set(self.nodes)

    def _fallback_greedy_solution(self) -> Solution:
        from .greedy import GreedyAlgorithm

        greedy = GreedyAlgorithm()
        return greedy.solve(self.graph, self.license_types)
