"""Moduł implementuje algorytm naive dla dystrybucji licencji.

Wejście zwykle obejmuje obiekt `networkx.Graph` oraz konfiguracje licencji (`LicenseType`, `LicenseGroup`).
"""

from src.core import LicenseType, Solution, Algorithm, LicenseGroup
from src.utils import SolutionBuilder
from typing import Any, List, Set, Tuple, Iterator
import networkx as nx
from itertools import product


class NaiveAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "naive_algorithm"

    def solve(
        self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any
    ) -> Solution:
        nodes = list(graph.nodes())
        n = len(nodes)

        if n > 10:
            raise ValueError(f"Graph too large for naive algorithm: {n} nodes > 10")

        if n == 0:
            return Solution(groups=[], total_cost=0.0, covered_nodes=set())

        best_solution = None
        best_cost = float("inf")

        for assignment in self._generate_all_assignments(nodes, graph, license_types):
            if assignment and self._is_valid_assignment(assignment, nodes, graph):
                cost = self._calculate_cost(assignment)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = assignment

        if best_solution is None:
            cheapest_individual = min(
                license_types,
                key=lambda lt: (
                    lt.cost if lt.min_capacity <= 1 <= lt.max_capacity else float("inf")
                ),
            )
            groups = [LicenseGroup(cheapest_individual, node, set()) for node in nodes]
            return SolutionBuilder.create_solution_from_groups(groups)

        return self._create_solution_from_assignment(best_solution)

    def _generate_all_assignments(
        self, nodes: List[Any], graph: nx.Graph, license_types: List[LicenseType]
    ) -> Iterator[List[Tuple[LicenseType, Any, Set[Any]]]]:
        for partition in self._generate_partitions(nodes):
            for assignment in self._generate_assignments_for_partition(
                partition, graph, license_types
            ):
                yield assignment

    def _generate_partitions(self, nodes: List[Any]) -> Iterator[List[Set[Any]]]:
        n = len(nodes)

        if n == 0:
            yield []
            return

        if n == 1:
            yield [{nodes[0]}]
            return

        first = nodes[0]
        rest = nodes[1:]

        for smaller_partition in self._generate_partitions(rest):
            yield [{first}] + smaller_partition

            for i, part in enumerate(smaller_partition):
                new_partition = smaller_partition.copy()
                new_partition[i] = part | {first}
                yield new_partition

    def _generate_assignments_for_partition(
        self,
        partition: List[Set[Any]],
        graph: nx.Graph,
        license_types: List[LicenseType],
    ) -> Iterator[List[Tuple[LicenseType, Any, Set[Any]]]]:
        if not partition:
            yield []
            return

        assignments = []
        for part in partition:
            part_assignments = []
            for license_type in license_types:
                if license_type.min_capacity <= len(part) <= license_type.max_capacity:
                    for owner in part:
                        if self._is_valid_group(owner, part - {owner}, graph):
                            part_assignments.append(
                                (license_type, owner, part - {owner})
                            )
            assignments.append(part_assignments)

        if all(assignments):
            for combination in product(*assignments):
                yield list(combination)

    def _is_valid_group(self, owner: Any, members: Set[Any], graph: nx.Graph) -> bool:
        owner_neighbors = set(graph.neighbors(owner))
        return all(member in owner_neighbors for member in members)

    def _is_valid_assignment(
        self,
        assignment: List[Tuple[LicenseType, Any, Set[Any]]],
        nodes: List[Any],
        graph: nx.Graph,
    ) -> bool:
        all_covered = set()

        for license_type, owner, members in assignment:
            group_nodes = {owner} | members

            if not self._is_valid_group(owner, members, graph):
                return False

            if (
                len(group_nodes) < license_type.min_capacity
                or len(group_nodes) > license_type.max_capacity
            ):
                return False

            if group_nodes & all_covered:
                return False

            all_covered.update(group_nodes)

        return all_covered == set(nodes)

    def _calculate_cost(
        self, assignment: List[Tuple[LicenseType, Any, Set[Any]]]
    ) -> float:
        return sum(license_type.cost for license_type, _, _ in assignment)

    def _create_solution_from_assignment(
        self, assignment: List[Tuple[LicenseType, Any, Set[Any]]]
    ) -> Solution:
        groups = []
        for license_type, owner, members in assignment:
            group = LicenseGroup(license_type, owner, members)
            groups.append(group)

        return SolutionBuilder.create_solution_from_groups(groups)
