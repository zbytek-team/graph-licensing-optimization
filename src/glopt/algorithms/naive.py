from itertools import product
from typing import Any, Iterator, List, Sequence, Set, Tuple

import networkx as nx

from ..core import Algorithm, LicenseGroup, LicenseType, Solution
from ..core.solution_builder import SolutionBuilder


Assignment = List[Tuple[LicenseType, Any, Set[Any]]]


class NaiveAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "naive_algorithm"

    def solve(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        nodes: List[Any] = list(graph.nodes())
        n = len(nodes)

        if n > 10:
            raise ValueError(f"graph too large for naive algorithm: {n} nodes > 10")

        if n == 0:
            return Solution(groups=())

        best: Assignment | None = None
        best_cost: float = float("inf")

        for assignment in self._generate_all_assignments(nodes, graph, license_types):
            if not assignment:
                continue
            if self._is_valid_assignment(assignment, nodes, graph):
                cost = self._calculate_cost(assignment)
                if cost < best_cost:
                    best_cost = cost
                    best = assignment

        if best is None:
            cheapest_single = min(
                license_types,
                key=lambda lt: lt.cost if lt.min_capacity <= 1 <= lt.max_capacity else float("inf"),
            )
            groups = [LicenseGroup(cheapest_single, node, frozenset()) for node in nodes]
            return SolutionBuilder.create_solution_from_groups(groups)

        return self._create_solution_from_assignment(best)

    def _generate_all_assignments(
        self,
        nodes: List[Any],
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Iterator[Assignment]:
        """Yield all license-owner assignments for every set partition of nodes."""
        for partition in self._generate_partitions(nodes):
            yield from self._generate_assignments_for_partition(partition, graph, license_types)

    def _generate_partitions(self, nodes: List[Any]) -> Iterator[List[Set[Any]]]:
        """All set partitions of 'nodes' as lists of disjoint non-empty sets."""
        n = len(nodes)
        if n == 0:
            yield []
            return
        if n == 1:
            yield [{nodes[0]}]
            return

        first, rest = nodes[0], nodes[1:]
        for smaller in self._generate_partitions(rest):
            yield [{first}] + smaller

            for i, block in enumerate(smaller):
                new_part = list(smaller)
                new_part[i] = set(block) | {first}
                yield new_part

    def _generate_assignments_for_partition(
        self,
        partition: List[Set[Any]],
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Iterator[Assignment]:
        """For each block in the partition, enumerate all (license, owner, members) choices."""
        if not partition:
            yield []
            return

        per_block: List[List[Tuple[LicenseType, Any, Set[Any]]]] = []
        for block in partition:
            block_choices: List[Tuple[LicenseType, Any, Set[Any]]] = []
            bsize = len(block)
            for lt in license_types:
                if not (lt.min_capacity <= bsize <= lt.max_capacity):
                    continue
                for owner in block:
                    members = block - {owner}
                    if self._is_valid_group(owner, members, graph):
                        block_choices.append((lt, owner, members))
            per_block.append(block_choices)

        if all(per_block):
            for combo in product(*per_block):
                yield list(combo)

    def _is_valid_group(self, owner: Any, members: Set[Any], graph: nx.Graph) -> bool:
        """All members must be neighbors of owner (closed neighborhood constraint)."""
        owner_neighbors = set(graph.neighbors(owner))
        return all(m in owner_neighbors for m in members)

    def _is_valid_assignment(self, assignment: Assignment, nodes: List[Any], graph: nx.Graph) -> bool:
        """No overlaps, each block respects license capacity and neighborhood, full coverage."""
        covered: Set[Any] = set()

        for lt, owner, members in assignment:
            group_nodes = {owner} | members

            if not self._is_valid_group(owner, members, graph):
                return False
            if not (lt.min_capacity <= len(group_nodes) <= lt.max_capacity):
                return False

            if covered & group_nodes:
                return False
            covered.update(group_nodes)

        return covered == set(nodes)

    def _calculate_cost(self, assignment: Assignment) -> float:
        return sum(lt.cost for lt, _, _ in assignment)

    def _create_solution_from_assignment(self, assignment: Assignment) -> Solution:
        groups = [LicenseGroup(lt, owner, frozenset(members)) for lt, owner, members in assignment]
        return SolutionBuilder.create_solution_from_groups(groups)
