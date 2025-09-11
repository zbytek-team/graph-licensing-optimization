import random
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder


class RandomizedAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "randomized_algorithm"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def solve(self, graph: nx.Graph, license_types: list[LicenseType], **kwargs: Any) -> Solution:
        if graph.number_of_nodes() == 0:
            return Solution(groups=())

        runtime_seed = kwargs.get("seed", self.seed)
        if runtime_seed is not None:
            random.seed(runtime_seed)

        nodes = list(graph.nodes())
        uncovered_nodes = set(nodes)
        groups = []

        random.shuffle(nodes)

        for node in nodes:
            if node not in uncovered_nodes:
                continue

            assignment = self._random_assignment(node, uncovered_nodes, graph, license_types)

            if assignment:
                license_type, group_members = assignment
                additional_members = group_members - {node}
                group = LicenseGroup(license_type, node, frozenset(additional_members))
                groups.append(group)
                uncovered_nodes -= group_members

        while uncovered_nodes:
            node = uncovered_nodes.pop()
            cheapest_single = self._find_cheapest_single_license(license_types)
            group = LicenseGroup(cheapest_single, node, frozenset())
            groups.append(group)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _random_assignment(
        self,
        node: Any,
        uncovered_nodes: set[Any],
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> tuple[LicenseType, set[Any]] | None:
        neighbors = set(graph.neighbors(node)) & uncovered_nodes
        available_nodes = neighbors | {node}

        compatible_licenses = [lt for lt in license_types if lt.min_capacity <= len(available_nodes)]

        if not compatible_licenses:
            return None

        random.shuffle(compatible_licenses)

        for license_type in compatible_licenses:
            max_possible_size = min(len(available_nodes), license_type.max_capacity)

            if max_possible_size < license_type.min_capacity:
                continue

            group_size = random.randint(license_type.min_capacity, max_possible_size)

            group_members = self._select_random_group_members(node, available_nodes, group_size)

            if len(group_members) >= license_type.min_capacity:
                return (license_type, group_members)

        return None

    def _select_random_group_members(self, owner: Any, available_nodes: set[Any], target_size: int) -> set[Any]:
        if target_size <= 0:
            return set()

        group_members = {owner}
        remaining_slots = target_size - 1

        if remaining_slots <= 0:
            return group_members

        candidates = list(available_nodes - {owner})

        if len(candidates) >= remaining_slots:
            selected_candidates = random.sample(candidates, remaining_slots)
            group_members.update(selected_candidates)
        else:
            group_members.update(candidates)

        return group_members

    def _find_cheapest_single_license(self, license_types: list[LicenseType]) -> LicenseType:
        single_licenses = [lt for lt in license_types if lt.min_capacity <= 1 <= lt.max_capacity]

        if not single_licenses:
            return min(license_types, key=lambda lt: lt.cost)

        return min(single_licenses, key=lambda lt: lt.cost)
