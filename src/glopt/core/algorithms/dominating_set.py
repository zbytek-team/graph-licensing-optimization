from collections.abc import Sequence
from typing import Any, cast

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder


class DominatingSetAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "dominating_set_algorithm"

    def solve(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        if graph.number_of_nodes() == 0:
            return Solution(groups=())
        dominating_set = self._find_cost_effective_dominating_set(
            graph, license_types
        )
        remaining_nodes = set(graph.nodes())
        groups = []
        degv = cast(Any, graph.degree)
        sorted_dominators = sorted(
            dominating_set, key=lambda n: int(degv[n]), reverse=True
        )
        for dominator in sorted_dominators:
            if dominator not in remaining_nodes:
                continue
            neighbors = set(graph.neighbors(dominator)) & remaining_nodes
            available_nodes = neighbors | {dominator}
            best_assignment = self._find_best_cost_assignment(
                graph, dominator, available_nodes, license_types
            )
            if best_assignment:
                license_type, group_members = best_assignment
                additional_members = group_members - {dominator}
                group = LicenseGroup(
                    license_type,
                    dominator,
                    frozenset(additional_members),
                )
                groups.append(group)
                remaining_nodes -= group_members
        remaining_sorted = sorted(
            remaining_nodes, key=lambda n: int(degv[n]), reverse=True
        )
        for node in remaining_sorted:
            if node not in remaining_nodes:
                continue
            neighbors = set(graph.neighbors(node)) & remaining_nodes
            available_nodes = neighbors | {node}
            best_assignment = self._find_best_cost_assignment(
                graph, node, available_nodes, license_types
            )
            if best_assignment:
                license_type, group_members = best_assignment
                additional_members = group_members - {node}
                group = LicenseGroup(
                    license_type, node, frozenset(additional_members)
                )
                groups.append(group)
                remaining_nodes -= group_members
            else:
                cheapest_single = self._find_cheapest_single_license(
                    license_types
                )
                group = LicenseGroup(cheapest_single, node, frozenset())
                groups.append(group)
                remaining_nodes.remove(node)
        return SolutionBuilder.create_solution_from_groups(groups)

    def _find_cost_effective_dominating_set(
        self, graph: nx.Graph, license_types: Sequence[LicenseType]
    ) -> set[Any]:
        nodes = set(graph.nodes())
        uncovered = nodes.copy()
        dominating_set = set()
        while uncovered:
            best_node = None
            best_score = -1
            for node in nodes:
                if node in dominating_set:
                    continue
                neighbors = set(graph.neighbors(node))
                coverage = (neighbors | {node}) & uncovered
                if len(coverage) == 0:
                    continue
                min_cost_per_node = self._calculate_min_cost_per_node(
                    len(coverage), license_types
                )
                score = (
                    len(coverage) / min_cost_per_node
                    if min_cost_per_node > 0
                    else len(coverage)
                )
                if score > best_score:
                    best_score = score
                    best_node = node
            if best_node is None:
                best_node = next(iter(uncovered))
            dominating_set.add(best_node)
            neighbors = set(graph.neighbors(best_node))
            covered_by_node = (neighbors | {best_node}) & uncovered
            uncovered -= covered_by_node
        return dominating_set

    def _calculate_min_cost_per_node(
        self, group_size: int, license_types: Sequence[LicenseType]
    ) -> float:
        min_cost = float("inf")
        for license_type in license_types:
            if (
                license_type.min_capacity
                <= group_size
                <= license_type.max_capacity
            ):
                cost_per_node = license_type.cost / group_size
                min_cost = min(min_cost, cost_per_node)
        return min_cost if min_cost != float("inf") else 0

    def _find_best_cost_assignment(
        self,
        graph: nx.Graph,
        owner: Any,
        available_nodes: set[Any],
        license_types: Sequence[LicenseType],
    ) -> tuple[LicenseType, set[Any]] | None:
        best_assignment = None
        best_efficiency = float("inf")
        for license_type in license_types:
            max_possible_size = min(
                len(available_nodes), license_type.max_capacity
            )
            for group_size in range(
                license_type.min_capacity, max_possible_size + 1
            ):
                if group_size > len(available_nodes):
                    break
                group_members = self._select_best_group_members(
                    graph, owner, available_nodes, group_size
                )
                if len(group_members) == group_size:
                    cost_per_node = license_type.cost / group_size
                    if cost_per_node < best_efficiency:
                        best_efficiency = cost_per_node
                        best_assignment = (
                            license_type,
                            group_members,
                        )
        return best_assignment

    def _select_best_group_members(
        self,
        graph: nx.Graph,
        owner: Any,
        available_nodes: set[Any],
        target_size: int,
    ) -> set[Any]:
        if target_size <= 0:
            return set()
        group_members = {owner}
        remaining_slots = target_size - 1
        if remaining_slots <= 0:
            return group_members
        candidates = list(available_nodes - {owner})
        degv = cast(Any, graph.degree)
        candidates.sort(key=lambda n: int(degv[n]), reverse=True)
        group_members.update(candidates[:remaining_slots])
        return group_members

    def _find_cheapest_single_license(
        self, license_types: Sequence[LicenseType]
    ) -> LicenseType:
        single_licenses = [
            lt
            for lt in license_types
            if lt.min_capacity <= 1 <= lt.max_capacity
        ]
        if not single_licenses:
            return min(license_types, key=lambda lt: lt.cost)
        return min(single_licenses, key=lambda lt: lt.cost)
