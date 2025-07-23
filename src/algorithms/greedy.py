from ..core.types import LicenseType, Solution, Algorithm, LicenseGroup

from typing import Any, List

import networkx as nx


class GreedyAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "greedy"

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        nodes = list(graph.nodes())
        uncovered_nodes = set(nodes)
        groups = []
        total_cost = 0.0

        sorted_license_types = sorted(license_types, key=lambda lt: lt.max_capacity, reverse=True)

        nodes_by_degree = sorted(nodes, key=lambda n: graph.degree(n), reverse=True)

        for node in nodes_by_degree:
            if node not in uncovered_nodes:
                continue

            neighbors = set(graph.neighbors(node)) | {node}
            available_neighbors = neighbors & uncovered_nodes

            if not available_neighbors:
                continue

            best_group = None
            best_efficiency = float("inf")

            for license_type in sorted_license_types:
                potential_members = list(available_neighbors)

                max_members = min(len(potential_members), license_type.max_capacity)

                if max_members < license_type.min_capacity:
                    continue

                potential_members.sort(key=lambda n: graph.degree(n), reverse=True)

                group_members = set(potential_members[:max_members])

                efficiency = license_type.cost / len(group_members)

                if efficiency < best_efficiency:
                    best_efficiency = efficiency
                    additional_members = group_members - {node}
                    best_group = LicenseGroup(license_type=license_type, owner=node, additional_members=additional_members)

            if best_group is not None:
                groups.append(best_group)
                total_cost += best_group.license_type.cost
                uncovered_nodes -= best_group.all_members

        while uncovered_nodes:
            node = next(iter(uncovered_nodes))
            neighbors = set(graph.neighbors(node)) | {node}
            available_neighbors = neighbors & uncovered_nodes

            best_group = None
            best_cost = float("inf")

            for license_type in license_types:
                if len(available_neighbors) >= license_type.min_capacity:
                    potential_members = list(available_neighbors)
                    potential_members.sort(key=lambda n: graph.degree(n), reverse=True)
                    group_size = min(len(potential_members), license_type.min_capacity)
                    group_members = set(potential_members[:group_size])

                    if license_type.cost < best_cost:
                        best_cost = license_type.cost
                        additional_members = group_members - {node}
                        best_group = LicenseGroup(license_type=license_type, owner=node, additional_members=additional_members)

            if best_group is not None:
                groups.append(best_group)
                total_cost += best_group.license_type.cost
                uncovered_nodes -= best_group.all_members
            else:
                cheapest_license = min(license_types, key=lambda lt: lt.cost)
                if cheapest_license.min_capacity == 1:
                    group = LicenseGroup(license_type=cheapest_license, owner=node, additional_members=set())
                    groups.append(group)
                    total_cost += cheapest_license.cost
                    uncovered_nodes.remove(node)
                else:
                    break

        covered_nodes = set()
        for group in groups:
            covered_nodes.update(group.all_members)

        return Solution(groups=groups, total_cost=total_cost, covered_nodes=covered_nodes)
