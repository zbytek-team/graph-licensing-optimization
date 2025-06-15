from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class NaiveAlgorithm(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__("Naive")

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":

        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return LicenseSolution.create_empty()
        if n > 12:
            msg = f"Graph too large for naive algorithm (n={n}). Use n <= 12."
            raise ValueError(msg)
        best_solution = None
        best_cost = float("inf")
        all_assignments = self._generate_all_assignments(nodes, graph, config)
        for assignment in all_assignments:
            solution = LicenseSolution(licenses=assignment)
            if solution.is_valid(graph, config):
                cost = solution.calculate_cost(config)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution

        if best_solution is None:
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                license_type, _ = best_solo
                licenses = {license_type: {node: [node] for node in nodes}}
                best_solution = LicenseSolution(licenses=licenses)
        return best_solution or LicenseSolution.create_empty()

    def _generate_all_assignments(self, nodes: list, graph: "nx.Graph", config: "LicenseConfig") -> list:
        if not nodes:
            return [{}]
        all_assignments = []
        self._recursive_assignment(nodes, graph, config, {}, set(), all_assignments)
        return all_assignments

    def _recursive_assignment(
        self,
        remaining_nodes: list,
        graph: "nx.Graph",
        config: "LicenseConfig",
        current_licenses: dict,
        assigned_nodes: set,
        all_assignments: list,
    ):
        from itertools import combinations

        if not remaining_nodes:
            all_assignments.append(current_licenses.copy())
            return
        node = remaining_nodes[0]
        remaining = remaining_nodes[1:]
        for license_type, license_config in config.license_types.items():
            for group_size in range(license_config.min_size, license_config.max_size + 1):
                if group_size == 1:
                    new_licenses = current_licenses.copy()
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node]
                    new_assigned = assigned_nodes | {node}
                    new_remaining = [n for n in remaining if n not in new_assigned]
                    self._recursive_assignment(
                        new_remaining, graph, config, new_licenses, new_assigned, all_assignments
                    )
                else:
                    available_neighbors = [
                        n for n in graph.neighbors(node) if n not in assigned_nodes and n in remaining_nodes
                    ]
                    if len(available_neighbors) >= group_size - 1:
                        for members_subset in combinations(available_neighbors, group_size - 1):
                            group_members = [node] + list(members_subset)
                            new_licenses = current_licenses.copy()
                            if license_type not in new_licenses:
                                new_licenses[license_type] = {}
                            new_licenses[license_type][node] = group_members
                            new_assigned = assigned_nodes | set(group_members)
                            new_remaining = [n for n in remaining if n not in new_assigned]
                            self._recursive_assignment(
                                new_remaining, graph, config, new_licenses, new_assigned, all_assignments
                            )
