from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class DominatingSetAlgorithm(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__("DominatingSet")

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        import networkx  # noqa: F401

        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        dominating_set = self._greedy_dominating_set(graph)

        unassigned = set(nodes)
        licenses = {}

        for dominator in dominating_set:
            if dominator not in unassigned:
                continue

            available_neighbors = [n for n in graph.neighbors(dominator) if n in unassigned]

            best_assignment = None
            best_cost_per_person = float("inf")

            for license_type, license_config in config.license_types.items():
                max_possible_size = min(len(available_neighbors) + 1, license_config.max_size)

                for group_size in range(license_config.min_size, max_possible_size + 1):
                    if config.is_size_beneficial(license_type, group_size):
                        cost_per_person = license_config.cost_per_person(group_size)
                        if cost_per_person < best_cost_per_person:
                            best_cost_per_person = cost_per_person
                            members = [dominator] + available_neighbors[: group_size - 1]
                            best_assignment = (license_type, members)

            if best_assignment:
                license_type, members = best_assignment
                if license_type not in licenses:
                    licenses[license_type] = {}
                licenses[license_type][dominator] = members

                for member in members:
                    unassigned.discard(member)
            else:
                best_solo = config.get_best_license_for_size(1)
                if best_solo:
                    license_type, _ = best_solo
                    if license_type not in licenses:
                        licenses[license_type] = {}
                    licenses[license_type][dominator] = [dominator]
                    unassigned.discard(dominator)

        if unassigned:
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                license_type, _ = best_solo
                if license_type not in licenses:
                    licenses[license_type] = {}
                for node in unassigned:
                    licenses[license_type][node] = [node]

        return LicenseSolution(licenses=licenses)

    def _greedy_dominating_set(self, graph: "nx.Graph") -> list[int]:
        dominating_set = []
        uncovered = set(graph.nodes())

        while uncovered:
            best_node = None
            best_coverage = 0

            for node in uncovered:
                coverage = 1 if node in uncovered else 0
                coverage += sum(1 for neighbor in graph.neighbors(node) if neighbor in uncovered)

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_node = node

            if best_node is not None:
                dominating_set.append(best_node)

                uncovered.discard(best_node)
                for neighbor in graph.neighbors(best_node):
                    uncovered.discard(neighbor)

            elif uncovered:
                node = uncovered.pop()
                dominating_set.append(node)

        return dominating_set
