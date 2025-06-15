from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class GreedyAlgorithm(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__("Greedy")

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration object."""
        return cls()

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        unassigned = set(nodes)
        licenses = {}

        sorted_nodes = sorted(nodes, key=lambda x: graph.degree(x), reverse=True)

        for node in sorted_nodes:
            if node not in unassigned:
                continue

            available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]

            best_assignment = self._find_best_license_assignment(node, available_neighbors, config, unassigned)

            if best_assignment:
                license_type, owner, members = best_assignment

                if license_type not in licenses:
                    licenses[license_type] = {}

                licenses[license_type][owner] = members

                for member in members:
                    unassigned.discard(member)
            else:
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    if cheapest_solo not in licenses:
                        licenses[cheapest_solo] = {}
                    licenses[cheapest_solo][node] = [node]
                    unassigned.discard(node)

        if unassigned:
            cheapest_solo = self._get_cheapest_solo_license(config)
            if cheapest_solo:
                if cheapest_solo not in licenses:
                    licenses[cheapest_solo] = {}
                for node in unassigned:
                    licenses[cheapest_solo][node] = [node]

        return LicenseSolution(licenses=licenses)

    def _find_best_license_assignment(
        self, node: int, available_neighbors: list[int], config: "LicenseConfig", unassigned: set[int]
    ) -> tuple[str, int, list[int]] | None:
        best_cost_per_person = float("inf")
        best_assignment = None

        for license_type, license_config in config.license_types.items():
            max_possible_size = min(license_config.max_size, len(available_neighbors) + 1, len(unassigned))

            for group_size in range(license_config.min_size, max_possible_size + 1):
                if group_size == 1:
                    cost_per_person = license_config.price
                    if cost_per_person < best_cost_per_person:
                        best_cost_per_person = cost_per_person
                        best_assignment = (license_type, node, [node])

                elif group_size > 1 and available_neighbors:
                    neighbors_by_degree = sorted(
                        available_neighbors,
                        key=lambda x: len([n for n in available_neighbors if n in unassigned]),
                        reverse=True,
                    )

                    selected_members = neighbors_by_degree[: group_size - 1]
                    if len(selected_members) == group_size - 1:
                        members = [node] + selected_members
                        cost_per_person = license_config.price / len(members)

                        if cost_per_person < best_cost_per_person:
                            best_cost_per_person = cost_per_person
                            best_assignment = (license_type, node, members)

        return best_assignment

    def _get_cheapest_solo_license(self, config: "LicenseConfig") -> str | None:
        cheapest_type = None
        cheapest_cost = float("inf")

        for license_type, license_config in config.license_types.items():
            if license_config.is_valid_size(1) and license_config.price < cheapest_cost:
                cheapest_cost = license_config.price
                cheapest_type = license_type

        return cheapest_type
