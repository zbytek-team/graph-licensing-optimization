"""Greedy approximation algorithm for licensing optimization."""

from typing import TYPE_CHECKING

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class GreedyAlgorithm(BaseAlgorithm):
    """Greedy approximation algorithm."""

    def __init__(self) -> None:
        """Initialize the greedy algorithm."""
        super().__init__("Greedy")

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using greedy heuristic.

        Args:
            graph: The social network graph.
            config: License configuration.
            **kwargs: Additional parameters (ignored).

        Returns:
            Approximate licensing solution.
        """
        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution(solo_nodes=[], group_owners={})

        unassigned = set(nodes)
        solo_nodes = []
        group_owners = {}

        # Sort nodes by degree (descending) for better group formation
        sorted_nodes = sorted(nodes, key=lambda x: graph.degree(x), reverse=True)

        for node in sorted_nodes:
            if node not in unassigned:
                continue

            # Get neighbors that are still unassigned
            available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]

            # Check if forming a group is beneficial
            potential_group_size = min(len(available_neighbors) + 1, config.group_size)

            if config.is_group_beneficial(potential_group_size):
                # Form a group with highest degree neighbors
                neighbors_by_degree = sorted(
                    available_neighbors,
                    key=lambda x: graph.degree(x),
                    reverse=True,
                )

                group_members = [node] + neighbors_by_degree[: config.group_size - 1]
                group_owners[node] = group_members

                # Remove assigned nodes
                for member in group_members:
                    unassigned.discard(member)
            else:
                # Assign solo license
                solo_nodes.append(node)
                unassigned.discard(node)

        # Assign remaining nodes to solo
        solo_nodes.extend(list(unassigned))

        return LicenseSolution(solo_nodes=solo_nodes, group_owners=group_owners)
