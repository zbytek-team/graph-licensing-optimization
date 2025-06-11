"""Dominating set based algorithm for licensing optimization."""

from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class DominatingSetAlgorithm(BaseAlgorithm):
    """Algorithm based on dominating set heuristics."""

    def __init__(self) -> None:
        """Initialize the dominating set algorithm."""
        super().__init__("DominatingSet")

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using dominating set approach.

        Args:
            graph: The social network graph.
            config: License configuration.
            warm_start: Previous solution (ignored for this algorithm).
            **kwargs: Additional parameters (ignored).

        Returns:
            Approximate licensing solution.
        """
        import networkx  # noqa: F401

        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution(solo_nodes=[], group_owners={})

        # Find a dominating set using greedy approach
        dominating_set = self._greedy_dominating_set(graph)

        unassigned = set(nodes)
        solo_nodes = []
        group_owners = {}

        # Try to form groups around dominating nodes
        for dominator in dominating_set:
            if dominator not in unassigned:
                continue

            # Get unassigned neighbors
            available_neighbors = [n for n in graph.neighbors(dominator) if n in unassigned]

            # Form group if beneficial
            potential_group_size = min(len(available_neighbors) + 1, config.group_size)

            if config.is_group_beneficial(potential_group_size):
                group_members = [dominator] + available_neighbors[: config.group_size - 1]
                group_owners[dominator] = group_members

                for member in group_members:
                    unassigned.discard(member)
            else:
                solo_nodes.append(dominator)
                unassigned.discard(dominator)

        # Handle remaining nodes
        solo_nodes.extend(list(unassigned))

        return LicenseSolution(solo_nodes=solo_nodes, group_owners=group_owners)

    def _greedy_dominating_set(self, graph: "nx.Graph") -> list[int]:
        """Find a dominating set using greedy heuristic.

        Args:
            graph: Input graph.

        Returns:
            List of nodes forming a dominating set.
        """
        dominating_set = []
        uncovered = set(graph.nodes())

        while uncovered:
            # Find node that covers the most uncovered nodes
            best_node = None
            best_coverage = 0

            for node in uncovered:
                # Count uncovered neighbors + node itself
                coverage = 1 if node in uncovered else 0
                coverage += sum(1 for neighbor in graph.neighbors(node) if neighbor in uncovered)

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_node = node

            if best_node is not None:
                dominating_set.append(best_node)
                # Remove covered nodes
                uncovered.discard(best_node)
                for neighbor in graph.neighbors(best_node):
                    uncovered.discard(neighbor)
            # Fallback: pick any remaining node
            elif uncovered:
                node = uncovered.pop()
                dominating_set.append(node)

        return dominating_set
