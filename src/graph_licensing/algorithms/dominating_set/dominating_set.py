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
            return LicenseSolution.create_empty()

        # Find a dominating set using greedy approach
        dominating_set = self._greedy_dominating_set(graph)

        unassigned = set(nodes)
        licenses = {}

        # Try to form groups around dominating nodes
        for dominator in dominating_set:
            if dominator not in unassigned:
                continue

            # Get unassigned neighbors
            available_neighbors = [n for n in graph.neighbors(dominator) if n in unassigned]

            # Try different license types and group sizes
            best_assignment = None
            best_cost_per_person = float('inf')
            
            # Try each license type
            for license_type, license_config in config.license_types.items():
                # Try different group sizes within the license constraints
                max_possible_size = min(len(available_neighbors) + 1, license_config.max_size)
                
                for group_size in range(license_config.min_size, max_possible_size + 1):
                    if config.is_size_beneficial(license_type, group_size):
                        cost_per_person = license_config.cost_per_person(group_size)
                        if cost_per_person < best_cost_per_person:
                            best_cost_per_person = cost_per_person
                            members = [dominator] + available_neighbors[:group_size - 1]
                            best_assignment = (license_type, members)
            
            # Apply best assignment
            if best_assignment:
                license_type, members = best_assignment
                if license_type not in licenses:
                    licenses[license_type] = {}
                licenses[license_type][dominator] = members
                
                for member in members:
                    unassigned.discard(member)
            else:
                # Fallback to solo with best license type
                best_solo = config.get_best_license_for_size(1)
                if best_solo:
                    license_type, _ = best_solo
                    if license_type not in licenses:
                        licenses[license_type] = {}
                    licenses[license_type][dominator] = [dominator]
                    unassigned.discard(dominator)

        # Handle remaining nodes with best solo license
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
