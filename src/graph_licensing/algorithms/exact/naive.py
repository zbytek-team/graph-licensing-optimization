"""Naive brute-force algorithm for exact optimization."""

from typing import TYPE_CHECKING

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class NaiveAlgorithm(BaseAlgorithm):
    """Naive brute-force algorithm for small graphs."""

    def __init__(self) -> None:
        """Initialize the naive algorithm."""
        super().__init__("Naive")

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using brute-force enumeration.

        Args:
            graph: The social network graph.
            config: License configuration.
            **kwargs: Additional parameters (ignored).

        Returns:
            Optimal licensing solution.
        """
        from itertools import combinations

        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        n = len(nodes)

        if n == 0:
            return LicenseSolution(solo_nodes=[], group_owners={})

        if n > 15:  # Prevent exponential explosion
            msg = f"Graph too large for naive algorithm (n={n}). Use n <= 15."
            raise ValueError(msg)

        best_solution = None
        best_cost = float("inf")

        # Try all possible combinations of solo and group assignments
        for num_groups in range(n + 1):
            # Choose which nodes will be group owners
            for group_owners in combinations(nodes, num_groups):
                remaining_nodes = [node for node in nodes if node not in group_owners]

                # For each group owner, try all possible group memberships
                group_assignments = {}
                valid = True

                for owner in group_owners:
                    # Get potential members (neighbors + owner)
                    potential_members = [owner, *list(graph.neighbors(owner))]
                    potential_members = [n for n in potential_members if n not in group_assignments]

                    # Try all possible subsets up to group_size
                    best_group = None
                    for group_size in range(
                        1,
                        min(config.group_size + 1, len(potential_members) + 1),
                    ):
                        for group_members in combinations(
                            potential_members,
                            group_size,
                        ):
                            if owner in group_members:
                                if best_group is None or len(group_members) > len(
                                    best_group,
                                ):
                                    best_group = list(group_members)

                    if best_group and len(best_group) >= 1:
                        group_assignments[owner] = best_group
                        # Remove assigned members from remaining nodes
                        remaining_nodes = [n for n in remaining_nodes if n not in best_group]
                    else:
                        valid = False
                        break

                if valid:
                    # Remaining nodes get solo licenses
                    solo_nodes = remaining_nodes

                    solution = LicenseSolution(
                        solo_nodes=solo_nodes,
                        group_owners=group_assignments,
                    )

                    if solution.is_valid(graph, config):
                        cost = solution.calculate_cost(config)
                        if cost < best_cost:
                            best_cost = cost
                            best_solution = solution

        return best_solution or LicenseSolution(solo_nodes=nodes, group_owners={})
