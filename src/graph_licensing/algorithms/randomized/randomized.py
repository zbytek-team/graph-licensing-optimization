"""Randomized algorithm for licensing optimization."""

import random
from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class RandomizedAlgorithm(BaseAlgorithm):
    """Randomized algorithm for comparison baseline."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the randomized algorithm.

        Args:
            seed: Random seed for reproducibility.
        """
        super().__init__("Randomized")
        self.seed = seed

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using randomized approach.

        Args:
            graph: The social network graph.
            config: License configuration.
            warm_start: Previous solution (ignored for randomized algorithm).
            **kwargs: Additional parameters (ignored).

        Returns:
            Random valid licensing solution.
        """
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution(solo_nodes=[], group_owners={})

        unassigned = set(nodes)
        solo_nodes = []
        group_owners = {}

        # Randomly shuffle nodes
        random.shuffle(nodes)

        for node in nodes:
            if node not in unassigned:
                continue

            # Randomly decide whether to try forming a group
            if random.random() < 0.5:
                # Try to form a group
                available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]

                if available_neighbors:
                    # Randomly select group members
                    max_additional = min(
                        len(available_neighbors),
                        config.group_size - 1,
                    )
                    num_members = random.randint(0, max_additional)

                    if num_members > 0:
                        selected_members = random.sample(
                            available_neighbors,
                            num_members,
                        )
                        group_members = [node, *selected_members]

                        # Check if group is beneficial
                        if config.is_group_beneficial(len(group_members)):
                            group_owners[node] = group_members
                            for member in group_members:
                                unassigned.discard(member)
                            continue

            # Assign solo license
            solo_nodes.append(node)
            unassigned.discard(node)

        return LicenseSolution(solo_nodes=solo_nodes, group_owners=group_owners)
