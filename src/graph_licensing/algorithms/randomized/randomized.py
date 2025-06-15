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
            return LicenseSolution.create_empty()

        unassigned = set(nodes)
        licenses = {}

        # Randomly shuffle nodes
        random.shuffle(nodes)

        for node in nodes:
            if node not in unassigned:
                continue

            # Get available license types
            available_license_types = list(config.license_types.keys())
            random.shuffle(available_license_types)

            assigned = False
            
            # Try each license type randomly
            for license_type in available_license_types:
                license_config = config.license_types[license_type]
                
                # Randomly decide group size within valid range
                max_size = min(license_config.max_size, len(unassigned))
                if max_size < license_config.min_size:
                    continue
                    
                group_size = random.randint(license_config.min_size, max_size)
                
                if group_size == 1:
                    # Solo assignment
                    if license_type not in licenses:
                        licenses[license_type] = {}
                    licenses[license_type][node] = [node]
                    unassigned.discard(node)
                    assigned = True
                    break
                else:
                    # Group assignment - find connected neighbors
                    available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]
                    
                    if len(available_neighbors) >= group_size - 1:
                        # Select random members
                        selected_members = random.sample(available_neighbors, group_size - 1)
                        group_members = [node] + selected_members
                        
                        # Check if this is a beneficial assignment
                        if config.is_size_beneficial(license_type, group_size):
                            if license_type not in licenses:
                                licenses[license_type] = {}
                            licenses[license_type][node] = group_members
                            
                            for member in group_members:
                                unassigned.discard(member)
                            assigned = True
                            break
            
            # If no assignment worked, use best solo license
            if not assigned:
                best_solo = config.get_best_license_for_size(1)
                if best_solo:
                    license_type, _ = best_solo
                    if license_type not in licenses:
                        licenses[license_type] = {}
                    licenses[license_type][node] = [node]
                    unassigned.discard(node)

        return LicenseSolution(licenses=licenses)
