"""Naive brute-force algorithm for exact optimization."""

from typing import TYPE_CHECKING, Optional

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
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using brute-force enumeration.

        Args:
            graph: The social network graph.
            config: License configuration.
            warm_start: Previous solution (ignored for exact algorithm).
            **kwargs: Additional parameters (ignored).

        Returns:
            Optimal licensing solution.
        """
        from itertools import combinations, product

        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        n = len(nodes)

        if n == 0:
            return LicenseSolution.create_empty()

        if n > 12:  # Prevent exponential explosion - reduced due to increased complexity
            msg = f"Graph too large for naive algorithm (n={n}). Use n <= 12."
            raise ValueError(msg)

        best_solution = None
        best_cost = float("inf")
        
        # Generate all possible valid assignments
        all_assignments = self._generate_all_assignments(nodes, graph, config)
        
        for assignment in all_assignments:
            solution = LicenseSolution(licenses=assignment)
            if solution.is_valid(graph, config):
                cost = solution.calculate_cost(config)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution

        # Fallback: assign all nodes to best solo license if no valid solution found
        if best_solution is None:
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                license_type, _ = best_solo
                licenses = {license_type: {node: [node] for node in nodes}}
                best_solution = LicenseSolution(licenses=licenses)
        
        return best_solution or LicenseSolution.create_empty()

    def _generate_all_assignments(self, nodes: list, graph: "nx.Graph", config: "LicenseConfig") -> list:
        """Generate all possible valid license assignments.
        
        Args:
            nodes: List of nodes.
            graph: The graph.
            config: License configuration.
            
        Returns:
            List of all possible license assignments.
        """
        if not nodes:
            return [{}]
        
        all_assignments = []
        
        # For small graphs, we can try all possible partitions
        # We'll use a recursive approach to generate all valid groupings
        self._recursive_assignment(nodes, graph, config, {}, set(), all_assignments)
        
        return all_assignments
    
    def _recursive_assignment(self, remaining_nodes: list, graph: "nx.Graph", config: "LicenseConfig", 
                             current_licenses: dict, assigned_nodes: set, all_assignments: list):
        """Recursively generate all possible assignments.
        
        Args:
            remaining_nodes: Nodes not yet assigned.
            graph: The graph.
            config: License configuration.
            current_licenses: Current partial assignment.
            assigned_nodes: Set of nodes already assigned.
            all_assignments: List to append complete assignments to.
        """
        from itertools import combinations
        
        if not remaining_nodes:
            # All nodes assigned, add to results
            all_assignments.append(current_licenses.copy())
            return
        
        # Take the first remaining node
        node = remaining_nodes[0]
        remaining = remaining_nodes[1:]
        
        # Try all license types
        for license_type, license_config in config.license_types.items():
            # Try different group sizes for this license type
            for group_size in range(license_config.min_size, license_config.max_size + 1):
                if group_size == 1:
                    # Solo assignment
                    new_licenses = current_licenses.copy()
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node]
                    
                    new_assigned = assigned_nodes | {node}
                    new_remaining = [n for n in remaining if n not in new_assigned]
                    
                    self._recursive_assignment(new_remaining, graph, config, new_licenses, new_assigned, all_assignments)
                
                else:
                    # Group assignment - find all possible groups of this size with node as owner
                    # Get neighbors that are still unassigned
                    available_neighbors = [n for n in graph.neighbors(node) if n not in assigned_nodes and n in remaining_nodes]
                    
                    # Try all combinations of neighbors for the group
                    if len(available_neighbors) >= group_size - 1:
                        for members_subset in combinations(available_neighbors, group_size - 1):
                            group_members = [node] + list(members_subset)
                            
                            new_licenses = current_licenses.copy()
                            if license_type not in new_licenses:
                                new_licenses[license_type] = {}
                            new_licenses[license_type][node] = group_members
                            
                            new_assigned = assigned_nodes | set(group_members)
                            new_remaining = [n for n in remaining if n not in new_assigned]
                            
                            self._recursive_assignment(new_remaining, graph, config, new_licenses, new_assigned, all_assignments)
