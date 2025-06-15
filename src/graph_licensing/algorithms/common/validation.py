"""Solution validation and adaptation utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution


class SolutionValidator:
    """Utilities for validating and adapting solutions."""

    @staticmethod
    def is_solution_valid(
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> bool:
        """Check if solution is valid for the given graph and config."""
        return solution.is_valid(graph, config)

    @staticmethod
    def adapt_solution_to_graph(
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Adapt a solution to work with a potentially different graph."""
        from ...models.license import LicenseSolution
        
        graph_nodes = set(graph.nodes())
        adapted_licenses = {}

        for license_type, groups in solution.licenses.items():
            if license_type not in config.license_types:
                continue
                
            license_config = config.license_types[license_type]
            adapted_groups = {}

            for owner, members in groups.items():
                # Keep only members that exist in the new graph
                valid_members = [m for m in members if m in graph_nodes]
                
                if not valid_members:
                    continue
                
                # Ensure owner is in the members list and exists in graph
                if owner in graph_nodes and owner not in valid_members:
                    valid_members.insert(0, owner)
                elif owner not in graph_nodes and valid_members:
                    # Pick new owner from existing members
                    owner = valid_members[0]

                # Check if group size is still valid
                if license_config.is_valid_size(len(valid_members)):
                    adapted_groups[owner] = valid_members
                elif len(valid_members) > license_config.max_size:
                    # Split oversized group
                    remaining_members = valid_members[:]
                    while remaining_members:
                        group_size = min(license_config.max_size, len(remaining_members))
                        group = remaining_members[:group_size]
                        group_owner = group[0]
                        adapted_groups[group_owner] = group
                        remaining_members = remaining_members[group_size:]
                elif len(valid_members) >= license_config.min_size:
                    # Group is too small but above minimum, keep it
                    adapted_groups[owner] = valid_members
                # If group is below minimum size, it gets dropped

            if adapted_groups:
                adapted_licenses[license_type] = adapted_groups

        # Handle nodes not covered by the adapted solution
        adapted_solution = LicenseSolution(licenses=adapted_licenses)
        covered_nodes = set(adapted_solution.get_all_nodes())
        uncovered_nodes = graph_nodes - covered_nodes

        if uncovered_nodes:
            # Add uncovered nodes using greedy approach
            from .initialization import SolutionInitializer
            
            # Create subgraph with uncovered nodes
            import networkx as nx
            subgraph = graph.subgraph(uncovered_nodes)
            
            # Get greedy solution for uncovered nodes
            greedy_solution = SolutionInitializer.create_greedy_solution(subgraph, config)
            
            # Merge with adapted solution
            for license_type, groups in greedy_solution.licenses.items():
                if license_type not in adapted_licenses:
                    adapted_licenses[license_type] = {}
                adapted_licenses[license_type].update(groups)

        return LicenseSolution(licenses=adapted_licenses)

    @staticmethod
    def calculate_solution_fitness(
        solution: "LicenseSolution", 
        config: "LicenseConfig",
        penalty_factor: float = 1000.0
    ) -> float:
        """Calculate fitness score for a solution (lower is better)."""
        base_cost = solution.calculate_cost(config)
        
        # Add penalties for invalid configurations
        penalty = 0.0
        
        for license_type, groups in solution.licenses.items():
            if license_type not in config.license_types:
                penalty += penalty_factor
                continue
                
            license_config = config.license_types[license_type]
            
            for owner, members in groups.items():
                group_size = len(members)
                if not license_config.is_valid_size(group_size):
                    # Penalty proportional to how far from valid range
                    if group_size < license_config.min_size:
                        penalty += penalty_factor * (license_config.min_size - group_size)
                    else:  # group_size > license_config.max_size
                        penalty += penalty_factor * (group_size - license_config.max_size)
                
                # Ensure owner is in members
                if owner not in members:
                    penalty += penalty_factor * 0.1

        return base_cost + penalty

    @staticmethod
    def repair_solution(
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Repair an invalid solution to make it valid."""
        from ...models.license import LicenseSolution
        
        repaired_licenses = {}
        
        for license_type, groups in solution.licenses.items():
            if license_type not in config.license_types:
                continue
                
            license_config = config.license_types[license_type]
            repaired_groups = {}

            for owner, members in groups.items():
                # Ensure owner is in members
                if owner not in members:
                    members = [owner] + [m for m in members if m != owner]

                # Fix group size
                if len(members) < license_config.min_size:
                    # Try to add neighbors to reach minimum size
                    graph_nodes = set(graph.nodes())
                    current_members = set(members)
                    
                    for member in members:
                        if len(current_members) >= license_config.min_size:
                            break
                        neighbors = [
                            n for n in graph.neighbors(member) 
                            if n in graph_nodes and n not in current_members
                        ]
                        # Add closest neighbors
                        needed = license_config.min_size - len(current_members)
                        to_add = neighbors[:needed]
                        current_members.update(to_add)
                    
                    members = list(current_members)

                elif len(members) > license_config.max_size:
                    # Trim to maximum size, keeping owner
                    members = [owner] + [m for m in members if m != owner][:license_config.max_size - 1]

                if license_config.is_valid_size(len(members)):
                    repaired_groups[owner] = members

            if repaired_groups:
                repaired_licenses[license_type] = repaired_groups

        return LicenseSolution(licenses=repaired_licenses)
