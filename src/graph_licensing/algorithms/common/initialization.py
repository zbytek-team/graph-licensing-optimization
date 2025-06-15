"""Solution initialization utilities."""

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Any

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution


class SolutionInitializer:
    """Handles different strategies for initializing solutions."""

    @staticmethod
    def create_random_solution(graph: "nx.Graph", config: "LicenseConfig") -> "LicenseSolution":
        """Create a random valid solution."""
        from ...models.license import LicenseSolution
        
        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        unassigned = set(nodes)
        licenses = {}

        while unassigned:
            # Pick a random unassigned node as owner
            owner = random.choice(list(unassigned))
            unassigned.remove(owner)

            # Pick a random license type
            license_type = random.choice(list(config.license_types.keys()))
            license_config = config.license_types[license_type]

            # Determine group size
            max_possible = min(license_config.max_size, len(unassigned) + 1)
            min_size = max(license_config.min_size, 1)
            
            if max_possible < min_size:
                # Fallback: create minimal group
                group_size = 1
            else:
                group_size = random.randint(min_size, max_possible)

            # Build group with neighbors when possible
            members = [owner]
            neighbors = [n for n in graph.neighbors(owner) if n in unassigned]
            
            # Add neighbors up to group size
            available_to_add = min(group_size - 1, len(neighbors))
            if available_to_add > 0:
                selected_neighbors = random.sample(neighbors, available_to_add)
                members.extend(selected_neighbors)
                for neighbor in selected_neighbors:
                    unassigned.remove(neighbor)

            # Add random nodes if still under group size
            remaining_slots = group_size - len(members)
            if remaining_slots > 0 and unassigned:
                additional_nodes = random.sample(
                    list(unassigned), 
                    min(remaining_slots, len(unassigned))
                )
                members.extend(additional_nodes)
                for node in additional_nodes:
                    unassigned.remove(node)

            # Add to licenses
            if license_type not in licenses:
                licenses[license_type] = {}
            licenses[license_type][owner] = members

        return LicenseSolution(licenses=licenses)

    @staticmethod
    def create_greedy_solution(graph: "nx.Graph", config: "LicenseConfig") -> "LicenseSolution":
        """Create a solution using greedy heuristic."""
        # Import here to avoid circular imports
        from ..greedy.greedy import GreedyAlgorithm
        
        greedy_algo = GreedyAlgorithm()
        return greedy_algo.solve(graph, config)

    @staticmethod
    def create_diverse_population(
        graph: "nx.Graph", 
        config: "LicenseConfig", 
        size: int,
        warm_start: Optional["LicenseSolution"] = None,
        greedy_ratio: float = 0.2
    ) -> List["LicenseSolution"]:
        """Create a diverse population of solutions."""
        population = []

        # Add warm start if provided
        if warm_start:
            population.append(warm_start)

        # Add greedy solutions (with some randomization)
        num_greedy = max(1, int(size * greedy_ratio))
        for _ in range(num_greedy):
            if len(population) >= size:
                break
            greedy_solution = SolutionInitializer.create_greedy_solution(graph, config)
            population.append(greedy_solution)

        # Fill the rest with random solutions
        while len(population) < size:
            random_solution = SolutionInitializer.create_random_solution(graph, config)
            population.append(random_solution)

        return population[:size]

    @staticmethod
    def encode_solution_as_dict(solution: "LicenseSolution") -> Dict[str, Any]:
        """Encode a solution as a dictionary for genetic algorithms."""
        assignments = {}
        
        for license_type, groups in solution.licenses.items():
            for owner, members in groups.items():
                for member in members:
                    assignments[member] = {
                        "license_type": license_type,
                        "owner": owner,
                        "is_owner": member == owner
                    }
        
        return {"assignments": assignments}

    @staticmethod
    def decode_dict_to_solution(
        encoded: Dict[str, Any], 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Decode a dictionary representation back to LicenseSolution."""
        from ...models.license import LicenseSolution
        
        licenses = {}
        assignments = encoded.get("assignments", {})
        
        # Group by license type and owner
        groups_by_type_owner = {}
        for node_id, assignment in assignments.items():
            license_type = assignment["license_type"]
            owner = assignment["owner"]
            
            key = (license_type, owner)
            if key not in groups_by_type_owner:
                groups_by_type_owner[key] = []
            groups_by_type_owner[key].append(int(node_id))
        
        # Convert to LicenseSolution format
        for (license_type, owner), members in groups_by_type_owner.items():
            if license_type not in licenses:
                licenses[license_type] = {}
            licenses[license_type][owner] = sorted(members)
        
        return LicenseSolution(licenses=licenses)
