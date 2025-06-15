"""Simulated Annealing algorithm for licensing optimization."""

import math
import random
from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class SimulatedAnnealingAlgorithm(BaseAlgorithm):
    """Simulated Annealing algorithm for licensing optimization."""

    def __init__(
        self,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        seed: int | None = None,
    ) -> None:
        """Initialize the simulated annealing algorithm.

        Args:
            initial_temp: Initial temperature.
            final_temp: Final temperature.
            cooling_rate: Temperature cooling rate.
            max_iterations: Maximum number of iterations.
            seed: Random seed for reproducibility.
        """
        super().__init__("SimulatedAnnealing")
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.seed = seed

    def supports_warm_start(self) -> bool:
        """Simulated annealing supports warm start initialization."""
        return True

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using simulated annealing.

        Args:
            graph: The social network graph.
            config: License configuration.
            warm_start: Previous solution to use as starting point.
            **kwargs: Additional parameters (ignored).

        Returns:
            Best licensing solution found.
        """
        from ...models.license import LicenseSolution
        from ...algorithms.greedy import GreedyAlgorithm

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        # Start with warm start if available, otherwise greedy solution
        if warm_start is not None:
            # Adapt warm start to current graph
            current_solution = self._adapt_solution_to_graph(warm_start, graph, config)
        else:
            greedy = GreedyAlgorithm()
            current_solution = greedy.solve(graph, config)
        
        current_cost = current_solution.calculate_cost(config)
        best_solution = current_solution
        best_cost = current_cost

        temperature = self.initial_temp

        for _iteration in range(self.max_iterations):
            if temperature < self.final_temp:
                break

            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(current_solution, graph, config)
            neighbor_cost = neighbor_solution.calculate_cost(config)

            # Accept or reject neighbor
            cost_diff = neighbor_cost - current_cost

            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

            # Cool down
            temperature *= self.cooling_rate

        return best_solution

    def _adapt_solution_to_graph(
        self, 
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Adapt a solution to work with a modified graph.
        
        Args:
            solution: Previous solution.
            graph: Current graph.
            config: License configuration.
            
        Returns:
            Adapted solution valid for current graph.
        """
        from ...models.license import LicenseSolution
        
        current_nodes = set(graph.nodes())
        new_licenses = {}
        assigned_nodes = set()
        
        # Process each license type and group
        for license_type, groups in solution.licenses.items():
            if license_type not in config.license_types:
                continue
                
            license_config = config.license_types[license_type]
            new_groups = {}
            
            for owner, members in groups.items():
                if owner in current_nodes:
                    # Keep only existing members that are connected to owner
                    valid_members = []
                    for member in members:
                        if member in current_nodes:
                            if member == owner or graph.has_edge(owner, member):
                                valid_members.append(member)
                    
                    # Check if group is still valid for this license type
                    if valid_members and license_config.is_valid_size(len(valid_members)):
                        new_groups[owner] = valid_members
                        assigned_nodes.update(valid_members)
            
            if new_groups:
                new_licenses[license_type] = new_groups
        
        # Handle unassigned nodes - assign them solo licenses with best license type
        unassigned = current_nodes - assigned_nodes
        if unassigned:
            # Find best solo license type
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                license_type, _ = best_solo
                if license_type not in new_licenses:
                    new_licenses[license_type] = {}
                for node in unassigned:
                    new_licenses[license_type][node] = [node]
        
        return LicenseSolution(licenses=new_licenses)

    def _generate_neighbor(
        self,
        solution: "LicenseSolution",
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> "LicenseSolution":
        """Generate a neighbor solution by making a small modification.

        Args:
            solution: Current solution.
            graph: The social network graph.
            config: License configuration.

        Returns:
            Neighbor solution.
        """
        from ...models.license import LicenseSolution
        import copy

        nodes = list(graph.nodes())
        if not nodes:
            return solution

        # Deep copy the current solution
        new_licenses = copy.deepcopy(solution.licenses)
        
        # Choose a random node to modify
        node = random.choice(nodes)
        
        # Find current assignment of this node
        current_license_type = None
        current_owner = None
        current_members = None
        
        for license_type, groups in new_licenses.items():
            for owner, members in groups.items():
                if node in members:
                    current_license_type = license_type
                    current_owner = owner
                    current_members = members
                    break
            if current_license_type:
                break
        
        if current_license_type is None:
            return solution  # Node not found, return original
        
        # Remove node from current assignment
        current_members.remove(node)
        if len(current_members) == 0:
            del new_licenses[current_license_type][current_owner]
            if len(new_licenses[current_license_type]) == 0:
                del new_licenses[current_license_type]
        elif current_owner == node:
            # Owner removed, need to reassign group
            if current_members:
                new_owner = current_members[0]
                new_licenses[current_license_type][new_owner] = current_members
            del new_licenses[current_license_type][current_owner]
        
        # Now try to assign node to a new license/group
        modification_types = ["solo", "join_group", "create_group"]
        modification = random.choice(modification_types)
        
        if modification == "solo":
            # Assign as solo with best license type for size 1
            best_license = config.get_best_license_for_size(1)
            if best_license:
                license_type, _ = best_license
                if license_type not in new_licenses:
                    new_licenses[license_type] = {}
                new_licenses[license_type][node] = [node]
        
        elif modification == "join_group":
            # Try to join an existing group
            neighbors = list(graph.neighbors(node))
            potential_groups = []
            
            for license_type, groups in new_licenses.items():
                license_config = config.license_types[license_type]
                for owner, members in groups.items():
                    if owner in neighbors and len(members) < license_config.max_size:
                        potential_groups.append((license_type, owner))
            
            if potential_groups:
                license_type, owner = random.choice(potential_groups)
                new_licenses[license_type][owner].append(node)
            else:
                # Can't join any group, make solo
                best_license = config.get_best_license_for_size(1)
                if best_license:
                    license_type, _ = best_license
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node]
        
        elif modification == "create_group":
            # Try to create a new group with neighbors
            neighbors = [n for n in graph.neighbors(node) if n != node]
            if neighbors:
                # Pick a neighbor to group with
                neighbor = random.choice(neighbors)
                
                # Remove neighbor from current assignment
                neighbor_license_type = None
                neighbor_owner = None
                neighbor_members = None
                
                for license_type, groups in new_licenses.items():
                    for owner, members in groups.items():
                        if neighbor in members:
                            neighbor_license_type = license_type
                            neighbor_owner = owner
                            neighbor_members = members
                            break
                    if neighbor_license_type:
                        break
                
                if neighbor_license_type:
                    neighbor_members.remove(neighbor)
                    if len(neighbor_members) == 0:
                        del new_licenses[neighbor_license_type][neighbor_owner]
                        if len(new_licenses[neighbor_license_type]) == 0:
                            del new_licenses[neighbor_license_type]
                    elif neighbor_owner == neighbor:
                        # Neighbor was owner, reassign
                        if neighbor_members:
                            new_owner = neighbor_members[0]
                            new_licenses[neighbor_license_type][new_owner] = neighbor_members
                        del new_licenses[neighbor_license_type][neighbor_owner]
                
                # Find best license type for group of 2
                best_license = config.get_best_license_for_size(2)
                if best_license:
                    license_type, _ = best_license
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node, neighbor]
                else:
                    # Can't form group, make both solo
                    best_solo = config.get_best_license_for_size(1)
                    if best_solo:
                        license_type, _ = best_solo
                        if license_type not in new_licenses:
                            new_licenses[license_type] = {}
                        new_licenses[license_type][node] = [node]
                        new_licenses[license_type][neighbor] = [neighbor]
            else:
                # No neighbors, make solo
                best_license = config.get_best_license_for_size(1)
                if best_license:
                    license_type, _ = best_license
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node]
        
        return LicenseSolution(licenses=new_licenses)
