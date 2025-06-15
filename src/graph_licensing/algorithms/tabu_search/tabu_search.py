"""Tabu Search algorithm for licensing optimization."""

import random
from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class TabuSearchAlgorithm(BaseAlgorithm):
    """Tabu Search algorithm for licensing optimization."""

    def __init__(
        self,
        max_iterations: int = 100,
        max_no_improvement: int = 20,
        seed: int | None = None,
    ) -> None:
        """Initialize the tabu search algorithm.

        Args:
            max_iterations: Maximum number of iterations.
            max_no_improvement: Stop after this many iterations without improvement.
            seed: Random seed for reproducibility.
        """
        super().__init__("TabuSearch")
        self.max_iterations = max_iterations
        self.max_no_improvement = max_no_improvement
        self.seed = seed

    def supports_warm_start(self) -> bool:
        """Tabu search supports warm start initialization."""
        return True

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using tabu search.

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

        self.tabu_size = graph.number_of_nodes() // 20

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

        tabu_list = []
        no_improvement_count = 0

        for _iteration in range(self.max_iterations):
            if no_improvement_count >= self.max_no_improvement:
                break

            # Generate all possible neighbors
            neighbors = self._generate_all_neighbors(current_solution, graph, config)

            # Filter out tabu moves
            allowed_neighbors = []
            for neighbor in neighbors:
                move = self._solution_to_move(current_solution, neighbor)
                if move not in tabu_list:
                    allowed_neighbors.append(neighbor)

            if not allowed_neighbors:
                break

            # Select best allowed neighbor
            best_neighbor = min(
                allowed_neighbors,
                key=lambda x: x.calculate_cost(config),
            )
            best_neighbor_cost = best_neighbor.calculate_cost(config)

            # Update tabu list
            move = self._solution_to_move(current_solution, best_neighbor)
            tabu_list.append(move)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)

            # Update current solution
            current_solution = best_neighbor
            current_cost = best_neighbor_cost

            # Update best solution
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1

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

    def _generate_all_neighbors(
        self,
        solution: "LicenseSolution",
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> list["LicenseSolution"]:
        """Generate all possible neighbor solutions.

        Args:
            solution: Current solution.
            graph: The social network graph.
            config: License configuration.

        Returns:
            List of neighbor solutions.
        """
        import copy
        neighbors = []
        nodes = list(graph.nodes())

        # For each node, try different assignments
        for node in nodes:
            current_info = solution.get_node_license_info(node)
            if current_info is None:
                continue
                
            current_license_type, current_owner = current_info

            # Try solo assignment with each license type that supports size 1
            for license_name, license_config in config.license_types.items():
                if license_config.is_valid_size(1) and license_name != current_license_type:
                    new_solution = self._assign_solo(solution, node, license_name, config)
                    if new_solution and new_solution.is_valid(graph, config):
                        neighbors.append(new_solution)

            # Try group assignments with neighbors
            for neighbor in graph.neighbors(node):
                neighbor_info = solution.get_node_license_info(neighbor)
                if neighbor_info is None:
                    continue
                    
                neighbor_license_type, neighbor_owner = neighbor_info
                
                # Try joining neighbor's group (if neighbor is owner and has space)
                if neighbor == neighbor_owner:
                    for license_name, license_config in config.license_types.items():
                        if license_name == neighbor_license_type:
                            current_group_size = len(solution.licenses[license_name][neighbor])
                            if current_group_size < license_config.max_size:
                                new_solution = self._assign_to_group(solution, node, neighbor, license_name, config)
                                if new_solution and new_solution.is_valid(graph, config):
                                    neighbors.append(new_solution)

                # Try creating new group with neighbor
                for license_name, license_config in config.license_types.items():
                    if license_config.is_valid_size(2):
                        new_solution = self._create_group(solution, node, neighbor, license_name, config)
                        if new_solution and new_solution.is_valid(graph, config):
                            neighbors.append(new_solution)

        return neighbors

    def _assign_solo(self, solution: "LicenseSolution", node: int, license_type: str, config: "LicenseConfig") -> "LicenseSolution | None":
        """Assign a node to solo license.

        Args:
            solution: Current solution.
            node: Node to assign.
            license_type: License type to assign.
            config: License configuration.

        Returns:
            New solution with node assigned to solo or None if invalid.
        """
        from ...models.license import LicenseSolution
        import copy

        if license_type not in config.license_types:
            return None
            
        license_config = config.license_types[license_type]
        if not license_config.is_valid_size(1):
            return None

        new_licenses = copy.deepcopy(solution.licenses)
        
        # Remove node from current assignment
        self._remove_node_from_solution(new_licenses, node)
        
        # Add to new solo license
        if license_type not in new_licenses:
            new_licenses[license_type] = {}
        new_licenses[license_type][node] = [node]

        return LicenseSolution(licenses=new_licenses)

    def _assign_to_group(
        self,
        solution: "LicenseSolution",
        node: int,
        owner: int,
        license_type: str,
        config: "LicenseConfig",
    ) -> "LicenseSolution | None":
        """Assign a node to a group.

        Args:
            solution: Current solution.
            node: Node to assign.
            owner: Group owner.
            license_type: License type for the group.
            config: License configuration.

        Returns:
            New solution or None if assignment is invalid.
        """
        from ...models.license import LicenseSolution
        import copy

        if license_type not in config.license_types:
            return None
            
        license_config = config.license_types[license_type]
        
        # Check if owner already has a group of this type
        if license_type not in solution.licenses or owner not in solution.licenses[license_type]:
            return None
            
        current_group = solution.licenses[license_type][owner]
        if len(current_group) >= license_config.max_size:
            return None

        new_licenses = copy.deepcopy(solution.licenses)
        
        # Remove node from current assignment
        self._remove_node_from_solution(new_licenses, node)
        
        # Add to the group
        new_licenses[license_type][owner].append(node)

        return LicenseSolution(licenses=new_licenses)

    def _create_group(
        self,
        solution: "LicenseSolution",
        node1: int,
        node2: int,
        license_type: str,
        config: "LicenseConfig",
    ) -> "LicenseSolution | None":
        """Create a new group with two nodes.

        Args:
            solution: Current solution.
            node1: First node (will be owner).
            node2: Second node.
            license_type: License type for the group.
            config: License configuration.

        Returns:
            New solution or None if assignment is invalid.
        """
        from ...models.license import LicenseSolution
        import copy

        if license_type not in config.license_types:
            return None
            
        license_config = config.license_types[license_type]
        if not license_config.is_valid_size(2):
            return None

        new_licenses = copy.deepcopy(solution.licenses)
        
        # Remove both nodes from current assignments
        self._remove_node_from_solution(new_licenses, node1)
        self._remove_node_from_solution(new_licenses, node2)
        
        # Create new group
        if license_type not in new_licenses:
            new_licenses[license_type] = {}
        new_licenses[license_type][node1] = [node1, node2]

        return LicenseSolution(licenses=new_licenses)

    def _remove_node_from_solution(self, licenses: dict, node: int) -> None:
        """Remove a node from all licenses in the solution.

        Args:
            licenses: License dictionary to modify.
            node: Node to remove.
        """
        to_delete = []
        
        for license_type, groups in licenses.items():
            for owner, members in list(groups.items()):
                if node in members:
                    members.remove(node)
                    
                    if len(members) == 0:
                        # Group is empty, remove it
                        to_delete.append((license_type, owner))
                    elif owner == node and len(members) > 0:
                        # Owner was removed, reassign to first member
                        new_owner = members[0]
                        groups[new_owner] = members
                        to_delete.append((license_type, owner))
        
        # Remove empty groups
        for license_type, owner in to_delete:
            if owner in licenses[license_type]:
                del licenses[license_type][owner]
            if not licenses[license_type]:
                del licenses[license_type]

    def _solution_to_move(
        self,
        from_solution: "LicenseSolution",
        to_solution: "LicenseSolution",
    ) -> tuple:
        """Convert solution difference to a move representation.

        Args:
            from_solution: Original solution.
            to_solution: Target solution.

        Returns:
            Move representation as tuple.
        """
        # Create a hashable representation of the solution
        solution_repr = []
        for license_type, groups in sorted(to_solution.licenses.items()):
            for owner, members in sorted(groups.items()):
                solution_repr.append((license_type, owner, tuple(sorted(members))))
        
        return tuple(solution_repr)
