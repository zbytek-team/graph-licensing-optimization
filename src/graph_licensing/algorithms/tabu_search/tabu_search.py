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
            return LicenseSolution(solo_nodes=[], group_owners={})

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
        
        # Filter solo nodes to only include existing nodes
        new_solo = [node for node in solution.solo_nodes if node in current_nodes]
        
        # Adapt groups
        new_groups = {}
        for owner, members in solution.group_owners.items():
            if owner in current_nodes:
                # Keep only existing members
                valid_members = [m for m in members if m in current_nodes]
                # Ensure all members are still connected to owner
                connected_members = [m for m in valid_members 
                                   if m == owner or graph.has_edge(owner, m)]
                
                if len(connected_members) > 1:  # Valid group
                    new_groups[owner] = connected_members
                elif connected_members:  # Only owner left
                    new_solo.append(owner)
        
        # Handle any nodes that weren't assigned
        assigned_nodes = set(new_solo) | set().union(*new_groups.values()) if new_groups else set(new_solo)
        unassigned = current_nodes - assigned_nodes
        new_solo.extend(list(unassigned))
        
        return LicenseSolution(solo_nodes=new_solo, group_owners=new_groups)

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
        neighbors = []
        nodes = list(graph.nodes())

        # For each node, try different assignments
        for node in nodes:
            current_type = solution.get_node_license_type(node)

            # Try solo assignment
            if current_type.value != "solo":
                new_solution = self._assign_solo(solution, node)
                if new_solution.is_valid(graph, config):
                    neighbors.append(new_solution)

            # Try group assignments with neighbors
            for neighbor in graph.neighbors(node):
                neighbor_type = solution.get_node_license_type(neighbor)

                # Try making neighbor the group owner
                if neighbor_type.value != "group_owner" or node not in solution.group_owners.get(neighbor, []):
                    new_solution = self._assign_to_group(solution, node, neighbor)
                    if new_solution and new_solution.is_valid(graph, config):
                        neighbors.append(new_solution)

        return neighbors

    def _assign_solo(self, solution: "LicenseSolution", node: int) -> "LicenseSolution":
        """Assign a node to solo license.

        Args:
            solution: Current solution.
            node: Node to assign.

        Returns:
            New solution with node assigned to solo.
        """
        from ...models.license import LicenseSolution

        new_solo = solution.solo_nodes.copy()
        new_groups = {k: v.copy() for k, v in solution.group_owners.items()}

        # Remove from current group if any
        for owner, members in new_groups.items():
            if node in members:
                members.remove(node)
                if len(members) == 0:
                    del new_groups[owner]
                elif owner == node:
                    # Node was owner, need to handle group
                    if len(members) > 0:
                        # Make first member the new owner
                        new_owner = members[0]
                        new_groups[new_owner] = members
                    del new_groups[owner]
                break

        # Add to solo
        if node not in new_solo:
            new_solo.append(node)

        return LicenseSolution(solo_nodes=new_solo, group_owners=new_groups)

    def _assign_to_group(
        self,
        solution: "LicenseSolution",
        node: int,
        owner: int,
    ) -> "LicenseSolution | None":
        """Assign a node to a group.

        Args:
            solution: Current solution.
            node: Node to assign.
            owner: Group owner.

        Returns:
            New solution or None if assignment is invalid.
        """
        from ...models.license import LicenseSolution

        new_solo = solution.solo_nodes.copy()
        new_groups = {k: v.copy() for k, v in solution.group_owners.items()}

        # Remove node from current assignment
        if node in new_solo:
            new_solo.remove(node)
        else:
            for group_owner, members in new_groups.items():
                if node in members:
                    members.remove(node)
                    if len(members) == 0:
                        del new_groups[group_owner]
                    break

        # Remove owner from solo if needed
        if owner in new_solo:
            new_solo.remove(owner)

        # Add to group
        if owner not in new_groups:
            new_groups[owner] = [owner]

        if node not in new_groups[owner]:
            new_groups[owner].append(node)

        return LicenseSolution(solo_nodes=new_solo, group_owners=new_groups)

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
        # Simple representation: hash of the solution
        return (
            tuple(sorted(to_solution.solo_nodes)),
            tuple(
                sorted((k, tuple(sorted(v))) for k, v in to_solution.group_owners.items()),
            ),
        )
