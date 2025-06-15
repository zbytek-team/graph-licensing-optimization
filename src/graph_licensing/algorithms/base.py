"""Enhanced base algorithm interface for licensing optimization."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List
import networkx as nx

if TYPE_CHECKING:
    from ..models.license import LicenseConfig, LicenseSolution


class BaseAlgorithm(ABC):
    """Abstract base class for licensing optimization algorithms."""

    def __init__(self, name: str) -> None:
        """Initialize the algorithm.

        Args:
            name: Name of the algorithm.
        """
        self.name = name

    @abstractmethod
    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve the licensing optimization problem.

        Args:
            graph: The social network graph.
            config: License configuration with pricing and constraints.
            warm_start: Previous solution to use as starting point (for dynamic problems).
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            Optimal or near-optimal licensing solution.
        """

    def solve_dynamic(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        iterations: int,
        modification_prob: float = 0.1,
        **kwargs,
    ) -> List["LicenseSolution"]:
        """Solve the dynamic version of the problem.

        Args:
            graph: Initial social network graph.
            config: License configuration.
            iterations: Number of dynamic iterations.
            modification_prob: Probability of modifying the graph at each iteration.
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            List of solutions for each iteration.
        """
        solutions = []
        current_graph = graph.copy()
        previous_solution = None

        for i in range(iterations):
            # Solve current graph, using previous solution as warm start
            solution = self.solve(
                current_graph, 
                config, 
                warm_start=previous_solution,
                **kwargs
            )
            solutions.append(solution)
            previous_solution = solution

            # Modify graph for next iteration (except last iteration)
            if i < iterations - 1:
                current_graph = self._modify_graph(current_graph, modification_prob)

        return solutions

    def _modify_graph(self, graph: "nx.Graph", modification_prob: float) -> "nx.Graph":
        """Modify the graph by adding/removing nodes and edges.

        Args:
            graph: Current graph.
            modification_prob: Probability of each type of modification (used as intensity factor).

        Returns:
            Modified graph.
        """
        import random

        modified_graph = graph.copy()
        nodes = list(modified_graph.nodes())
        
        # Calculate number of modifications based on probability as intensity
        # Higher modification_prob means more changes per iteration
        base_changes = max(1, int(len(nodes) * modification_prob * 0.15))  # ~3 changes for 20 nodes at prob=1.0
        
        # Node modifications (add/remove nodes)
        node_changes = random.randint(0, base_changes)
        for _ in range(node_changes):
            if len(nodes) > 1 and random.random() < 0.5:  # 60% chance to remove
                node_to_remove = random.choice(nodes)
                modified_graph.remove_node(node_to_remove)
                nodes.remove(node_to_remove)
            else:  # Add node
                new_node = max(nodes) + 1 if nodes else 0
                modified_graph.add_node(new_node)
                nodes.append(new_node)
                # Connect to 1-3 random existing nodes
                if len(nodes) > 1:
                    other_nodes = [n for n in nodes if n != new_node]
                    num_connections = min(random.randint(1, 3), len(other_nodes))
                    targets = random.sample(other_nodes, num_connections)
                    for target in targets:
                        modified_graph.add_edge(new_node, target)

        # Edge modifications (add/remove edges)
        edge_changes = random.randint(0, base_changes * 2)  # More edge changes than node changes
        for _ in range(edge_changes):
            current_nodes = list(modified_graph.nodes())
            if len(current_nodes) > 1:
                if random.random() < 0.5 and modified_graph.number_of_edges() > 0:
                    # Remove edge
                    edge_to_remove = random.choice(list(modified_graph.edges()))
                    modified_graph.remove_edge(*edge_to_remove)
                else:
                    # Add edge
                    non_edges = list(nx.non_edges(modified_graph))
                    if non_edges:
                        edge_to_add = random.choice(non_edges)
                        modified_graph.add_edge(*edge_to_add)

        return modified_graph

    def supports_warm_start(self) -> bool:
        """Check if the algorithm supports warm start initialization.
        
        Returns:
            True if the algorithm can use previous solutions as starting points.
        """
        return False
