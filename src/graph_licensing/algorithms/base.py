"""Base algorithm interface for licensing optimization."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
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
        **kwargs,
    ) -> "LicenseSolution":
        """Solve the licensing optimization problem.

        Args:
            graph: The social network graph.
            config: License configuration with pricing and constraints.
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
    ) -> list["LicenseSolution"]:
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

        for i in range(iterations):
            # Solve current graph
            solution = self.solve(current_graph, config, **kwargs)
            solutions.append(solution)

            # Modify graph for next iteration (except last iteration)
            if i < iterations - 1:
                current_graph = self._modify_graph(current_graph, modification_prob)

        return solutions

    def _modify_graph(self, graph: "nx.Graph", modification_prob: float) -> "nx.Graph":
        """Modify the graph by adding/removing nodes and edges.

        Args:
            graph: Current graph.
            modification_prob: Probability of each type of modification.

        Returns:
            Modified graph.
        """
        import random

        modified_graph = graph.copy()
        nodes = list(modified_graph.nodes())

        # Add/remove nodes
        if random.random() < modification_prob and len(nodes) > 1:
            if random.random() < 0.5:  # Remove node
                node_to_remove = random.choice(nodes)
                modified_graph.remove_node(node_to_remove)
            else:  # Add node
                new_node = max(nodes) + 1 if nodes else 0
                modified_graph.add_node(new_node)
                # Connect to 1-3 random existing nodes
                if nodes:
                    num_connections = min(random.randint(1, 3), len(nodes))
                    targets = random.sample(nodes, num_connections)
                    for target in targets:
                        modified_graph.add_edge(new_node, target)

        # Add/remove edges
        nodes = list(modified_graph.nodes())
        if len(nodes) > 1 and random.random() < modification_prob:
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
