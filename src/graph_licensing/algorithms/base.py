from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import networkx as nx

if TYPE_CHECKING:
    from ..models.license import LicenseConfig, LicenseSolution


class BaseAlgorithm(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        pass

    def solve_dynamic(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        iterations: int,
        modification_prob: float = 0.1,
        **kwargs,
    ) -> List["LicenseSolution"]:
        solutions = []
        current_graph = graph.copy()
        previous_solution = None

        for i in range(iterations):
            solution = self.solve(current_graph, config, warm_start=previous_solution, **kwargs)
            solutions.append(solution)
            previous_solution = solution

            if i < iterations - 1:
                current_graph = self._modify_graph(current_graph, modification_prob)

        return solutions

    def _modify_graph(self, graph: "nx.Graph", modification_prob: float) -> "nx.Graph":
        import random

        modified_graph = graph.copy()
        nodes = list(modified_graph.nodes())

        base_changes = max(1, int(len(nodes) * modification_prob * 0.15))  # ~3 changes for 20 nodes at prob=1.0

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

                if len(nodes) > 1:
                    other_nodes = [n for n in nodes if n != new_node]
                    num_connections = min(random.randint(1, 3), len(other_nodes))
                    targets = random.sample(other_nodes, num_connections)
                    for target in targets:
                        modified_graph.add_edge(new_node, target)

        edge_changes = random.randint(0, base_changes * 2)  # More edge changes than node changes
        for _ in range(edge_changes):
            current_nodes = list(modified_graph.nodes())
            if len(current_nodes) > 1:
                if random.random() < 0.5 and modified_graph.number_of_edges() > 0:
                    edge_to_remove = random.choice(list(modified_graph.edges()))
                    modified_graph.remove_edge(*edge_to_remove)
                else:
                    non_edges = list(nx.non_edges(modified_graph))
                    if non_edges:
                        edge_to_add = random.choice(non_edges)
                        modified_graph.add_edge(*edge_to_add)

        return modified_graph

    def supports_warm_start(self) -> bool:
        return False
