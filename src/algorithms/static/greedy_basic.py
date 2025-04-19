from __future__ import annotations
from ..base import BaseSolver, Solution
import networkx as nx


class GreedyBasicSolver(BaseSolver):
    def solve(self, graph: nx.Graph, c_single: float, c_group: float, group_size: int) -> Solution:
        covered: set[int] = set()
        solution: Solution = {"singles": [], "groups": []}

        sorted_nodes: list[int] = sorted(graph.nodes, key=lambda v: graph.degree[v], reverse=True)

        for node in sorted_nodes:
            if node in covered:
                continue

            potential: set[int] = set(graph.neighbors(node)) - covered
            selected: set[int] = {node}

            if potential:
                if len(potential) > group_size - 1:
                    top_neighbors: list[int] = sorted(
                        potential,
                        key=lambda v: graph.degree[v],
                        reverse=True,
                    )[: group_size - 1]
                    selected.update(top_neighbors)
                else:
                    selected.update(potential)

            if len(selected) >= 2:
                solution["groups"].append({"license_holder": node, "members": list(selected)})
            else:
                solution["singles"].append(node)

            covered.update(selected)

        return solution
