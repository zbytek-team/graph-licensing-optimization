from __future__ import annotations
import itertools
import networkx as nx
from src.algorithms.base import BaseSolver, Solution
from src.utils.solution_utils import calculate_cost


class BruteForceSolver(BaseSolver):
    def solve(self, graph: nx.Graph, c_single: float, c_group: float, group_size: int) -> Solution:
        n = graph.number_of_nodes()

        best_cost = float("inf")
        best_sol: Solution = {"singles": [], "groups": []}
        nodes = list(graph.nodes)

        for holder_mask in range(1 << n):
            holders = [nodes[i] for i in range(n) if holder_mask & (1 << i)]
            member_options: list[list[list[int]]] = []
            feasible = True

            for h in holders:
                neigh = [u for u in graph.neighbors(h) if u not in holders]
                subsets = [
                    [h, *combo]
                    for k in range(1, min(group_size - 1, len(neigh)) + 1)
                    for combo in itertools.combinations(neigh, k)
                ]

                if not subsets:
                    feasible = False
                    break
                member_options.append(subsets)

            if not feasible and holders:
                continue

            for member_choice in itertools.product(*member_options) if holders else [[]]:
                covered = set()
                groups = []
                for members in member_choice:
                    covered.update(members)
                    groups.append({"license_holder": members[0], "members": list(members)})
                singles = [v for v in nodes if v not in covered]
                covered.update(singles)
                if covered != set(nodes):
                    continue
                cost = calculate_cost({"singles": singles, "groups": groups}, c_single, c_group)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = {"singles": singles, "groups": groups}

        return best_sol
