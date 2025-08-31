from typing import Any, Dict, List, Tuple
import networkx as nx

from ..core import Algorithm, LicenseGroup, LicenseType, Solution
from ..core.solution_builder import SolutionBuilder


class TreeDynamicProgramming(Algorithm):
    """Dynamic programming algorithm optimized for tree graphs."""

    @property
    def name(self) -> str:
        return "tree_dp"

    def solve(
        self,
        graph: nx.Graph,
        license_types: List[LicenseType],
        **_: Any,
    ) -> Solution:
        if not nx.is_tree(graph):
            raise ValueError("TreeDynamicProgramming requires a tree graph")

        if len(graph.nodes()) == 0:
            return Solution(groups=())

        if len(graph.nodes()) == 1:
            node = list(graph.nodes())[0]
            cheapest = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 <= lt.max_capacity else float("inf"))
            group = LicenseGroup(cheapest, node, frozenset())
            return SolutionBuilder.create_solution_from_groups([group])

        root = list(graph.nodes())[0]
        memo = {}
        cost, groups = self._solve_subtree(graph, root, None, license_types, memo)
        return SolutionBuilder.create_solution_from_groups(groups)

    def _solve_subtree(self, graph: nx.Graph, node: Any, parent: Any, license_types: List[LicenseType], memo: Dict) -> Tuple[float, List[LicenseGroup]]:
        children = [child for child in graph.neighbors(node) if child != parent]

        state_key = (node, tuple(sorted(children)))
        if state_key in memo:
            return memo[state_key]

        if not children:
            cheapest = min(
                license_types,
                key=lambda lt: lt.cost if lt.min_capacity <= 1 <= lt.max_capacity else float("inf"),
            )
            group = LicenseGroup(cheapest, node, frozenset())
            result = (cheapest.cost, [group])
            memo[state_key] = result
            return result

        child_solutions = {}
        for child in children:
            child_solutions[child] = self._solve_subtree(graph, child, node, license_types, memo)

        best_cost = float("inf")
        best_groups = []

        for license_type in license_types:
            min_capacity = license_type.min_capacity
            max_capacity = license_type.max_capacity

            if max_capacity < 1:
                continue

            for num_children in range(min(len(children), max_capacity - 1) + 1):
                if min_capacity > num_children + 1:
                    continue

                from itertools import combinations

                if num_children == 0:
                    child_combinations = [()]
                else:
                    child_combinations = combinations(children, num_children)

                for child_combination in child_combinations:
                    included_children = set(child_combination)
                    remaining_children = [c for c in children if c not in included_children]

                    cost = license_type.cost
                    groups = [LicenseGroup(license_type, node, frozenset(included_children))]

                    for child in remaining_children:
                        child_cost, child_groups = child_solutions[child]
                        cost += child_cost
                        groups.extend(child_groups)

                    for child in included_children:
                        subtree_cost = self._solve_child_subtree(graph, child, node, license_types, memo)
                        cost += subtree_cost[0]
                        groups.extend(subtree_cost[1])

                    if cost < best_cost:
                        best_cost = cost
                        best_groups = groups

        result = (best_cost, best_groups)
        memo[state_key] = result
        return result

    def _solve_child_subtree(self, graph: nx.Graph, child: Any, parent: Any, license_types: List[LicenseType], memo: Dict) -> Tuple[float, List[LicenseGroup]]:
        """Solve the subtree rooted at child, where child is already covered by parent's license."""

        grandchildren = [gc for gc in graph.neighbors(child) if gc != parent]

        if not grandchildren:

            return (0.0, [])

        total_cost = 0.0
        all_groups = []

        for grandchild in grandchildren:
            gc_cost, gc_groups = self._solve_subtree(graph, grandchild, child, license_types, memo)
            total_cost += gc_cost
            all_groups.extend(gc_groups)

        return (total_cost, all_groups)
