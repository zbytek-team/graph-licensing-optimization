from typing import Any, List, Set
import networkx as nx

from ..core import Algorithm, LicenseGroup, LicenseType, Solution
from ..core.solution_builder import SolutionBuilder


class GreedyAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "greedy"

    def solve(
        self,
        graph: nx.Graph,
        license_types: List[LicenseType],
        **_: Any,
    ) -> Solution:
        licenses = sorted(license_types, key=lambda lt: (-lt.max_capacity, lt.cost))

        nodes: List[Any] = list(graph.nodes())
        uncovered: Set[Any] = set(nodes)
        groups: List[LicenseGroup] = []

        for owner in sorted(nodes, key=lambda n: graph.degree(n), reverse=True):
            if owner not in uncovered:
                continue

            avail = SolutionBuilder.get_owner_neighbors_with_self(graph, owner) & uncovered
            if not avail:
                continue

            best_group = self._best_group_for_owner(owner, avail, graph, licenses)
            if best_group is None:
                continue

            groups.append(best_group)
            uncovered -= best_group.all_members

        while uncovered:
            owner = next(iter(uncovered))
            avail = SolutionBuilder.get_owner_neighbors_with_self(graph, owner) & uncovered

            fallback = self._cheapest_feasible_group(owner, avail, graph, license_types)
            if fallback is not None:
                groups.append(fallback)
                uncovered -= fallback.all_members
                continue

            cheapest = min(license_types, key=lambda lt: lt.cost)
            if cheapest.min_capacity == 1:
                groups.append(LicenseGroup(license_type=cheapest, owner=owner, additional_members=frozenset()))
                uncovered.remove(owner)
            else:
                break

        return SolutionBuilder.create_solution_from_groups(groups)

    def _best_group_for_owner(
        self,
        owner: Any,
        avail: Set[Any],
        graph: nx.Graph,
        licenses: List[LicenseType],
    ) -> LicenseGroup | None:
        """
        Pick the most efficient group for 'owner' from available neighbors.
        Efficiency = cost / group_size. Respect license [min, max].
        """
        ordered = sorted(avail, key=lambda n: graph.degree(n), reverse=True)

        best: LicenseGroup | None = None
        best_eff = float("inf")

        for lt in licenses:
            cap_additional = max(0, lt.max_capacity - 1)
            pool = [n for n in ordered if n != owner]
            take = min(len(pool), cap_additional)

            additional = pool[:take]
            size_with_owner = 1 + len(additional)
            if size_with_owner < lt.min_capacity:
                continue

            grp = LicenseGroup(license_type=lt, owner=owner, additional_members=frozenset(additional))
            eff = lt.cost / grp.size
            if eff < best_eff:
                best_eff = eff
                best = grp

        return best

    def _cheapest_feasible_group(
        self,
        owner: Any,
        avail: Set[Any],
        graph: nx.Graph,
        license_types: List[LicenseType],
    ) -> LicenseGroup | None:
        """
        Build the cheapest valid group for 'owner' at exactly min_capacity if possible.
        """
        for lt in sorted(license_types, key=lambda x: (x.cost, -x.max_capacity)):
            if len(avail) < lt.min_capacity:
                continue

            need_additional = max(0, lt.min_capacity - 1)
            pool = sorted((n for n in avail if n != owner), key=lambda n: graph.degree(n), reverse=True)
            chosen = pool[:need_additional]

            return LicenseGroup(license_type=lt, owner=owner, additional_members=frozenset(chosen))

        return None
