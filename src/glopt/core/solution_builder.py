from typing import List, Optional, Set, Sequence, Hashable
import networkx as nx
from .models import Solution, LicenseGroup, LicenseType

N = Hashable


class SolutionBuilder:
    """Small helpers to assemble solutions and query licenses."""

    @staticmethod
    def create_solution_from_groups(groups: List[LicenseGroup]) -> Solution:
        """Return Solution; cost/coverage are computed by Solution properties."""
        return Solution(groups=tuple(groups))

    @staticmethod
    def get_compatible_license_types(
        group_size: int,
        license_types: Sequence[LicenseType],
        exclude: Optional[LicenseType] = None,
    ) -> List[LicenseType]:
        """Licenses that allow given size, optionally excluding one."""
        out: List[LicenseType] = []
        for lt in license_types:
            if exclude and lt == exclude:
                continue
            if lt.min_capacity <= group_size <= lt.max_capacity:
                out.append(lt)
        return out

    @staticmethod
    def get_owner_neighbors_with_self(graph: nx.Graph, owner: N) -> Set[N]:
        """Closed neighborhood of owner."""
        return set(graph.neighbors(owner)) | {owner}

    @staticmethod
    def merge_groups(
        group1: LicenseGroup,
        group2: LicenseGroup,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Optional[LicenseGroup]:
        """Try to merge two groups into one valid group."""
        members = group1.all_members | group2.all_members
        size = len(members)

        for lt in license_types:
            if lt.min_capacity <= size <= lt.max_capacity:
                for owner in members:
                    neigh = SolutionBuilder.get_owner_neighbors_with_self(graph, owner)
                    if members.issubset(neigh):
                        return LicenseGroup(lt, owner, frozenset(members - {owner}))
        return None

    @staticmethod
    def find_cheapest_single_license(license_types: Sequence[LicenseType]) -> LicenseType:
        """Cheapest license that can cover a single node; else cheapest overall."""
        singles = [lt for lt in license_types if lt.min_capacity <= 1]
        return min(singles or list(license_types), key=lambda lt: lt.cost)

    @staticmethod
    def find_cheapest_license_for_size(size: int, license_types: Sequence[LicenseType]) -> Optional[LicenseType]:
        """Cheapest license for a given size or None if incompatible."""
        compat = [lt for lt in license_types if lt.min_capacity <= size <= lt.max_capacity]
        return min(compat, key=lambda lt: lt.cost) if compat else None
