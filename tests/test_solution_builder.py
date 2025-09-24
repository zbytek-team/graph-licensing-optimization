from __future__ import annotations

import networkx as nx

from glopt.core.solution_builder import SolutionBuilder
from glopt.license_config import LicenseConfigFactory


def test_find_cheapest_license_for_size_prefers_lower_cost() -> None:
    lts = LicenseConfigFactory.get_config("roman_p_3_0")
    # size=2 should pick Group (cost=3.0) over Solo (incompatible)
    lt = SolutionBuilder.find_cheapest_license_for_size(2, lts)
    assert lt is not None and lt.name == "Group"


def test_get_compatible_license_types_respects_bounds() -> None:
    lts = LicenseConfigFactory.get_config("duolingo_p_2_0")
    compat = SolutionBuilder.get_compatible_license_types(6, lts)
    names = {lt.name for lt in compat}
    assert names == {"Family"}


def test_merge_groups_when_neighbors_allows_merge() -> None:
    # Star graph center=0, leaves 1..4, owner can cover neighbors
    G = nx.star_graph(4)
    lts = LicenseConfigFactory.get_config("duolingo_p_2_0")
    # two groups size=2 each; should merge into size=4 if allowed by license (cap=6)
    g1 = SolutionBuilder  # alias for brevity
    from glopt.core.models import LicenseGroup

    fam = SolutionBuilder.find_cheapest_license_for_size(2, lts)
    assert fam is not None
    a = LicenseGroup(fam, 0, frozenset({1}))
    b = LicenseGroup(fam, 0, frozenset({2}))
    merged = SolutionBuilder.merge_groups(a, b, G, lts)
    assert merged is not None
    assert merged.size == 3  # owner 0 + {1,2}
    # Merge merged with another leaf
    c = LicenseGroup(fam, 0, frozenset({3}))
    merged2 = SolutionBuilder.merge_groups(merged, c, G, lts)
    assert merged2 is not None and merged2.size == 4

