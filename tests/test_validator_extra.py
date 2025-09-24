from __future__ import annotations

import networkx as nx
import pytest

from glopt.core.models import LicenseGroup, Solution
from glopt.core.solution_validator import SolutionValidator
from glopt.license_config import LicenseConfigFactory


def make_line_graph(n: int) -> nx.Graph:
    G = nx.path_graph(n)
    # ensure int nodes 0..n-1 (already true for path_graph)
    return G


def test_validator_detects_overlap_and_non_neighbors() -> None:
    G = make_line_graph(4)  # 0-1-2-3
    lts = LicenseConfigFactory.get_config("duolingo_p_2_0")
    fam = next(lt for lt in lts if lt.name == "Family")
    # two groups overlapping on node 1 and including non-neighbor 3 under owner 0
    g1 = LicenseGroup(fam, 0, frozenset({1}))
    g2 = LicenseGroup(fam, 1, frozenset({2, 3}))  # 3 is not neighbor of 1
    ok, issues = SolutionValidator(debug=True).validate(Solution(groups=(g1, g2)), G)
    assert not ok
    codes = {i.code for i in issues}
    assert "OVERLAP" in codes and "DISCONNECTED_MEMBER" in codes


def test_validator_accepts_correct_cover() -> None:
    G = make_line_graph(3)  # 0-1-2
    lts = LicenseConfigFactory.get_config("duolingo_p_2_0")
    fam = next(lt for lt in lts if lt.name == "Family")
    sol = Solution(groups=(LicenseGroup(fam, 1, frozenset({0})), LicenseGroup(fam, 1, frozenset({2}))))
    ok, issues = SolutionValidator().validate(sol, G)
    assert ok, f"unexpected issues: {issues}"

