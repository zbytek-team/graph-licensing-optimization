from __future__ import annotations

import math
import os
from pathlib import Path

import networkx as nx
import pytest

try:
    from hypothesis import given, strategies as st
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not installed", allow_module_level=True)

from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core.solution_validator import SolutionValidator
from glopt.license_config import LicenseConfigFactory


def gen_random_graph(n: int, p: float, seed: int | None) -> nx.Graph:
    G = nx.gnp_random_graph(n=n, p=p, seed=seed)
    if not all(isinstance(v, int) for v in G.nodes()):
        mapping = {v: i for i, v in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)
    return G


@given(
    n=st.integers(min_value=5, max_value=30),
    p=st.floats(min_value=0.05, max_value=0.4, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_greedy_solution_is_valid_random(n: int, p: float, seed: int) -> None:
    G = gen_random_graph(n, p, seed)
    lts = LicenseConfigFactory.get_config("roman_domination")
    sol = GreedyAlgorithm().solve(G, lts, seed=seed)
    ok, issues = SolutionValidator(debug=True).validate(sol, G)
    assert ok, f"invalid solution: {issues}"


def test_smoke_real_ego_if_available() -> None:
    """Load one Facebook ego graph if present and run greedy smoke test."""
    base = Path("data/facebook")
    edges = sorted(base.glob("*.edges"))
    if not edges:
        pytest.skip("no facebook ego data available")
    ego_id = edges[0].stem
    # minimal load: only edges file
    G = nx.Graph()
    with edges[0].open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    lts = LicenseConfigFactory.get_config("roman_domination")
    sol = GreedyAlgorithm().solve(G, lts)
    ok, issues = SolutionValidator(debug=False).validate(sol, G)
    assert ok, f"invalid solution on ego {ego_id}: {issues}"

