import networkx as nx
import pytest

from src.algorithms.naive import NaiveAlgorithm
from src.core import LicenseType


def create_license_types():
    """Helper to create basic license types used in tests."""
    single = LicenseType(name="single", cost=5.0, min_capacity=1, max_capacity=1)
    pair = LicenseType(name="pair", cost=6.0, min_capacity=2, max_capacity=2)
    return [single, pair]


def test_naive_algorithm_two_node_graph():
    """Naive algorithm should cover a simple two-node graph optimally."""
    graph = nx.Graph()
    graph.add_edge(0, 1)
    algo = NaiveAlgorithm()

    solution = algo.solve(graph, create_license_types())

    assert solution.total_cost == pytest.approx(6.0)
    assert solution.covered_nodes == {0, 1}
    assert len(solution.groups) == 1
    group = solution.groups[0]
    assert group.all_members == {0, 1}


def test_naive_algorithm_too_large_graph_raises():
    """Algorithm should refuse to solve graphs with more than ten nodes."""
    graph = nx.path_graph(11)
    algo = NaiveAlgorithm()

    with pytest.raises(ValueError):
        algo.solve(graph, create_license_types())

