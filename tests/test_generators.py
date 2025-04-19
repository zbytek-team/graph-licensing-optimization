import networkx as nx
import pytest
from src.data.generators import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
)


@pytest.mark.parametrize(
    "gen, kwargs, n_expected",
    [
        (generate_erdos_renyi, {"n": 50, "p": 0.05, "seed": 1}, 50),
        (generate_barabasi_albert, {"n": 40, "m": 3, "seed": 2}, 40),
        (generate_watts_strogatz, {"n": 30, "k": 4, "p": 0.1, "seed": 3}, 30),
    ],
)
def test_graph_size(gen, kwargs, n_expected):
    g: nx.Graph = gen(**kwargs)
    assert g.number_of_nodes() == n_expected
