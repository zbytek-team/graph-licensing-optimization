import random

import networkx as nx

from src.logger import get_logger

logger = get_logger(__name__)

random.seed(42)


def generate_clustered_graph(N=300, p_ws=0.02, extra_links=3, num_subgroups=15) -> nx.Graph:
    G = nx.Graph()

    for _ in range(num_subgroups):
        subgroup_size = random.randint(5, 10)
        subgraph = nx.watts_strogatz_graph(subgroup_size, max(2, subgroup_size // 4), p_ws)
        mapping = {i: random.randint(0, N - 1) for i in range(subgroup_size)}
        H = nx.relabel_nodes(subgraph, mapping)
        G = nx.compose(G, H)

    for _ in range(extra_links):
        a, b = random.sample(range(N), 2)
        G.add_edge(a, b)

    return G


def generate_tree_graph(N=300) -> nx.Graph:
    G = nx.Graph()
    for i in range(1, N):
        G.add_edge(i, random.randint(0, i - 1))
    return G


def generate_star_graph(N=300) -> nx.Graph:
    G = nx.Graph()
    for i in range(1, N):
        G.add_edge(0, i)
    return G


def generate_complete_graph(N=300) -> nx.Graph:
    G = nx.complete_graph(N)
    return G


def generate_random_graph(N=300, p=0.05) -> nx.Graph:
    G = nx.erdos_renyi_graph(N, p)
    return G
