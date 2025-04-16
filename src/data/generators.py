import networkx as nx
from typing import Optional


def generate_erdos_renyi(n: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    return nx.gnp_random_graph(n, p, seed=seed)


def generate_barabasi_albert(n: int, m: int, seed: Optional[int] = None) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=seed)


def generate_watts_strogatz(n: int, k: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    return nx.watts_strogatz_graph(n, k, p, seed=seed)
