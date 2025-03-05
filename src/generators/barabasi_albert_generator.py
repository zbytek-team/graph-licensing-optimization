import networkx as nx

from .base import BaseGenerator


class BarabasiAlbertGenerator(BaseGenerator):
    def __init__(self, m: int = 2, seed: int | None = None):
        self.m = m
        self.seed = seed

    def generate(self, num_nodes: int) -> nx.Graph:
        return nx.barabasi_albert_graph(n=num_nodes, m=self.m, seed=self.seed)
