import networkx as nx

from .base import BaseGenerator


class WattsStrogatzGenerator(BaseGenerator):
    def __init__(self, k: int = 4, p: float = 0.1, seed: int | None = None):
        self.k = k
        self.p = p
        self.seed = seed

    def generate(self, num_nodes: int) -> nx.Graph:
        return nx.watts_strogatz_graph(n=num_nodes, k=self.k, p=self.p, seed=self.seed)
