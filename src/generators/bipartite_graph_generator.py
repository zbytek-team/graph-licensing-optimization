import networkx as nx

from .base import BaseGenerator


class BipartiteGraphGenerator(BaseGenerator):
    def __init__(self, p: float = 0.1, n1: int | None = None, seed: int | None = None):
        self.p = p
        self.n1 = n1
        self.seed = seed

    def generate(self, num_nodes: int) -> nx.Graph:
        n1 = self.n1 if self.n1 is not None else num_nodes // 2
        n2 = num_nodes - n1
        graph = nx.bipartite.generators.random_graph(n1, n2, self.p, seed=self.seed)
        return graph
