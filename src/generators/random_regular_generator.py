import networkx as nx

from .base import BaseGenerator


class RandomRegularGenerator(BaseGenerator):
    def __init__(self, d: int = 3, seed: int | None = None):
        self.d = d
        self.seed = seed

    def generate(self, num_nodes: int) -> nx.Graph:
        return nx.random_regular_graph(d=self.d, n=num_nodes, seed=self.seed)
