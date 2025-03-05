import networkx as nx

from .base import BaseGenerator


class TreeGenerator(BaseGenerator):
    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate(self, num_nodes: int) -> nx.Graph:
        return nx.random_tree(n=num_nodes, seed=self.seed)
