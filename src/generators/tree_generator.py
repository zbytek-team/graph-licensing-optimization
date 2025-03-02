import networkx as nx

from .base import Generator


class TreeGenerator(Generator):
    def generate(self, num_nodes: int, **_) -> nx.Graph:
        return nx.random_tree(num_nodes)
