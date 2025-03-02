import networkx as nx

from .base import Generator


class CompleteGraphGenerator(Generator):
    def generate(self, num_nodes: int, **_) -> nx.Graph:
        return nx.complete_graph(num_nodes)
