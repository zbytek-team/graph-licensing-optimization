import networkx as nx

from .base import BaseGenerator


class CompleteGraphGenerator(BaseGenerator):
    def generate(self, num_nodes: int) -> nx.Graph:
        return nx.complete_graph(num_nodes)
