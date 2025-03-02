import networkx as nx

from .base import Generator


class RandomGraphGenerator(Generator):
    def generate(self, num_nodes: int, **kwargs) -> nx.Graph:
        p = kwargs.get("p", 0.5)
        return nx.erdos_renyi_graph(num_nodes, p)
