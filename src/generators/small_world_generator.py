import networkx as nx

from .base import Generator


class SmallWorldGenerator(Generator):
    def generate(self, num_nodes: int, **kwargs) -> nx.Graph:
        k = kwargs.get("k", 4)
        p = kwargs.get("p", 0.1)
        return nx.watts_strogatz_graph(num_nodes, k, p)
