import networkx as nx

from .base import Generator


class ScaleFreeGenerator(Generator):
    def generate(self, num_nodes: int, **kwargs) -> nx.Graph:
        m = kwargs.get("m", 2)
        return nx.barabasi_albert_graph(num_nodes, m)
