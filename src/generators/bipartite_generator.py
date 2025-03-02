import networkx as nx

from .base import Generator


class BipartiteGenerator(Generator):
    def generate(self, num_nodes: int, **_) -> nx.Graph:
        top_nodes = num_nodes // 2
        return nx.complete_bipartite_graph(top_nodes, num_nodes - top_nodes)
