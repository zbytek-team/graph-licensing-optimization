import random

import networkx as nx

from .base import Generator


class PlannarGraphGenerator(Generator):
    def generate(self, num_nodes: int, **_) -> nx.Graph:
        G = nx.Graph()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < 0.3:
                    G.add_edge(i, j)
        return G if nx.check_planarity(G)[0] else self.generate(num_nodes, **_)
