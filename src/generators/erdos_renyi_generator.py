import networkx as nx

from .base import BaseGenerator


class ErdosRenyiGenerator(BaseGenerator):
    def __init__(self, p: float = 0.1, seed: int | None = None):
        self.p = p
        self.seed = seed

    def generate(self, num_nodes: int) -> nx.Graph:
        return nx.erdos_renyi_graph(n=num_nodes, p=self.p, seed=self.seed)
