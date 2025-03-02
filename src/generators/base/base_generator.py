from abc import ABC, abstractmethod

import networkx as nx


class Generator(ABC):
    @abstractmethod
    def generate(self, num_nodes: int, **kwargs) -> nx.Graph:
        pass
