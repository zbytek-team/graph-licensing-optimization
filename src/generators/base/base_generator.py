from abc import ABC, abstractmethod

import networkx as nx


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, num_nodes: int) -> nx.Graph:
        pass
