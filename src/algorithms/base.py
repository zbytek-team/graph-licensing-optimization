"""Moduł implementuje algorytm base dla dystrybucji licencji.

Wejście zwykle obejmuje obiekt `networkx.Graph` oraz konfiguracje licencji (`LicenseType`, `LicenseGroup`).
"""

from abc import ABC, abstractmethod
from typing import List
import networkx as nx
from src.core import LicenseType, Solution


class BaseAlgorithm(ABC):
    @abstractmethod
    def solve(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        pass
