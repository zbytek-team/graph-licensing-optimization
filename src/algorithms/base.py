from typing import TypedDict
from abc import ABC, abstractmethod
import networkx as nx


class LicenseGroup(TypedDict):
    license_holder: int
    members: list[int]


class Solution(TypedDict):
    singles: list[int]
    groups: list[LicenseGroup]


class BaseSolver(ABC):
    @abstractmethod
    def solve(self, graph: nx.Graph, c_single: float, c_group: float, group_size: int) -> Solution:
        pass
