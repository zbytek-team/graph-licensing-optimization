from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Set
import networkx as nx


@dataclass(frozen=True)
class LicenseType:
    name: str
    cost: float
    min_capacity: int
    max_capacity: int
    color: str = "#000000"


@dataclass(frozen=True)
class LicenseGroup:
    license_type: LicenseType
    owner: Any
    additional_members: Set[Any]

    @property
    def all_members(self) -> Set[Any]:
        return self.additional_members | {self.owner}

    @property
    def size(self) -> int:
        return len(self.all_members)


@dataclass
class Solution:
    groups: List[LicenseGroup]
    total_cost: float
    covered_nodes: Set[Any]


class Algorithm(ABC):
    @abstractmethod
    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
