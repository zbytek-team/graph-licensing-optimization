from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, Generic, Hashable, Sequence, Tuple, TypeVar, Set
import networkx as nx

N = TypeVar("N", bound=Hashable)


@dataclass(frozen=True, slots=True)
class LicenseType:
    name: str
    cost: float
    min_capacity: int
    max_capacity: int
    color: str = "#000000"

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("cost must be >= 0")
        if self.min_capacity < 1:
            raise ValueError("min_capacity must be >= 1")
        if self.max_capacity < self.min_capacity:
            raise ValueError("max_capacity must be >= min_capacity")


@dataclass(frozen=True, slots=True)
class LicenseGroup(Generic[N]):
    license_type: LicenseType
    owner: N
    additional_members: FrozenSet[N] = frozenset()

    @property
    def all_members(self) -> FrozenSet[N]:
        if self.owner in self.additional_members:
            return self.additional_members
        return frozenset({self.owner, *self.additional_members})

    @property
    def size(self) -> int:
        return len(self.all_members)

    def __post_init__(self) -> None:
        s = self.size
        if not (self.license_type.min_capacity <= s <= self.license_type.max_capacity):
            raise ValueError(f"group size {s} violates [{self.license_type.min_capacity}, {self.license_type.max_capacity}] for {self.license_type.name}")


@dataclass(slots=True)
class Solution(Generic[N]):
    groups: Tuple[LicenseGroup[N], ...] = ()

    @property
    def total_cost(self) -> float:
        return sum(g.license_type.cost for g in self.groups)

    @property
    def covered_nodes(self) -> Set[N]:
        covered: Set[N] = set()
        for g in self.groups:
            covered.update(g.all_members)
        return covered


class Algorithm(ABC, Generic[N]):
    @abstractmethod
    def solve(self, graph: nx.Graph, license_types: Sequence[LicenseType], **kwargs) -> Solution[N]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
