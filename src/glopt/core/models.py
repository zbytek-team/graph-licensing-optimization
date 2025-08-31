from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

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
            msg = "cost must be >= 0"
            raise ValueError(msg)
        if self.min_capacity < 1:
            msg = "min_capacity must be >= 1"
            raise ValueError(msg)
        if self.max_capacity < self.min_capacity:
            msg = "max_capacity must be >= min_capacity"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LicenseGroup[N: Hashable]:
    license_type: LicenseType
    owner: N
    additional_members: frozenset[N] = frozenset()

    @property
    def all_members(self) -> frozenset[N]:
        if self.owner in self.additional_members:
            return self.additional_members
        return frozenset({self.owner, *self.additional_members})

    @property
    def size(self) -> int:
        return len(self.all_members)

    def __post_init__(self) -> None:
        s = self.size
        if not (self.license_type.min_capacity <= s <= self.license_type.max_capacity):
            msg = f"group size {s} violates [{self.license_type.min_capacity}, {self.license_type.max_capacity}] for {self.license_type.name}"
            raise ValueError(
                msg
            )


@dataclass(slots=True)
class Solution[N: Hashable]:
    groups: tuple[LicenseGroup[N], ...] = ()

    @property
    def total_cost(self) -> float:
        return sum(g.license_type.cost for g in self.groups)




class Algorithm[N: Hashable](ABC):
    @abstractmethod
    def solve(self, graph: nx.Graph, license_types: Sequence[LicenseType], **kwargs: Any) -> Solution[N]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
