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
    color: str = "#000000"  # Default black color


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

    def is_valid(self, graph: nx.Graph, all_nodes: Set[Any]) -> bool:
        # check if all nodes are covered
        if self.covered_nodes != all_nodes:
            missing_nodes = all_nodes - self.covered_nodes
            extra_nodes = self.covered_nodes - all_nodes

            error_msg = "Node coverage mismatch."
            if missing_nodes:
                error_msg += f" Missing nodes: {missing_nodes}."
            if extra_nodes:
                error_msg += f" Extra nodes: {extra_nodes}."

            raise ValueError(error_msg)

        for group in self.groups:
            owner_neighbors = set(graph.neighbors(group.owner)) | {group.owner}

            # check if group is of correct size
            if not (group.license_type.min_capacity <= group.size <= group.license_type.max_capacity):
                raise ValueError(
                    f"Group with owner {group.owner} has size {group.size}, "
                    f"but license type '{group.license_type.name}' requires "
                    f"capacity between {group.license_type.min_capacity} and {group.license_type.max_capacity}"
                )

            # check for valid group ownership - members must be neighbors of the owner
            invalid_members = group.additional_members - owner_neighbors
            if invalid_members:
                raise ValueError(f"Group with owner {group.owner} contains invalid members {invalid_members} that are not neighbors of the owner")

        # check for overlapping groups
        all_covered = set()
        for group in self.groups:
            overlapping_members = all_covered & group.all_members
            if overlapping_members:
                raise ValueError(f"Group with owner {group.owner} has overlapping members {overlapping_members} with previous groups")
            all_covered.update(group.all_members)

        return True


class Algorithm(ABC):
    @abstractmethod
    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class GraphGenerator(ABC):
    @abstractmethod
    def generate(self, n_nodes: int, **kwargs: Any) -> nx.Graph:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
