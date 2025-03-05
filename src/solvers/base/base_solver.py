from abc import ABC, abstractmethod
from typing import TypedDict, final

import networkx as nx

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Assignment(TypedDict):
    individual: set[int]
    group: dict[int, set[int]]


class SolverOutput(TypedDict):
    assignment: Assignment
    total_cost: float
    execution_time: float


class BaseSolver(ABC):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int):
        self.individual_cost = individual_cost
        self.group_cost = group_cost
        self.group_size = group_size

    @abstractmethod
    def _solve(self, graph: nx.Graph) -> Assignment:
        pass

    @abstractmethod
    def run(self, graph: nx.Graph) -> SolverOutput:
        pass

    @final
    def _verify(self, graph: nx.Graph, result: Assignment) -> None:
        all_nodes = set(graph.nodes)
        assigned_nodes = set()
        covered_nodes = set(result["individual"])

        for holder, members in result["group"].items():
            if len(members) < 2 or len(members) > self.group_size:
                raise ValueError(
                    f"{result}\n\nGroup led by {holder} has {len(members)} members, expected between 2 and {self.group_size}."
                )

            if holder not in members:
                raise ValueError(f"{result}\n\nLicense holder {holder} is not in its own group.")

            for member in members:
                if member in assigned_nodes:
                    raise ValueError(f"{result}\n\nNode {member} appears in multiple groups.")
                assigned_nodes.add(member)

                if member != holder and member not in graph.neighbors(holder):
                    raise ValueError(f"{result}\n\nNode {member} is in group of {holder} but is not connected.")

            covered_nodes.update(members)

        if covered_nodes != all_nodes:
            missing = all_nodes - covered_nodes
            raise ValueError(f"{result}\n\nNot all nodes are covered: {missing}")

    @final
    def calculate_total_cost(self, result: Assignment) -> float:
        return len(result["individual"]) * self.individual_cost + len(result["group"]) * self.group_cost

    def __str__(self) -> str:
        return self.__class__.__name__.strip("Solver")
