from abc import ABC, abstractmethod
from typing import TypedDict, final

import networkx as nx

from src.logger import get_logger

logger = get_logger(__name__)


class AssignmentResult(TypedDict):
    individual: set[int]
    group: dict[int, set[int]]


class SolverOutput(TypedDict):
    assignment: AssignmentResult
    total_cost: float


class Solver(ABC):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int):
        logger.info(
            f"Initializing solver {self.__class__.__name__} with: "
            f"individual_cost={individual_cost}, group_cost={group_cost}, group_size={group_size}."
        )
        self.individual_cost = individual_cost
        self.group_cost = group_cost
        self.group_size = group_size

    @abstractmethod
    def _solve(self, graph: nx.Graph) -> AssignmentResult:
        pass

    @abstractmethod
    def run(self, graph: nx.Graph) -> SolverOutput:
        pass

    @final
    def _verify(self, graph: nx.Graph, result: AssignmentResult) -> None:
        all_nodes = set(graph.nodes)
        assigned_nodes = set()
        covered_nodes = set(result["individual"])

        for holder, members in result["group"].items():
            if len(members) < 2 or len(members) > self.group_size:
                raise ValueError(
                    f"Group led by {holder} has {len(members)} members, expected between 2 and {self.group_size}."
                )
            if holder not in members:
                raise ValueError(f"License holder {holder} is not in its own group.")

            for member in members:
                if member in assigned_nodes:
                    raise ValueError(f"Node {member} appears in multiple groups.")
                assigned_nodes.add(member)

                if member != holder and member not in graph.neighbors(holder):
                    raise ValueError(f"Node {member} is in group of {holder} but is not connected.")

            covered_nodes.update(members)

        if covered_nodes != all_nodes:
            missing = all_nodes - covered_nodes
            raise ValueError(f"Not all nodes are covered: {missing}")

    @final
    def calculate_total_cost(self, result: AssignmentResult) -> float:
        return len(result["individual"]) * self.individual_cost + len(result["group"]) * self.group_cost
