from abc import ABC, abstractmethod
from typing import TypedDict, final

import networkx as nx

from src.logger import get_logger

logger = get_logger(__name__)


class SolverResult(TypedDict):
    individual: set[int]
    group: dict[int, set[int]]


class Solver(ABC):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int):
        logger.info(
            f"Initializing solver {self.__class__.__name__} with: individual_cost={individual_cost}, group_cost={group_cost}, group_size={group_size}."
        )
        self.individual_cost = individual_cost
        self.group_cost = group_cost
        self.group_size = group_size

    @abstractmethod
    def _solve(self, graph: nx.Graph) -> SolverResult:
        pass

    @final
    def run(self, graph: nx.Graph) -> tuple[SolverResult, float]:
        logger.info(f"Running solver {self.__class__.__name__}...")
        result = self._solve(graph)
        logger.info("Running result verification...")
        self.verify(graph, result)
        logger.info("Calculating total cost...")
        total_cost = self.calculate_total_cost(result)

        return (result, total_cost)

    @final
    def verify(self, graph: nx.Graph, result: SolverResult) -> None:
        all_nodes = set(graph.nodes)
        covered_nodes = result["individual"].copy()

        for holder, members in result["group"].items():
            if len(members) < 2 or len(members) > self.group_size:
                raise ValueError(
                    f"Group led by {holder} has {len(members)} members, expected between 2 and {self.group_size}."
                )
            if holder not in members:
                raise ValueError(f"License holder {holder} is not in its own group.")

            covered_nodes.update(members)

        if covered_nodes < all_nodes:
            raise ValueError("Not all nodes are covered.")

        all_assigned_nodes = [node for nodes in result["group"].values() for node in nodes]
        all_assigned_nodes.extend(result["individual"])

        if len(all_assigned_nodes) != len(covered_nodes):
            raise ValueError(
                "Duplicate nodes detected in individual and group assignments."
            )

    def calculate_total_cost(self, result: SolverResult) -> float:
        total_cost = 0.0
        total_cost += len(result["individual"]) * self.individual_cost
        total_cost += len(result["group"]) * self.group_cost
        return total_cost
