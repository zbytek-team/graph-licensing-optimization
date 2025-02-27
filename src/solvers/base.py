from abc import ABC, abstractmethod
import networkx as nx
from typing import TypedDict


class SolverResult(TypedDict):
    individual: set[int]
    group: dict[int, set[int]]


class BaseSolver(ABC):
    graph: nx.Graph
    individual_cost: float
    group_cost: float
    group_size: int
    result: SolverResult

    def __new__(cls, graph: nx.Graph, individual_cost: float, group_cost: float, group_size: int):
        instance = super().__new__(cls)
        instance.graph = graph
        instance.individual_cost = individual_cost
        instance.group_cost = group_cost
        instance.group_size = group_size

        instance.result = instance.run()
        instance.verify()

        return instance.result

    @abstractmethod
    def run(self) -> SolverResult:
        pass

    def verify(self):
        all_nodes = set(self.graph.nodes)
        covered_nodes = self.result["individual"].copy()

        for holder, members in self.result["group"].items():
            if len(members) < 2 or len(members) > self.group_size:
                raise ValueError(
                    f"Group led by {holder} has {len(members)} members, expected between 2 and {self.group_size}."
                )
            if holder not in members:
                raise ValueError(f"License holder {holder} is not in its own group.")

            covered_nodes.update(members)

        if covered_nodes != all_nodes:
            raise ValueError("Not all nodes are covered.")

        all_assigned_nodes = self.result["individual"].union(*self.result["group"].values())

        if len(all_assigned_nodes) != len(covered_nodes):
            raise ValueError("Duplicate nodes detected in individual and group assignments.")
