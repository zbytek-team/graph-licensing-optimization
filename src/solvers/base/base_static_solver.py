import time

import networkx as nx

from src.utils.logger import get_logger

from .base_solver import BaseSolver, SolverOutput

logger = get_logger(__name__)


class BaseStaticSolver(BaseSolver):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int):
        super().__init__(individual_cost, group_cost, group_size)

    def run(self, graph: nx.Graph) -> SolverOutput:
        execution_time = time.time()
        result = self._solve(graph)
        execution_time = time.time() - execution_time

        self._verify(graph, result)

        total_cost = self.calculate_total_cost(result)

        return {"assignment": result, "total_cost": total_cost, "execution_time": execution_time}
