import time

import networkx as nx

from src.utils.logger import get_logger

from .base_solver import BaseSolver, SolverOutput

logger = get_logger(__name__)


class BaseDynamicSolver(BaseSolver):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int):
        super().__init__(individual_cost, group_cost, group_size)

    def run(self, graph: nx.Graph, iterations_num: int = 10) -> SolverOutput:
        logger.info(f"Running solver {self.__class__.__name__}...")

        total_cost = None
        result = None

        time_taken = time.time()

        for iteration in range(iterations_num):
            logger.info(f"Running iteration {iteration + 1}/{iterations_num}...")
            result = self._solve(graph)
            logger.info("Running result verification...")
            self._verify(graph, result)
            logger.info("Calculating total cost...")
            total_cost = self.calculate_total_cost(result)

        time_taken = time.time() - time_taken

        if total_cost is None:
            raise ValueError("Total cost is None.")

        if result is None:
            raise ValueError("Result is None.")

        return {"assignment": result, "total_cost": total_cost, "time_taken": time_taken}
