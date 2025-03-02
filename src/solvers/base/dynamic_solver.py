import networkx as nx

from src.logger import get_logger

from .solver import Solver, SolverOutput

logger = get_logger(__name__)


class DynamicSolver(Solver):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int):
        super().__init__(individual_cost, group_cost, group_size)

    def run(self, graph: nx.Graph, iterations_num: int = 10) -> SolverOutput:
        logger.info(f"Running solver {self.__class__.__name__}...")

        total_cost = None
        result = None

        for iteration in range(iterations_num):
            logger.info(f"Running iteration {iteration + 1}/{iterations_num}...")
            result = self._solve(graph)
            logger.info("Running result verification...")
            self._verify(graph, result)
            logger.info("Calculating total cost...")
            total_cost = self.calculate_total_cost(result)

        if total_cost is None:
            raise ValueError("Total cost is None.")

        if result is None:
            raise ValueError("Result is None.")

        return {"assignment": result, "total_cost": total_cost}
