import networkx as nx

from src.logger import get_logger

from .solver import Solver, SolverOutput

logger = get_logger(__name__)


class StaticSolver(Solver):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int):
        super().__init__(individual_cost, group_cost, group_size)

    def run(self, graph: nx.Graph) -> SolverOutput:
        logger.info(f"Running solver {self.__class__.__name__}...")

        result = self._solve(graph)

        logger.info("Running result verification...")
        self._verify(graph, result)

        logger.info("Calculating total cost...")
        total_cost = self.calculate_total_cost(result)

        return {"assignment": result, "total_cost": total_cost}
