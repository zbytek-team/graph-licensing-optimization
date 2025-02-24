from app.models.solver_models import SolverType
from app.solvers.greedy.greedy_solver import greedy_solver


def get_solver(solver_type: SolverType):
    match solver_type:
        case SolverType.GREEDY:
            return greedy_solver
        case _:
            return None
