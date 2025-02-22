from fastapi import APIRouter, HTTPException
from app.models.solve import SolveRequest, SolveResponse, SolverType, AntSolverType
from app.solvers.greedy.greedy_solver import greedy_solver
from app.solvers.ant.ant_solver import ant_solver
from app.solvers.ant.ant_solver_multiprocessing import ant_solver_multiprocessing

router = APIRouter()


@router.post("/")
async def solve(request: SolveRequest) -> SolveResponse:
    licenses = request.licenses
    solver = request.solver
    graph = request.graph

    match solver:
        case SolverType.GREEDY:
            assignments = greedy_solver(graph, licenses)
        case SolverType.ANTS:
            assignments = ant_solver(
                graph,
                licenses,
                ants=8,
                solution_type=AntSolverType.PATH_AND_LICENSES,
            )
        case SolverType.ANTS_MULTIPROCESSING:
            assignments = ant_solver_multiprocessing(
                graph,
                licenses,
                ants=16,
                solution_type=AntSolverType.PATH_AND_LICENSES,
            )
        case _:
            raise HTTPException(status_code=400, detail="Invalid solver")

    return SolveResponse(assignments=assignments, licenses=licenses)
