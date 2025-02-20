from fastapi import APIRouter, HTTPException
from app.models.solve import SolveRequest, SolveResponse, SolverType
from app.solvers.greedy.greedy_solver import greedy_solver
from app.solvers.ant.ant_solver import ant_solver
router = APIRouter()


@router.post("/")
async def solve(request: SolveRequest) -> SolveResponse:
    licenses = request.licenses
    solver = request.solver
    graph = request.graph

    match solver:
        case SolverType.GREEDY:
            assignments = greedy_solver(graph, licenses)
        case SolverType.ANT:
            assignments = ant_solver(graph, licenses, ants=10, iterations=100)
        case _:
            raise HTTPException(status_code=400, detail="Invalid solver")

    return SolveResponse(assignments=assignments, licenses=licenses)
