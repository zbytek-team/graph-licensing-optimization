from fastapi import APIRouter, HTTPException
from app.models.solver_models import SolverType, SolverRequest, SolverResponse
from app.solvers.solver_factory import get_solver
import networkx as nx

router = APIRouter()


@router.get("/", response_model=list[str])
def get_solvers() -> list[str]:
    return [solver.value for solver in SolverType]


@router.post("/solve", response_model=SolverResponse)
def solve(request: SolverRequest) -> SolverResponse:
    graph = request.graph
    license_params = request.license_params
    solver_type = request.solver_type
    solver_params = request.solver_params

    solver = get_solver(solver_type)

    if solver is None:
        raise HTTPException(status_code=400, detail="Invalid solver type")

    G = nx.Graph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges)

    return solver(G, license_params, solver_params)
