from fastapi import FastAPI, HTTPException
import uvicorn
from src.models import Graph, SimulationRequest, SimulationResponse
from src.solvers import greedy_solver
import time
import networkx as nx
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/simulate")
async def simulate(request: SimulationRequest) -> SimulationResponse:
    # G = nx.watts_strogatz_graph(10000, 4, 0.1)
    # graph = Graph(nodes=list(G.nodes), edges=list(G.edges))

    license_types = request.license_types
    match request.algorithm:
        case "greedy":
            start = time.time()
            licenses = greedy_solver(request.graph, license_types)
            end = time.time()
            print(end - start)
        case _:
            raise HTTPException(status_code=400, detail="Invalid algorithm")

    return SimulationResponse(root=licenses)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
