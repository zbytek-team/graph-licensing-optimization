from fastapi import APIRouter, HTTPException
from app.models.graph import GraphRequest, GraphResponse, GraphType, Graph
import networkx as nx

router = APIRouter()


@router.post("/")
async def graph(request: GraphRequest) -> GraphResponse:
    graph_type = request.graph_type

    match graph_type:
        case GraphType.WATTS_STROGATZ:
            graph = nx.watts_strogatz_graph(100, 4, 0.1)
        case GraphType.BARABASI_ALBERT:
            graph = nx.barabasi_albert_graph(100, 4)
        case GraphType.ERDOS_RENYI:
            graph = nx.gnp_random_graph(100, 0.1)
        case _:
            raise HTTPException(status_code=400, detail="Invalid graph")

    graph = Graph(nodes=list(graph.nodes), edges=list(graph.edges))

    return GraphResponse(graph=graph)
