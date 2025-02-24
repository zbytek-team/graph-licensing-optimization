from fastapi import APIRouter, HTTPException
import networkx as nx
from app.models.graph_models import GraphRequest, GraphResponse, GraphType, Graph
from app.utils.graph_generators import generate_clustered_graph

router = APIRouter()


@router.get("/", response_model=list[str])
def get_graphs() -> list[str]:
    return [graph.value for graph in GraphType]


@router.post("/generate", response_model=GraphResponse)
def generate_graph(request: GraphRequest) -> GraphResponse:
    graph_type = request.graph_type
    params = request.params

    match graph_type:
        case GraphType.CLUSTERED:
            N: int = params.get("N", 300)
            p_ws: float = params.get("p_ws", 0.02)
            extra_links: int = params.get("extra_links", 3)
            num_subgroups: int = params.get("num_subgroups", 15)

            graph = generate_clustered_graph(N, p_ws, extra_links, num_subgroups)
        case GraphType.SMALL_WORLD:
            n: int = params.get("n", 30)
            k: int = params.get("k", 4)
            p: float = params.get("p", 0.1)

            graph = nx.watts_strogatz_graph(n, k, p)
        case GraphType.RANDOM:
            n: int = params.get("n", 30)
            p: float = params.get("p", 0.1)

            graph = nx.fast_gnp_random_graph(n, p)
        case GraphType.TREE:
            n: int = params.get("n", 15)

            graph = nx.balanced_tree(r=2, h=int((n - 1).bit_length()))
        case _:
            raise HTTPException(status_code=400, detail="Invalid graph type")

    return GraphResponse.model_validate(
        Graph(
            nodes=list(graph.nodes()),
            edges=[list(graph.neighbors(node)) for node in graph.nodes()],
        )
    )
