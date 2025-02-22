from fastapi import APIRouter, HTTPException
from app.models.graph import GraphRequest, GraphResponse, GraphType, Graph
import networkx as nx
import random

router = APIRouter()


def generate_social_network(
    N=300, m=1, K=4, p_ws=0.02, extra_links=3, num_subgroups=15
):
    G = nx.Graph()

    for _ in range(num_subgroups):
        subgroup_size = random.randint(5, 10)
        subgraph = nx.watts_strogatz_graph(
            subgroup_size, max(2, subgroup_size // 4), p_ws
        )
        mapping = {i: random.randint(0, N - 1) for i in range(subgroup_size)}
        H = nx.relabel_nodes(subgraph, mapping)
        G = nx.compose(G, H)

    for _ in range(extra_links):
        a, b = random.sample(range(N), 2)
        G.add_edge(a, b)

    return G


@router.post("/")
async def graph(request: GraphRequest) -> GraphResponse:
    graph_type = request.graph_type

    match graph_type:
        case GraphType.WATTS_STROGATZ:
            graph = generate_social_network()
        case GraphType.BARABASI_ALBERT:
            graph = nx.barabasi_albert_graph(100, 4)
        case GraphType.ERDOS_RENYI:
            graph = nx.gnp_random_graph(100, 0.1)
        case _:
            raise HTTPException(status_code=400, detail="Invalid graph")

    graph = Graph(nodes=list(graph.nodes), edges=list(graph.edges))

    return GraphResponse(graph=graph)
