from fastapi import APIRouter, HTTPException
from app.models.graph import GraphRequest, GraphResponse, GraphType, Graph
import networkx as nx
import random

router = APIRouter()

def generate_sparse_community_graph(N=300, p_ws=0.02, extra_links=3, num_subgroups=15
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
    match request.graph_type:
        case GraphType.SMALL_WORLD:
            G = nx.watts_strogatz_graph(100, k=6, p=0.1)
        case GraphType.SCALE_FREE:
            G = nx.barabasi_albert_graph(100, m=3)
        case GraphType.RANDOM:
            G = nx.erdos_renyi_graph(100, p=0.05)
        case GraphType.SOCIAL_CIRCLE:
            G = nx.relaxed_caveman_graph(10, 10, p=0.3)
        case GraphType.COMMUNITY:
            G = nx.connected_caveman_graph(5, 20)
        case GraphType.SPARSE_COMMUNITY:
            G = generate_sparse_community_graph(N=300, p_ws=0.02, extra_links=3, num_subgroups=15)
        case GraphType.BIPARTITE:
            G = nx.bipartite.random_graph(50, 50, p=0.1)
        case GraphType.COMPLETE:
            G = nx.complete_graph(20)
        case GraphType.TREE:
            G = nx.balanced_tree(2, 5)
        case GraphType.GRID:
            G = nx.grid_2d_graph(10, 10)
        case GraphType.LATTICE:
            G = nx.hexagonal_lattice_graph(5, 5)
        case _:
            raise HTTPException(status_code=400, detail="Invalid graph")

    graph = Graph(nodes=list(G.nodes), edges=list(G.edges))

    return GraphResponse(graph=graph)
