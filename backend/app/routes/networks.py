from fastapi import APIRouter
import networkx as nx

router = APIRouter(prefix="/networks", tags=["Networks"])


def generate_graph(graph, scale, seed=42):
    """Generates nodes and edges for a given NetworkX graph."""
    node_style = {
        "width": 50,
        "height": 50,
        "borderRadius": 50,
        "fontSize": 18,
        "backgroundColor": "#333333",
    }

    positions = nx.spring_layout(graph, scale=scale, seed=seed)

    nodes = [
        {
            "id": str(node),
            "data": {"label": str(i)},
            "position": {"x": pos[0], "y": pos[1]},
            "style": node_style,
        }
        for i, (node, pos) in enumerate(positions.items())
    ]
    edges = [
        {"id": f"e{source}-{target}", "source": str(source), "target": str(target)}
        for source, target in graph.edges
    ]

    return {"nodes": nodes, "edges": edges}


@router.get("/")
def get_networks():
    return [
        "Basic Graph",
        "Star Graph",
        "Florentine Families Graph",
        "Les Miserables Graph",
    ]


@router.get("/basic-graph")
def get_basic_graph():
    graph = nx.erdos_renyi_graph(10, 0.3, seed=42)
    return generate_graph(graph, scale=300)


@router.get("/star-graph")
def get_star_graph():
    graph = nx.star_graph(13)
    return generate_graph(graph, scale=200)


@router.get("/florentine-families-graph")
def get_florentine_families_graph():
    graph = nx.florentine_families_graph()
    return generate_graph(graph, scale=500)


@router.get("/les-miserables")
def get_les_miserables_graph():
    graph = nx.les_miserables_graph()
    return generate_graph(graph, scale=1000)
