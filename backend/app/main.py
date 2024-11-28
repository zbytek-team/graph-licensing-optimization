from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

from algorithms.greedy_solver import greedy_license_distribution
from algorithms.greedy_advanced_solver import greedy_advanced_license_distribution
from config.config import LicenseType


class Node(BaseModel):
    data: dict
    id: str
    measured: dict
    position: dict
    style: dict


class Edge(BaseModel):
    id: str
    source: str
    target: str


class Graph(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/algorithms")
def get_available_algorithms():
    return ["Greedy", "Greedy Advanced"]


@app.get("/networks")
def get_networks():
    return ["Les Miserables", "Network 2"]


@app.get("/networks/les-miserables")
def get_network():
    nodeStyle = {
        "width": 50,
        "height": 50,
        "borderRadius": 50,
        "fontSize": 18,
        "backgroundColor": "#333333",
    }

    graph = nx.les_miserables_graph()

    initNodes = [
        {
            "id": str(node),
            "data": {"label": str(node)},
            "position": {"x": 0, "y": 0},
            "style": nodeStyle,
        }
        for node in graph.nodes
    ]

    initEdges = [
        {"id": f"e{source}-{target}", "source": str(source), "target": str(target)}
        for source, target in graph.edges
    ]

    return {
        "nodes": initNodes,
        "edges": initEdges,
    }


@app.post("/run-simulation")
def run_simulation(
    graph: Graph,
    max_group_size: int,
    prices: tuple[float, float],
    selected_algorithm: str,
):
    nx_graph = nx.Graph()

    for node in graph.nodes:
        nx_graph.add_node(node.id, **node.data)

    for edge in graph.edges:
        nx_graph.add_edge(edge.source, edge.target)

    match selected_algorithm:
        case "Greedy":
            licenses = greedy_license_distribution(nx_graph, prices, max_group_size)
        case "Greedy Advanced":
            licenses = greedy_advanced_license_distribution(
                nx_graph, prices, max_group_size
            )
        case _:
            raise ValueError("Invalid algorithm selected")

    response = {
        "individual": licenses[LicenseType.INDIVIDUAL],
        "group_owner": licenses[LicenseType.GROUP_OWNER],
        "group_member": licenses[LicenseType.GROUP_MEMBER],
    }

    return response
