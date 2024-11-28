from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

# Import your function and LicenseType
from algorithms.greedy_solver import greedy_license_distribution
from config.config import LicenseType


# Define a Node model
class Node(BaseModel):
    data: dict
    id: str
    measured: dict
    position: dict
    style: dict


# Define an Edge model
class Edge(BaseModel):
    id: str
    source: str
    target: str


# Extend the Graph model to include edges
class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/run-simulation")
def run_simulation(graph: Graph):
    # Step 1: Build a NetworkX graph from the input data
    nx_graph = nx.Graph()

    for node in graph.nodes:
        nx_graph.add_node(node.id, **node.data)

    for edge in graph.edges:
        nx_graph.add_edge(edge.source, edge.target)

    # Step 2: Run the greedy solver
    licenses = greedy_license_distribution(nx_graph, max_group_size=6)

    # Step 3: Format the response
    response = {
        "individual": licenses[LicenseType.INDIVIDUAL],
        "group_owner": licenses[LicenseType.GROUP_OWNER],
        "group_member": licenses[LicenseType.GROUP_MEMBER],
    }

    return response
