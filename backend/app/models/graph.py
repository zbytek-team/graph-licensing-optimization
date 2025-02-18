from pydantic import BaseModel
from enum import Enum


class GraphType(str, Enum):
    WATTS_STROGATZ = "watts_strogatz"
    BARABASI_ALBERT = "barabasi_albert"
    ERDOS_RENYI = "erdos_renyi"


Node = int
Edge = tuple[Node, Node]


class Graph(BaseModel):
    nodes: list[Node]
    edges: list[Edge]

    def get_neighbors(self, node: Node) -> list[Node]:
        neighbors: list[Node] = []
        for edge in self.edges:
            if node in edge:
                neighbor = edge[0] if edge[1] == node else edge[1]
                neighbors.append(neighbor)
        return neighbors


class GraphRequest(BaseModel):
    graph_type: GraphType


class GraphResponse(BaseModel):
    graph: Graph
