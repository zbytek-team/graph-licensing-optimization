from pydantic import BaseModel


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
