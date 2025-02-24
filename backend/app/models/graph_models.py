from enum import Enum
from typing import Any

from pydantic import BaseModel, RootModel


class GraphType(str, Enum):
    CLUSTERED = "clustered"
    TREE = "tree"
    RANDOM = "random"
    SMALL_WORLD = "small_world"


class Graph(BaseModel):
    nodes: list[int]
    edges: list[list[int]]


class GraphRequest(BaseModel):
    graph_type: GraphType
    params: dict[str, Any]


class GraphResponse(RootModel):
    root: Graph
