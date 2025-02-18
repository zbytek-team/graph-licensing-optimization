from enum import Enum

from app.models.graph import Graph, Node
from pydantic import BaseModel


class SolverType(str, Enum):
    GREEDY = "greedy"


class License(BaseModel):
    name: str
    cost: float
    limit: int


class SolveRequest(BaseModel):
    graph: Graph
    licenses: list[License]
    solver: SolverType


class LicenseAssignment(BaseModel):
    license_holder: Node
    covered_nodes: list[Node]


Assignments = dict[str, list[LicenseAssignment]]


class SolveResponse(BaseModel):
    assignments: Assignments
    licenses: list[License]
