from enum import Enum
from typing import Any

from app.models.graph_models import Graph
from pydantic import BaseModel


class SolverType(str, Enum):
    GREEDY = "greedy"
    ANTS = "ants"


class LicenseParams(BaseModel):
    individual_cost: float
    group_cost: float
    group_size: int


class SolverRequest(BaseModel):
    graph: Graph
    license_params: LicenseParams
    solver_type: SolverType
    solver_params: dict[str, Any]


class LicenseHolder(BaseModel):
    holder: int
    covered_nodes: list[int]


class SolverResponse(BaseModel):
    individual: list[LicenseHolder]
    group: list[LicenseHolder]
