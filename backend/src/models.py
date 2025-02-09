from pydantic import BaseModel, RootModel, Field 

example_algorithm = "greedy"
example_license_types = [
    {"name": "individual", "limit": 1, "cost": 1.0},
    {"name": "group", "limit": 6, "cost": 2.5},
    {"name": "mega", "limit": 10, "cost": 4.0},
]
example_graph = {"nodes": [0, 1, 2, 3, 4], "edges": [(0, 1), (0, 2), (0, 3), (1, 3), (1, 4)]}


class LicenseType(BaseModel):
    name: str
    limit: int
    cost: float


class Graph(BaseModel):
    nodes: list[int]
    edges: list[tuple[int, int]]

    def neighbors(self, node: int) -> list[int]:
        result: list[int] = []
        for edge in self.edges:
            if node in edge:
                neighbor = edge[0] if edge[1] == node else edge[1]
                result.append(neighbor)
        return result


class SimulationRequest(BaseModel):
    algorithm: str = Field(..., examples=[example_algorithm])
    license_types: list[LicenseType] = Field(..., examples=[example_license_types])
    graph: Graph = Field(..., examples=[example_graph])


class LicenseAssignmentItem(BaseModel):
    owner: int
    users: list[int]


class LicenseAssignment(BaseModel):
    license_type: str
    item: list[LicenseAssignmentItem]


class SimulationResponse(RootModel):
    root: list[LicenseAssignment]
