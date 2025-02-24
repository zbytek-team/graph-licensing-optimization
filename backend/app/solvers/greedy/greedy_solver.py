import networkx as nx
from app.models.solver_models import LicenseParams, SolverResponse, LicenseHolder
from typing import Any


def greedy_solver(G: nx.Graph, license_params: LicenseParams, _: dict[str, Any]) -> SolverResponse:
    individual_cost = license_params.individual_cost
    group_cost = license_params.group_cost
    group_size = license_params.group_size

    uncovered = set(G.nodes)
    individual_list: list[LicenseHolder] = []
    group_list: list[LicenseHolder] = []

    while uncovered:
        best_saving = 0.0
        best_vertex = None
        best_covered_set = set()

        for vertex in uncovered:
            neighbors = set(G.neighbors(vertex))

            available_neighbors = list(neighbors)
            num_neighbors_to_cover = min(len(available_neighbors), group_size - 1)
            potential_coverage = 1 + num_neighbors_to_cover

            saving = potential_coverage * individual_cost - group_cost

            if saving > best_saving:
                best_saving = saving
                best_vertex = vertex
                best_covered_set = {vertex} | set(available_neighbors[:num_neighbors_to_cover])

        if best_vertex is None or best_saving <= 0:
            break

        group_license_holder = LicenseHolder(holder=best_vertex, covered_nodes=list(best_covered_set))

        group_list.append(group_license_holder)
        uncovered -= best_covered_set

    for vertex in uncovered:
        individual_list.append(LicenseHolder(holder=vertex, covered_nodes=[vertex]))

    return SolverResponse(individual=individual_list, group=group_list)
