from fastapi import APIRouter
from models.graph import Graph
from config.config import LicenseType
import networkx as nx

# Import naszych solverów
from algorithms.greedy_solver import greedy_license_distribution
from algorithms.greedy_multi_group_solver import greedy_multi_group_license_distribution
from algorithms.ilp_solver import ilp_license_distribution
from algorithms.genetic_solver import genetic_license_distribution

router = APIRouter(prefix="/run-simulation", tags=["Simulation"])


@router.post("/")
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
        case "Greedy Multi-Group":
            licenses = greedy_multi_group_license_distribution(
                nx_graph, prices, max_group_size
            )
        case "ILP":
            # Załóżmy prices = (C1, Ck)
            licenses = ilp_license_distribution(
                nx_graph, prices[0], prices[1], max_group_size
            )
        case "GA":
            licenses = genetic_license_distribution(nx_graph, prices, max_group_size)
        case _:
            raise ValueError("Invalid algorithm selected")

    response = {
        "individual": licenses[LicenseType.INDIVIDUAL],
        "group_owner": licenses[LicenseType.GROUP_OWNER],
        "group_member": licenses[LicenseType.GROUP_MEMBER],
    }
    return response
