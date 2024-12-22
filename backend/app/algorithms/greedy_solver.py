import networkx as nx
import math
from config.config import LicenseType
from utils.graph_utils import key


def greedy_license_distribution(
    graph: nx.Graph, prices: tuple[float, float], max_group_size
):
    licenses = {
        LicenseType.INDIVIDUAL: [],
        LicenseType.GROUP_OWNER: [],
        LicenseType.GROUP_MEMBER: [],
    }
    covered_nodes = set()

    nodes_sorted = sorted(graph.nodes, key=lambda n: key(n, graph), reverse=True)

    for node in nodes_sorted:
        if node in covered_nodes:
            continue

        neighbors = [n for n in graph.neighbors(node) if n not in covered_nodes]

        if len(neighbors) + 1 <= math.ceil(prices[1] / prices[0]):
            licenses[LicenseType.INDIVIDUAL].append(node)
            covered_nodes.add(node)
        else:
            licenses[LicenseType.GROUP_OWNER].append(node)
            covered_nodes.add(node)

            group_members = neighbors[: max_group_size - 1]
            licenses[LicenseType.GROUP_MEMBER].extend(group_members)
            licenses[LicenseType.GROUP_MEMBER].append(node)
            covered_nodes.update(group_members)

    return licenses
