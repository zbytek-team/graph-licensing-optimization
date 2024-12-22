import networkx as nx
import math
from config.config import LicenseType
from utils.graph_utils import key


def greedy_multi_group_license_distribution(
    graph: nx.Graph, prices: tuple[float, float], max_group_size: int
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
            group_capacity = max_group_size

            if node not in licenses[LicenseType.GROUP_MEMBER]:
                licenses[LicenseType.GROUP_MEMBER].append(node)
                group_capacity -= 1

            group_members = neighbors[:group_capacity]
            licenses[LicenseType.GROUP_MEMBER].extend(group_members)
            covered_nodes.update(group_members)
            covered_nodes.add(node)

            neighbors = neighbors[group_capacity:]

            while neighbors:
                if len(neighbors) + 1 <= math.ceil(prices[1] / prices[0]):
                    licenses[LicenseType.INDIVIDUAL].extend(neighbors)
                    covered_nodes.update(neighbors)
                    break

                group_capacity = max_group_size
                licenses[LicenseType.GROUP_OWNER].append(node)

                group_members = neighbors[:group_capacity]
                licenses[LicenseType.GROUP_MEMBER].extend(group_members)
                covered_nodes.update(group_members)

                neighbors = neighbors[group_capacity:]

    return licenses
