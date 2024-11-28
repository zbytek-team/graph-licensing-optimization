import networkx as nx
import math

from config.config import LicenseType


def greedy_advanced_license_distribution(
    graph: nx.Graph, prices: tuple[float, float], max_group_size: int = 6
):
    def key(n: int):
        if isinstance(graph.degree, nx.classes.reportviews.DegreeView):
            return graph.degree[n]
        else:
            return graph.degree

    licenses = {
        LicenseType.INDIVIDUAL: [],
        LicenseType.GROUP_OWNER: [],
        LicenseType.GROUP_MEMBER: [],
    }

    covered_nodes = set()

    nodes_sorted = sorted(graph.nodes, key=key, reverse=True)

    for node in nodes_sorted:
        if node in covered_nodes:
            continue

        neighbors = list(graph.neighbors(node))

        uncovered_neighbors = [n for n in neighbors if n not in covered_nodes]

        if len(uncovered_neighbors) < math.ceil(prices[1] / prices[0]):
            licenses[LicenseType.INDIVIDUAL].append(node)
            covered_nodes.add(node)
        else:
            groups_count = 0

            while uncovered_neighbors:
                group_members = uncovered_neighbors[: max_group_size - 1]
                if len(group_members) < math.ceil(prices[1] / prices[0]):
                    licenses[LicenseType.INDIVIDUAL].extend(group_members)
                    covered_nodes.update(group_members)
                    uncovered_neighbors = uncovered_neighbors[max_group_size - 1 :]
                    continue
                for member in group_members:
                    licenses[LicenseType.GROUP_MEMBER].append(member)

                covered_nodes.update(group_members)
                uncovered_neighbors = uncovered_neighbors[max_group_size - 1 :]

                groups_count += 1

            if groups_count == 0:
                licenses[LicenseType.INDIVIDUAL].append(node)
                covered_nodes.add(node)
                continue

            for _ in range(groups_count):
                licenses[LicenseType.GROUP_OWNER].append(node)
                covered_nodes.add(node)

    return licenses
