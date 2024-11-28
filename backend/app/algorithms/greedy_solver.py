import networkx as nx

from config.config import LicenseType


def greedy_license_distribution(graph: nx.Graph, max_group_size: int = 6):
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

        uncovered_neighbors_sorted = sorted(uncovered_neighbors, key=key)[
            : max_group_size - 1
        ]

        if len(uncovered_neighbors_sorted) < 3:
            licenses[LicenseType.INDIVIDUAL].append(node)
            covered_nodes.add(node)

        else:
            licenses[LicenseType.GROUP_OWNER].append(node)
            covered_nodes.add(node)

            for uncovered_neighbor in uncovered_neighbors_sorted:
                licenses[LicenseType.GROUP_MEMBER].append(uncovered_neighbor)

            covered_nodes.update(uncovered_neighbors_sorted)

    return licenses
