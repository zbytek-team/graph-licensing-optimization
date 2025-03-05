import networkx as nx


def get_degree(graph: nx.Graph, node: int) -> int:
    degree = graph.degree

    if isinstance(degree, int):
        return degree
    else:
        return int(degree[node])
