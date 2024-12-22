import networkx as nx


def key(n: int, graph: nx.Graph):
    if isinstance(graph.degree, nx.classes.reportviews.DegreeView):
        return graph.degree[n]
    else:
        return graph.degree
