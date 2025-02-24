import networkx as nx
import random


def generate_clustered_graph(N=300, p_ws=0.02, extra_links=3, num_subgroups=15):
    G = nx.Graph()

    for _ in range(num_subgroups):
        subgroup_size = random.randint(5, 10)
        subgraph = nx.watts_strogatz_graph(subgroup_size, max(2, subgroup_size // 4), p_ws)
        mapping = {i: random.randint(0, N - 1) for i in range(subgroup_size)}
        H = nx.relabel_nodes(subgraph, mapping)
        G = nx.compose(G, H)

    for _ in range(extra_links):
        a, b = random.sample(range(N), 2)
        G.add_edge(a, b)

    return G
