import random

import networkx as nx
import itertools

from src.logger import get_logger

logger = get_logger(__name__)

random.seed(42)


def generate_clustered_graph(
    N=300,
    p_ws=0.1,
    extra_links=5,
    num_subgroups=15,
    min_subgroup_size=5,
    max_subgroup_size=12,
    inter_cluster_prob=0.05,
) -> nx.Graph:
    G = nx.Graph()
    available_nodes = list(range(N))
    random.shuffle(available_nodes)

    cluster_nodes = []

    for _ in range(num_subgroups):
        if len(available_nodes) < min_subgroup_size:
            break

        subgroup_size = random.randint(min_subgroup_size, max_subgroup_size)
        if subgroup_size > len(available_nodes):
            subgroup_size = len(available_nodes)

        nodes = [available_nodes.pop() for _ in range(subgroup_size)]
        cluster_nodes.append(nodes)

        subgraph = nx.watts_strogatz_graph(
            subgroup_size, max(2, subgroup_size // 3), p_ws
        )
        mapping = {i: nodes[i] for i in range(subgroup_size)}
        H = nx.relabel_nodes(subgraph, mapping)
        G = nx.compose(G, H)

    for cluster_a, cluster_b in itertools.combinations(cluster_nodes, 2):
        if random.random() < inter_cluster_prob:
            a, b = random.choice(cluster_a), random.choice(cluster_b)
            G.add_edge(a, b)

    existing_nodes = list(G.nodes())
    for _ in range(extra_links):
        if len(existing_nodes) > 1:
            a, b = random.sample(existing_nodes, 2)
            G.add_edge(a, b)

    return G


def generate_tree_graph(N=300) -> nx.Graph:
    G = nx.Graph()
    for i in range(1, N):
        G.add_edge(i, random.randint(0, i - 1))
    return G


def generate_star_graph(N=300) -> nx.Graph:
    G = nx.Graph()
    for i in range(1, N):
        G.add_edge(0, i)
    return G


def generate_complete_graph(N=300) -> nx.Graph:
    G = nx.complete_graph(N)
    return G


def generate_random_graph(N=300, p=0.05) -> nx.Graph:
    G = nx.erdos_renyi_graph(N, p)
    return G
