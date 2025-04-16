import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional
from src.algorithms.base import Solution


def visualize_graph(
    G: nx.Graph,
    solution: Optional[Solution] = None,
    title: str = "",
    show: bool = True,
    save_path: Optional[str] = None,
):
    pos = nx.spring_layout(G, iterations=25, seed=42)

    nx.draw_networkx_edges(G, pos, edge_color="#b9bcb6")

    node_colors = {node: "lightgray" for node in G.nodes()}
    node_sizes = {node: 150 for node in G.nodes()}
    highlight_edges = []

    if solution is not None:
        singles = set(solution["singles"])
        group_holders = set(g["license_holder"] for g in solution["groups"])
        group_members = set()

        for grp in solution["groups"]:
            h = grp["license_holder"]
            for m in grp["members"]:
                group_members.add(m)
                if G.has_edge(h, m):
                    e = (h, m) if h < m else (m, h)
                    highlight_edges.append(e)

        for n in G.nodes():
            if n in singles:
                node_colors[n] = "#a7192e"
            if n in group_holders:
                node_colors[n] = "#003057"
                node_sizes[n] = 250
            if n in group_members and n not in group_holders:
                node_colors[n] = "#003057"

    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color="#003057")

    c_map = {}
    for n, c in node_colors.items():
        if c not in c_map:
            c_map[c] = []
        c_map[c].append(n)

    for c, nl in c_map.items():
        sz = [node_sizes[x] for x in nl]
        nx.draw_networkx_nodes(G, pos, nodelist=nl, node_color=c, node_size=sz)

    labels = {n: str(n) for n in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=labels, font_color="white", font_size=8, font_family="sans-serif")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved graph visualization to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
