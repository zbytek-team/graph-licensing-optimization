import matplotlib.pyplot as plt
import networkx as nx
import random
from .solvers.base import AssignmentResult

colors = {"individual": "#c3102f", "group": "#003667"}


def vary_color(hex_color: str, variation: int = 30) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    def clamp(value):
        return max(0, min(255, value))

    r = clamp(r + random.randint(-variation, variation))
    g = clamp(g + random.randint(-variation, variation))
    b = clamp(b + random.randint(-variation, variation))

    return f"#{r:02x}{g:02x}{b:02x}"


def visualize_graph(graph: nx.Graph, result: AssignmentResult, output_path: str | None = None) -> None:
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)

    nx.draw(graph, pos, node_size=10, edge_color="gray", alpha=0.5, width=0.5)

    for node in result["individual"]:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[node],
            node_color=vary_color(colors["individual"]),
            node_size=10,
        )

    for holder, members in result["group"].items():
        group_color = vary_color(colors["group"])
        nx.draw_networkx_nodes(graph, pos, nodelist=[holder], node_color=group_color, node_size=20)
        nx.draw_networkx_nodes(graph, pos, nodelist=members, node_color=group_color, node_size=10)

        for member in members:
            if member == holder:
                continue

            nx.draw_networkx_edges(graph, pos, edgelist=[(holder, member)], edge_color=group_color, width=2)

    plt.title("Colored Graph Visualization")

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
