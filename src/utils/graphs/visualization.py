import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import os
from typing import List, Optional
from datetime import datetime

from src.core.types import Solution


class GraphVisualizer:
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        self.default_edge_color = "#808080"
        self.owner_size = 500
        self.member_size = 300
        self.solo_size = 400

    def visualize_solution(
        self, graph: nx.Graph, solution: Solution, solver_name: str, timestamp_folder: Optional[str] = None, save_path: Optional[str] = None
    ) -> None:
        _, ax = plt.subplots(figsize=self.figsize)
        pos = nx.spring_layout(graph, seed=42)

        node_colors, node_sizes = self._get_node_properties(graph, solution)
        edge_colors = self._get_edge_colors(graph, solution)

        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, alpha=0.7, width=1.5, ax=ax)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

        self._add_legend(ax, solution)
        ax.axis("off")

        if save_path is None:
            if timestamp_folder is None:
                now = datetime.now()
                timestamp_folder = now.strftime("%Y%m%d_%H%M%S")

            n_nodes = len(graph.nodes())
            n_edges = len(graph.edges())
            save_path = f"results/graphs/{timestamp_folder}/{solver_name}_{n_nodes}n_{n_edges}e.png"

            os.makedirs(f"results/graphs/{timestamp_folder}", exist_ok=True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
        plt.close()

    def _get_node_properties(self, graph: nx.Graph, solution: Solution) -> tuple:
        node_colors = []
        node_sizes = []

        node_to_group = {}
        owners = set()

        for group in solution.groups:
            node_to_group[group.owner] = group
            owners.add(group.owner)
            for member in group.additional_members:
                node_to_group[member] = group

        for node in graph.nodes():
            if node in node_to_group:
                group = node_to_group[node]
                color = group.license_type.color
                size = self.owner_size if node in owners else self.member_size
            else:
                color = "#cccccc"
                size = self.solo_size

            node_colors.append(color)
            node_sizes.append(size)

        return node_colors, node_sizes

    def _get_edge_colors(self, graph: nx.Graph, solution: Solution) -> List[str]:
        edge_colors = []
        node_to_group = {}

        for group in solution.groups:
            for member in group.all_members:
                node_to_group[member] = group

        for edge in graph.edges():
            node1, node2 = edge
            if node1 in node_to_group and node2 in node_to_group and node_to_group[node1] == node_to_group[node2]:
                color = node_to_group[node1].license_type.color
            else:
                color = self.default_edge_color
            edge_colors.append(color)

        return edge_colors

    def _add_legend(self, ax, solution: Solution) -> None:
        license_types = list(set(group.license_type for group in solution.groups))
        license_types.sort(key=lambda x: x.name)

        legend_elements = []

        for lt in license_types:
            legend_elements.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lt.color, markersize=10, label=f"{lt.name}"))

        legend_elements.append(plt.Line2D([0], [0], color=self.default_edge_color, linewidth=2, label="Other Edges"))

        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))
