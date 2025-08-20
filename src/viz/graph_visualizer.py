import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import matplotlib

matplotlib.use("Agg")  # headless backend for WSL / servers

import matplotlib.pyplot as plt
import networkx as nx
from src.core import Solution


class GraphVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.default_edge_color = "#808080"
        self.owner_size = 500
        self.member_size = 300
        self.solo_size = 400

    def visualize_solution(
        self,
        graph: nx.Graph,
        solution: Solution,
        solver_name: str,
        timestamp_folder: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> str:
        _, ax = plt.subplots(figsize=self.figsize)
        pos = nx.spring_layout(graph, seed=42)

        node_to_group = self._map_nodes_to_groups(solution)

        node_colors, node_sizes = self._get_node_properties(graph, node_to_group)
        edge_colors = self._get_edge_colors(graph, node_to_group)

        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, alpha=0.7, width=1.5, ax=ax)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

        self._add_legend(ax, solution)
        ax.axis("off")

        if save_path is None:
            if timestamp_folder is None:
                timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
            save_path = f"results/graphs/{timestamp_folder}/{solver_name}_{n_nodes}n_{n_edges}e.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    # --- helpers ---

    def _map_nodes_to_groups(self, solution: Solution) -> Dict:
        mapping = {}
        for group in solution.groups:
            for member in group.all_members:
                mapping[member] = group
        return mapping

    def _get_node_properties(self, graph: nx.Graph, node_to_group: Dict) -> Tuple[List[str], List[int]]:
        colors, sizes = [], []
        owners = {g.owner for g in node_to_group.values()}

        for node in graph.nodes():
            group = node_to_group.get(node)
            if group is None:
                colors.append("#cccccc")
                sizes.append(self.solo_size)
            else:
                colors.append(group.license_type.color)
                sizes.append(self.owner_size if node in owners else self.member_size)
        return colors, sizes

    def _get_edge_colors(self, graph: nx.Graph, node_to_group: Dict) -> List[str]:
        colors = []
        for u, v in graph.edges():
            g1, g2 = node_to_group.get(u), node_to_group.get(v)
            if g1 is not None and g1 == g2:
                colors.append(g1.license_type.color)
            else:
                colors.append(self.default_edge_color)
        return colors

    def _add_legend(self, ax, solution: Solution) -> None:
        license_types = sorted({g.license_type for g in solution.groups}, key=lambda lt: lt.name)
        legend_elements = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lt.color, markersize=10, label=lt.name) for lt in license_types]
        legend_elements.append(plt.Line2D([0], [0], color=self.default_edge_color, linewidth=2, label="Other Edges"))
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))
