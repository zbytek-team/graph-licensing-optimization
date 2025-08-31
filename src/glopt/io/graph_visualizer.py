from datetime import datetime
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import pathlib

import matplotlib.pyplot as plt
import networkx as nx

from glopt.core import Solution


class GraphVisualizer:
    def __init__(
        self,
        figsize: tuple[int, int] = (12, 8),
        layout_seed: int = 42,
        reuse_layout: bool = True,
    ) -> None:
        self.figsize = figsize
        self.layout_seed = layout_seed
        self.reuse_layout = reuse_layout

        self.default_edge_color = "#808080"
        self.owner_size = 500
        self.member_size = 300
        self.solo_size = 400

        self._pos: dict[Any, tuple[float, float]] | None = None

    def visualize_solution(
        self,
        graph: nx.Graph,
        solution: Solution,
        solver_name: str,
        timestamp_folder: str | None = None,
        save_path: str | None = None,
    ) -> str:
        if not self.reuse_layout or self._pos is None:
            self._pos = nx.spring_layout(graph, seed=self.layout_seed)
        else:
            self._update_positions_for_graph(graph)

        node_to_group = self._map_nodes_to_groups(solution)
        node_colors, node_sizes = self._get_node_properties(graph, node_to_group)
        edge_colors = self._get_edge_colors(graph, node_to_group)

        _, ax = plt.subplots(figsize=self.figsize)
        nx.draw_networkx_edges(graph, self._pos, edge_color=edge_colors, alpha=0.7, width=1.5, ax=ax)
        nx.draw_networkx_nodes(graph, self._pos, node_color=node_colors, node_size=node_sizes, ax=ax)

        self._add_legend(ax, solution)
        ax.axis("off")

        if save_path is None:
            if timestamp_folder is None:
                timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
            save_path = f"runs/graphs/{timestamp_folder}/{solver_name}_{n_nodes}n_{n_edges}e.png"
        pathlib.Path(pathlib.Path(save_path).parent).mkdir(exist_ok=True, parents=True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    # reset_layout and set_layout were unused; removed

    def _update_positions_for_graph(self, graph: nx.Graph) -> None:
        assert self._pos is not None
        g_nodes = set(graph.nodes())
        pos_nodes = set(self._pos.keys())

        for n in pos_nodes - g_nodes:
            self._pos.pop(n, None)

        new_nodes = g_nodes - pos_nodes
        if not new_nodes:
            return

        for n in new_nodes:
            neigh = [v for v in graph.neighbors(n) if v in self._pos]
            if neigh:
                anchor = random_choice(neigh)
                ax, ay = self._pos[anchor]

                self._pos[n] = (ax + jitter(), ay + jitter())
            else:
                self._pos[n] = (jitter(scale=0.25), jitter(scale=0.25))

    def _map_nodes_to_groups(self, solution: Solution) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for group in solution.groups:
            for member in group.all_members:
                mapping[member] = group
        return mapping

    def _get_node_properties(self, graph: nx.Graph, node_to_group: dict[Any, Any]) -> tuple[list[str], list[int]]:
        colors: list[str] = []
        sizes: list[int] = []
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

    def _get_edge_colors(self, graph: nx.Graph, node_to_group: dict[Any, Any]) -> list[str]:
        colors: list[str] = []
        for u, v in graph.edges():
            g1 = node_to_group.get(u)
            g2 = node_to_group.get(v)
            if g1 is not None and g1 == g2:
                colors.append(g1.license_type.color)
            else:
                colors.append(self.default_edge_color)
        return colors

    def _add_legend(self, ax, solution: Solution) -> None:
        license_types = sorted({g.license_type for g in solution.groups}, key=lambda lt: lt.name)
        if not license_types:
            return
        elems = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lt.color, markersize=10, label=lt.name) for lt in license_types]
        elems.append(plt.Line2D([0], [0], color=self.default_edge_color, linewidth=2, label="Other Edges"))
        ax.legend(handles=elems, loc="upper right", bbox_to_anchor=(0.98, 0.98))


def jitter(scale: float = 0.08) -> float:
    import random as _r

    return (_r.random() - 0.5) * 2.0 * scale


def random_choice(seq):
    import random as _r

    return _r.choice(list(seq))
