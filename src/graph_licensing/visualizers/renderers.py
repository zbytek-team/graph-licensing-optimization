"""Core rendering functions for graph visualization."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from ..models.license import LicenseSolution

from .constants import COLOR_MAP, DEFAULT_EDGE_WIDTH_UNASSIGNED, DEFAULT_EDGE_WIDTH_GROUP, DEFAULT_DPI
from .utils import get_node_colors_and_sizes, get_edge_lists_and_colors


def draw_edges(graph: "nx.Graph", pos: dict, group_edges: list, unassigned_edges: list, 
               group_edge_colors: list[str], ax=None) -> None:
    """Draw both unassigned and group edges with appropriate styling."""
    # Draw unassigned edges with dashed style and gray color
    if unassigned_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=unassigned_edges,
            edge_color=COLOR_MAP["unassigned"],
            width=DEFAULT_EDGE_WIDTH_UNASSIGNED,
            alpha=1,
            style="--",
            ax=ax,
        )

    # Draw group edges with solid lines and appropriate colors
    if group_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=group_edges,
            edge_color=group_edge_colors,
            width=DEFAULT_EDGE_WIDTH_GROUP,
            alpha=1,
            style="solid",
            ax=ax,
        )


def create_legend(solution: "LicenseSolution") -> list:
    """Create legend elements for the visualization."""
    legend_elements = []
    used_license_types = set()

    for license_type, groups in solution.licenses.items():
        if license_type not in used_license_types:
            color = COLOR_MAP.get(license_type, COLOR_MAP["unassigned"])
            count = len(groups)
            total_people = sum(len(members) for members in groups.values())

            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=f"{license_type.title()}: {count} licenses ({total_people} people)",
                )
            )
            used_license_types.add(license_type)

    return legend_elements


def save_figure(save_path: str | None, default_subdir: str, default_filename: str) -> None:
    """Save figure with consistent logic."""
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        default_path = Path(f"results/{default_subdir}") / default_filename
        default_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(default_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        print(f"Visualization saved to: {default_path}")


def render_single_solution(graph: "nx.Graph", solution: "LicenseSolution", 
                          pos: dict, ax=None, node_size: int = None) -> None:
    """Render a single solution (nodes and edges) on given axes."""
    from .constants import DEFAULT_NODE_SIZE
    
    # Get node colors and sizes
    node_colors, node_sizes = get_node_colors_and_sizes(graph, solution)
    
    # Override node sizes if specified
    if node_size is not None:
        node_sizes = [node_size] * len(node_colors)

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
    )

    # Get edge lists and colors
    group_edges, unassigned_edges, group_edge_colors = get_edge_lists_and_colors(graph, solution)
    
    # Draw edges
    draw_edges(graph, pos, group_edges, unassigned_edges, group_edge_colors, ax=ax)
