"""Functions for creating animated visualizations."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from ..models.license import LicenseConfig, LicenseSolution

from .constants import COLOR_MAP, DEFAULT_FIGSIZE, DEFAULT_NODE_SIZE
from .renderers import draw_edges
from .utils import (
    get_edge_lists_and_colors,
    get_node_colors_and_sizes,
    calculate_solution_stats,
    calculate_frame_changes,
    calculate_node_position,
)


def create_dynamic_gif(
    graph_states: list["nx.Graph"],
    solutions: list["LicenseSolution"],
    config: "LicenseConfig",
    algorithm_name: str = "Algorithm",
    title: str = "Dynamic Graph Evolution",
    save_path: str | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    duration: float = 1.0,
    show_changes: bool = True,
) -> None:
    """Create an animated GIF showing graph evolution."""
    try:
        import matplotlib.animation as animation
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib.animation required for GIF creation")
        return

    if len(graph_states) != len(solutions):
        print("Error: Number of graph states must match number of solutions")
        return

    if len(graph_states) < 2:
        print("Error: Need at least 2 frames for animation")
        return

    initial_graph = graph_states[0]
    if len(initial_graph.nodes()) > 0:
        base_pos = nx.spring_layout(initial_graph, seed=42, k=2, iterations=50)
    else:
        base_pos = {}

    pos_cache = [base_pos.copy()]  # Store positions for each frame

    for i in range(1, len(graph_states)):
        prev_graph = graph_states[i - 1]
        current_graph = graph_states[i]
        prev_pos = pos_cache[i - 1]
        current_pos = prev_pos.copy()

        new_nodes = set(current_graph.nodes()) - set(prev_graph.nodes())
        for new_node in new_nodes:
            current_pos[new_node] = calculate_node_position(new_node, current_graph, prev_pos)

        removed_nodes = set(prev_graph.nodes()) - set(current_graph.nodes())
        for removed_node in removed_nodes:
            current_pos.pop(removed_node, None)

        pos_cache.append(current_pos)

    fig, ax = plt.subplots(figsize=figsize)

    def animate(frame):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")

        current_graph = graph_states[frame]
        current_solution = solutions[frame]
        current_pos = pos_cache[frame]

        # Get edge lists and colors
        group_edges, unassigned_edges, group_edge_colors = get_edge_lists_and_colors(current_graph, current_solution)

        # Draw edges
        draw_edges(current_graph, current_pos, group_edges, unassigned_edges, group_edge_colors)

        # Get node colors and sizes
        node_colors, node_sizes = get_node_colors_and_sizes(current_graph, current_solution)

        # Draw nodes with consistent size
        nx.draw_networkx_nodes(
            current_graph,
            current_pos,
            node_color=node_colors,
            node_size=[DEFAULT_NODE_SIZE] * len(node_colors),
        )

        # Calculate statistics
        stats = calculate_solution_stats(current_solution, config)
        num_nodes = current_graph.number_of_nodes()
        num_edges = current_graph.number_of_edges()

        ax.set_title(f"{title} - {algorithm_name}\nIteration {frame}", fontsize=16, fontweight="bold", pad=20)

        stats_text = f"Nodes: {num_nodes} | Edges: {num_edges}\nCost: ${stats['total_cost']:.2f} | Solo: {stats['num_solo']} | Groups: {stats['num_groups']}"

        if show_changes and frame > 0:
            prev_graph = graph_states[frame - 1]
            changes_text = calculate_frame_changes(current_graph, prev_graph)
            stats_text += f"\n{changes_text}"

        text_box = FancyBboxPatch(
            (0.02, 0.02),
            0.4,
            0.15,
            boxstyle="round,pad=0.01",
            facecolor="white",
            edgecolor="black",
            alpha=0.9,
            transform=ax.transAxes,
        )
        ax.add_patch(text_box)
        ax.text(
            0.03,
            0.09,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.0),
        )

        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP["solo"], markersize=8, label="Solo"),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_MAP["duo"],
                markersize=8,
                label="Duo Member",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLOR_MAP["family"],
                markersize=8,
                label="Family Member",
            ),
            plt.Line2D([0], [0], color=COLOR_MAP["duo"], linewidth=3, label="Duo Connection"),
            plt.Line2D([0], [0], color=COLOR_MAP["family"], linewidth=3, label="Family Connection"),
            plt.Line2D(
                [0], [0], color=COLOR_MAP["unassigned"], linewidth=1, linestyle="dashed", label="Unassigned Connection"
            ),
        ]

        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.9)

    frames = len(graph_states)
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=duration * 1000, repeat=True, blit=False)

    if save_path is None:
        save_path = f"results/dynamic/dynamic_{algorithm_name.lower()}_evolution.gif"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating GIF with {frames} frames...")
    try:
        anim.save(
            save_path,
            writer="pillow",
            fps=1 / duration,
            savefig_kwargs={"bbox_inches": "tight", "facecolor": "white"},
        )
        print(f"Dynamic GIF saved to: {save_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")

        try:
            mp4_path = save_path.replace(".gif", ".mp4")
            anim.save(
                mp4_path,
                writer="ffmpeg",
                fps=1 / duration,
                savefig_kwargs={"bbox_inches": "tight", "facecolor": "white"},
            )
            print(f"Saved as MP4 instead: {mp4_path}")
        except Exception as e2:
            print(f"Could not save as MP4 either: {e2}")

    plt.close(fig)
