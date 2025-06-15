from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from ..models.license import LicenseConfig, LicenseSolution

# Constants
COLOR_MAP = {
    "solo": "#f6d700", 
    "duo": "#c3102f",  
    "family": "#003667",
    "default": "#000001",
    "unassigned": "#cccccc",
}

SIZE_MAP = {"owner": 500, "member": 300, "solo": 400}

# Default visualization settings
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_DPI = 300
DEFAULT_NODE_SIZE = 500
DEFAULT_EDGE_WIDTH_UNASSIGNED = 1
DEFAULT_EDGE_WIDTH_GROUP = 3


def get_node_colors_and_sizes(graph: "nx.Graph", solution: "LicenseSolution") -> tuple[list[str], list[int]]:
    """Get node colors and sizes based on solution."""
    node_colors = []
    node_sizes = []

    for node in graph.nodes():
        license_info = solution.get_node_license_info(node)
        if license_info:
            license_type, owner = license_info
            color = COLOR_MAP.get(license_type, COLOR_MAP["unassigned"])
            node_colors.append(color)

            if node == owner:
                node_sizes.append(SIZE_MAP["owner"])
            else:
                node_sizes.append(SIZE_MAP["member"])
        else:
            node_colors.append(COLOR_MAP["unassigned"])
            node_sizes.append(SIZE_MAP["solo"])

    return node_colors, node_sizes


def get_edge_lists_and_colors(graph: "nx.Graph", solution: "LicenseSolution") -> tuple[list, list, list[str]]:
    """Separate edges into group edges and unassigned edges with their colors."""
    group_edges = []
    group_edge_colors = []
    
    for license_type, groups in solution.licenses.items():
        license_color = COLOR_MAP.get(license_type, COLOR_MAP["unassigned"])
        for owner, members in groups.items():
            if len(members) > 1:  # Only for multi-member groups
                for member in members:
                    if member != owner and graph.has_edge(owner, member):
                        group_edges.append((owner, member))
                        group_edge_colors.append(license_color)

    # Get all edges that are not group edges (unassigned edges)
    all_edges = list(graph.edges())
    unassigned_edges = [
        edge for edge in all_edges 
        if edge not in group_edges and tuple(reversed(edge)) not in group_edges
    ]

    return group_edges, unassigned_edges, group_edge_colors


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


def calculate_solution_stats(solution: "LicenseSolution", config: "LicenseConfig") -> dict:
    """Calculate statistics for a solution."""
    total_cost = solution.calculate_cost(config)
    total_licenses = sum(len(groups) for groups in solution.licenses.values())
    total_people = len(solution.get_all_nodes())
    
    num_solo = sum(
        1
        for license_type, groups in solution.licenses.items()
        for members in groups.values()
        if len(members) == 1
    )
    num_groups = sum(
        1
        for license_type, groups in solution.licenses.items()
        for members in groups.values()
        if len(members) > 1
    )
    
    return {
        "total_cost": total_cost,
        "total_licenses": total_licenses,
        "total_people": total_people,
        "cost_per_person": total_cost / total_people if total_people > 0 else 0,
        "num_solo": num_solo,
        "num_groups": num_groups,
    }


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
                          pos: dict, ax=None, node_size: int = DEFAULT_NODE_SIZE) -> None:
    """Render a single solution (nodes and edges) on given axes."""
    # Get node colors and sizes
    node_colors, node_sizes = get_node_colors_and_sizes(graph, solution)
    
    # Override node sizes if specified
    if node_size != DEFAULT_NODE_SIZE:
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


def visualize_solution(
    graph: "nx.Graph",
    solution: "LicenseSolution",
    config: "LicenseConfig",
    title: str = "Licensing Solution",
    save_path: str | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
) -> None:
    """Create a visualization of a single solution."""
    plt.figure(figsize=figsize)
    print("spring layouting")

    pos = nx.spring_layout(graph, seed=42)
    print("done springing")

    # Render the solution
    render_single_solution(graph, solution, pos)

    # Create and display legend
    legend_elements = create_legend(solution)
    if legend_elements:
        plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=10)

    # Add title with statistics
    stats = calculate_solution_stats(solution, config)
    plt.title(
        f"{title}\nTotal Cost: ${stats['total_cost']:.2f} | {stats['total_licenses']} licenses | {stats['total_people']} people | ${stats['cost_per_person']:.2f}/person",
        fontsize=14,
        pad=20,
    )

    plt.axis("off")
    plt.tight_layout()

    # Save figure
    save_figure(save_path, "single", "visualization.png")

    plt.close()


def calculate_frame_changes(current_graph: "nx.Graph", prev_graph: "nx.Graph") -> str:
    """Calculate changes between two graph frames."""
    added_nodes = list(set(current_graph.nodes()) - set(prev_graph.nodes()))
    removed_nodes = list(set(prev_graph.nodes()) - set(current_graph.nodes()))
    added_edges = list(set(current_graph.edges()) - set(prev_graph.edges()))
    removed_edges = list(set(prev_graph.edges()) - set(current_graph.edges()))

    changes_text = "Changes: "
    if added_nodes:
        changes_text += f"+{len(added_nodes)} nodes "
    if removed_nodes:
        changes_text += f"-{len(removed_nodes)} nodes "
    if added_edges:
        changes_text += f"+{len(added_edges)} edges "
    if removed_edges:
        changes_text += f"-{len(removed_edges)} edges"
    if changes_text == "Changes: ":
        changes_text += "None"
    
    return changes_text


def calculate_node_position(new_node, current_graph: "nx.Graph", prev_pos: dict) -> tuple[float, float]:
    """Calculate position for a new node based on its neighbors."""
    import random
    
    neighbors = list(current_graph.neighbors(new_node))
    if neighbors:
        neighbor_positions = [prev_pos[n] for n in neighbors if n in prev_pos]
        if neighbor_positions:
            avg_x = sum(pos[0] for pos in neighbor_positions) / len(neighbor_positions)
            avg_y = sum(pos[1] for pos in neighbor_positions) / len(neighbor_positions)
            
            offset_x = random.uniform(-0.1, 0.1)
            offset_y = random.uniform(-0.1, 0.1)
            return (avg_x + offset_x, avg_y + offset_y)
    
    return (random.uniform(-1, 1), random.uniform(-1, 1))


def compare_solutions(
    graph: "nx.Graph",
    solutions: dict[str, "LicenseSolution"],
    config: "LicenseConfig",
    title: str = "Solution Comparison",
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (20, 12),
) -> None:
    """Compare multiple solutions side by side."""
    n_solutions = len(solutions)
    if n_solutions == 0:
        return

    cols = min(3, n_solutions)
    rows = (n_solutions + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if hasattr(axes, "flatten"):
        axes_list = list(axes.flatten())
    else:
        axes_list = [axes]

    pos = nx.spring_layout(graph, seed=42)

    for idx, (algorithm_name, solution) in enumerate(solutions.items()):
        ax = axes_list[idx]

        # Render the solution
        render_single_solution(graph, solution, pos, ax=ax, node_size=DEFAULT_NODE_SIZE)

        # Add title with statistics
        stats = calculate_solution_stats(solution, config)
        ax.set_title(
            f"{algorithm_name}\nCost: ${stats['total_cost']:.2f} | {stats['total_licenses']} licenses | {stats['total_people']} people"
        )
        ax.axis("off")

    for ax in axes_list[n_solutions:]:
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save figure
    save_figure(save_path, "compare", "solution_comparison.png")

    plt.close()


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
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP["solo"], markersize=8, label="Solo"
            ),
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
            plt.Line2D([0], [0], color=COLOR_MAP["unassigned"], linewidth=1, linestyle="dashed", label="Unassigned Connection"),
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
