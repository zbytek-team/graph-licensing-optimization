"""Visualization utilities for licensing optimization results."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from ..models.license import LicenseConfig, LicenseSolution


class GraphVisualizer:
    """Visualizer for graph licensing solutions."""

    def __init__(self) -> None:
        """Initialize the visualizer."""
        # Enhanced color map for different license types
        self.color_map = {
            "solo": "#f7ca00",         # Yellow for solo
            "individual": "#f7ca00",   # Yellow for individual  
            "duo": "#ff6b6b",          # Red for duo
            "trio": "#4ecdc4",         # Teal for trio
            "family": "#45b7d1",       # Blue for family
            "small_team": "#96ceb4",   # Light green for small team
            "large_team": "#003667",   # Dark blue for large team
            "small": "#96ceb4",        # Light green
            "medium": "#45b7d1",       # Blue
            "large": "#003667",        # Dark blue
            "group": "#003667",        # Default group color
        }

        self.size_map = {
            "owner": 500,
            "member": 300,
            "solo": 400
        }

    def visualize_solution(
        self,
        graph: "nx.Graph",
        solution: "LicenseSolution",
        config: "LicenseConfig",
        title: str = "Licensing Solution",
        save_path: str | None = None,
        figsize: tuple[int, int] = (12, 8),
    ) -> None:
        """Visualize a licensing solution.

        Args:
            graph: The social network graph.
            solution: Licensing solution to visualize.
            config: License configuration.
            title: Plot title.
            save_path: Path to save the plot (optional).
            figsize: Figure size as (width, height).
        """
        plt.figure(figsize=figsize)
        print("spring layouting")
        # Calculate layout
        pos = nx.spring_layout(graph, seed=42)
        print("done springing")
        
        # Prepare node colors and sizes based on new license structure
        node_colors = []
        node_sizes = []
        
        for node in graph.nodes():
            license_info = solution.get_node_license_info(node)
            if license_info:
                license_type, owner = license_info
                # Get color for license type
                color = self.color_map.get(license_type, "#cccccc")
                node_colors.append(color)
                
                # Set size based on role (owner vs member)
                if node == owner:
                    node_sizes.append(self.size_map["owner"])
                else:
                    node_sizes.append(self.size_map["member"])
            else:
                node_colors.append("#cccccc")  # Gray for unassigned
                node_sizes.append(self.size_map["solo"])

        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
        )

        # Draw regular edges
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color="gray",
            width=1,
            alpha=0.5,
            style='dashed',
        )

        # Draw license group edges with colors
        license_colors = ["#ff4444", "#44ff44", "#4444ff", "#ffff44", "#ff44ff", "#44ffff"]
        color_idx = 0
        
        for license_type, groups in solution.licenses.items():
            for owner, members in groups.items():
                if len(members) > 1:  # Only for multi-member groups
                    group_edges = []
                    for member in members:
                        if member != owner and graph.has_edge(owner, member):
                            group_edges.append((owner, member))
                    
                    if group_edges:
                        edge_color = license_colors[color_idx % len(license_colors)]
                        nx.draw_networkx_edges(
                            graph,
                            pos,
                            edgelist=group_edges,
                            edge_color=edge_color,
                            width=3,
                            alpha=0.8,
                        )
                        color_idx += 1

        # Draw node labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")

        # Create legend for license types
        legend_elements = []
        used_license_types = set()
        
        for license_type, groups in solution.licenses.items():
            if license_type not in used_license_types:
                color = self.color_map.get(license_type, "#cccccc")
                count = len(groups)
                total_people = sum(len(members) for members in groups.values())
                
                legend_elements.append(
                    plt.Line2D(
                        [0], [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=10,
                        label=f"{license_type.title()}: {count} licenses ({total_people} people)",
                    )
                )
                used_license_types.add(license_type)

        if legend_elements:
            plt.legend(
                handles=legend_elements,
                loc="upper right",
                bbox_to_anchor=(1.0, 1.0),
                fontsize=10
            )

        # Add comprehensive cost information as title
        total_cost = solution.calculate_cost(config)
        total_licenses = sum(len(groups) for groups in solution.licenses.values())
        total_people = len(solution.get_all_nodes())
        
        plt.title(
            f"{title}\nTotal Cost: ${total_cost:.2f} | {total_licenses} licenses | {total_people} people | ${total_cost/total_people:.2f}/person", 
            fontsize=14, pad=20
        )

        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {save_path}")
        else:
            # Generate a default save path if none provided
            from pathlib import Path
            default_path = Path("results/single") / "visualization.png"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {default_path}")

        plt.close()

    def compare_solutions(
        self,
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

        # Layout of subplots
        cols = min(3, n_solutions)
        rows = (n_solutions + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # --- normalize axes into a flat list of Axes instances ---
        if hasattr(axes, "flatten"):
            axes_list = list(axes.flatten())
        else:
            axes_list = [axes]

        # compute positions once
        pos = nx.spring_layout(graph, seed=42)

        for idx, (algorithm_name, solution) in enumerate(solutions.items()):
            ax = axes_list[idx]

            # Prepare node colors based on new license structure
            node_colors = []
            for node in graph.nodes():
                license_info = solution.get_node_license_info(node)
                if license_info:
                    license_type, owner = license_info
                    color = self.color_map.get(license_type, "#cccccc")
                    node_colors.append(color)
                else:
                    node_colors.append("#cccccc")  # Gray for unassigned

            # Draw everything on this Axes
            nx.draw_networkx_nodes(
                graph,
                pos,
                node_color=node_colors,
                node_size=500,
                alpha=0.8,
                ax=ax,
            )
            nx.draw_networkx_edges(
                graph,
                pos,
                edge_color="gray",
                width=1,
                alpha=0.7,
                style='dashed',
                ax=ax,
            )

            # Highlight group‐member edges using new structure
            group_edges = []
            for license_type, groups in solution.licenses.items():
                for owner, members in groups.items():
                    if len(members) > 1:  # Multi-member group
                        for member in members:
                            if member != owner and graph.has_edge(owner, member):
                                group_edges.append((owner, member))
            
            if group_edges:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=group_edges,
                    edge_color="#013865",
                    width=2,
                    alpha=0.8,
                    ax=ax,
                )

            # Labels and title with new license structure info
            total_cost = solution.calculate_cost(config)
            total_licenses = sum(len(groups) for groups in solution.licenses.values())
            total_people = len(solution.get_all_nodes())
            ax.set_title(f"{algorithm_name}\nCost: ${total_cost:.2f} | {total_licenses} licenses | {total_people} people")
            ax.axis("off")

        # Turn off any unused subplots
        for ax in axes_list[n_solutions:]:
            ax.axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparison visualization saved to: {save_path}")
        else:
            # Generate a default save path if none provided
            default_path = Path("results/compare") / "solution_comparison.png"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(default_path, dpi=300, bbox_inches="tight")
            print(f"Comparison visualization saved to: {default_path}")

        plt.close()

    def plot_cost_comparison(
        self,
        results: dict[str, dict[str, float]],
        title: str = "Algorithm Cost Comparison",
        save_path: str | None = None,
        show: bool = True,
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """Plot cost comparison across algorithms and graph types.

        Args:
            results: Nested dict {graph_type: {algorithm: cost}}.
            title: Plot title.
            save_path: Path to save the plot (optional).
            show: Whether to display the plot.
            figsize: Figure size as (width, height).
        """
        import numpy as np

        plt.figure(figsize=figsize)

        graph_types = list(results.keys())
        algorithms = list(next(iter(results.values())).keys())

        x = np.arange(len(graph_types))
        width = 0.8 / len(algorithms)

        for i, algorithm in enumerate(algorithms):
            costs = [results[graph_type].get(algorithm, 0) for graph_type in graph_types]
            plt.bar(x + i * width, costs, width, label=algorithm, alpha=0.8)

        plt.xlabel("Graph Type")
        plt.ylabel("Total Cost")
        plt.title(title)
        plt.xticks(x + width * (len(algorithms) - 1) / 2, graph_types, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

    def plot_runtime_comparison(
        self,
        results: dict[str, dict[str, float]],
        title: str = "Algorithm Runtime Comparison",
        save_path: str | None = None,
        show: bool = True,
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """Plot runtime comparison across algorithms and graph sizes.

        Args:
            results: Nested dict {graph_size: {algorithm: runtime}}.
            title: Plot title.
            save_path: Path to save the plot (optional).
            show: Whether to display the plot.
            figsize: Figure size as (width, height).
        """
        plt.figure(figsize=figsize)

        graph_sizes = sorted([int(k) for k in results])
        algorithms = list(next(iter(results.values())).keys())

        for algorithm in algorithms:
            runtimes = [results[str(size)].get(algorithm, 0) for size in graph_sizes]
            plt.plot(graph_sizes, runtimes, marker="o", label=algorithm, linewidth=2)

        plt.xlabel("Graph Size (number of nodes)")
        plt.ylabel("Runtime (seconds)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

    def create_dynamic_gif(
        self,
        graph_states: list["nx.Graph"],
        solutions: list["LicenseSolution"],
        config: "LicenseConfig",
        algorithm_name: str = "Algorithm",
        title: str = "Dynamic Graph Evolution",
        save_path: str | None = None,
        figsize: tuple[int, int] = (12, 8),
        duration: float = 1.0,
        show_changes: bool = True,
    ) -> None:
        """Create animated GIF showing graph evolution over time.

        Args:
            graph_states: List of graph states at each iteration.
            solutions: List of solutions for each iteration.
            config: License configuration.
            algorithm_name: Name of algorithm used.
            title: Base title for the animation.
            save_path: Path to save the GIF (optional).
            figsize: Figure size as (width, height).
            duration: Duration per frame in seconds.
            show_changes: Whether to show statistics about changes.
        """
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

        # Calculate initial layout using spring layout for the first graph
        # Then maintain positions and add new nodes intelligently
        initial_graph = graph_states[0]
        if len(initial_graph.nodes()) > 0:
            base_pos = nx.spring_layout(initial_graph, seed=42, k=2, iterations=50)
        else:
            base_pos = {}
        
        pos_cache = [base_pos.copy()]  # Store positions for each frame
        
        # Calculate positions for each subsequent frame
        for i in range(1, len(graph_states)):
            prev_graph = graph_states[i-1]
            current_graph = graph_states[i]
            prev_pos = pos_cache[i-1]
            current_pos = prev_pos.copy()
            
            # Handle new nodes - place them near their neighbors
            new_nodes = set(current_graph.nodes()) - set(prev_graph.nodes())
            for new_node in new_nodes:
                neighbors = list(current_graph.neighbors(new_node))
                if neighbors:
                    # Place new node near average position of its neighbors
                    neighbor_positions = [prev_pos[n] for n in neighbors if n in prev_pos]
                    if neighbor_positions:
                        avg_x = sum(pos[0] for pos in neighbor_positions) / len(neighbor_positions)
                        avg_y = sum(pos[1] for pos in neighbor_positions) / len(neighbor_positions)
                        # Add small random offset to avoid exact overlap
                        import random
                        offset_x = random.uniform(-0.1, 0.1)
                        offset_y = random.uniform(-0.1, 0.1)
                        current_pos[new_node] = (avg_x + offset_x, avg_y + offset_y)
                    else:
                        # No neighbor positions available, place randomly
                        current_pos[new_node] = (random.uniform(-1, 1), random.uniform(-1, 1))
                else:
                    # No neighbors, place randomly
                    import random
                    current_pos[new_node] = (random.uniform(-1, 1), random.uniform(-1, 1))
            
            # Remove positions for deleted nodes
            removed_nodes = set(prev_graph.nodes()) - set(current_graph.nodes())
            for removed_node in removed_nodes:
                current_pos.pop(removed_node, None)
            
            pos_cache.append(current_pos)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.axis('off')
            
            current_graph = graph_states[frame]
            current_solution = solutions[frame]
            current_pos = pos_cache[frame]
            
            # Find group edges first
            group_edges = []
            for owner, members in current_solution.group_owners.items():
                for member in members:
                    if member != owner and current_graph.has_edge(owner, member):
                        group_edges.append((owner, member))

            # Draw non-group edges with darker gray and dashed lines
            non_group_edges = [edge for edge in current_graph.edges() if edge not in group_edges and tuple(reversed(edge)) not in group_edges]
            if non_group_edges:
                nx.draw_networkx_edges(
                    current_graph,
                    current_pos,
                    edgelist=non_group_edges,
                    edge_color="#666666",  # Darker gray instead of lightgray
                    width=1,
                    alpha=0.6,
                    style="dashed",  # Dashed lines
                )

            # Draw group edges with thick blue lines
            if group_edges:
                nx.draw_networkx_edges(
                    current_graph,
                    current_pos,
                    edgelist=group_edges,
                    edge_color="#013865",
                    width=3,
                    alpha=0.8,
                )

            # Prepare node colors and sizes based on solution
            node_colors = []
            node_sizes = []
            for node in current_graph.nodes():
                license_type = current_solution.get_node_license_type(node).value
                node_colors.append(self.color_map[license_type])
                node_sizes.append(500)  # Consistent size for all nodes

            # Draw nodes
            nx.draw_networkx_nodes(
                current_graph,
                current_pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.9,
            )

            # Add iteration info and stats
            total_cost = current_solution.calculate_cost(config)
            num_solo = len(current_solution.solo_nodes)
            num_groups = len(current_solution.group_owners)
            num_nodes = current_graph.number_of_nodes()
            num_edges = current_graph.number_of_edges()
            
            # Main title
            ax.set_title(f"{title} - {algorithm_name}\nIteration {frame}", 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Stats box
            stats_text = f"Nodes: {num_nodes} | Edges: {num_edges}\nCost: ${total_cost:.2f} | Solo: {num_solo} | Groups: {num_groups}"
            
            # Changes info
            if show_changes and frame > 0:
                prev_graph = graph_states[frame - 1]
                added_nodes = list(set(current_graph.nodes()) - set(prev_graph.nodes()))
                removed_nodes = list(set(prev_graph.nodes()) - set(current_graph.nodes()))
                added_edges = list(set(current_graph.edges()) - set(prev_graph.edges()))
                removed_edges = list(set(prev_graph.edges()) - set(current_graph.edges()))
                
                changes_text = f"Changes: "
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
                stats_text += f"\n{changes_text}"
            
            # Add text box with stats
            text_box = FancyBboxPatch((0.02, 0.02), 0.4, 0.15,
                                    boxstyle="round,pad=0.01",
                                    facecolor='white',
                                    edgecolor='black',
                                    alpha=0.9,
                                    transform=ax.transAxes)
            ax.add_patch(text_box)
            ax.text(0.03, 0.09, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.0))

            # Add legend - simplified without "new node/edge" indicators
            legend_elements = [
                plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=self.color_map["solo"], markersize=8, label="Solo"),
                plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=self.color_map["group_owner"], markersize=8, label="Group Member"),
                plt.Line2D([0], [0], color="#013865", linewidth=3, label="Group Connection"),
                plt.Line2D([0], [0], color="#666666", linewidth=1, linestyle="dashed", label="Other Connection"),
            ]
            
            ax.legend(handles=legend_elements, loc="upper right", 
                     bbox_to_anchor=(0.98, 0.98), framealpha=0.9)

        # Create animation
        frames = len(graph_states)
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=duration*1000, repeat=True, blit=False)
        
        # Save as GIF
        if save_path is None:
            save_path = f"results/dynamic/dynamic_{algorithm_name.lower()}_evolution.gif"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating GIF with {frames} frames...")
        try:
            # Try to save as GIF using pillow writer
            anim.save(save_path, writer='pillow', fps=1/duration, 
                     savefig_kwargs={'bbox_inches': 'tight', 'facecolor': 'white'})
            print(f"Dynamic GIF saved to: {save_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            # Fallback: save as MP4 if available
            try:
                mp4_path = save_path.replace('.gif', '.mp4')
                anim.save(mp4_path, writer='ffmpeg', fps=1/duration,
                         savefig_kwargs={'bbox_inches': 'tight', 'facecolor': 'white'})
                print(f"Saved as MP4 instead: {mp4_path}")
            except Exception as e2:
                print(f"Could not save as MP4 either: {e2}")
        
        plt.close(fig)
