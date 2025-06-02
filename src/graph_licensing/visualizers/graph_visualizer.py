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
        self.color_map = {
            "solo": "#c20d31",
            "group_owner": "#013865", 
            "group_member": "#013865",
        }

        self.size_map = {
            "solo": 20,
            "group_owner": 40,
            "group_member": 20
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
            show: Whether to display the plot.
            figsize: Figure size as (width, height).
        """
        plt.figure(figsize=figsize)
        print("spring layouting")
        # Calculate layout
        pos = nx.spring_layout(graph, seed=42)
        # pos = nx.spring_layout(graph, seed=42, iterations=200, k=0.15, scale=2.0)
        # pos = nx.spectral_layout(graph)
        print("done springing")
        # Prepare node colors and labels
        node_colors = []
        node_sizes = []

        for node in graph.nodes():
            license_type = solution.get_node_license_type(node)
            node_colors.append(self.color_map[license_type.value])
            node_sizes.append(self.size_map[license_type.value])

        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=1,
        )

        # Draw regular edges
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color="gray",
            width=1,
            alpha=0.6,
            style="dashed"
        )

        # Draw group edges with different colors
        group_edges = []
        for owner, members in solution.group_owners.items():
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
            )

        # Draw labels
        # nx.draw_networkx_labels(graph, pos, node_labels, font_size=8)

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.color_map["solo"],
                markersize=10,
                label=f"Solo",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.color_map["group_owner"],
                markersize=10,
                label="Group Member",
            ),
            plt.Line2D([0], [0], color="#013865", linewidth=3, label="Group Connection"),
        ]
        # plt.legend(
        #     handles=legend_elements,
        #     loc="upper right",
        #     bbox_to_anchor=(1.0, 1.0),
        # )

        # Add cost information
        total_cost = solution.calculate_cost(config)
        num_solo = len(solution.solo_nodes)
        num_groups = len(solution.group_owners)
        cost_info = f"Total Cost: ${total_cost:.2f}\nSolo Licenses: {num_solo}\nGroup Licenses: {num_groups}"
        # plt.text(
        #     0.02,
        #     0.98,
        #     cost_info,
        #     size='large',
        #     transform=plt.gca().transAxes,
        #     verticalalignment="top",
        #     bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.2},
        # )

        plt.axis("off")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

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

            # Prepare node colors
            node_colors = [self.color_map[solution.get_node_license_type(node).value] for node in graph.nodes()]

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
                edge_color="lightgray",
                width=1,
                alpha=0.6,
                ax=ax,
            )

            # Highlight group‐member edges
            group_edges = []
            for owner, members in solution.group_owners.items():
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

            # Labels and title
            # nx.draw_networkx_labels(
            #     graph,
            #     pos,
            #     {n: str(n) for n in graph.nodes()},
            #     font_size=8,
            #     ax=ax,
            # )
            total_cost = solution.calculate_cost(config)
            num_solo = len(solution.solo_nodes)
            num_groups = len(solution.group_owners)
            ax.set_title(f"{algorithm_name}\nCost: ${total_cost:.2f}  |  Solo: {num_solo}  |  Groups: {num_groups}")
            ax.axis("off")

        # Turn off any unused subplots
        for ax in axes_list[n_solutions:]:
            ax.axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

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
