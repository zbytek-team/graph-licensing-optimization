"""Functions for visualizing single solutions."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from ..models.license import LicenseConfig, LicenseSolution

from .constants import DEFAULT_FIGSIZE, DEFAULT_NODE_SIZE
from .renderers import render_single_solution, create_legend, save_figure
from .utils import calculate_solution_stats


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
