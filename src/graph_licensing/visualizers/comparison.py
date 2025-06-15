"""Functions for comparing multiple solutions."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:
    from ..models.license import LicenseConfig, LicenseSolution

from .constants import DEFAULT_NODE_SIZE
from .renderers import render_single_solution, save_figure
from .utils import calculate_solution_stats


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
