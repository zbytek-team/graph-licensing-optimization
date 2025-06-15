"""
Graph visualization module - main entry point for backward compatibility.

This module re-exports all visualization functions from their respective modules.
"""

# Import constants
from .constants import (
    COLOR_MAP,
    SIZE_MAP,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    DEFAULT_NODE_SIZE,
    DEFAULT_EDGE_WIDTH_UNASSIGNED,
    DEFAULT_EDGE_WIDTH_GROUP,
)

# Import utility functions
from .utils import (
    get_node_colors_and_sizes,
    get_edge_lists_and_colors,
    calculate_solution_stats,
    calculate_frame_changes,
    calculate_node_position,
)

# Import renderer functions
from .renderers import (
    draw_edges,
    create_legend,
    save_figure,
    render_single_solution,
)

# Import single solution visualization
from .single_solution import visualize_solution

# Import comparison functions
from .comparison import compare_solutions

# Import animation functions
from .animation import create_dynamic_gif

# Re-export all functions for backward compatibility
__all__ = [
    # Constants
    "COLOR_MAP",
    "SIZE_MAP", 
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
    "DEFAULT_NODE_SIZE",
    "DEFAULT_EDGE_WIDTH_UNASSIGNED",
    "DEFAULT_EDGE_WIDTH_GROUP",
    # Utility functions
    "get_node_colors_and_sizes",
    "get_edge_lists_and_colors",
    "calculate_solution_stats",
    "calculate_frame_changes",
    "calculate_node_position",
    # Renderer functions
    "draw_edges",
    "create_legend",
    "save_figure",
    "render_single_solution",
    # Main visualization functions
    "visualize_solution",
    "compare_solutions",
    "create_dynamic_gif",
]
