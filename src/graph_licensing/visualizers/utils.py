"""Utility functions for graph visualization."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    from ..models.license import LicenseConfig, LicenseSolution

from .constants import COLOR_MAP, SIZE_MAP


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
