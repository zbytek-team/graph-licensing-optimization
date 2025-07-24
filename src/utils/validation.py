"""
Common validation utilities for graph licensing optimization algorithms.
"""

from typing import Set
import networkx as nx
from ..core.types import Solution


class SolutionValidator:
    """Centralized validation logic for solutions"""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def is_valid_solution(self, solution: Solution, graph: nx.Graph) -> bool:
        """
        Check if a solution is valid with optional debug output.

        Args:
            solution: The solution to validate
            graph: The graph being solved

        Returns:
            bool: True if solution is valid, False otherwise
        """
        try:
            all_nodes = set(graph.nodes())

            # Check if all nodes are covered exactly once
            covered = set()
            for group in solution.groups:
                if group.all_members & covered:  # Node overlap
                    if self.debug:
                        overlap = group.all_members & covered
                        print(f"    Validation failed: Node overlap in group with owner {group.owner}: {overlap}")
                    return False
                covered.update(group.all_members)

            if covered != all_nodes:
                if self.debug:
                    missing = all_nodes - covered
                    extra = covered - all_nodes
                    print(f"    Validation failed: Coverage mismatch. Missing: {missing}, Extra: {extra}")
                return False

            # Check group constraints
            for group in solution.groups:
                # Size constraints
                if not (group.license_type.min_capacity <= group.size <= group.license_type.max_capacity):
                    if self.debug:
                        print(
                            f"    Validation failed: Group with owner {group.owner} has size {group.size}, "
                            f"license {group.license_type.name} requires {group.license_type.min_capacity}-{group.license_type.max_capacity}"
                        )
                    return False

                # Connectivity constraint - each member must be connected to owner
                owner_neighbors = set(graph.neighbors(group.owner)) | {group.owner}
                if not group.all_members.issubset(owner_neighbors):
                    if self.debug:
                        disconnected = group.all_members - owner_neighbors
                        print(f"    Validation failed: Group {group.owner} has disconnected nodes: {disconnected}")
                    return False

            return True
        except Exception as e:
            if self.debug:
                print(f"    Validation failed: Exception {e}")
            return False

    def validate_solution_strict(self, solution: Solution, graph: nx.Graph, all_nodes: Set = None) -> bool:
        """
        Strict validation that raises exceptions on validation failures.
        Uses the same validation logic as is_valid_solution but raises exceptions.

        Args:
            solution: The solution to validate
            graph: The graph being solved
            all_nodes: Set of all nodes that should be covered (optional, defaults to graph nodes)

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        if all_nodes is None:
            all_nodes = set(graph.nodes())

        # Check if all nodes are covered
        if solution.covered_nodes != all_nodes:
            missing_nodes = all_nodes - solution.covered_nodes
            extra_nodes = solution.covered_nodes - all_nodes

            error_msg = "Node coverage mismatch."
            if missing_nodes:
                error_msg += f" Missing nodes: {missing_nodes}."
            if extra_nodes:
                error_msg += f" Extra nodes: {extra_nodes}."

            raise ValueError(error_msg)

        for group in solution.groups:
            owner_neighbors = set(graph.neighbors(group.owner)) | {group.owner}

            # Check if group is of correct size
            if not (group.license_type.min_capacity <= group.size <= group.license_type.max_capacity):
                raise ValueError(
                    f"Group with owner {group.owner} has size {group.size}, "
                    f"but license type '{group.license_type.name}' requires "
                    f"capacity between {group.license_type.min_capacity} and {group.license_type.max_capacity}"
                )

            # Check for valid group ownership - members must be neighbors of the owner
            invalid_members = group.additional_members - owner_neighbors
            if invalid_members:
                raise ValueError(f"Group with owner {group.owner} contains invalid members {invalid_members} that are not neighbors of the owner")

        # Check for overlapping groups
        all_covered = set()
        for group in solution.groups:
            overlapping_members = all_covered & group.all_members
            if overlapping_members:
                raise ValueError(f"Group with owner {group.owner} has overlapping members {overlapping_members} with previous groups")
            all_covered.update(group.all_members)

        return True
