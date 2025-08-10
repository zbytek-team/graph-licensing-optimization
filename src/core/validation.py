from typing import Set
import networkx as nx
from .models import Solution


class SolutionValidator:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def is_valid_solution(self, solution: Solution, graph: nx.Graph) -> bool:
        try:
            all_nodes = set(graph.nodes())
            covered = set()
            for group in solution.groups:
                if group.all_members & covered:
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
            for group in solution.groups:
                if not (group.license_type.min_capacity <= group.size <= group.license_type.max_capacity):
                    if self.debug:
                        print(
                            f"    Validation failed: Group with owner {group.owner} has size {group.size}, "
                            f"license {group.license_type.name} requires {group.license_type.min_capacity}-{group.license_type.max_capacity}"
                        )
                    return False
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
        if all_nodes is None:
            all_nodes = set(graph.nodes())
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
            if not (group.license_type.min_capacity <= group.size <= group.license_type.max_capacity):
                raise ValueError(
                    f"Group with owner {group.owner} has size {group.size}, "
                    f"but license type '{group.license_type.name}' requires "
                    f"capacity between {group.license_type.min_capacity} and {group.license_type.max_capacity}"
                )
            invalid_members = group.additional_members - owner_neighbors
            if invalid_members:
                raise ValueError(f"Group with owner {group.owner} contains invalid members {invalid_members} that are not neighbors of the owner")
        all_covered = set()
        for group in solution.groups:
            overlapping_members = all_covered & group.all_members
            if overlapping_members:
                raise ValueError(f"Group with owner {group.owner} has overlapping members {overlapping_members} with previous groups")
            all_covered.update(group.all_members)
        return True
