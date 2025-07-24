"""
Common solution manipulation utilities for graph licensing optimization algorithms.
"""

from typing import List, Set, Any
import networkx as nx
from ..core.types import Solution, LicenseGroup, LicenseType


class SolutionBuilder:
    """Utility class for creating and manipulating solutions"""

    @staticmethod
    def create_solution_from_groups(groups: List[LicenseGroup]) -> Solution:
        """
        Create a Solution object from a list of LicenseGroups.

        Args:
            groups: List of license groups

        Returns:
            Solution: Complete solution with calculated cost and coverage
        """
        total_cost = sum(group.license_type.cost for group in groups)
        covered_nodes = set()
        for group in groups:
            covered_nodes.update(group.all_members)

        return Solution(groups, total_cost, covered_nodes)

    @staticmethod
    def get_compatible_license_types(group_size: int, license_types: List[LicenseType], exclude: LicenseType = None) -> List[LicenseType]:
        """
        Get license types that are compatible with a given group size.

        Args:
            group_size: Size of the group
            license_types: Available license types
            exclude: License type to exclude from results

        Returns:
            List[LicenseType]: Compatible license types
        """
        compatible = []
        for lt in license_types:
            if lt != exclude and lt.min_capacity <= group_size <= lt.max_capacity:
                compatible.append(lt)
        return compatible

    @staticmethod
    def find_cheapest_single_license(license_types: List[LicenseType]) -> LicenseType:
        """
        Find the cheapest license type that can accommodate a single node.

        Args:
            license_types: Available license types

        Returns:
            LicenseType: Cheapest single-node license
        """
        single_licenses = [lt for lt in license_types if lt.min_capacity <= 1]
        if single_licenses:
            return min(single_licenses, key=lambda lt: lt.cost)
        else:
            return min(license_types, key=lambda lt: lt.cost)

    @staticmethod
    def get_owner_neighbors_with_self(graph: nx.Graph, owner: Any) -> Set[Any]:
        """
        Get all neighbors of a node including the node itself.

        Args:
            graph: The graph
            owner: The node

        Returns:
            Set[Any]: Neighbors + self
        """
        return set(graph.neighbors(owner)) | {owner}

    @staticmethod
    def can_merge_groups(group1: LicenseGroup, group2: LicenseGroup, graph: nx.Graph, license_types: List[LicenseType]) -> bool:
        """
        Check if two groups can be merged while maintaining connectivity.

        Args:
            group1: First group
            group2: Second group
            graph: The graph
            license_types: Available license types

        Returns:
            bool: True if groups can be merged
        """
        all_members = group1.all_members | group2.all_members

        # Check if any license type can accommodate the merged group
        compatible_licenses = SolutionBuilder.get_compatible_license_types(len(all_members), license_types)

        if not compatible_licenses:
            return False

        # Check connectivity for both potential owners
        for potential_owner in [group1.owner, group2.owner]:
            owner_neighbors = SolutionBuilder.get_owner_neighbors_with_self(graph, potential_owner)
            if all_members.issubset(owner_neighbors):
                return True

        return False

    @staticmethod
    def merge_groups(group1: LicenseGroup, group2: LicenseGroup, graph: nx.Graph, license_types: List[LicenseType]) -> LicenseGroup:
        """
        Merge two groups into one, choosing the best owner and license type.

        Args:
            group1: First group
            group2: Second group
            graph: The graph
            license_types: Available license types

        Returns:
            LicenseGroup: Merged group, or None if merge is not possible
        """
        all_members = group1.all_members | group2.all_members

        compatible_licenses = SolutionBuilder.get_compatible_license_types(len(all_members), license_types)

        if not compatible_licenses:
            return None

        # Try each potential owner
        for potential_owner in [group1.owner, group2.owner]:
            owner_neighbors = SolutionBuilder.get_owner_neighbors_with_self(graph, potential_owner)
            if all_members.issubset(owner_neighbors):
                # Choose cheapest compatible license
                best_license = min(compatible_licenses, key=lambda lt: lt.cost)
                additional_members = all_members - {potential_owner}
                return LicenseGroup(best_license, potential_owner, additional_members)

        return None


class RandomSolutionGenerator:
    """Utility for generating random solutions with different strategies"""

    @staticmethod
    def generate_random_greedy_solution(graph: nx.Graph, license_types: List[LicenseType], strategy: int = 0) -> Solution:
        """
        Generate a solution using randomized greedy approach.

        Args:
            graph: The graph to solve
            license_types: Available license types
            strategy: Strategy number for different randomization approaches

        Returns:
            Solution: Generated solution
        """
        import random

        nodes = list(graph.nodes())
        uncovered_nodes = set(nodes)
        groups = []

        # Different node ordering strategies
        if strategy % 3 == 0:
            # Strategy 0: Random order
            random.shuffle(nodes)
        elif strategy % 3 == 1:
            # Strategy 1: Degree-based order (high degree first)
            nodes.sort(key=lambda n: graph.degree(n), reverse=True)
            # Add some randomness
            for i in range(0, len(nodes), 3):
                end_idx = min(i + 3, len(nodes))
                sublist = nodes[i:end_idx]
                random.shuffle(sublist)
                nodes[i:end_idx] = sublist
        else:
            # Strategy 2: Low degree first with randomness
            nodes.sort(key=lambda n: graph.degree(n))
            for i in range(0, len(nodes), 3):
                end_idx = min(i + 3, len(nodes))
                sublist = nodes[i:end_idx]
                random.shuffle(sublist)
                nodes[i:end_idx] = sublist

        for node in nodes:
            if node not in uncovered_nodes:
                continue

            neighbors = SolutionBuilder.get_owner_neighbors_with_self(graph, node)
            available_neighbors = neighbors & uncovered_nodes

            if not available_neighbors:
                continue

            # License type selection strategy
            if strategy % 2 == 0:
                # Prefer cheaper licenses
                sorted_licenses = sorted(license_types, key=lambda lt: lt.cost)
            else:
                # Random license selection
                sorted_licenses = license_types.copy()
                random.shuffle(sorted_licenses)

            group_created = False
            for license_type in sorted_licenses:
                potential_members = list(available_neighbors)
                max_members = min(len(potential_members), license_type.max_capacity)

                if max_members < license_type.min_capacity:
                    continue

                # Member selection strategy
                if strategy % 4 == 0:
                    # Random selection
                    random.shuffle(potential_members)
                    group_members = set(potential_members[:max_members])
                elif strategy % 4 == 1:
                    # Prefer high degree neighbors
                    potential_members.sort(key=lambda n: graph.degree(n), reverse=True)
                    group_members = set(potential_members[:max_members])
                elif strategy % 4 == 2:
                    # Prefer low degree neighbors
                    potential_members.sort(key=lambda n: graph.degree(n))
                    group_members = set(potential_members[:max_members])
                else:
                    # Mixed strategy
                    random.shuffle(potential_members)
                    mid_point = max_members // 2
                    group_members = set(potential_members[:mid_point])
                    if mid_point < max_members:
                        remaining = [n for n in potential_members[mid_point:] if n not in group_members]
                        if remaining:
                            group_members.update(random.sample(remaining, min(len(remaining), max_members - len(group_members))))

                additional_members = group_members - {node}
                group = LicenseGroup(license_type, node, additional_members)
                groups.append(group)
                uncovered_nodes -= group_members
                group_created = True
                break

            if not group_created and node in uncovered_nodes:
                # Create single node group as fallback
                license_type = SolutionBuilder.find_cheapest_single_license(license_types)
                group = LicenseGroup(license_type, node, set())
                groups.append(group)
                uncovered_nodes.remove(node)

        return SolutionBuilder.create_solution_from_groups(groups)
