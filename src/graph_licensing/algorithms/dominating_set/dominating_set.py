from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class DominatingSetAlgorithm(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__("DominatingSet")

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        import networkx  # noqa: F401

        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        # Get dominating set as starting point
        dominating_set = self._cost_aware_dominating_set(graph, config)

        unassigned = set(nodes)
        licenses = {}

        # Process dominating set nodes first
        for dominator in dominating_set:
            if dominator not in unassigned:
                continue

            available_neighbors = [n for n in graph.neighbors(dominator) if n in unassigned]
            best_assignment = self._find_best_cost_assignment(dominator, available_neighbors, config, unassigned)

            if best_assignment:
                license_type, members = best_assignment
                if license_type not in licenses:
                    licenses[license_type] = {}
                licenses[license_type][dominator] = members

                for member in members:
                    unassigned.discard(member)
            else:
                # Fallback to solo license
                best_solo = config.get_best_license_for_size(1)
                if best_solo:
                    license_type, _ = best_solo
                    if license_type not in licenses:
                        licenses[license_type] = {}
                    licenses[license_type][dominator] = [dominator]
                    unassigned.discard(dominator)

        # Handle remaining unassigned nodes with greedy approach
        remaining_nodes = sorted(unassigned, key=lambda x: graph.degree(x), reverse=True)
        for node in remaining_nodes:
            if node not in unassigned:
                continue
                
            available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]
            best_assignment = self._find_best_cost_assignment(node, available_neighbors, config, unassigned)
            
            if best_assignment:
                license_type, members = best_assignment
                if license_type not in licenses:
                    licenses[license_type] = {}
                licenses[license_type][node] = members

                for member in members:
                    unassigned.discard(member)

        # Assign solo licenses to any remaining nodes
        if unassigned:
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                license_type, _ = best_solo
                if license_type not in licenses:
                    licenses[license_type] = {}
                for node in unassigned:
                    licenses[license_type][node] = [node]

        return LicenseSolution(licenses=licenses)

    def _cost_aware_dominating_set(self, graph: "nx.Graph", config: "LicenseConfig") -> list[int]:
        """Create dominating set considering cost efficiency, not just coverage."""
        dominating_set = []
        uncovered = set(graph.nodes())

        while uncovered:
            best_node = None
            best_score = float("-inf")

            for node in uncovered:
                # Calculate coverage
                coverage = 1 if node in uncovered else 0
                coverage += sum(1 for neighbor in graph.neighbors(node) if neighbor in uncovered)
                
                # Calculate potential cost efficiency
                neighbors_in_uncovered = [n for n in graph.neighbors(node) if n in uncovered]
                max_group_size = min(len(neighbors_in_uncovered) + 1, 6)  # Assume max group size 6
                
                # Find best cost per person for this node
                best_cost_per_person = float("inf")
                for license_type, license_config in config.license_types.items():
                    for size in range(1, min(max_group_size + 1, license_config.max_size + 1)):
                        if license_config.is_valid_size(size):
                            cost_per_person = license_config.cost_per_person(size)
                            best_cost_per_person = min(best_cost_per_person, cost_per_person)
                
                # Score combines coverage with cost efficiency
                score = coverage / best_cost_per_person if best_cost_per_person > 0 else coverage

                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node is not None:
                dominating_set.append(best_node)
                uncovered.discard(best_node)
                for neighbor in graph.neighbors(best_node):
                    uncovered.discard(neighbor)

        return dominating_set

    def _find_best_cost_assignment(self, node: int, available_neighbors: list, config: "LicenseConfig", unassigned: set) -> tuple | None:
        """Find the most cost-effective license assignment."""
        best_cost_per_person = float("inf")
        best_assignment = None

        for license_type, license_config in config.license_types.items():
            max_possible_size = min(len(available_neighbors) + 1, license_config.max_size, len(unassigned))

            for group_size in range(license_config.min_size, max_possible_size + 1):
                if group_size == 1:
                    cost_per_person = license_config.price
                    if cost_per_person < best_cost_per_person:
                        best_cost_per_person = cost_per_person
                        best_assignment = (license_type, [node])

                elif group_size > 1 and available_neighbors:
                    # Select best neighbors by degree (connectivity heuristic)
                    sorted_neighbors = sorted(available_neighbors, 
                                            key=lambda x: len([n for n in available_neighbors if n in unassigned]), 
                                            reverse=True)
                    selected_members = sorted_neighbors[:group_size - 1]
                    
                    if len(selected_members) == group_size - 1:
                        members = [node] + selected_members
                        cost_per_person = license_config.price / len(members)
                        
                        if cost_per_person < best_cost_per_person:
                            best_cost_per_person = cost_per_person
                            best_assignment = (license_type, members)

        return best_assignment
