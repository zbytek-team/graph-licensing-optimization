import random
from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class RandomizedAlgorithm(BaseAlgorithm):
    def __init__(self, seed: int | None = None, greedy_probability: float = 0.3) -> None:
        super().__init__("Randomized")
        self.seed = seed
        self.greedy_probability = greedy_probability  # Probability of making greedy choice

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)
        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()
        unassigned = set(nodes)
        licenses = {}
        random.shuffle(nodes)

        for node in nodes:
            if node not in unassigned:
                continue
                
            # Sometimes make greedy choice, sometimes random
            if random.random() < self.greedy_probability:
                # Greedy choice: find best cost-effective assignment
                best_assignment = self._find_best_assignment(node, graph, config, unassigned)
                if best_assignment:
                    license_type, owner, members = best_assignment
                    if license_type not in licenses:
                        licenses[license_type] = {}
                    licenses[license_type][owner] = members
                    for member in members:
                        unassigned.discard(member)
                    continue
            
            # Random choice (original logic)
            available_license_types = list(config.license_types.keys())
            random.shuffle(available_license_types)
            assigned = False
            for license_type in available_license_types:
                license_config = config.license_types[license_type]
                max_size = min(license_config.max_size, len(unassigned))
                if max_size < license_config.min_size:
                    continue
                group_size = random.randint(license_config.min_size, max_size)
                if group_size == 1:
                    if license_type not in licenses:
                        licenses[license_type] = {}
                    licenses[license_type][node] = [node]
                    unassigned.discard(node)
                    assigned = True
                    break
                else:
                    available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]
                    if len(available_neighbors) >= group_size - 1:
                        selected_members = random.sample(available_neighbors, group_size - 1)
                        group_members = [node] + selected_members
                        if config.is_size_beneficial(license_type, group_size):
                            if license_type not in licenses:
                                licenses[license_type] = {}
                            licenses[license_type][node] = group_members
                            for member in group_members:
                                unassigned.discard(member)
                            assigned = True
                            break
            if not assigned:
                best_solo = config.get_best_license_for_size(1)
                if best_solo:
                    license_type, _ = best_solo
                    if license_type not in licenses:
                        licenses[license_type] = {}
                    licenses[license_type][node] = [node]
                    unassigned.discard(node)
        return LicenseSolution(licenses=licenses)

    def _find_best_assignment(self, node: int, graph: "nx.Graph", config: "LicenseConfig", unassigned: set) -> tuple | None:
        """Find the most cost-effective license assignment for a node."""
        best_cost_per_person = float("inf")
        best_assignment = None
        
        available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]
        
        for license_type, license_config in config.license_types.items():
            max_possible_size = min(license_config.max_size, len(available_neighbors) + 1, len(unassigned))
            
            for group_size in range(license_config.min_size, max_possible_size + 1):
                if group_size == 1:
                    cost_per_person = license_config.price
                    if cost_per_person < best_cost_per_person:
                        best_cost_per_person = cost_per_person
                        best_assignment = (license_type, node, [node])
                        
                elif group_size > 1 and available_neighbors:
                    # Select best neighbors based on degree (simple heuristic)
                    neighbors_by_degree = sorted(available_neighbors, key=lambda x: graph.degree(x), reverse=True)
                    selected_members = neighbors_by_degree[:group_size - 1]
                    
                    if len(selected_members) == group_size - 1:
                        members = [node] + selected_members
                        cost_per_person = license_config.price / len(members)
                        
                        if cost_per_person < best_cost_per_person:
                            best_cost_per_person = cost_per_person
                            best_assignment = (license_type, node, members)
        
        return best_assignment
