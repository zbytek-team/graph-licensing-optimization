import math
import random
from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class SimulatedAnnealingAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        seed: int | None = None,
    ) -> None:
        super().__init__("SimulatedAnnealing")
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.seed = seed

    def supports_warm_start(self) -> bool:
        return True

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        from ...algorithms.greedy import GreedyAlgorithm
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)
        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()
        if warm_start is not None:
            current_solution = self._adapt_solution_to_graph(warm_start, graph, config)
        else:
            greedy = GreedyAlgorithm()
            current_solution = greedy.solve(graph, config)
        current_cost = current_solution.calculate_cost(config)
        best_solution = current_solution
        best_cost = current_cost
        temperature = self.initial_temp
        for _iteration in range(self.max_iterations):
            if temperature < self.final_temp:
                break
            neighbor_solution = self._generate_neighbor(current_solution, graph, config)
            neighbor_cost = neighbor_solution.calculate_cost(config)
            cost_diff = neighbor_cost - current_cost
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
            temperature *= self.cooling_rate

        return best_solution

    def _adapt_solution_to_graph(
        self, solution: "LicenseSolution", graph: "nx.Graph", config: "LicenseConfig"
    ) -> "LicenseSolution":
        from ...models.license import LicenseSolution

        current_nodes = set(graph.nodes())
        new_licenses = {}
        assigned_nodes = set()
        for license_type, groups in solution.licenses.items():
            if license_type not in config.license_types:
                continue
            license_config = config.license_types[license_type]
            new_groups = {}
            for owner, members in groups.items():
                if owner in current_nodes:
                    valid_members = []
                    for member in members:
                        if member in current_nodes:
                            if member == owner or graph.has_edge(owner, member):
                                valid_members.append(member)
                    if valid_members and license_config.is_valid_size(len(valid_members)):
                        new_groups[owner] = valid_members
                        assigned_nodes.update(valid_members)
            if new_groups:
                new_licenses[license_type] = new_groups
        unassigned = current_nodes - assigned_nodes
        if unassigned:
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                license_type, _ = best_solo
                if license_type not in new_licenses:
                    new_licenses[license_type] = {}
                for node in unassigned:
                    new_licenses[license_type][node] = [node]
        return LicenseSolution(licenses=new_licenses)

    def _generate_neighbor(
        self,
        solution: "LicenseSolution",
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> "LicenseSolution":
        import copy

        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return solution
        new_licenses = copy.deepcopy(solution.licenses)
        node = random.choice(nodes)
        current_license_type = None
        current_owner = None
        current_members = None
        for license_type, groups in new_licenses.items():
            for owner, members in groups.items():
                if node in members:
                    current_license_type = license_type
                    current_owner = owner
                    current_members = members
                    break
            if current_license_type:
                break
        if current_license_type is None:
            return solution
        current_members.remove(node)
        if len(current_members) == 0:
            del new_licenses[current_license_type][current_owner]
            if len(new_licenses[current_license_type]) == 0:
                del new_licenses[current_license_type]
        elif current_owner == node:
            if current_members:
                new_owner = current_members[0]
                new_licenses[current_license_type][new_owner] = current_members
            del new_licenses[current_license_type][current_owner]
        modification_types = ["solo", "join_group", "create_group"]
        modification = random.choice(modification_types)
        if modification == "solo":
            best_license = config.get_best_license_for_size(1)
            if best_license:
                license_type, _ = best_license
                if license_type not in new_licenses:
                    new_licenses[license_type] = {}
                new_licenses[license_type][node] = [node]
        elif modification == "join_group":
            neighbors = list(graph.neighbors(node))
            potential_groups = []
            for license_type, groups in new_licenses.items():
                license_config = config.license_types[license_type]
                for owner, members in groups.items():
                    if owner in neighbors and len(members) < license_config.max_size:
                        potential_groups.append((license_type, owner))
            if potential_groups:
                license_type, owner = random.choice(potential_groups)
                new_licenses[license_type][owner].append(node)
            else:
                best_license = config.get_best_license_for_size(1)
                if best_license:
                    license_type, _ = best_license
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node]
        elif modification == "create_group":
            neighbors = [n for n in graph.neighbors(node) if n != node]
            if neighbors:
                neighbor = random.choice(neighbors)
                neighbor_license_type = None
                neighbor_owner = None
                neighbor_members = None
                for license_type, groups in new_licenses.items():
                    for owner, members in groups.items():
                        if neighbor in members:
                            neighbor_license_type = license_type
                            neighbor_owner = owner
                            neighbor_members = members
                            break
                    if neighbor_license_type:
                        break
                if neighbor_license_type:
                    neighbor_members.remove(neighbor)
                    if len(neighbor_members) == 0:
                        del new_licenses[neighbor_license_type][neighbor_owner]
                        if len(new_licenses[neighbor_license_type]) == 0:
                            del new_licenses[neighbor_license_type]
                    elif neighbor_owner == neighbor:
                        if neighbor_members:
                            new_owner = neighbor_members[0]
                            new_licenses[neighbor_license_type][new_owner] = neighbor_members
                        del new_licenses[neighbor_license_type][neighbor_owner]
                best_license = config.get_best_license_for_size(2)
                if best_license:
                    license_type, _ = best_license
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node, neighbor]
                else:
                    best_solo = config.get_best_license_for_size(1)
                    if best_solo:
                        license_type, _ = best_solo
                        if license_type not in new_licenses:
                            new_licenses[license_type] = {}
                        new_licenses[license_type][node] = [node]
                        new_licenses[license_type][neighbor] = [neighbor]
            else:
                best_license = config.get_best_license_for_size(1)
                if best_license:
                    license_type, _ = best_license
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node]
        return LicenseSolution(licenses=new_licenses)
