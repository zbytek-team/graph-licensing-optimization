import random
import math
from typing import Any, List, Optional
import networkx as nx
from src.core import Solution, LicenseGroup, LicenseType, Algorithm
from src.core import SolutionValidator
from src.utils import SolutionBuilder


class SimulatedAnnealing(Algorithm):
    @property
    def name(self) -> str:
        return "simulated_annealing"

    def __init__(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.999,
        min_temperature: float = 0.01,
        max_iterations: int = 30000,
        max_iterations_without_improvement: int = 3000,
    ):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.validator = SolutionValidator()

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        from .greedy import GreedyAlgorithm

        current_solution = GreedyAlgorithm().solve(graph, license_types)
        if not self.validator.is_valid_solution(current_solution, graph):
            return Solution([], 0, set())

        best_solution = current_solution
        temperature = self.initial_temperature
        iterations_without_improvement = 0

        for i in range(self.max_iterations):
            if temperature < self.min_temperature:
                break

            neighbor = self._generate_neighbor(current_solution, graph, license_types)

            if neighbor and self.validator.is_valid_solution(neighbor, graph):
                delta_cost = neighbor.total_cost - current_solution.total_cost
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                    current_solution = neighbor
                    if current_solution.total_cost < best_solution.total_cost:
                        best_solution = current_solution
                        iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1

            temperature *= self.cooling_rate

            if iterations_without_improvement >= self.max_iterations_without_improvement:
                temperature = best_solution.total_cost / 10.0
                iterations_without_improvement = 0

        return best_solution

    def _generate_neighbor(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        if not solution.groups:
            return None

        mutation_strategies = [
            self._change_license_type,
            self._move_node,
            self._swap_nodes,
            self._merge_groups,
            self._split_group,
        ]

        for _ in range(10):
            strategy = random.choice(mutation_strategies)
            new_solution = strategy(solution, graph, license_types)
            if new_solution:
                return new_solution
        return None

    def _change_license_type(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        group = random.choice(solution.groups)
        new_license = SolutionBuilder.find_cheapest_license_for_size(group.size, license_types)
        if not new_license or new_license.name == group.license_type.name:
            return None

        new_groups = [g for g in solution.groups if g != group]
        new_group = LicenseGroup(new_license, group.owner, group.additional_members)
        new_groups.append(new_group)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    def _move_node(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        source_candidates = [g for g in solution.groups if g.size > g.license_type.min_capacity]
        if not source_candidates:
            return None
        source_group = random.choice(source_candidates)
        node_to_move = random.choice(list(source_group.all_members))

        target_candidates = [g for g in solution.groups if g != source_group and g.size < g.license_type.max_capacity]
        if not target_candidates:
            return None
        target_group = random.choice(target_candidates)

        new_source_members = source_group.all_members - {node_to_move}
        new_target_members = target_group.all_members | {node_to_move}

        if not nx.is_connected(graph.subgraph(new_source_members)) or not nx.is_connected(graph.subgraph(new_target_members)):
            return None

        new_source_license = SolutionBuilder.find_cheapest_license_for_size(len(new_source_members), license_types)
        new_target_license = SolutionBuilder.find_cheapest_license_for_size(len(new_target_members), license_types)

        if not new_source_license or not new_target_license:
            return None

        new_groups = [g for g in solution.groups if g not in [source_group, target_group]]
        new_owner_source = next(iter(new_source_members))
        new_owner_target = next(iter(new_target_members))

        new_groups.append(LicenseGroup(new_source_license, new_owner_source, new_source_members - {new_owner_source}))
        new_groups.append(LicenseGroup(new_target_license, new_owner_target, new_target_members - {new_owner_target}))

        return SolutionBuilder.create_solution_from_groups(new_groups)

    def _swap_nodes(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        if len(solution.groups) < 2:
            return None
        group1, group2 = random.sample(solution.groups, 2)

        node1 = random.choice(list(group1.all_members))
        node2 = random.choice(list(group2.all_members))

        new_g1_members = (group1.all_members - {node1}) | {node2}
        new_g2_members = (group2.all_members - {node2}) | {node1}

        if not nx.is_connected(graph.subgraph(new_g1_members)) or not nx.is_connected(graph.subgraph(new_g2_members)):
            return None

        new_groups = [g for g in solution.groups if g not in [group1, group2]]
        owner1 = next(iter(new_g1_members))
        owner2 = next(iter(new_g2_members))
        
        license1 = SolutionBuilder.find_cheapest_license_for_size(len(new_g1_members), license_types)
        license2 = SolutionBuilder.find_cheapest_license_for_size(len(new_g2_members), license_types)

        if not license1 or not license2:
            return None

        new_groups.append(LicenseGroup(license1, owner1, new_g1_members - {owner1}))
        new_groups.append(LicenseGroup(license2, owner2, new_g2_members - {owner2}))

        return SolutionBuilder.create_solution_from_groups(new_groups)

    def _merge_groups(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        if len(solution.groups) < 2:
            return None
        group1, group2 = random.sample(solution.groups, 2)
        merged_nodes = group1.all_members | group2.all_members

        if not nx.is_connected(graph.subgraph(merged_nodes)):
            return None

        new_license = SolutionBuilder.find_cheapest_license_for_size(len(merged_nodes), license_types)
        if not new_license:
            return None

        new_groups = [g for g in solution.groups if g not in [group1, group2]]
        new_owner = next(iter(merged_nodes))
        new_groups.append(LicenseGroup(new_license, new_owner, merged_nodes - {new_owner}))
        return SolutionBuilder.create_solution_from_groups(new_groups)

    def _split_group(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        large_groups = [g for g in solution.groups if g.size > 2]
        if not large_groups:
            return None
        group_to_split = random.choice(large_groups)
        
        nodes = list(group_to_split.all_members)
        subgraph = graph.subgraph(nodes)
        
        if not nx.is_connected(subgraph):
             return None

        cut = nx.minimum_edge_cut(subgraph)
        if not cut:
            return None
            
        G_prime = subgraph.copy()
        G_prime.remove_edges_from(cut)
        components = list(nx.connected_components(G_prime))

        if len(components) < 2:
            return None

        new_groups = [g for g in solution.groups if g != group_to_split]
        for component in components:
            license = SolutionBuilder.find_cheapest_license_for_size(len(component), license_types)
            if not license:
                return None
            owner = next(iter(component))
            new_groups.append(LicenseGroup(license, owner, component - {owner}))
        
        return SolutionBuilder.create_solution_from_groups(new_groups)

        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.validator = SolutionValidator()

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        from .greedy import GreedyAlgorithm

        current_solution = GreedyAlgorithm().solve(graph, license_types)
        if not self.validator.is_valid_solution(current_solution, graph):
            return Solution([], 0, set())

        best_solution = current_solution
        temperature = self.initial_temperature
        iterations_without_improvement = 0

        for i in range(self.max_iterations):
            if temperature < self.min_temperature:
                break

            neighbor = self._generate_neighbor(current_solution, graph, license_types)

            if neighbor and self.validator.is_valid_solution(neighbor, graph):
                delta_cost = neighbor.total_cost - current_solution.total_cost
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                    current_solution = neighbor
                    if current_solution.total_cost < best_solution.total_cost:
                        best_solution = current_solution
                        iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1

            temperature *= self.cooling_rate

            if iterations_without_improvement >= self.max_iterations_without_improvement:
                temperature = self.initial_temperature * (1 - (i / self.max_iterations)) ** 2
                iterations_without_improvement = 0

        return best_solution

    def _generate_neighbor(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        if not solution.groups:
            return None

        mutation_strategies = [
            self._change_license_type,
            self._move_node,
            self._swap_nodes,
            self._merge_groups,
            self._split_group,
        ]

        for _ in range(10):
            strategy = random.choice(mutation_strategies)
            new_solution = strategy(solution, graph, license_types)
            if new_solution:
                return new_solution
        return None

    def _change_license_type(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        group = random.choice(solution.groups)
        new_license = SolutionBuilder.find_cheapest_license_for_size(group.size, license_types)
        if not new_license or new_license.name == group.license_type.name:
            return None

        new_groups = [g for g in solution.groups if g != group]
        new_group = LicenseGroup(new_license, group.owner, group.additional_members)
        new_groups.append(new_group)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    def _move_node(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        source_candidates = [g for g in solution.groups if g.size > g.license_type.min_capacity]
        if not source_candidates:
            return None
        source_group = random.choice(source_candidates)
        node_to_move = random.choice(list(source_group.all_members))

        target_candidates = [g for g in solution.groups if g != source_group and g.size < g.license_type.max_capacity]
        if not target_candidates:
            return None
        target_group = random.choice(target_candidates)

        new_source_members = source_group.all_members - {node_to_move}
        new_target_members = target_group.all_members | {node_to_move}

        if not nx.is_connected(graph.subgraph(new_source_members)) or not nx.is_connected(graph.subgraph(new_target_members)):
            return None

        new_source_license = SolutionBuilder.find_cheapest_license_for_size(len(new_source_members), license_types)
        new_target_license = SolutionBuilder.find_cheapest_license_for_size(len(new_target_members), license_types)

        if not new_source_license or not new_target_license:
            return None

        new_groups = [g for g in solution.groups if g not in [source_group, target_group]]
        new_owner_source = next(iter(new_source_members))
        new_owner_target = next(iter(new_target_members))

        new_groups.append(LicenseGroup(new_source_license, new_owner_source, new_source_members - {new_owner_source}))
        new_groups.append(LicenseGroup(new_target_license, new_owner_target, new_target_members - {new_owner_target}))

        return SolutionBuilder.create_solution_from_groups(new_groups)

    def _swap_nodes(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        if len(solution.groups) < 2:
            return None
        group1, group2 = random.sample(solution.groups, 2)

        node1 = random.choice(list(group1.all_members))
        node2 = random.choice(list(group2.all_members))

        new_g1_members = (group1.all_members - {node1}) | {node2}
        new_g2_members = (group2.all_members - {node2}) | {node1}

        if not nx.is_connected(graph.subgraph(new_g1_members)) or not nx.is_connected(graph.subgraph(new_g2_members)):
            return None

        new_groups = [g for g in solution.groups if g not in [group1, group2]]
        owner1 = next(iter(new_g1_members))
        owner2 = next(iter(new_g2_members))
        
        license1 = SolutionBuilder.find_cheapest_license_for_size(len(new_g1_members), license_types)
        license2 = SolutionBuilder.find_cheapest_license_for_size(len(new_g2_members), license_types)

        if not license1 or not license2:
            return None

        new_groups.append(LicenseGroup(license1, owner1, new_g1_members - {owner1}))
        new_groups.append(LicenseGroup(license2, owner2, new_g2_members - {owner2}))

        return SolutionBuilder.create_solution_from_groups(new_groups)

    def _merge_groups(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        if len(solution.groups) < 2:
            return None
        group1, group2 = random.sample(solution.groups, 2)
        merged_nodes = group1.all_members | group2.all_members

        if not nx.is_connected(graph.subgraph(merged_nodes)):
            return None

        new_license = SolutionBuilder.find_cheapest_license_for_size(len(merged_nodes), license_types)
        if not new_license:
            return None

        new_groups = [g for g in solution.groups if g not in [group1, group2]]
        new_owner = next(iter(merged_nodes))
        new_groups.append(LicenseGroup(new_license, new_owner, merged_nodes - {new_owner}))
        return SolutionBuilder.create_solution_from_groups(new_groups)

    def _split_group(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Optional[Solution]:
        large_groups = [g for g in solution.groups if g.size > 2]
        if not large_groups:
            return None
        group_to_split = random.choice(large_groups)
        
        nodes = list(group_to_split.all_members)
        subgraph = graph.subgraph(nodes)
        
        if not nx.is_connected(subgraph):
             return None

        cut = nx.minimum_edge_cut(subgraph)
        if not cut:
            return None
            
        G_prime = subgraph.copy()
        G_prime.remove_edges_from(cut)
        components = list(nx.connected_components(G_prime))

        if len(components) < 2:
            return None

        new_groups = [g for g in solution.groups if g != group_to_split]
        for component in components:
            license = SolutionBuilder.find_cheapest_license_for_size(len(component), license_types)
            if not license:
                return None
            owner = next(iter(component))
            new_groups.append(LicenseGroup(license, owner, component - {owner}))
        
        return SolutionBuilder.create_solution_from_groups(new_groups)
