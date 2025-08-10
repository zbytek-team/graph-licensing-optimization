from src.core import LicenseType, Solution, LicenseGroup
from src.core import SolutionValidator
from src.utils import SolutionBuilder
from src.utils import MutationOperators
from typing import List, Tuple
import random
import networkx as nx


class GeneticOperators:
    def __init__(self, validator: SolutionValidator):
        self.validator = validator

    def tournament_selection(self, population: List[Solution], fitness_scores: List[float], tournament_size: int = 3) -> Solution:
        if not population:
            return population[0] if population else None
        tournament_size = min(tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def crossover(self, parent1: Solution, parent2: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Tuple[Solution, Solution]:
        all_groups = parent1.groups + parent2.groups
        child1_groups = self._select_groups_for_child(all_groups, graph, 0.7)
        child2_groups = self._select_groups_for_child(all_groups, graph, 0.3)

        child1 = self._repair_solution(child1_groups, graph, license_types)
        child2 = self._repair_solution(child2_groups, graph, license_types)

        if not self.validator.is_valid_solution(child1, graph):
            child1 = self._force_repair_solution(child1, graph, license_types)
        if not self.validator.is_valid_solution(child2, graph):
            child2 = self._force_repair_solution(child2, graph, license_types)

        return child1, child2

    def mutate(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        mutation_operators = [
            self._mutate_change_license,
            self._mutate_reassign_member,
            self._mutate_merge_groups,
            self._mutate_split_group,
        ]

        for _ in range(3):  # Try up to 3 times to get a valid mutation
            operator = random.choice(mutation_operators)
            mutated = operator(solution, graph, license_types)
            if mutated and self.validator.is_valid_solution(mutated, graph):
                return mutated

        # If no valid mutation found, return original solution
        return solution

    def intensive_mutate(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        mutated = solution
        num_mutations = random.randint(2, 4)

        for _ in range(num_mutations):
            mutation_operators = [
                self._mutate_change_license,
                self._mutate_reassign_member,
                self._mutate_merge_groups,
                self._mutate_split_group,
            ]

            for _ in range(2):  # Try up to 2 times per mutation
                operator = random.choice(mutation_operators)
                new_mutated = operator(mutated, graph, license_types)
                if new_mutated and self.validator.is_valid_solution(new_mutated, graph):
                    mutated = new_mutated
                    break

        return mutated

    def intensive_local_search(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        current = solution
        improved = True
        iterations = 0
        max_iterations = 10

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            license_optimized = self._local_search_license_optimization(current, graph, license_types)
            if license_optimized.total_cost < current.total_cost:
                current = license_optimized
                improved = True

            merge_optimized = self._local_search_group_merging(current, graph, license_types)
            if merge_optimized.total_cost < current.total_cost:
                current = merge_optimized
                improved = True

            reassignment_optimized = self._local_search_member_reassignment(current, graph, license_types)
            if reassignment_optimized.total_cost < current.total_cost:
                current = reassignment_optimized
                improved = True

        return current

    def _select_groups_for_child(self, all_groups: List[LicenseGroup], graph: nx.Graph, bias: float) -> List[LicenseGroup]:
        selected_groups = []
        covered_nodes = set()

        for group in all_groups:
            if not (group.all_members & covered_nodes):
                if random.random() < bias:
                    selected_groups.append(group)
                    covered_nodes.update(group.all_members)

        return selected_groups

    def _repair_solution(self, groups: List[LicenseGroup], graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        all_nodes = set(graph.nodes())
        covered_nodes = set()
        valid_groups = []

        for group in groups:
            if not (group.all_members & covered_nodes):
                valid_groups.append(group)
                covered_nodes.update(group.all_members)

        uncovered = all_nodes - covered_nodes

        while uncovered:
            node = random.choice(list(uncovered))
            neighbors = set(graph.neighbors(node)) & uncovered
            available = neighbors | {node}

            best_group = None
            best_efficiency = float("inf")

            for license_type in license_types:
                max_size = min(len(available), license_type.max_capacity)
                if max_size >= license_type.min_capacity:
                    efficiency = license_type.cost / max_size
                    if efficiency < best_efficiency:
                        best_efficiency = efficiency
                        members = list(available)[:max_size]
                        additional_members = set(members) - {node}
                        best_group = LicenseGroup(license_type, node, additional_members)

            if best_group:
                valid_groups.append(best_group)
                uncovered -= best_group.all_members
            else:
                cheapest = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                group = LicenseGroup(cheapest, node, set())
                valid_groups.append(group)
                uncovered.remove(node)

        return SolutionBuilder.create_solution_from_groups(valid_groups)

    def _force_repair_solution(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        all_nodes = set(graph.nodes())
        covered_nodes = set()
        valid_groups = []

        for group in solution.groups:
            owner = group.owner
            members = group.all_members

            valid_members = {owner}
            for member in members:
                if member != owner:
                    if graph.has_edge(owner, member) or member == owner:
                        valid_members.add(member)

            license_type = group.license_type
            if license_type.min_capacity <= len(valid_members) <= license_type.max_capacity:
                if not (valid_members & covered_nodes):
                    additional_members = valid_members - {owner}
                    repaired_group = LicenseGroup(license_type, owner, additional_members)
                    valid_groups.append(repaired_group)
                    covered_nodes.update(valid_members)

        uncovered = all_nodes - covered_nodes

        while uncovered:
            node = random.choice(list(uncovered))
            neighbors = {n for n in graph.neighbors(node) if n in uncovered}
            available = neighbors | {node}

            cheapest_license = min(license_types, key=lambda lt: lt.cost)
            max_size = min(len(available), cheapest_license.max_capacity)

            if max_size >= cheapest_license.min_capacity:
                members = list(available)[:max_size]
                additional_members = set(members) - {node}
                group = LicenseGroup(cheapest_license, node, additional_members)
            else:
                single_license = next((lt for lt in license_types if lt.min_capacity <= 1), cheapest_license)
                group = LicenseGroup(single_license, node, set())

            valid_groups.append(group)
            uncovered -= group.all_members

        return SolutionBuilder.create_solution_from_groups(valid_groups)

    def _mutate_change_license(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        return MutationOperators.change_license_type(solution, graph, license_types)

    def _mutate_reassign_member(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        return MutationOperators.reassign_member(solution, graph, license_types)

    def _mutate_merge_groups(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        return MutationOperators.merge_groups(solution, graph, license_types)

    def _mutate_split_group(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        return MutationOperators.split_group(solution, graph, license_types)

    def _local_search_license_optimization(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        best_solution = solution
        for i, group in enumerate(solution.groups):
            for license_type in license_types:
                if license_type != group.license_type:
                    if license_type.min_capacity <= len(group.all_members) <= license_type.max_capacity:
                        new_groups = solution.groups.copy()
                        new_group = LicenseGroup(license_type, group.owner, group.additional_members)
                        new_groups[i] = new_group
                        new_solution = SolutionBuilder.create_solution_from_groups(new_groups)
                        if new_solution.total_cost < best_solution.total_cost:
                            best_solution = new_solution
        return best_solution

    def _local_search_group_merging(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        best_solution = solution
        for i in range(len(solution.groups)):
            for j in range(i + 1, len(solution.groups)):
                group1, group2 = solution.groups[i], solution.groups[j]
                combined_members = group1.all_members | group2.all_members

                for license_type in license_types:
                    if license_type.min_capacity <= len(combined_members) <= license_type.max_capacity:
                        subgraph = graph.subgraph(combined_members)
                        if nx.is_connected(subgraph):
                            new_groups = [g for k, g in enumerate(solution.groups) if k not in [i, j]]
                            owner = random.choice(list(combined_members))
                            additional = combined_members - {owner}
                            merged_group = LicenseGroup(license_type, owner, additional)
                            new_groups.append(merged_group)
                            new_solution = SolutionBuilder.create_solution_from_groups(new_groups)
                            if new_solution.total_cost < best_solution.total_cost:
                                best_solution = new_solution
        return best_solution

    def _local_search_member_reassignment(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        best_solution = solution
        for i, group in enumerate(solution.groups):
            if len(group.additional_members) > 0:
                for member in list(group.additional_members):
                    for j, other_group in enumerate(solution.groups):
                        if i != j and member not in other_group.all_members:
                            neighbors_other = set(graph.neighbors(other_group.owner))
                            if member in neighbors_other:
                                new_other_members = other_group.all_members | {member}
                                for license_type in license_types:
                                    if license_type.min_capacity <= len(new_other_members) <= license_type.max_capacity:
                                        new_groups = solution.groups.copy()
                                        new_group1 = LicenseGroup(group.license_type, group.owner, group.additional_members - {member})
                                        new_group2 = LicenseGroup(license_type, other_group.owner, (other_group.additional_members | {member}))
                                        new_groups[i] = new_group1
                                        new_groups[j] = new_group2
                                        new_solution = SolutionBuilder.create_solution_from_groups(new_groups)
                                        if self.validator.is_valid_solution(new_solution, graph) and new_solution.total_cost < best_solution.total_cost:
                                            best_solution = new_solution
        return best_solution
