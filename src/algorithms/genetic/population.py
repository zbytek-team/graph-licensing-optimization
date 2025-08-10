from src.core import LicenseType, Solution, LicenseGroup
from ..greedy import GreedyAlgorithm
from src.core import SolutionValidator
from src.utils import SolutionBuilder
from typing import List
import random
import networkx as nx


class PopulationManager:
    def __init__(self, validator: SolutionValidator):
        self.validator = validator

    def initialize_population(self, graph: nx.Graph, license_types: List[LicenseType], population_size: int) -> List[Solution]:
        population = []
        greedy_solver = GreedyAlgorithm()
        greedy_solution = greedy_solver.solve(graph, license_types)
        population.append(greedy_solution)

        license_preference_count = max(1, population_size // 10)
        for _ in range(license_preference_count):
            biased_license_types = license_types.copy()
            random.shuffle(biased_license_types)
            solution = self._generate_biased_solution(graph, biased_license_types)
            if solution and self.validator.is_valid_solution(solution, graph):
                population.append(solution)

        clustering_count = max(1, population_size // 5)
        for _ in range(clustering_count):
            solution = self._generate_clustering_based_solution(graph, license_types)
            if solution and self.validator.is_valid_solution(solution, graph):
                population.append(solution)

        random_count = max(1, int(population_size * 0.3))
        for _ in range(random_count):
            solution = self._generate_random_order_solution(graph, license_types)
            if solution and self.validator.is_valid_solution(solution, graph):
                population.append(solution)

        while len(population) < population_size:
            solution = self._generate_completely_random_solution(graph, license_types)
            if solution and self.validator.is_valid_solution(solution, graph):
                population.append(solution)
            else:
                mutated = self._intensive_mutate_fallback(greedy_solution, graph, license_types)
                population.append(mutated)

        return population

    def generate_truly_random_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []
        max_attempts = len(nodes) * 2
        attempts = 0

        while uncovered and attempts < max_attempts:
            attempts += 1
            if not uncovered:
                break

            owner = random.choice(list(uncovered))
            owner_neighbors = set(graph.neighbors(owner)) | {owner}
            available = owner_neighbors & uncovered

            if not available:
                license_type = random.choice([lt for lt in license_types if lt.min_capacity <= 1])
                group = LicenseGroup(license_type, owner, set())
                groups.append(group)
                uncovered.remove(owner)
                continue

            random.shuffle(license_types)
            group_created = False

            for license_type in license_types:
                max_size = min(len(available), license_type.max_capacity)
                min_size = license_type.min_capacity

                if max_size >= min_size:
                    group_size = random.randint(min_size, max_size)
                    if random.random() < 0.5:
                        available_list = list(available)
                        available_list.sort(key=lambda n: graph.degree(n), reverse=True)
                        members = available_list[:group_size]
                    else:
                        members = random.sample(list(available), group_size)

                    additional_members = set(members) - {owner}
                    group = LicenseGroup(license_type, owner, additional_members)
                    groups.append(group)
                    uncovered -= set(members)
                    group_created = True
                    break

            if not group_created:
                license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                group = LicenseGroup(license_type, owner, set())
                groups.append(group)
                uncovered.remove(owner)

        while uncovered:
            node = uncovered.pop()
            license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
            group = LicenseGroup(license_type, node, set())
            groups.append(group)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _generate_biased_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        uncovered = set(nodes)
        groups = []
        preferred_license = license_types[0]

        while uncovered:
            owner = random.choice(list(uncovered))
            neighbors = set(graph.neighbors(owner)) & uncovered
            available = neighbors | {owner}

            if preferred_license.min_capacity <= len(available) <= preferred_license.max_capacity:
                group_size = min(len(available), preferred_license.max_capacity)
                members = random.sample(list(available), group_size)
                additional_members = set(members) - {owner}
                group = LicenseGroup(preferred_license, owner, additional_members)
                groups.append(group)
                uncovered -= set(members)
            else:
                for license_type in license_types:
                    max_size = min(len(available), license_type.max_capacity)
                    if max_size >= license_type.min_capacity:
                        group_size = random.randint(license_type.min_capacity, max_size)
                        members = random.sample(list(available), group_size)
                        additional_members = set(members) - {owner}
                        group = LicenseGroup(license_type, owner, additional_members)
                        groups.append(group)
                        uncovered -= set(members)
                        break
                else:
                    license_type = min(license_types, key=lambda lt: (lt.cost if lt.min_capacity <= 1 else float("inf")))
                    group = LicenseGroup(license_type, owner, set())
                    groups.append(group)
                    uncovered.remove(owner)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _generate_clustering_based_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []
        nodes_by_degree = sorted(nodes, key=lambda n: graph.degree(n), reverse=True)

        for owner in nodes_by_degree:
            if owner not in uncovered:
                continue

            neighbors = set(graph.neighbors(owner)) & uncovered
            available = neighbors | {owner}
            best_group = None
            best_efficiency = 0

            for license_type in license_types:
                max_size = min(len(available), license_type.max_capacity)
                if max_size >= license_type.min_capacity:
                    efficiency = max_size / license_type.cost
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        members = list(available)[:max_size]
                        additional_members = set(members) - {owner}
                        best_group = LicenseGroup(license_type, owner, additional_members)

            if best_group:
                groups.append(best_group)
                uncovered -= best_group.all_members

        return SolutionBuilder.create_solution_from_groups(groups)

    def _generate_random_order_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        uncovered = set(nodes)
        groups = []

        for owner in nodes:
            if owner not in uncovered:
                continue

            neighbors = set(graph.neighbors(owner)) & uncovered
            available = neighbors | {owner}
            license_type = random.choice(license_types)
            max_size = min(len(available), license_type.max_capacity)

            if max_size >= license_type.min_capacity:
                group_size = random.randint(license_type.min_capacity, max_size)
                members = random.sample(list(available), group_size)
                additional_members = set(members) - {owner}
                group = LicenseGroup(license_type, owner, additional_members)
                groups.append(group)
                uncovered -= set(members)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _intensive_mutate_fallback(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        from src.utils import MutationOperators

        mutated = solution
        for _ in range(2):
            mutated = MutationOperators.apply_random_mutation(mutated, graph, license_types)
            if not mutated:
                break
        return mutated or solution

    def _generate_completely_random_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []
        max_attempts = 1000
        attempts = 0

        while uncovered and attempts < max_attempts:
            attempts += 1
            owner = random.choice(list(uncovered))
            neighbors = set(graph.neighbors(owner)) & uncovered
            available = neighbors | {owner}
            license_type = random.choice(license_types)
            max_size = min(len(available), license_type.max_capacity)

            if max_size >= license_type.min_capacity:
                group_size = random.randint(license_type.min_capacity, max_size)
                members = random.sample(list(available), group_size)
                additional_members = set(members) - {owner}
                group = LicenseGroup(license_type, owner, additional_members)
                groups.append(group)
                uncovered -= set(members)

        while uncovered:
            node = uncovered.pop()
            license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
            group = LicenseGroup(license_type, node, set())
            groups.append(group)

        return SolutionBuilder.create_solution_from_groups(groups)
