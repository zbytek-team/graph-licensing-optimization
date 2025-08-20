import random
from typing import Dict, List, Tuple, Any
import networkx as nx
from src.core import Solution, LicenseGroup, LicenseType, Algorithm
from src.core import SolutionValidator
from src.utils import SolutionBuilder


class AntColonyOptimization(Algorithm):
    @property
    def name(self) -> str:
        return "ant_colony_optimization"

    def __init__(self, alpha=1.0, beta=2.0, evaporation_rate=0.5, q0=0.9, num_ants=20, max_iterations=100):
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q0 = q0
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.validator = SolutionValidator()

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        pheromones = self._initialize_pheromones(graph, license_types)
        heuristics = self._calculate_heuristics(graph, license_types)

        # Initialize with random solution instead of greedy
        best_solution = self._generate_random_initial_solution(graph, license_types)
        best_cost = best_solution.total_cost

        self._deposit_pheromones_for_solution(pheromones, best_solution, graph)

        no_improvement_streak = 0
        max_no_improvement = 20

        for _ in range(self.max_iterations):
            current_best_cost = best_cost

            for _ in range(self.num_ants):
                solution = self._construct_ant_solution(graph, license_types, pheromones, heuristics)
                if self.validator.is_valid_solution(solution, graph):
                    solution = self._local_search(solution, graph, license_types)
                    if solution.total_cost < best_cost:
                        best_solution = solution
                        best_cost = solution.total_cost

            self._update_pheromones(pheromones, best_solution, graph)

            if best_cost < current_best_cost:
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1

            if no_improvement_streak >= max_no_improvement:
                break

        return best_solution

    def _initialize_pheromones(self, graph: nx.Graph, license_types: List[LicenseType]) -> Dict[Tuple, float]:
        pheromones = {}
        initial_pheromone = 1.0
        for node in graph.nodes():
            for lt in license_types:
                pheromones[(node, lt.name)] = initial_pheromone
        for u, v in graph.edges():
            for lt in license_types:
                pheromones[tuple(sorted((u, v))) + (lt.name,)] = initial_pheromone
        return pheromones

    def _calculate_heuristics(self, graph: nx.Graph, license_types: List[LicenseType]) -> Dict[Tuple, float]:
        heuristics = {}
        for node in graph.nodes():
            degree = graph.degree(node)
            for lt in license_types:
                efficiency = (lt.max_capacity / lt.cost) if lt.cost > 0 else float("inf")
                heuristics[(node, lt.name)] = efficiency * (1 + degree / 10.0)
        for u, v in graph.edges():
            for lt in license_types:
                benefit = 1.0 / lt.cost if lt.cost > 0 else float("inf")
                heuristics[tuple(sorted((u, v))) + (lt.name,)] = benefit
        return heuristics

    def _construct_ant_solution(self, graph: nx.Graph, license_types: List[LicenseType], pheromones: Dict, heuristics: Dict) -> Solution:
        uncovered = set(graph.nodes())
        groups = []
        while uncovered:
            start_node = self._select_node(uncovered, license_types, pheromones, heuristics)
            if start_node is None:
                start_node = list(uncovered)[0]

            license_type = self._select_license(start_node, license_types, pheromones, heuristics)

            group_nodes = {start_node}
            candidates = list(set(graph.neighbors(start_node)) & uncovered - {start_node})

            while len(group_nodes) < license_type.max_capacity and candidates:
                next_member = self._select_member(start_node, candidates, license_type, pheromones, heuristics)
                if next_member is None:
                    break
                group_nodes.add(next_member)
                candidates.remove(next_member)

            if len(group_nodes) >= license_type.min_capacity:
                groups.append(LicenseGroup(license_type, start_node, group_nodes - {start_node}))
                uncovered -= group_nodes
            else:
                cheapest_single = SolutionBuilder.find_cheapest_single_license(license_types)
                groups.append(LicenseGroup(cheapest_single, start_node, set()))
                uncovered.remove(start_node)
        return SolutionBuilder.create_solution_from_groups(groups)

    def _select_component(self, choices: List[Any], probabilities: Dict[Any, float]) -> Any:
        if not choices:
            return None

        if random.random() < self.q0:
            max_prob = -1
            best_choice = None
            for choice in choices:
                if probabilities.get(choice, 0) > max_prob:
                    max_prob = probabilities[choice]
                    best_choice = choice
            return best_choice
        else:
            total_prob = sum(probabilities.values())
            if total_prob == 0:
                return random.choice(choices) if choices else None

            r = random.uniform(0, total_prob)
            upto = 0
            for choice in choices:
                upto += probabilities.get(choice, 0)
                if upto >= r:
                    return choice
            return random.choice(choices) if choices else None

    def _get_choice_probabilities(self, choices: List[Any], pheromones: Dict, heuristics: Dict, key_func) -> Dict[Any, float]:
        probabilities = {}
        for choice in choices:
            key = key_func(choice)
            tau = pheromones.get(key, 1.0)
            eta = heuristics.get(key, 1.0)
            probabilities[choice] = (tau**self.alpha) * (eta**self.beta)
        return probabilities

    def _select_node(self, uncovered: set, license_types: List[LicenseType], pheromones: Dict, heuristics: Dict) -> Any:
        nodes = list(uncovered)
        probabilities = {}
        for node in nodes:
            avg_prob = (
                sum((pheromones.get((node, lt.name), 1.0) ** self.alpha) * (heuristics.get((node, lt.name), 1.0) ** self.beta) for lt in license_types)
                / len(license_types)
                if license_types
                else 0
            )
            probabilities[node] = avg_prob
        return self._select_component(nodes, probabilities)

    def _select_license(self, node: Any, license_types: List[LicenseType], pheromones: Dict, heuristics: Dict) -> LicenseType:
        probabilities = self._get_choice_probabilities(license_types, pheromones, heuristics, lambda lt: (node, lt.name))
        return self._select_component(license_types, probabilities)

    def _select_member(self, owner: Any, candidates: List[Any], license_type: LicenseType, pheromones: Dict, heuristics: Dict) -> Any:
        probabilities = self._get_choice_probabilities(candidates, pheromones, heuristics, lambda c: tuple(sorted((owner, c))) + (license_type.name,))
        return self._select_component(candidates, probabilities)

    def _update_pheromones(self, pheromones: Dict, best_solution: Solution, graph: nx.Graph):
        for key in pheromones:
            pheromones[key] *= 1 - self.evaporation_rate

        self._deposit_pheromones_for_solution(pheromones, best_solution, graph)

    def _deposit_pheromones_for_solution(self, pheromones: Dict, solution: Solution, graph: nx.Graph):
        if solution.total_cost == 0:
            return
        quality = 1.0 / solution.total_cost
        for group in solution.groups:
            for node in group.all_members:
                key = (node, group.license_type.name)
                if key in pheromones:
                    pheromones[key] += quality

            members = list(group.all_members)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if graph.has_edge(members[i], members[j]):
                        key = tuple(sorted((members[i], members[j]))) + (group.license_type.name,)
                        if key in pheromones:
                            pheromones[key] += quality

    def _local_search(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        current_solution = solution
        for _ in range(3):
            improved = False
            groups = list(current_solution.groups)

            for i, group in enumerate(groups):
                sorted_licenses = sorted(license_types, key=lambda lt: lt.cost)
                for new_license_type in sorted_licenses:
                    if (
                        new_license_type.name != group.license_type.name
                        and group.size >= new_license_type.min_capacity
                        and group.size <= new_license_type.max_capacity
                        and new_license_type.cost < group.license_type.cost
                    ):
                        groups[i] = LicenseGroup(new_license_type, group.owner, group.additional_members)
                        current_solution = SolutionBuilder.create_solution_from_groups(groups)
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return current_solution

    def _generate_random_initial_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        nodes = list(graph.nodes())
        random.shuffle(nodes)

        covered_nodes = set()
        license_groups = []

        while len(covered_nodes) < len(nodes):
            uncovered = [n for n in nodes if n not in covered_nodes]
            if not uncovered:
                break

            center = random.choice(uncovered)
            license_type = random.choice(license_types)

            # License covers owner + direct neighbors (range = 1)
            neighbors = set(graph.neighbors(center)) | {center}

            additional = neighbors - {center}
            license_groups.append(LicenseGroup(license_type, center, additional))
            covered_nodes.update(neighbors)

        return SolutionBuilder.create_solution_from_groups(license_groups)
