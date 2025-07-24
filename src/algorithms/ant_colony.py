"""
Ant Colony Optim    def __init__(self, alpha=1.0, beta=2.0, evaporation_rate=0.5, num_ants=20, max_iterations=100):
        self.alpha = alpha  # Pheromone importance
        self.beta = beta   # Heuristic importance
        self.evaporation_rate = evaporation_rate
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.validator = SolutionValidator()
"""

import random
from typing import Dict, List, Tuple, Any
import networkx as nx

from ..core.types import Solution, LicenseGroup, LicenseType, Algorithm
from ..utils.validation import SolutionValidator
from ..utils.solution_utils import SolutionBuilder


class AntColonyOptimization(Algorithm):
    """Ant Colony Optimization for license assignment"""

    @property
    def name(self) -> str:
        return "ant_colony_optimization"

    def __init__(self, alpha=1.0, beta=2.0, evaporation_rate=0.5, num_ants=20, max_iterations=100):
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic importance
        self.evaporation_rate = evaporation_rate
        self.num_ants = num_ants
        self.max_iterations = max_iterations

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        """Main ACO algorithm"""
        # Initialize utilities
        self.validator = SolutionValidator()

        pheromones = self._initialize_pheromones(graph, license_types)
        heuristics = self._calculate_heuristics(graph, license_types)

        best_solution = None
        best_cost = float("inf")

        for iteration in range(self.max_iterations):
            iteration_solutions = []

            # Generate solutions with all ants
            for ant in range(self.num_ants):
                try:
                    solution = self._construct_ant_solution(graph, license_types, pheromones, heuristics)

                    if solution and self.validator.is_valid_solution(solution, graph):
                        iteration_solutions.append(solution)

                        if solution.total_cost < best_cost:
                            best_solution = solution
                            best_cost = solution.total_cost
                except Exception:
                    continue

            # Update pheromones
            self._update_pheromones(pheromones, iteration_solutions)

        if best_solution is None:
            # Fallback to greedy solution
            from .greedy import GreedyAlgorithm

            greedy = GreedyAlgorithm()
            return greedy.solve(graph, license_types)

        return best_solution

    def _initialize_pheromones(self, graph: nx.Graph, license_types: List[LicenseType]) -> Dict[Tuple, float]:
        """Initialize pheromone trails"""
        pheromones = {}
        initial_pheromone = 1.0

        # Pheromone for node-license type combinations
        for node in graph.nodes():
            for license_type in license_types:
                key = (node, license_type.name)
                pheromones[key] = initial_pheromone

        # Pheromone trails for grouping connected nodes
        for edge in graph.edges():
            for license_type in license_types:
                key = tuple(sorted(edge) + [license_type.name])
                pheromones[key] = initial_pheromone

        return pheromones

    def _calculate_heuristics(self, graph: nx.Graph, license_types: List[LicenseType]) -> Dict[Tuple, float]:
        """Calculate heuristic information"""
        heuristics = {}

        # Heuristic for node-license combinations
        for node in graph.nodes():
            node_degree = graph.degree(node)
            for license_type in license_types:
                efficiency = license_type.max_capacity / license_type.cost
                degree_bonus = 1.0 + (node_degree / 10.0)
                heuristics[(node, license_type.name)] = efficiency * degree_bonus

        # Heuristic for grouping nodes
        for edge in graph.edges():
            for license_type in license_types:
                benefit = 1.0 / license_type.cost
                key = tuple(sorted(edge) + [license_type.name])
                heuristics[key] = benefit

        return heuristics

    def _construct_ant_solution(
        self, graph: nx.Graph, license_types: List[LicenseType], pheromones: Dict[Tuple, float], heuristics: Dict[Tuple, float]
    ) -> Solution:
        """Construct solution by building connected groups"""
        uncovered = set(graph.nodes())
        groups = []

        while uncovered:
            # Start new group with random uncovered node
            start_node = random.choice(list(uncovered))

            # Choose license type probabilistically
            license_type = self._choose_license_type(start_node, license_types, pheromones, heuristics)

            # Build connected group around this node
            group_nodes = self._build_connected_group(start_node, license_type, uncovered, graph, pheromones, heuristics)

            if len(group_nodes) >= license_type.min_capacity:
                # Create valid group
                owner = start_node
                additional = group_nodes - {owner}
                group = LicenseGroup(license_type, owner, additional)
                groups.append(group)
                uncovered -= group_nodes
            else:
                # Single node with smallest valid license
                license_type = SolutionBuilder.find_cheapest_single_license(license_types)
                group = LicenseGroup(license_type, start_node, set())
                groups.append(group)
                uncovered.remove(start_node)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _choose_license_type(self, node: Any, license_types: List[LicenseType], pheromones: Dict[Tuple, float], heuristics: Dict[Tuple, float]) -> LicenseType:
        """Choose license type probabilistically"""
        probabilities = {}

        for license_type in license_types:
            pheromone_key = (node, license_type.name)
            heuristic_key = (node, license_type.name)

            tau = pheromones.get(pheromone_key, 1.0)
            eta = heuristics.get(heuristic_key, 1.0)

            probabilities[license_type] = (tau**self.alpha) * (eta**self.beta)

        # Normalize and select
        total = sum(probabilities.values())
        if total == 0:
            return random.choice(license_types)

        r = random.random() * total
        cumulative = 0.0
        for license_type, prob in probabilities.items():
            cumulative += prob
            if r <= cumulative:
                return license_type

        return random.choice(license_types)

    def _build_connected_group(
        self, start_node: Any, license_type: LicenseType, uncovered: set, graph: nx.Graph, pheromones: Dict[Tuple, float], heuristics: Dict[Tuple, float]
    ) -> set:
        """Build a group where all members are direct neighbors of the owner"""
        group = {start_node}
        owner = start_node

        # Get all uncovered direct neighbors of the owner
        owner_neighbors = set(graph.neighbors(owner)) & uncovered
        candidates = list(owner_neighbors - group)

        while len(group) < license_type.max_capacity and candidates:
            # Choose next node to add probabilistically from direct neighbors only
            probabilities = {}
            for candidate in candidates:
                # Use pheromone from edge between owner and candidate
                edge_key = tuple(sorted([owner, candidate]) + [license_type.name])
                pheromone = pheromones.get(edge_key, 1.0)

                # Heuristic: prefer nodes with higher connectivity within remaining uncovered
                eta = len(set(graph.neighbors(candidate)) & uncovered)

                probabilities[candidate] = (pheromone**self.alpha) * ((1.0 + eta) ** self.beta)

            # Select candidate
            total = sum(probabilities.values())
            if total == 0:
                break

            r = random.random() * total
            cumulative = 0.0
            selected = None
            for candidate, prob in probabilities.items():
                cumulative += prob
                if r <= cumulative:
                    selected = candidate
                    break

            if selected:
                group.add(selected)
                candidates.remove(selected)
            else:
                break

        return group

    def _update_pheromones(self, pheromones: Dict[Tuple, float], solutions: List[Solution]):
        """Update pheromone trails based on solution quality"""
        # Evaporation
        for key in pheromones:
            pheromones[key] *= 1.0 - self.evaporation_rate

        # Reinforcement from good solutions
        for solution in solutions:
            quality = 1.0 / solution.total_cost  # Better solutions = higher quality

            for group in solution.groups:
                # Reinforce node-license pheromones
                all_nodes = {group.owner} | group.additional_members
                for node in all_nodes:
                    key = (node, group.license_type.name)
                    if key in pheromones:
                        pheromones[key] += quality

                # Reinforce edge pheromones within group
                nodes_list = list(all_nodes)
                for i in range(len(nodes_list)):
                    for j in range(i + 1, len(nodes_list)):
                        key = tuple(sorted([nodes_list[i], nodes_list[j]]) + [group.license_type.name])
                        if key in pheromones:
                            pheromones[key] += quality
