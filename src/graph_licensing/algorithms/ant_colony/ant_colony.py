"""Refactored Ant Colony Optimization algorithm using common components."""

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from ..common.population_base import PopulationBasedAlgorithm
from ..common.initialization import SolutionInitializer
from ..common.validation import SolutionValidator

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution, LicenseTypeConfig
    from ..common.config import AntColonyConfig


class AntColonyAlgorithm(PopulationBasedAlgorithm):
    """Ant Colony Optimization algorithm for license optimization."""

    def __init__(
        self,
        num_ants: int = 50,
        max_iterations: int = 100,
        alpha: float = 1.0,  # pheromone importance
        beta: float = 2.0,  # heuristic importance  
        rho: float = 0.5,  # evaporation rate
        q0: float = 0.9,  # exploitation vs exploration parameter
        initial_pheromone: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__("Ant Colony", num_ants, max_iterations, seed)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone

        # Pheromone matrices
        self.license_pheromones: Dict[Tuple[str, int], float] = {}  # (license_type, node) -> pheromone
        self.group_pheromones: Dict[Tuple[str, int, int], float] = {}  # (license_type, owner, member) -> pheromone

    def _initialize_population(
        self, graph: "nx.Graph", config: "LicenseConfig", warm_start: Optional["LicenseSolution"] = None
    ) -> List["LicenseSolution"]:
        """Initialize population using ant construction."""
        return []  # ACO constructs solutions during each iteration

    def _initialize_algorithm_state(
        self, graph: "nx.Graph", config: "LicenseConfig", population: List["LicenseSolution"]
    ) -> None:
        """Initialize pheromone matrices."""
        self._initialize_pheromones(graph, config)

    def _evolve_population(
        self,
        population: List["LicenseSolution"],
        fitness_scores: List[float],
        graph: "nx.Graph",
        config: "LicenseConfig",
        generation: int,
    ) -> List["LicenseSolution"]:
        """Construct new solutions using ants."""
        new_solutions = []
        
        for _ in range(self.population_size):
            solution = self._construct_solution(graph, config)
            if solution:
                new_solutions.append(solution)
        
        # Update pheromones based on solution quality
        if new_solutions:
            fitness_scores = [SolutionValidator.calculate_solution_fitness(sol, config) for sol in new_solutions]
            self._update_pheromones(list(zip(new_solutions, fitness_scores)))
        
        return new_solutions

    def _update_algorithm_state(
        self, population: List["LicenseSolution"], fitness_scores: List[float], generation: int
    ) -> None:
        """Update pheromones and apply evaporation."""
        self._evaporate_pheromones()

    def _initialize_pheromones(self, graph: "nx.Graph", config: "LicenseConfig") -> None:
        """Initialize pheromone matrices."""
        nodes = list(graph.nodes())
        
        # Initialize license assignment pheromones
        for license_type in config.license_types.keys():
            for node in nodes:
                self.license_pheromones[(license_type, node)] = self.initial_pheromone
        
        # Initialize group membership pheromones
        for license_type in config.license_types.keys():
            for owner in nodes:
                for member in graph.neighbors(owner):
                    self.group_pheromones[(license_type, owner, member)] = self.initial_pheromone

    def _construct_solution(self, graph: "nx.Graph", config: "LicenseConfig") -> Optional["LicenseSolution"]:
        """Construct a solution using ant behavior."""
        from ...models.license import LicenseSolution
        
        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()
        
        unassigned = set(nodes)
        licenses = {}
        
        while unassigned:
            # Select starting node based on heuristic
            start_node = self._select_node_probabilistically(unassigned, graph, config)
            unassigned.remove(start_node)
            
            # Select license type for this node
            license_type = self._select_license_type(start_node, config)
            license_config = config.license_types[license_type]
            
            # Build group around this node
            group = [start_node]
            
            # Add members to group based on pheromones and heuristics
            while len(group) < license_config.max_size and unassigned:
                next_member = self._select_group_member(start_node, group, unassigned, graph, license_type, config)
                if next_member is None:
                    break
                group.append(next_member)
                unassigned.remove(next_member)
            
            # Check if group size is valid
            if license_config.is_valid_size(len(group)):
                if license_type not in licenses:
                    licenses[license_type] = {} 
                licenses[license_type][start_node] = group
            else:
                # If group is too small, try to create individual licenses
                for member in group:
                    best_solo_license = config.get_best_license_for_size(1)
                    if best_solo_license:
                        solo_license_type = best_solo_license[0]
                        if solo_license_type not in licenses:
                            licenses[solo_license_type] = {}
                        licenses[solo_license_type][member] = [member]
        
        return LicenseSolution(licenses=licenses)

    def _select_node_probabilistically(
        self, available_nodes: Set[int], graph: "nx.Graph", config: "LicenseConfig"
    ) -> int:
        """Select a node probabilistically based on heuristics."""
        if len(available_nodes) == 1:
            return next(iter(available_nodes))
        
        # Use degree as heuristic (higher degree = better starting point)
        node_scores = {}
        for node in available_nodes:
            degree = graph.degree(node)
            # Add small random component to break ties
            node_scores[node] = degree + random.random() * 0.1
        
        # Select based on scores (higher is better)
        total_score = sum(node_scores.values())
        if total_score == 0:
            return random.choice(list(available_nodes))
        
        rand_val = random.random() * total_score
        cumulative = 0
        
        for node, score in node_scores.items():
            cumulative += score
            if rand_val <= cumulative:
                return node
        
        return list(available_nodes)[-1]  # Fallback

    def _select_license_type(self, node: int, config: "LicenseConfig") -> str:
        """Select license type for a node based on pheromones."""
        license_types = list(config.license_types.keys())
        
        if len(license_types) == 1:
            return license_types[0]
            
        # Calculate probabilities based on pheromones
        probabilities = {}
        for license_type in license_types:
            pheromone = self.license_pheromones.get((license_type, node), self.initial_pheromone)
            # Simple heuristic: prefer licenses with better cost-per-person ratio
            license_config = config.license_types[license_type]
            heuristic = 1.0 / license_config.cost_per_person(license_config.min_size)
            
            probabilities[license_type] = (pheromone ** self.alpha) * (heuristic ** self.beta)
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob == 0:
            return random.choice(license_types)
        
        # Roulette wheel selection
        rand_val = random.random() * total_prob
        cumulative = 0
        
        for license_type, prob in probabilities.items():
            cumulative += prob
            if rand_val <= cumulative:
                return license_type
        
        return license_types[-1]  # Fallback

    def _select_group_member(
        self,
        owner: int,
        current_group: List[int],
        available_nodes: Set[int],
        graph: "nx.Graph",
        license_type: str,
        config: "LicenseConfig"
    ) -> Optional[int]:
        """Select next member for a group."""
        # Only consider neighbors for group formation
        candidates = [n for n in available_nodes if n in graph.neighbors(owner)]
        
        if not candidates:
            return None
        
        # Calculate probabilities for each candidate
        probabilities = {}
        
        for candidate in candidates:
            # Pheromone component
            pheromone = self.group_pheromones.get((license_type, owner, candidate), self.initial_pheromone)
            
            # Heuristic component (preference for nodes with many connections to current group)
            shared_connections = sum(1 for member in current_group if candidate in graph.neighbors(member))
            heuristic = shared_connections + 1  # +1 to avoid zero
            
            probabilities[candidate] = (pheromone ** self.alpha) * (heuristic ** self.beta)
        
        # Apply exploitation vs exploration
        if random.random() < self.q0:
            # Exploitation: choose best candidate
            return max(candidates, key=lambda c: probabilities[c])
        else:
            # Exploration: probabilistic selection
            total_prob = sum(probabilities.values())
            if total_prob == 0:
                return random.choice(candidates)
            
            rand_val = random.random() * total_prob
            cumulative = 0
            
            for candidate, prob in probabilities.items():
                cumulative += prob
                if rand_val <= cumulative:
                    return candidate
            
            return candidates[-1]  # Fallback

    def _update_pheromones(self, solutions_with_costs: List[Tuple["LicenseSolution", float]]) -> None:
        """Update pheromones based on solution quality."""
        if not solutions_with_costs:
            return
        
        # Find best solution in this iteration
        best_solution, best_cost = min(solutions_with_costs, key=lambda x: x[1])
        
        # Update pheromones for all solutions (with different intensities)
        for solution, cost in solutions_with_costs:
            # Calculate pheromone deposit (inversely proportional to cost)
            if cost > 0:
                deposit = 1.0 / cost
            else:
                deposit = 1.0
                
            # Extra deposit for best solution
            if solution == best_solution:
                deposit *= 2.0
            
            self._deposit_pheromones(solution, deposit)

    def _deposit_pheromones(self, solution: "LicenseSolution", deposit: float) -> None:
        """Deposit pheromones for a given solution."""
        for license_type, groups in solution.licenses.items():
            for owner, members in groups.items():
                # Deposit on license type assignment
                current_pheromone = self.license_pheromones.get((license_type, owner), self.initial_pheromone)
                self.license_pheromones[(license_type, owner)] = current_pheromone + deposit
                
                # Deposit on group membership
                for member in members:
                    if member != owner:
                        key = (license_type, owner, member)
                        current_pheromone = self.group_pheromones.get(key, self.initial_pheromone)
                        self.group_pheromones[key] = current_pheromone + deposit

    def _evaporate_pheromones(self) -> None:
        """Apply pheromone evaporation."""
        # Evaporate license assignment pheromones
        for key in self.license_pheromones:
            self.license_pheromones[key] *= (1 - self.rho)
            # Ensure minimum pheromone level
            self.license_pheromones[key] = max(self.license_pheromones[key], self.initial_pheromone * 0.1)
        
        # Evaporate group membership pheromones
        for key in self.group_pheromones:
            self.group_pheromones[key] *= (1 - self.rho)
            self.group_pheromones[key] = max(self.group_pheromones[key], self.initial_pheromone * 0.1)

    @classmethod
    def from_config(cls, config: "AntColonyConfig"):
        """Create instance from configuration object."""
        return cls(
            num_ants=config.num_ants,
            max_iterations=config.max_iterations,
            alpha=config.alpha,
            beta=config.beta,
            rho=config.rho,
            q0=config.q0,
            initial_pheromone=config.initial_pheromone,
            seed=config.seed
        )
