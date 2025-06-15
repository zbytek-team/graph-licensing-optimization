"""Ant Colony Optimization Algorithm for licensing optimization."""

import math
import random
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Set

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution, LicenseTypeConfig


class AntColonyAlgorithm(BaseAlgorithm):
    """Ant Colony Optimization Algorithm for graph licensing optimization.
    
    This algorithm uses artificial ants to find optimal licensing solutions by
    following pheromone trails that represent the quality of licensing decisions.
    """

    def __init__(
        self,
        num_ants: int = 50,
        max_iterations: int = 100,
        alpha: float = 1.0,  # pheromone importance
        beta: float = 2.0,   # heuristic importance
        rho: float = 0.5,    # evaporation rate
        q0: float = 0.9,     # exploitation vs exploration parameter
        initial_pheromone: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize the Ant Colony Optimization algorithm.

        Args:
            num_ants: Number of ants in the colony.
            max_iterations: Maximum number of iterations.
            alpha: Pheromone importance factor (higher = more pheromone influence).
            beta: Heuristic importance factor (higher = more heuristic influence).
            rho: Pheromone evaporation rate (0-1, higher = faster evaporation).
            q0: Exploitation vs exploration parameter (0-1, higher = more exploitation).
            initial_pheromone: Initial pheromone level on all edges/decisions.
            seed: Random seed for reproducibility.
        """
        super().__init__("Ant Colony")
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone
        self.seed = seed
        
        # Pheromone matrices
        self.solo_pheromones: Dict[int, float] = {}  # node -> pheromone for solo license
        self.group_pheromones: Dict[Tuple[int, int], float] = {}  # (owner, member) -> pheromone
        
        self._best_solution = None
        self._best_cost = float('inf')

    def supports_warm_start(self) -> bool:
        """Ant Colony algorithm supports warm start initialization."""
        return True

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using Ant Colony Optimization.

        Args:
            graph: The social network graph.
            config: License configuration.
            warm_start: Previous solution to use as starting point.
            **kwargs: Additional parameters.

        Returns:
            Best licensing solution found.
        """
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        # Initialize pheromone trails
        self._initialize_pheromones(graph, config, warm_start)
        
        # Initialize best solution
        if warm_start:
            self._best_solution = warm_start
            self._best_cost = warm_start.calculate_cost(config)
        else:
            self._best_solution = None
            self._best_cost = float('inf')

        # Main ACO loop
        for iteration in range(self.max_iterations):
            # Construct solutions with all ants
            iteration_solutions = []
            for ant in range(self.num_ants):
                solution = self._construct_solution(graph, config)
                if solution:
                    cost = solution.calculate_cost(config)
                    iteration_solutions.append((solution, cost))
                    
                    # Update best solution
                    if cost < self._best_cost:
                        self._best_cost = cost
                        self._best_solution = solution

            # Update pheromones
            self._update_pheromones(iteration_solutions)
            
            # Optional: early stopping if no improvement
            if iteration > 20 and iteration % 10 == 0:
                # Check for convergence (can be added later)
                pass

        return self._best_solution or self._create_fallback_solution(graph, config)

    def _initialize_pheromones(
        self, 
        graph: "nx.Graph", 
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None
    ) -> None:
        """Initialize pheromone trails."""
        nodes = list(graph.nodes())
        
        # Initialize pheromones for each license type and potential groups
        self.license_pheromones = {}  # (license_type, node) -> pheromone
        self.group_pheromones = {}    # (license_type, owner, member) -> pheromone
        
        for license_type in config.license_types.keys():
            # Solo-type licenses (min_size = max_size = 1)
            for node in nodes:
                self.license_pheromones[(license_type, node)] = self.initial_pheromone
            
            # Group-type licenses
            for node1 in nodes:
                for node2 in graph.neighbors(node1):
                    if node1 != node2:
                        self.group_pheromones[(license_type, node1, node2)] = self.initial_pheromone
                        
        # Boost pheromones based on warm start solution
        if warm_start:
            boost_factor = 2.0
            
            for license_type, groups in warm_start.licenses.items():
                for owner, members in groups.items():
                    if len(members) == 1:
                        # Solo license
                        key = (license_type, owner)
                        if key in self.license_pheromones:
                            self.license_pheromones[key] *= boost_factor
                    else:
                        # Group license
                        for member in members:
                            if member != owner:
                                key = (license_type, owner, member)
                                if key in self.group_pheromones:
                                    self.group_pheromones[key] *= boost_factor

    def _construct_solution(
        self, 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> Optional["LicenseSolution"]:
        """Construct a solution using ant decision rules."""
        from ...models.license import LicenseSolution
        
        nodes = list(graph.nodes())
        unassigned_nodes = set(nodes)
        solution_licenses = {}
        
        while unassigned_nodes:
            # Select a node to make decision for
            current_node = random.choice(list(unassigned_nodes))
            
            # Choose best license type and group configuration
            best_choice = self._choose_license_assignment(
                current_node, graph, config, unassigned_nodes
            )
            
            if best_choice:
                license_type, owner, members = best_choice
                
                # Add to solution
                if license_type not in solution_licenses:
                    solution_licenses[license_type] = {}
                
                solution_licenses[license_type][owner] = members
                
                # Remove assigned nodes
                for member in members:
                    unassigned_nodes.discard(member)
            else:
                # Fallback: assign solo license with cheapest available type
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    license_type = cheapest_solo
                    if license_type not in solution_licenses:
                        solution_licenses[license_type] = {}
                    solution_licenses[license_type][current_node] = [current_node]
                
                unassigned_nodes.remove(current_node)
        
        return LicenseSolution(licenses=solution_licenses)

    def _update_pheromones(self, solutions: List[Tuple["LicenseSolution", float]]) -> None:
        """Update pheromone trails based on solutions quality."""
        if not solutions:
            return
            
        # Evaporation
        for key in self.license_pheromones:
            self.license_pheromones[key] *= (1.0 - self.rho)
            
        for key in self.group_pheromones:
            self.group_pheromones[key] *= (1.0 - self.rho)
        
        # Pheromone deposit
        for solution, cost in solutions:
            if cost <= 0:
                continue
                
            # Calculate pheromone deposit amount (inverse of cost)
            deposit = 1.0 / cost
            
            # Boost deposit for better solutions
            if cost == self._best_cost:
                deposit *= 2.0  # Best solution gets double pheromone
                
            # Deposit pheromones for all license assignments
            for license_type, groups in solution.licenses.items():
                for owner, members in groups.items():
                    if len(members) == 1:
                        # Solo license
                        key = (license_type, owner)
                        if key in self.license_pheromones:
                            self.license_pheromones[key] += deposit
                    else:
                        # Group license
                        for member in members:
                            if member != owner:
                                key = (license_type, owner, member)
                                if key in self.group_pheromones:
                                    self.group_pheromones[key] += deposit

    def _choose_license_assignment(
        self,
        node: int,
        graph: "nx.Graph",
        config: "LicenseConfig",
        unassigned_nodes: Set[int]
    ) -> Optional[Tuple[str, int, List[int]]]:
        """Choose the best license assignment for a node.
        
        Returns:
            Tuple of (license_type, owner, members) or None
        """
        candidates = []
        
        # Evaluate all possible license types
        for license_type, license_config in config.license_types.items():
            # Try different group sizes within the license constraints
            for group_size in range(license_config.min_size, min(license_config.max_size + 1, len(unassigned_nodes) + 1)):
                if group_size == 1:
                    # Solo assignment
                    pheromone = self.license_pheromones.get((license_type, node), self.initial_pheromone)
                    heuristic = self._calculate_solo_heuristic(node, license_config, graph)
                    attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    candidates.append((attractiveness, license_type, node, [node]))
                
                elif group_size > 1:
                    # Group assignment
                    potential_members = self._find_potential_group_members(
                        node, graph, unassigned_nodes, group_size - 1
                    )
                    
                    if len(potential_members) >= group_size - 1:
                        members = [node] + potential_members[:group_size - 1]
                        pheromone = self._calculate_group_pheromone(license_type, node, members[1:])
                        heuristic = self._calculate_group_heuristic(license_config, members, graph)
                        attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)
                        candidates.append((attractiveness, license_type, node, members))
        
        if not candidates:
            return None
        
        # Choose based on exploitation vs exploration
        if random.random() < self.q0:
            # Exploitation: choose best candidate
            best_candidate = max(candidates, key=lambda x: x[0])
            return (best_candidate[1], best_candidate[2], best_candidate[3])
        else:
            # Exploration: probabilistic selection
            total_attractiveness = sum(candidate[0] for candidate in candidates)
            if total_attractiveness > 0:
                rand_val = random.random() * total_attractiveness
                cumulative = 0.0
                for attractiveness, license_type, owner, members in candidates:
                    cumulative += attractiveness
                    if cumulative >= rand_val:
                        return (license_type, owner, members)
            
            # Fallback to random selection
            chosen = random.choice(candidates)
            return (chosen[1], chosen[2], chosen[3])

    def _get_cheapest_solo_license(self, config: "LicenseConfig") -> Optional[str]:
        """Find the cheapest license type that allows solo assignment."""
        cheapest_type = None
        cheapest_cost = float('inf')
        
        for license_type, license_config in config.license_types.items():
            if license_config.is_valid_size(1) and license_config.price < cheapest_cost:
                cheapest_cost = license_config.price
                cheapest_type = license_type
        
        return cheapest_type

    def _find_potential_group_members(
        self,
        node: int,
        graph: "nx.Graph",
        unassigned_nodes: Set[int],
        max_members: int
    ) -> List[int]:
        """Find potential group members for a node."""
        neighbors = [n for n in graph.neighbors(node) if n in unassigned_nodes]
        
        # Sort by degree (prefer high-degree nodes for group stability)
        neighbors.sort(key=lambda x: graph.degree(x), reverse=True)
        
        return neighbors[:max_members]

    def _calculate_solo_heuristic(
        self,
        node: int,
        license_config: "LicenseTypeConfig",
        graph: "nx.Graph"
    ) -> float:
        """Calculate heuristic value for solo license assignment."""
        # Higher heuristic for nodes with few neighbors (less potential for groups)
        degree = graph.degree(node)
        return 1.0 / (1.0 + degree * 0.1)

    def _calculate_group_pheromone(
        self,
        license_type: str,
        owner: int,
        members: List[int]
    ) -> float:
        """Calculate combined pheromone for group formation."""
        total_pheromone = 0.0
        count = 0
        
        for member in members:
            key = (license_type, owner, member)
            if key in self.group_pheromones:
                total_pheromone += self.group_pheromones[key]
                count += 1
        
        return total_pheromone / max(count, 1)

    def _calculate_group_heuristic(
        self,
        license_config: "LicenseTypeConfig",
        members: List[int],
        graph: "nx.Graph"
    ) -> float:
        """Calculate heuristic value for group license assignment."""
        group_size = len(members)
        if not license_config.is_valid_size(group_size):
            return 0.0
        
        # Higher heuristic for cost-effective groups
        cost_per_person = license_config.cost_per_person(group_size)
        
        # Consider connectivity within the group
        owner = members[0]
        connectivity = sum(1 for member in members[1:] if graph.has_edge(owner, member))
        connectivity_factor = connectivity / max(len(members) - 1, 1)
        
        return (1.0 / cost_per_person) * connectivity_factor

    def _create_fallback_solution(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Create a fallback solution using cheapest solo licenses."""
        from ...models.license import LicenseSolution
        
        cheapest_type = self._get_cheapest_solo_license(config)
        if not cheapest_type:
            return LicenseSolution.create_empty()
        
        nodes = list(graph.nodes())
        licenses = {
            cheapest_type: {node: [node] for node in nodes}
        }
        
        return LicenseSolution(licenses=licenses)

    def get_algorithm_info(self) -> dict:
        """Get information about the algorithm and its current state."""
        return {
            "name": self.name,
            "num_ants": self.num_ants,
            "max_iterations": self.max_iterations,
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "q0": self.q0,
            "best_cost": self._best_cost if self._best_cost != float('inf') else None,
            "pheromone_levels": {
                "license_avg": sum(self.license_pheromones.values()) / len(self.license_pheromones) 
                               if self.license_pheromones else 0,
                "group_avg": sum(self.group_pheromones.values()) / len(self.group_pheromones)
                            if self.group_pheromones else 0,
            }
        }
