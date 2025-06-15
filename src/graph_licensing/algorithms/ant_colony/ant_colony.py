"""Ant Colony Optimization Algorithm for licensing optimization."""

import math
import random
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Set

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution


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
            return LicenseSolution(solo_nodes=[], group_owners={})

        # Initialize pheromone trails
        self._initialize_pheromones(graph, warm_start)
        
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

        return self._best_solution or LicenseSolution(solo_nodes=nodes, group_owners={})

    def _initialize_pheromones(
        self, 
        graph: "nx.Graph", 
        warm_start: Optional["LicenseSolution"] = None
    ) -> None:
        """Initialize pheromone trails."""
        nodes = list(graph.nodes())
        
        # Initialize solo pheromones
        for node in nodes:
            self.solo_pheromones[node] = self.initial_pheromone
            
        # Initialize group pheromones for all possible connections
        for node1 in nodes:
            for node2 in graph.neighbors(node1):
                if node1 != node2:
                    self.group_pheromones[(node1, node2)] = self.initial_pheromone
                    
        # Boost pheromones based on warm start solution
        if warm_start:
            boost_factor = 2.0
            
            # Boost solo decisions
            for node in warm_start.solo_nodes:
                self.solo_pheromones[node] *= boost_factor
                
            # Boost group decisions
            for owner, members in warm_start.group_owners.items():
                for member in members:
                    if member != owner and (owner, member) in self.group_pheromones:
                        self.group_pheromones[(owner, member)] *= boost_factor

    def _construct_solution(
        self, 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> Optional["LicenseSolution"]:
        """Construct a solution using ant decision rules."""
        from ...models.license import LicenseSolution
        
        nodes = list(graph.nodes())
        unassigned_nodes = set(nodes)
        solo_nodes = []
        group_owners = {}
        
        while unassigned_nodes:
            # Select a node to make decision for
            current_node = random.choice(list(unassigned_nodes))
            
            # Decide between solo license or becoming group owner
            if self._should_choose_solo(current_node, graph, config, unassigned_nodes):
                solo_nodes.append(current_node)
                unassigned_nodes.remove(current_node)
            else:
                # Try to form a group
                group_members = self._form_group(
                    current_node, graph, config, unassigned_nodes
                )
                if group_members and len(group_members) >= 2:  # At least owner + 1 member
                    group_owners[current_node] = group_members
                    for member in group_members:
                        unassigned_nodes.discard(member)
                else:
                    # Fallback to solo if group formation failed
                    solo_nodes.append(current_node)
                    unassigned_nodes.remove(current_node)
        
        return LicenseSolution(solo_nodes=solo_nodes, group_owners=group_owners)

    def _should_choose_solo(
        self,
        node: int,
        graph: "nx.Graph",
        config: "LicenseConfig",
        unassigned_nodes: Set[int]
    ) -> bool:
        """Decide whether a node should get a solo license."""
        # Calculate probability of solo vs group decision
        solo_pheromone = self.solo_pheromones.get(node, self.initial_pheromone)
        
        # Heuristic: solo is better if node has few unassigned neighbors
        neighbors = [n for n in graph.neighbors(node) if n in unassigned_nodes]
        
        # Heuristic value: inverse of potential group benefit
        if len(neighbors) == 0:
            solo_heuristic = 1.0  # No neighbors, solo is only option
        else:
            # Consider cost benefit of potential group
            potential_group_size = min(len(neighbors) + 1, config.group_size)
            if config.is_group_beneficial(potential_group_size):
                solo_heuristic = 0.1  # Group would be beneficial, discourage solo
            else:
                solo_heuristic = 0.9  # Group wouldn't be beneficial, encourage solo
        
        # Calculate group formation attractiveness
        group_attractiveness = 0.0
        if neighbors:
            for neighbor in neighbors[:min(len(neighbors), config.group_size - 1)]:
                pheromone = self.group_pheromones.get((node, neighbor), self.initial_pheromone)
                group_attractiveness += pheromone ** self.alpha
        
        # Calculate solo attractiveness
        solo_attractiveness = (solo_pheromone ** self.alpha) * (solo_heuristic ** self.beta)
        
        # Make probabilistic decision
        if group_attractiveness > 0:
            solo_probability = solo_attractiveness / (solo_attractiveness + group_attractiveness)
        else:
            solo_probability = 1.0
            
        # Apply exploitation vs exploration
        if random.random() < self.q0:
            # Exploitation: choose best option
            return solo_attractiveness > group_attractiveness
        else:
            # Exploration: probabilistic choice
            return random.random() < solo_probability

    def _form_group(
        self,
        owner: int,
        graph: "nx.Graph",
        config: "LicenseConfig",
        unassigned_nodes: Set[int]
    ) -> List[int]:
        """Form a group with the given owner."""
        group_members = [owner]
        candidates = [n for n in graph.neighbors(owner) if n in unassigned_nodes]
        
        if not candidates:
            return group_members
        
        # Select members based on pheromone and heuristic
        while len(group_members) < config.group_size and candidates:
            # Calculate attractiveness for each candidate
            attractiveness = {}
            for candidate in candidates:
                pheromone = self.group_pheromones.get((owner, candidate), self.initial_pheromone)
                
                # Heuristic: prefer nodes with fewer neighbors (easier to include)
                candidate_neighbors = len([n for n in graph.neighbors(candidate) 
                                         if n in unassigned_nodes and n != owner])
                heuristic = 1.0 / (1.0 + candidate_neighbors * 0.1)
                
                attractiveness[candidate] = (pheromone ** self.alpha) * (heuristic ** self.beta)
            
            if not attractiveness:
                break
                
            # Select member based on attractiveness
            if random.random() < self.q0:
                # Exploitation: select best candidate
                selected = max(attractiveness.keys(), key=lambda x: attractiveness[x])
            else:
                # Exploration: probabilistic selection
                total_attractiveness = sum(attractiveness.values())
                if total_attractiveness > 0:
                    rand_val = random.random() * total_attractiveness
                    cumulative = 0.0
                    selected = None
                    for candidate, attr in attractiveness.items():
                        cumulative += attr
                        if cumulative >= rand_val:
                            selected = candidate
                            break
                    if selected is None:
                        selected = random.choice(list(attractiveness.keys()))
                else:
                    selected = random.choice(candidates)
            
            group_members.append(selected)
            candidates.remove(selected)
            
            # Stop if group wouldn't be beneficial with more members
            if not config.is_group_beneficial(len(group_members) + 1):
                break
        
        return group_members

    def _update_pheromones(self, solutions: List[Tuple["LicenseSolution", float]]) -> None:
        """Update pheromone trails based on solutions quality."""
        if not solutions:
            return
            
        # Evaporation
        for node in self.solo_pheromones:
            self.solo_pheromones[node] *= (1.0 - self.rho)
            
        for edge in self.group_pheromones:
            self.group_pheromones[edge] *= (1.0 - self.rho)
        
        # Pheromone deposit
        for solution, cost in solutions:
            if cost <= 0:
                continue
                
            # Calculate pheromone deposit amount (inverse of cost)
            deposit = 1.0 / cost
            
            # Boost deposit for better solutions
            if cost == self._best_cost:
                deposit *= 2.0  # Best solution gets double pheromone
                
            # Deposit pheromones for solo decisions
            for node in solution.solo_nodes:
                if node in self.solo_pheromones:
                    self.solo_pheromones[node] += deposit
                    
            # Deposit pheromones for group decisions
            for owner, members in solution.group_owners.items():
                for member in members:
                    if member != owner and (owner, member) in self.group_pheromones:
                        self.group_pheromones[(owner, member)] += deposit

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
                "solo_avg": sum(self.solo_pheromones.values()) / len(self.solo_pheromones) 
                           if self.solo_pheromones else 0,
                "group_avg": sum(self.group_pheromones.values()) / len(self.group_pheromones)
                            if self.group_pheromones else 0,
            }
        }
