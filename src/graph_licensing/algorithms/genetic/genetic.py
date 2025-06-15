"""Enhanced Genetic Algorithm for flexible licensing optimization."""

import random
from typing import TYPE_CHECKING, List, Optional, Dict, Tuple, Any

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution


class GeneticAlgorithm(BaseAlgorithm):
    """Enhanced Genetic Algorithm supporting flexible license types."""

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.85,
        elite_size: int = 5,
        seed: int | None = None,
    ) -> None:
        """Initialize the genetic algorithm.

        Args:
            population_size: Size of the population.
            generations: Number of generations to evolve.
            mutation_rate: Probability of mutation.
            crossover_rate: Probability of crossover.
            elite_size: Number of elite individuals to preserve.
            seed: Random seed for reproducibility.
        """
        super().__init__("Genetic")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.seed = seed
        self._best_solution = None

    def supports_warm_start(self) -> bool:
        """Genetic algorithm supports warm start initialization."""
        return True

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using enhanced genetic algorithm.

        Args:
            graph: The social network graph.
            config: License configuration.
            warm_start: Previous solution for warm start.
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

        # Initialize population
        population = self._initialize_population(graph, config, warm_start)

        best_fitness = float('inf')
        stagnation_count = 0
        max_stagnation = 20

        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual, config) for individual in population]
            
            # Track best fitness
            current_best = min(fitness_scores)
            if current_best < best_fitness:
                best_fitness = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Early stopping if stagnant
            if stagnation_count >= max_stagnation:
                break

            # Evolve population
            population = self._evolve_population(population, fitness_scores, graph, config)

        # Return best solution
        best_individual = min(population, key=lambda x: self._evaluate_fitness(x, config))
        self._best_solution = self._decode_individual(best_individual, graph, config)
        return self._best_solution

    def _initialize_population(
        self, 
        graph: "nx.Graph", 
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None
    ) -> List[Dict[str, Any]]:
        """Initialize population with diverse individuals.
        
        Individual representation: {
            'assignments': {node_id: (license_type, owner_id)}
        }
        """
        population = []
        
        # Add warm start solution if available
        if warm_start:
            warm_individual = self._encode_solution(warm_start)
            population.append(warm_individual)
        
        # Add greedy solution
        from ..greedy import GreedyAlgorithm
        greedy_algo = GreedyAlgorithm()
        greedy_solution = greedy_algo.solve(graph, config)
        greedy_individual = self._encode_solution(greedy_solution)
        population.append(greedy_individual)
        
        # Fill rest with random individuals
        while len(population) < self.population_size:
            random_individual = self._create_random_individual(graph, config)
            population.append(random_individual)
            
        return population

    def _encode_solution(self, solution: "LicenseSolution") -> Dict[str, Any]:
        """Encode a LicenseSolution into individual representation."""
        assignments = {}
        
        for license_type, groups in solution.licenses.items():
            for owner, members in groups.items():
                for member in members:
                    assignments[member] = (license_type, owner)
        
        return {'assignments': assignments}

    def _decode_individual(
        self, 
        individual: Dict[str, Any], 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Decode individual to LicenseSolution."""
        from ...models.license import LicenseSolution
        
        licenses = {}
        assignments = individual['assignments']
        
        # Group by license type and owner
        for node, (license_type, owner) in assignments.items():
            if license_type not in licenses:
                licenses[license_type] = {}
            if owner not in licenses[license_type]:
                licenses[license_type][owner] = []
            if node not in licenses[license_type][owner]:
                licenses[license_type][owner].append(node)
        
        # Ensure owners appear in their member lists
        for license_type in licenses:
            for owner in licenses[license_type]:
                if owner not in licenses[license_type][owner]:
                    licenses[license_type][owner].insert(0, owner)
        
        return LicenseSolution(licenses=licenses)

    def _create_random_individual(
        self, 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> Dict[str, Any]:
        """Create a random valid individual."""
        nodes = list(graph.nodes())
        assignments = {}
        unassigned = set(nodes)
        
        while unassigned:
            node = random.choice(list(unassigned))
            
            # Randomly choose a license type
            license_type = random.choice(list(config.license_types.keys()))
            license_config = config.license_types[license_type]
            
            # Decide group size randomly within constraints
            available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]
            max_group_size = min(
                license_config.max_size,
                len(available_neighbors) + 1
            )
            
            if max_group_size >= license_config.min_size:
                group_size = random.randint(license_config.min_size, max_group_size)
                
                if group_size == 1:
                    assignments[node] = (license_type, node)
                    unassigned.remove(node)
                else:
                    # Form a group
                    selected_neighbors = random.sample(
                        available_neighbors, 
                        min(group_size - 1, len(available_neighbors))
                    )
                    group_members = [node] + selected_neighbors
                    
                    for member in group_members:
                        assignments[member] = (license_type, node)
                        unassigned.discard(member)
            else:
                # Fallback to cheapest solo license
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    assignments[node] = (cheapest_solo, node)
                unassigned.remove(node)
        
        return {'assignments': assignments}

    def _evaluate_fitness(self, individual: Dict[str, Any], config: "LicenseConfig") -> float:
        """Evaluate fitness of an individual (lower is better)."""
        assignments = individual['assignments']
        licenses = {}
        
        # Group by license type and owner
        for node, (license_type, owner) in assignments.items():
            if license_type not in licenses:
                licenses[license_type] = {}
            if owner not in licenses[license_type]:
                licenses[license_type][owner] = []
            licenses[license_type][owner].append(node)
        
        # Calculate total cost
        total_cost = 0.0
        for license_type, groups in licenses.items():
            if license_type in config.license_types:
                license_config = config.license_types[license_type]
                for owner, members in groups.items():
                    if license_config.is_valid_size(len(members)):
                        total_cost += license_config.price
                    else:
                        total_cost += float('inf')  # Invalid group size penalty
        
        return total_cost

    def _evolve_population(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        graph: "nx.Graph",
        config: "LicenseConfig"
    ) -> List[Dict[str, Any]]:
        """Evolve the population for one generation."""
        new_population = []
        
        # Keep elite individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:self.elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, graph, config)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1, graph, config)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2, graph, config)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]

    def _tournament_selection(
        self, 
        population: List[Dict[str, Any]], 
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> Dict[str, Any]:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()

    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        graph: "nx.Graph",
        config: "LicenseConfig"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover two parents to create offspring."""
        nodes = list(graph.nodes())
        
        # Simple crossover: randomly inherit assignments from each parent
        child1_assignments = {}
        child2_assignments = {}
        
        for node in nodes:
            if random.random() < 0.5:
                child1_assignments[node] = parent1['assignments'].get(node, self._get_default_assignment(node, config))
                child2_assignments[node] = parent2['assignments'].get(node, self._get_default_assignment(node, config))
            else:
                child1_assignments[node] = parent2['assignments'].get(node, self._get_default_assignment(node, config))
                child2_assignments[node] = parent1['assignments'].get(node, self._get_default_assignment(node, config))
        
        child1 = {'assignments': child1_assignments}
        child2 = {'assignments': child2_assignments}
        
        # Repair children to ensure validity
        child1 = self._repair_individual(child1, graph, config)
        child2 = self._repair_individual(child2, graph, config)
        
        return child1, child2

    def _mutate(
        self,
        individual: Dict[str, Any],
        graph: "nx.Graph",
        config: "LicenseConfig"
    ) -> Dict[str, Any]:
        """Mutate an individual."""
        mutated = {'assignments': individual['assignments'].copy()}
        nodes = list(graph.nodes())
        
        # Randomly change some assignments
        num_mutations = max(1, int(len(nodes) * 0.1))  # Mutate 10% of nodes
        
        for _ in range(num_mutations):
            node = random.choice(nodes)
            
            # Get available license types
            available_types = list(config.license_types.keys())
            new_license_type = random.choice(available_types)
            
            # Assign to solo or random neighbor as owner
            if random.random() < 0.5 or not list(graph.neighbors(node)):
                # Solo assignment
                mutated['assignments'][node] = (new_license_type, node)
            else:
                # Group assignment with random neighbor as owner
                neighbor = random.choice(list(graph.neighbors(node)))
                mutated['assignments'][node] = (new_license_type, neighbor)
        
        return self._repair_individual(mutated, graph, config)

    def _repair_individual(
        self,
        individual: Dict[str, Any],
        graph: "nx.Graph",
        config: "LicenseConfig"
    ) -> Dict[str, Any]:
        """Repair an individual to ensure validity."""
        assignments = individual['assignments'].copy()
        
        # Group assignments by license type and owner
        groups = {}
        for node, (license_type, owner) in assignments.items():
            key = (license_type, owner)
            if key not in groups:
                groups[key] = []
            groups[key].append(node)
        
        # Fix invalid groups
        for (license_type, owner), members in groups.items():
            if license_type not in config.license_types:
                # Invalid license type - reassign to cheapest solo
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    for member in members:
                        assignments[member] = (cheapest_solo, member)
                continue
            
            license_config = config.license_types[license_type]
            
            # Check group size constraints
            if not license_config.is_valid_size(len(members)):
                if len(members) > license_config.max_size:
                    # Group too large - split into smaller groups
                    excess_members = members[license_config.max_size:]
                    cheapest_solo = self._get_cheapest_solo_license(config)
                    if cheapest_solo:
                        for member in excess_members:
                            assignments[member] = (cheapest_solo, member)
                elif len(members) < license_config.min_size:
                    # Group too small - convert to solo licenses
                    cheapest_solo = self._get_cheapest_solo_license(config)
                    if cheapest_solo:
                        for member in members:
                            assignments[member] = (cheapest_solo, member)
            
            # Check connectivity for multi-member groups
            if len(members) > 1:
                valid_members = [owner]  # Owner is always valid
                for member in members:
                    if member != owner and graph.has_edge(owner, member):
                        valid_members.append(member)
                
                # Reassign disconnected members
                disconnected = set(members) - set(valid_members)
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    for member in disconnected:
                        assignments[member] = (cheapest_solo, member)
        
        return {'assignments': assignments}

    def _get_default_assignment(self, node: int, config: "LicenseConfig") -> Tuple[str, int]:
        """Get default assignment for a node."""
        cheapest_solo = self._get_cheapest_solo_license(config)
        return (cheapest_solo, node) if cheapest_solo else (list(config.license_types.keys())[0], node)

    def _get_cheapest_solo_license(self, config: "LicenseConfig") -> str | None:
        """Find the cheapest license type that allows solo assignment."""
        cheapest_type = None
        cheapest_cost = float('inf')
        
        for license_type, license_config in config.license_types.items():
            if license_config.is_valid_size(1) and license_config.price < cheapest_cost:
                cheapest_cost = license_config.price
                cheapest_type = license_type
        
        return cheapest_type
