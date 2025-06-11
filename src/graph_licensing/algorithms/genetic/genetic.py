"""Improved Genetic Algorithm for licensing optimization."""

import random
from typing import TYPE_CHECKING, List, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution


class GeneticAlgorithm(BaseAlgorithm):
    """Enhanced Genetic Algorithm with better group formation."""

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.85,
        elite_size: int = 5,
        group_bias: float = 0.8,
        seed: int | None = None,
    ) -> None:
        """Initialize the genetic algorithm.

        Args:
            population_size: Size of the population.
            generations: Number of generations to evolve.
            mutation_rate: Probability of mutation.
            crossover_rate: Probability of crossover.
            elite_size: Number of elite individuals to preserve.
            group_bias: Bias towards group formation (0.0-1.0).
            seed: Random seed for reproducibility.
        """
        super().__init__("Genetic")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.group_bias = group_bias
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
            **kwargs: Additional parameters.

        Returns:
            Best licensing solution found.
        """
        from ...models.license import LicenseSolution
        from ..greedy import GreedyAlgorithm

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution(solo_nodes=[], group_owners={})

        # Initialize population with greedy solution and diversity
        population = self._initialize_population(graph, config, nodes, warm_start)

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

            # Adaptive mutation rate
            adaptive_mutation = self.mutation_rate * (1 + stagnation_count / max_stagnation)

            # Selection and reproduction
            new_population = self._evolve_population(
                population, fitness_scores, graph, config, adaptive_mutation
            )
            population = new_population

        # Return best solution
        best_individual = min(population, key=lambda x: self._evaluate_fitness(x, config))
        self._best_solution = self._decode_individual(best_individual, nodes)
        return self._best_solution

    def _initialize_population(
        self, 
        graph: "nx.Graph", 
        config: "LicenseConfig", 
        nodes: List[int],
        warm_start: Optional["LicenseSolution"] = None
    ) -> List[List[int]]:
        """Initialize population with greedy seed and diverse individuals."""
        population = []
        
        # Start with warm start if available, otherwise greedy solution
        if warm_start is not None:
            # Adapt warm start to current graph
            adapted_solution = self._adapt_solution_to_graph(warm_start, graph, config)
            warm_individual = self._encode_solution(adapted_solution, nodes)
            population.append(warm_individual)
        else:
            # Start with greedy solution
            from ..greedy import GreedyAlgorithm
            greedy_algorithm = GreedyAlgorithm()
            greedy_solution = greedy_algorithm.solve(graph, config)
            greedy_individual = self._encode_solution(greedy_solution, nodes)
            population.append(greedy_individual)
        
        # Add variations of the seed solution
        seed_individual = population[0]
        for _ in range(min(5, self.population_size - 1)):
            varied = self._create_greedy_variation(seed_individual, graph, config)
            population.append(varied)
        
        # Fill rest with diverse random individuals
        for _ in range(self.population_size - len(population)):
            population.append(self._create_smart_random_individual(graph, config))
            
        return population

    def _create_greedy_variation(
        self, 
        greedy_individual: List[int], 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> List[int]:
        """Create variation of greedy solution."""
        varied = greedy_individual.copy()
        nodes = list(graph.nodes())
        
        # Randomly modify 10-20% of assignments
        num_changes = random.randint(len(nodes) // 10, len(nodes) // 5)
        for _ in range(num_changes):
            node_idx = random.randint(0, len(nodes) - 1)
            varied[node_idx] = self._get_smart_assignment(node_idx, nodes, graph, config)
            
        return self._repair_individual(varied, graph, config)

    def _create_smart_random_individual(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> List[int]:
        """Create random individual with intelligent group formation."""
        nodes = list(graph.nodes())
        individual = [0] * len(nodes)
        processed = set()
        
        # Sort nodes by degree for better group formation
        nodes_by_degree = sorted(
            enumerate(nodes), 
            key=lambda x: graph.degree(x[1]), 
            reverse=True
        )
        
        for i, node in nodes_by_degree:
            if node in processed:
                continue
            
            # Probability based on degree and group bias
            degree = graph.degree(node)
            group_prob = self.group_bias * (1 + degree / (max(1, graph.number_of_nodes())))
            
            if random.random() < group_prob:
                # Try to form a group
                neighbors = [n for n in graph.neighbors(node) if n not in processed]
                if neighbors and config.is_group_beneficial(2):
                    # Create group
                    max_members = min(config.group_size - 1, len(neighbors))
                    num_members = random.randint(1, max(1, max_members))
                    selected = random.sample(neighbors, num_members)
                    
                    # Assign group
                    individual[i] = i + 1  # Owner
                    processed.add(node)
                    
                    for member in selected:
                        member_idx = nodes.index(member)
                        individual[member_idx] = i + 1
                        processed.add(member)
                    continue
            
            # Solo assignment
            individual[i] = 0
            processed.add(node)
            
        return self._repair_individual(individual, graph, config)

    def _get_smart_assignment(
        self, 
        node_idx: int, 
        nodes: List[int], 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> int:
        """Get smart assignment for a node."""
        node = nodes[node_idx]
        assignments = [0]  # Solo option
        
        # Add neighbor group options
        for neighbor in graph.neighbors(node):
            if neighbor in nodes:
                neighbor_idx = nodes.index(neighbor)
                assignments.append(neighbor_idx + 1)
        
        # Add self as group owner option
        assignments.append(node_idx + 1)
        
        # Weight assignments based on potential benefit
        weights = []
        for assignment in assignments:
            if assignment == 0:
                weights.append(1.0)  # Solo baseline
            else:
                # Higher weight for beneficial groups
                weights.append(2.0 if config.is_group_beneficial(2) else 0.5)
        
        return random.choices(assignments, weights=weights)[0]

    def _evolve_population(
        self, 
        population: List[List[int]], 
        fitness_scores: List[float],
        graph: "nx.Graph",
        config: "LicenseConfig",
        mutation_rate: float
    ) -> List[List[int]]:
        """Evolve population with elitism and diversity."""
        new_population = []
        
        # Elitism: preserve best individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:self.elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            if random.random() < self.crossover_rate:
                child = self._enhanced_crossover(parent1, parent2, graph, config)
            else:
                child = parent1.copy() if random.random() < 0.5 else parent2.copy()
            
            if random.random() < mutation_rate:
                child = self._enhanced_mutation(child, graph, config)
            
            new_population.append(child)
        
        return new_population[:self.population_size]

    def _enhanced_crossover(
        self,
        parent1: List[int],
        parent2: List[int],
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> List[int]:
        """Enhanced crossover preserving group structures."""
        # Multi-point crossover with group awareness
        length = len(parent1)
        child = [0] * length
        
        # Random crossover points
        num_points = random.randint(2, 4)
        crossover_points = sorted(random.sample(range(1, length), num_points))
        crossover_points = [0] + crossover_points + [length]
        
        # Alternate between parents
        for i in range(len(crossover_points) - 1):
            start, end = crossover_points[i], crossover_points[i + 1]
            parent = parent1 if i % 2 == 0 else parent2
            child[start:end] = parent[start:end]
        
        return self._repair_individual(child, graph, config)

    def _enhanced_mutation(
        self,
        individual: List[int],
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> List[int]:
        """Enhanced mutation with group-aware operations."""
        mutated = individual.copy()
        nodes = list(graph.nodes())
        
        # Multiple mutation operations
        num_mutations = random.randint(1, max(1, len(nodes) // 20))
        
        for _ in range(num_mutations):
            mutation_type = random.choice(["reassign", "split_group", "merge_groups", "optimize_group"])
            
            if mutation_type == "reassign":
                node_idx = random.randint(0, len(nodes) - 1)
                mutated[node_idx] = self._get_smart_assignment(node_idx, nodes, graph, config)
                
            elif mutation_type == "split_group" and self._has_groups(mutated):
                self._split_random_group(mutated, nodes, graph, config)
                
            elif mutation_type == "merge_groups":
                self._try_merge_groups(mutated, nodes, graph, config)
                
            elif mutation_type == "optimize_group":
                self._optimize_random_group(mutated, nodes, graph, config)
        
        return self._repair_individual(mutated, graph, config)

    def _has_groups(self, individual: List[int]) -> bool:
        """Check if individual has any groups."""
        return any(assignment > 0 for assignment in individual)

    def _split_random_group(
        self, individual: List[int], nodes: List[int], graph: "nx.Graph", config: "LicenseConfig"
    ) -> None:
        """Split a random group."""
        # Find groups
        groups = {}
        for i, assignment in enumerate(individual):
            if assignment > 0:
                owner = assignment - 1
                if owner not in groups:
                    groups[owner] = []
                groups[owner].append(i)
        
        if groups:
            # Select random group to split
            owner = random.choice(list(groups.keys()))
            members = groups[owner]
            
            if len(members) > 2:
                # Split group
                split_point = random.randint(1, len(members) - 1)
                for i in range(split_point, len(members)):
                    individual[members[i]] = 0  # Convert to solo

    def _try_merge_groups(
        self, individual: List[int], nodes: List[int], graph: "nx.Graph", config: "LicenseConfig"
    ) -> None:
        """Try to merge compatible groups."""
        # Find adjacent groups
        groups = self._get_groups_from_individual(individual)
        
        for owner1, members1 in groups.items():
            for owner2, members2 in groups.items():
                if owner1 >= owner2:
                    continue
                    
                # Check if groups can be merged
                if len(members1) + len(members2) <= config.group_size:
                    # Check connectivity
                    node1, node2 = nodes[owner1], nodes[owner2]
                    if graph.has_edge(node1, node2):
                        # Merge groups
                        for member_idx in members2:
                            individual[member_idx] = owner1 + 1
                        return

    def _optimize_random_group(
        self, individual: List[int], nodes: List[int], graph: "nx.Graph", config: "LicenseConfig"
    ) -> None:
        """Optimize a random group by adding/removing members."""
        groups = self._get_groups_from_individual(individual)
        
        if groups:
            owner = random.choice(list(groups.keys()))
            members = groups[owner]
            
            # Try to add a neighbor
            owner_node = nodes[owner]
            for neighbor in graph.neighbors(owner_node):
                if neighbor in nodes:
                    neighbor_idx = nodes.index(neighbor)
                    if individual[neighbor_idx] == 0 and len(members) < config.group_size:
                        individual[neighbor_idx] = owner + 1
                        break

    def _get_groups_from_individual(self, individual: List[int]) -> dict:
        """Get groups from individual representation."""
        groups = {}
        for i, assignment in enumerate(individual):
            if assignment > 0:
                owner = assignment - 1
                if owner not in groups:
                    groups[owner] = []
                groups[owner].append(i)
        return groups

    def _adapt_solution_to_graph(
        self, 
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Adapt a solution to work with a modified graph.
        
        Args:
            solution: Previous solution.
            graph: Current graph.
            config: License configuration.
            
        Returns:
            Adapted solution valid for current graph.
        """
        from ...models.license import LicenseSolution
        
        current_nodes = set(graph.nodes())
        
        # Filter solo nodes to only include existing nodes
        new_solo = [node for node in solution.solo_nodes if node in current_nodes]
        
        # Adapt groups
        new_groups = {}
        for owner, members in solution.group_owners.items():
            if owner in current_nodes:
                # Keep only existing members
                valid_members = [m for m in members if m in current_nodes]
                # Ensure all members are still connected to owner
                connected_members = [m for m in valid_members 
                                   if m == owner or graph.has_edge(owner, m)]
                
                if len(connected_members) > 1:  # Valid group
                    new_groups[owner] = connected_members
                elif connected_members:  # Only owner left
                    new_solo.append(owner)
        
        # Handle any nodes that weren't assigned
        assigned_nodes = set(new_solo) | set().union(*new_groups.values()) if new_groups else set(new_solo)
        unassigned = current_nodes - assigned_nodes
        new_solo.extend(list(unassigned))
        
        return LicenseSolution(solo_nodes=new_solo, group_owners=new_groups)

    # ...existing methods from original implementation...
    def _encode_solution(
        self,
        solution: "LicenseSolution",
        nodes: List[int],
    ) -> List[int]:
        """Encode LicenseSolution to individual representation."""
        individual = [0] * len(nodes)
        
        # Encode solo nodes
        for node in solution.solo_nodes:
            if node in nodes:
                individual[nodes.index(node)] = 0
        
        # Encode group memberships
        for owner, members in solution.group_owners.items():
            if owner in nodes:
                owner_idx = nodes.index(owner)
                for member in members:
                    if member in nodes:
                        member_idx = nodes.index(member)
                        individual[member_idx] = owner_idx + 1
        
        return individual

    def _repair_individual(
        self,
        individual: List[int],
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> List[int]:
        """Repair an individual to ensure validity."""
        nodes = list(graph.nodes())
        repaired = individual.copy()

        # Count group sizes
        group_sizes = {}
        for i, assignment in enumerate(repaired):
            if assignment > 0:
                owner_idx = assignment - 1
                if owner_idx not in group_sizes:
                    group_sizes[owner_idx] = 0
                group_sizes[owner_idx] += 1

        # Fix group size violations
        for owner_idx, size in group_sizes.items():
            if size > config.group_size:
                members = [i for i, assignment in enumerate(repaired) if assignment == owner_idx + 1]
                excess = size - config.group_size
                for _ in range(excess):
                    member_idx = random.choice(members)
                    repaired[member_idx] = 0
                    members.remove(member_idx)

        # Fix connectivity violations
        for i, assignment in enumerate(repaired):
            if assignment > 0:
                owner_idx = assignment - 1
                node = nodes[i]
                owner_node = nodes[owner_idx]

                if node != owner_node and not graph.has_edge(node, owner_node):
                    repaired[i] = 0

        return repaired

    def _evaluate_fitness(
        self,
        individual: List[int],
        config: "LicenseConfig",
    ) -> float:
        """Evaluate fitness with additional quality metrics."""
        # Basic cost
        solo_count = sum(1 for assignment in individual if assignment == 0)
        group_owners = set()
        for assignment in individual:
            if assignment > 0:
                group_owners.add(assignment - 1)
        group_count = len(group_owners)
        
        base_cost = solo_count * config.solo_price + group_count * config.group_price
        
        # Penalty for fragmentation (many small groups)
        fragmentation_penalty = 0
        if group_count > 0:
            avg_group_size = (len(individual) - solo_count) / group_count
            if avg_group_size < 2.5:  # Prefer larger groups
                fragmentation_penalty = group_count * 0.1
        
        return base_cost + fragmentation_penalty

    def _tournament_selection(
        self,
        population: List[List[int]],
        fitness_scores: List[float],
    ) -> List[int]:
        """Tournament selection with adaptive tournament size."""
        tournament_size = max(2, min(5, len(population) // 10))
        tournament_indices = random.sample(
            range(len(population)),
            min(tournament_size, len(population)),
        )
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def _decode_individual(
        self,
        individual: List[int],
        nodes: List[int],
    ) -> "LicenseSolution":
        """Decode individual to LicenseSolution."""
        from ...models.license import LicenseSolution

        solo_nodes = []
        group_owners = {}

        for i, assignment in enumerate(individual):
            node = nodes[i]
            if assignment == 0:
                solo_nodes.append(node)
            else:
                owner_idx = assignment - 1
                owner_node = nodes[owner_idx]

                if owner_node not in group_owners:
                    group_owners[owner_node] = []
                group_owners[owner_node].append(node)

        return LicenseSolution(solo_nodes=solo_nodes, group_owners=group_owners)
