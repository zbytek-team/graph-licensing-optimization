from ..core.types import LicenseType, Solution, Algorithm, LicenseGroup
from .greedy import GreedyAlgorithm
from ..utils.validation import SolutionValidator
from ..utils.solution_utils import SolutionBuilder
from ..utils.mutation_operators import MutationOperators

from typing import Any, List, Tuple
import random
import networkx as nx


class GeneticAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "genetic_algorithm"

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        # Initialize utilities
        self.validator = SolutionValidator()

        population_size = kwargs.get("population_size", 50)
        max_generations = kwargs.get("max_generations", 200)  # Increased from 100
        mutation_rate = kwargs.get("mutation_rate", 0.3)  # Increased mutation rate
        crossover_rate = kwargs.get("crossover_rate", 0.7)
        elitism_count = kwargs.get("elitism_count", 3)  # Reduced elitism

        # Initialize population with more diversity
        population = self._initialize_population(graph, license_types, population_size)

        best_solution = min(population, key=lambda sol: sol.total_cost)

        # Track generations without improvement
        generations_without_improvement = 0
        max_stagnation = 20

        for generation in range(max_generations):
            # Evaluate fitness
            fitness_scores = [self._calculate_fitness(solution) for solution in population]

            # Track best solution and apply local search if needed
            current_best = min(population, key=lambda sol: sol.total_cost)
            if current_best.total_cost < best_solution.total_cost:
                best_solution = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

                # Apply intensive local search to best solution when stagnating
                if generations_without_improvement > 8 and generation % 3 == 0:
                    improved_best = self._intensive_local_search(best_solution, graph, license_types)
                    if improved_best.total_cost < best_solution.total_cost:
                        best_solution = improved_best
                        generations_without_improvement = 0
                        # Replace worst solution in population with improved best
                        worst_idx = min(range(len(population)), key=lambda i: fitness_scores[i])
                        population[worst_idx] = best_solution

            # Adaptive mutation rate - increase when stagnating
            adaptive_mutation_rate = mutation_rate
            if generations_without_improvement > 10:
                adaptive_mutation_rate = min(0.9, mutation_rate * 3)  # More aggressive
            elif generations_without_improvement > 5:
                adaptive_mutation_rate = min(0.7, mutation_rate * 2)

            # Early stopping if no improvement for too long
            if generations_without_improvement > max_stagnation:
                break

            # Selection and reproduction
            new_population = []

            # Elitism - keep more best solutions when stagnating
            elitism_size = elitism_count
            if generations_without_improvement > 5:
                elitism_size = min(elitism_count * 2, population_size // 4)  # Increase elitism when stagnating

            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elitism_size]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Generate offspring
            while len(new_population) < population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, graph, license_types)
                else:
                    child1, child2 = parent1, parent2

                # Mutation with adaptive rate
                if random.random() < adaptive_mutation_rate:
                    child1 = self._mutate(child1, graph, license_types)
                if random.random() < adaptive_mutation_rate:
                    child2 = self._mutate(child2, graph, license_types)

                # Add valid children to new population
                if child1 and self._is_valid_solution(child1, graph):
                    new_population.append(child1)
                if child2 and self._is_valid_solution(child2, graph) and len(new_population) < population_size:
                    new_population.append(child2)

            # Fill population if needed with random solutions
            while len(new_population) < population_size:
                solution = self._generate_truly_random_solution(graph, license_types)
                if solution and self._is_valid_solution(solution, graph):
                    new_population.append(solution)
                else:
                    # Last resort: duplicate a good solution with mutation
                    base_solution = random.choice(new_population[:elitism_count] if new_population else [best_solution])
                    mutated = self._intensive_mutate(base_solution, graph, license_types)
                    new_population.append(mutated)

            population = new_population[:population_size]

        return best_solution

    def _initialize_population(self, graph: nx.Graph, license_types: List[LicenseType], population_size: int) -> List[Solution]:
        """Initialize population with truly diverse solutions"""
        population = []

        # Method 1: Add pure greedy solution (only 1)
        greedy_solver = GreedyAlgorithm()
        greedy_solution = greedy_solver.solve(graph, license_types)
        population.append(greedy_solution)

        # Method 2: Generate solutions with different license preferences (10% of population)
        license_preference_count = max(1, population_size // 10)
        for _ in range(license_preference_count):
            # Bias towards different license types
            biased_license_types = license_types.copy()
            random.shuffle(biased_license_types)
            solution = self._generate_biased_solution(graph, biased_license_types)
            if solution and self._is_valid_solution(solution, graph):
                population.append(solution)

        # Method 3: Generate solutions with different clustering strategies (20% of population)
        clustering_count = max(1, population_size // 5)
        for _ in range(clustering_count):
            solution = self._generate_clustering_based_solution(graph, license_types)
            if solution and self._is_valid_solution(solution, graph):
                population.append(solution)

        # Method 4: Generate solutions with random node ordering (30% of population)
        random_count = max(1, int(population_size * 0.3))
        for _ in range(random_count):
            solution = self._generate_random_order_solution(graph, license_types)
            if solution and self._is_valid_solution(solution, graph):
                population.append(solution)

        # Method 5: Generate completely random valid solutions for remaining slots
        while len(population) < population_size:
            solution = self._generate_completely_random_solution(graph, license_types)
            if solution and self._is_valid_solution(solution, graph):
                population.append(solution)
            else:
                # Fallback: take greedy and mutate it heavily
                mutated = self._intensive_mutate(greedy_solution, graph, license_types)
                population.append(mutated)

        return population

    def _generate_truly_random_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Generate a completely random solution using different approach"""
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []

        # Try to create random groups by picking random nodes and their neighborhoods
        max_attempts = len(nodes) * 2
        attempts = 0

        while uncovered and attempts < max_attempts:
            attempts += 1

            if not uncovered:
                break

            # Pick random uncovered node as potential owner
            owner = random.choice(list(uncovered))

            # Get all neighbors of this node
            owner_neighbors = set(graph.neighbors(owner)) | {owner}
            available = owner_neighbors & uncovered

            if not available:
                # Create single node group
                license_type = random.choice([lt for lt in license_types if lt.min_capacity <= 1])
                group = LicenseGroup(license_type, owner, set())
                groups.append(group)
                uncovered.remove(owner)
                continue

            # Try different license types randomly
            random.shuffle(license_types)
            group_created = False

            for license_type in license_types:
                max_size = min(len(available), license_type.max_capacity)
                min_size = license_type.min_capacity

                if max_size >= min_size:
                    # Random group size within constraints
                    group_size = random.randint(min_size, max_size)

                    # Randomly select members with different strategies
                    if random.random() < 0.5:
                        # Strategy 1: Pick highest degree nodes
                        available_list = list(available)
                        available_list.sort(key=lambda n: graph.degree(n), reverse=True)
                        members = available_list[:group_size]
                    else:
                        # Strategy 2: Completely random
                        members = random.sample(list(available), group_size)

                    additional_members = set(members) - {owner}
                    group = LicenseGroup(license_type, owner, additional_members)
                    groups.append(group)
                    uncovered -= set(members)
                    group_created = True
                    break

            if not group_created:
                # Force create single node group
                license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                group = LicenseGroup(license_type, owner, set())
                groups.append(group)
                uncovered.remove(owner)

        # Handle any remaining nodes
        while uncovered:
            node = uncovered.pop()
            license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
            group = LicenseGroup(license_type, node, set())
            groups.append(group)

        return self._create_solution_from_groups(groups)

    def _calculate_diversity(self, population: List[Solution]) -> float:
        """Calculate population diversity based on cost variance"""
        if len(population) <= 1:
            return 0.0

        costs = [sol.total_cost for sol in population]
        mean_cost = sum(costs) / len(costs)
        variance = sum((cost - mean_cost) ** 2 for cost in costs) / len(costs)
        return variance**0.5  # Standard deviation

    def _intensive_mutate(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Apply multiple mutations to create more diverse solutions"""
        mutated = solution

        # Apply 2-4 random mutations
        num_mutations = random.randint(2, 4)

        for _ in range(num_mutations):
            mutation_operators = [self._mutate_change_license, self._mutate_reassign_member, self._mutate_merge_groups, self._mutate_split_group]

            operator = random.choice(mutation_operators)
            new_mutated = operator(mutated, graph, license_types)

            if new_mutated and self._is_valid_solution(new_mutated, graph):
                mutated = new_mutated

        return mutated

    def _generate_biased_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Generate solution with bias towards specific license types"""
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        uncovered = set(nodes)
        groups = []

        # Prefer the first license type in the shuffled list
        preferred_license = license_types[0]

        while uncovered:
            # Pick random uncovered node as owner
            owner = random.choice(list(uncovered))

            # Try to use preferred license first
            neighbors = set(graph.neighbors(owner)) & uncovered
            available = neighbors | {owner}

            if preferred_license.min_capacity <= len(available) <= preferred_license.max_capacity:
                # Use preferred license
                group_size = min(len(available), preferred_license.max_capacity)
                members = random.sample(list(available), group_size)
                additional_members = set(members) - {owner}
                group = LicenseGroup(preferred_license, owner, additional_members)
                groups.append(group)
                uncovered -= set(members)
            else:
                # Fall back to any compatible license
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
                    # Force single node group
                    license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                    group = LicenseGroup(license_type, owner, set())
                    groups.append(group)
                    uncovered.remove(owner)

        return self._create_solution_from_groups(groups)

    def _generate_clustering_based_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Generate solution based on graph clustering"""
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []

        # Use node degree as clustering heuristic
        nodes_by_degree = sorted(nodes, key=lambda n: graph.degree(n), reverse=True)

        for owner in nodes_by_degree:
            if owner not in uncovered:
                continue

            # Get all connected uncovered neighbors
            neighbors = set(graph.neighbors(owner)) & uncovered
            available = neighbors | {owner}

            # Try to create largest possible group
            best_group = None
            best_efficiency = 0

            for license_type in license_types:
                max_size = min(len(available), license_type.max_capacity)
                if max_size >= license_type.min_capacity:
                    # Calculate efficiency (coverage per cost)
                    efficiency = max_size / license_type.cost
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        group_size = max_size
                        members = list(available)[:group_size]
                        additional_members = set(members) - {owner}
                        best_group = LicenseGroup(license_type, owner, additional_members)

            if best_group:
                groups.append(best_group)
                uncovered -= best_group.all_members
            else:
                # Fallback single node
                license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                group = LicenseGroup(license_type, owner, set())
                groups.append(group)
                uncovered.remove(owner)

        return self._create_solution_from_groups(groups)

    def _generate_random_order_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Generate solution by processing nodes in random order"""
        nodes = list(graph.nodes())
        random.shuffle(nodes)  # Different random order each time
        uncovered = set(nodes)
        groups = []

        for owner in nodes:
            if owner not in uncovered:
                continue

            # Get available neighbors
            neighbors = set(graph.neighbors(owner)) & uncovered
            available = neighbors | {owner}

            # Choose random license type and group size
            compatible_licenses = [lt for lt in license_types if lt.min_capacity <= len(available) <= lt.max_capacity]

            if compatible_licenses:
                license_type = random.choice(compatible_licenses)
                max_size = min(len(available), license_type.max_capacity)
                group_size = random.randint(license_type.min_capacity, max_size)
                members = random.sample(list(available), group_size)
                additional_members = set(members) - {owner}
                group = LicenseGroup(license_type, owner, additional_members)
                groups.append(group)
                uncovered -= set(members)
            else:
                # Single node fallback
                license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                group = LicenseGroup(license_type, owner, set())
                groups.append(group)
                uncovered.remove(owner)

        return self._create_solution_from_groups(groups)

    def _generate_completely_random_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Generate completely random valid solution with high diversity"""
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []

        max_attempts = 1000
        attempts = 0

        while uncovered and attempts < max_attempts:
            attempts += 1

            # Pick completely random uncovered node
            owner = random.choice(list(uncovered))

            # Get random subset of neighbors (not necessarily connected)
            all_uncovered = list(uncovered)
            available = {owner}

            # Add random nodes (may not be neighbors - this creates more diversity)
            num_to_add = random.randint(0, min(5, len(all_uncovered) - 1))
            if num_to_add > 0:
                additional_nodes = random.sample([n for n in all_uncovered if n != owner], num_to_add)
                available.update(additional_nodes)

            # Choose random license
            license_type = random.choice(license_types)

            # Adjust group size to fit license constraints
            max_size = min(len(available), license_type.max_capacity)
            min_size = license_type.min_capacity

            if max_size >= min_size:
                group_size = random.randint(min_size, max_size)
                members = random.sample(list(available), group_size)
                additional_members = set(members) - {owner}
                group = LicenseGroup(license_type, owner, additional_members)
                groups.append(group)
                uncovered -= set(members)
            else:
                # Single node group
                license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                group = LicenseGroup(license_type, owner, set())
                groups.append(group)
                uncovered.remove(owner)

        return self._create_solution_from_groups(groups)

    def _generate_random_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Generate a random valid solution"""
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []

        max_attempts = 1000
        attempts = 0

        while uncovered and attempts < max_attempts:
            attempts += 1

            # Pick a random uncovered node as owner
            if not uncovered:
                break

            owner = random.choice(list(uncovered))

            # Get neighbors within the uncovered set
            owner_neighbors = set(graph.neighbors(owner)) | {owner}
            available = owner_neighbors & uncovered

            if not available:
                # Force single node group
                license_type = random.choice([lt for lt in license_types if lt.min_capacity <= 1])
                group = LicenseGroup(license_type, owner, set())
                groups.append(group)
                uncovered.remove(owner)
                continue

            # Choose random license type
            license_type = random.choice(license_types)

            # Determine group size
            max_size = min(len(available), license_type.max_capacity)
            min_size = max(1, license_type.min_capacity)

            if max_size < min_size:
                # Find a license type that fits
                compatible_licenses = [lt for lt in license_types if lt.min_capacity <= len(available) <= lt.max_capacity]
                if compatible_licenses:
                    license_type = random.choice(compatible_licenses)
                    max_size = min(len(available), license_type.max_capacity)
                    min_size = license_type.min_capacity
                else:
                    continue

            group_size = random.randint(min_size, max_size)
            members = random.sample(list(available), group_size)

            additional_members = set(members) - {owner}
            group = LicenseGroup(license_type, owner, additional_members)
            groups.append(group)

            uncovered -= set(members)

        # Handle remaining uncovered nodes
        while uncovered:
            node = uncovered.pop()
            license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
            group = LicenseGroup(license_type, node, set())
            groups.append(group)

        return self._create_solution_from_groups(groups)

    def _generate_randomized_greedy(self, graph: nx.Graph, license_types: List[LicenseType], strategy: int = 0) -> Solution:
        """Generate solution using randomized greedy approach with different strategies"""
        nodes = list(graph.nodes())
        uncovered_nodes = set(nodes)
        groups = []

        # Different node ordering strategies
        if strategy % 3 == 0:
            # Strategy 0: Random order
            random.shuffle(nodes)
        elif strategy % 3 == 1:
            # Strategy 1: Degree-based order (high degree first)
            nodes.sort(key=lambda n: graph.degree(n), reverse=True)
            # Add some randomness
            for i in range(0, len(nodes), 3):
                end_idx = min(i + 3, len(nodes))
                sublist = nodes[i:end_idx]
                random.shuffle(sublist)
                nodes[i:end_idx] = sublist
        else:
            # Strategy 2: Low degree first with randomness
            nodes.sort(key=lambda n: graph.degree(n))
            for i in range(0, len(nodes), 3):
                end_idx = min(i + 3, len(nodes))
                sublist = nodes[i:end_idx]
                random.shuffle(sublist)
                nodes[i:end_idx] = sublist

        for node in nodes:
            if node not in uncovered_nodes:
                continue

            neighbors = set(graph.neighbors(node)) | {node}
            available_neighbors = neighbors & uncovered_nodes

            if not available_neighbors:
                continue

            # License type selection strategy
            if strategy % 2 == 0:
                # Prefer cheaper licenses
                sorted_licenses = sorted(license_types, key=lambda lt: lt.cost)
            else:
                # Random license selection
                sorted_licenses = license_types.copy()
                random.shuffle(sorted_licenses)

            group_created = False
            for license_type in sorted_licenses:
                potential_members = list(available_neighbors)
                max_members = min(len(potential_members), license_type.max_capacity)

                if max_members < license_type.min_capacity:
                    continue

                # Member selection strategy
                if strategy % 4 == 0:
                    # Random selection
                    random.shuffle(potential_members)
                    group_members = set(potential_members[:max_members])
                elif strategy % 4 == 1:
                    # Prefer high degree neighbors
                    potential_members.sort(key=lambda n: graph.degree(n), reverse=True)
                    group_members = set(potential_members[:max_members])
                elif strategy % 4 == 2:
                    # Prefer low degree neighbors
                    potential_members.sort(key=lambda n: graph.degree(n))
                    group_members = set(potential_members[:max_members])
                else:
                    # Mixed strategy
                    random.shuffle(potential_members)
                    mid_point = max_members // 2
                    group_members = set(potential_members[:mid_point])
                    if mid_point < max_members:
                        remaining = [n for n in potential_members[mid_point:] if n not in group_members]
                        if remaining:
                            group_members.update(random.sample(remaining, min(len(remaining), max_members - len(group_members))))

                additional_members = group_members - {node}
                group = LicenseGroup(license_type, node, additional_members)
                groups.append(group)
                uncovered_nodes -= group_members
                group_created = True
                break

            if not group_created and node in uncovered_nodes:
                # Create single node group as fallback
                license_type = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                group = LicenseGroup(license_type, node, set())
                groups.append(group)
                uncovered_nodes.remove(node)

        return self._create_solution_from_groups(groups)

    def _calculate_fitness(self, solution: Solution) -> float:
        """Calculate fitness score (higher is better)"""
        # Use inverse of cost as fitness, add small constant to avoid division by zero
        return 1.0 / (solution.total_cost + 1.0)

    def _tournament_selection(self, population: List[Solution], fitness_scores: List[float], tournament_size: int = 3) -> Solution:
        """Select parent using tournament selection"""
        if not population:
            raise ValueError("Population is empty")

        tournament_size = min(tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def _crossover(self, parent1: Solution, parent2: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Tuple[Solution, Solution]:
        """Advanced crossover combining different regions of solutions"""

        # Method 1: Node-based crossover - divide nodes into regions
        all_nodes = list(graph.nodes())
        random.shuffle(all_nodes)

        # Split nodes roughly in half
        split_point = len(all_nodes) // 2
        region1_nodes = set(all_nodes[:split_point])
        region2_nodes = set(all_nodes[split_point:])

        # Create child 1: take groups from parent1 that cover region1, parent2 for region2
        child1_groups = []
        child1_covered = set()

        # Add groups from parent1 that have majority of nodes in region1
        for group in parent1.groups:
            if len(group.all_members & region1_nodes) > len(group.all_members & region2_nodes):
                child1_groups.append(group)
                child1_covered.update(group.all_members)

        # Add groups from parent2 that cover uncovered nodes
        for group in parent2.groups:
            if group.all_members - child1_covered:  # If group covers some uncovered nodes
                # Check if we can add this group (no overlap)
                if not (group.all_members & child1_covered):
                    child1_groups.append(group)
                    child1_covered.update(group.all_members)

        # Create child 2: opposite regions
        child2_groups = []
        child2_covered = set()

        # Add groups from parent2 that have majority of nodes in region1
        for group in parent2.groups:
            if len(group.all_members & region1_nodes) > len(group.all_members & region2_nodes):
                child2_groups.append(group)
                child2_covered.update(group.all_members)

        # Add groups from parent1 that cover uncovered nodes
        for group in parent1.groups:
            if group.all_members - child2_covered:  # If group covers some uncovered nodes
                # Check if we can add this group (no overlap)
                if not (group.all_members & child2_covered):
                    child2_groups.append(group)
                    child2_covered.update(group.all_members)

        # Repair both children to ensure all nodes are covered
        child1 = self._repair_solution(child1_groups, graph, license_types)
        child2 = self._repair_solution(child2_groups, graph, license_types)

        return child1, child2

    def _select_groups_for_child(self, all_groups: List[LicenseGroup], graph: nx.Graph, bias: float) -> List[LicenseGroup]:
        """Select groups for child solution with some bias"""
        if not all_groups:
            return []

        nodes = set(graph.nodes())
        covered = set()
        selected_groups = []

        # Shuffle groups and select based on bias
        shuffled_groups = all_groups.copy()
        random.shuffle(shuffled_groups)

        for group in shuffled_groups:
            # Check if this group would create overlap
            if group.all_members & covered:
                continue

            # Select group based on bias (probability)
            if random.random() < bias:
                selected_groups.append(group)
                covered.update(group.all_members)

            # Stop if all nodes are covered
            if covered == nodes:
                break

        return selected_groups

    def _repair_solution(self, groups: List[LicenseGroup], graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Repair solution to ensure all nodes are covered"""
        all_nodes = set(graph.nodes())
        covered = set()

        for group in groups:
            covered.update(group.all_members)

        uncovered = all_nodes - covered
        repaired_groups = groups.copy()

        # Cover uncovered nodes
        while uncovered:
            node = uncovered.pop()

            # Try to add to existing group
            added = False
            for i, group in enumerate(repaired_groups):
                if group.size < group.license_type.max_capacity and node in set(graph.neighbors(group.owner)) | {group.owner}:
                    new_additional = group.additional_members | {node}
                    new_group = LicenseGroup(group.license_type, group.owner, new_additional)
                    repaired_groups[i] = new_group
                    added = True
                    break

            if not added:
                # Create new single-node group
                single_node_licenses = [lt for lt in license_types if lt.min_capacity <= 1]
                if single_node_licenses:
                    license_type = min(single_node_licenses, key=lambda lt: lt.cost)
                else:
                    license_type = min(license_types, key=lambda lt: lt.cost)
                new_group = LicenseGroup(license_type, node, set())
                repaired_groups.append(new_group)

        return self._create_solution_from_groups(repaired_groups)

    def _mutate(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Mutate solution with exploration vs exploitation balance"""

        # 80% chance for exploitative mutations (improve or maintain quality)
        # 20% chance for explorative mutations (accept worse solutions for diversity)
        exploit_probability = 0.8

        if random.random() < exploit_probability:
            # EXPLOITATIVE MUTATION: Try to improve or maintain quality
            best_mutations = []

            # Generate multiple mutation candidates
            for _ in range(5):
                mutation_operators = [
                    self._mutate_change_license,
                    self._mutate_reassign_member,
                    self._mutate_merge_groups,
                    self._mutate_split_group,
                    self._mutate_local_search,
                ]

                operator = random.choice(mutation_operators)
                mutated = operator(solution, graph, license_types)

                if mutated and self._is_valid_solution(mutated, graph):
                    best_mutations.append(mutated)

            # Return the best mutation (or allow slightly worse for diversity)
            if best_mutations:
                best_mutation = min(best_mutations, key=lambda sol: sol.total_cost)
                if best_mutation.total_cost <= solution.total_cost * 1.02:  # Allow 2% worse
                    return best_mutation
        else:
            # EXPLORATIVE MUTATION: Accept potentially worse solutions for diversity
            explorative_mutations = [self._mutate_destructive_reconstruct, self._mutate_random_shake, self._mutate_license_shuffle]

            operator = random.choice(explorative_mutations)
            mutated = operator(solution, graph, license_types)

            if mutated and self._is_valid_solution(mutated, graph):
                # Accept even if worse (for exploration)
                return mutated

        return solution

    def _mutate_local_search(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Advanced mutation using local search principles"""
        # Try to improve the solution by looking for better license assignments
        best_solution = solution

        # Check if we can improve any group by changing its license type
        for i, group in enumerate(solution.groups):
            for license_type in license_types:
                if license_type != group.license_type and license_type.min_capacity <= group.size <= license_type.max_capacity:
                    # Create new solution with changed license
                    new_groups = solution.groups.copy()
                    new_group = LicenseGroup(license_type, group.owner, group.additional_members)
                    new_groups[i] = new_group

                    candidate = self._create_solution_from_groups(new_groups)
                    if candidate.total_cost < best_solution.total_cost:
                        best_solution = candidate

        # Try to merge small expensive groups
        if len(solution.groups) > 1:
            # Find expensive small groups
            expensive_groups = [(i, g) for i, g in enumerate(solution.groups) if g.size <= 2 and g.license_type.cost > 20]

            if len(expensive_groups) >= 2:
                # Try to merge two of them
                (i1, g1), (i2, g2) = random.sample(expensive_groups, 2)
                all_members = g1.all_members | g2.all_members

                # Find suitable license and owner
                for license_type in license_types:
                    if license_type.min_capacity <= len(all_members) <= license_type.max_capacity:
                        for potential_owner in [g1.owner, g2.owner]:
                            owner_neighbors = set(graph.neighbors(potential_owner)) | {potential_owner}
                            if all_members.issubset(owner_neighbors):
                                # Create merged group
                                new_additional = all_members - {potential_owner}
                                new_group = LicenseGroup(license_type, potential_owner, new_additional)

                                # Create new solution
                                new_groups = [g for j, g in enumerate(solution.groups) if j not in [i1, i2]]
                                new_groups.append(new_group)

                                candidate = self._create_solution_from_groups(new_groups)
                                if candidate.total_cost < best_solution.total_cost:
                                    best_solution = candidate
                                    break
                        if best_solution != solution:
                            break
                    if best_solution != solution:
                        break

        return best_solution

    def _mutate_change_license(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Mutation: change license type of a random group"""
        if not solution.groups:
            return solution

        group = random.choice(solution.groups)
        compatible_licenses = [lt for lt in license_types if lt != group.license_type and lt.min_capacity <= group.size <= lt.max_capacity]

        if not compatible_licenses:
            return solution

        new_license = random.choice(compatible_licenses)
        new_groups = []

        for g in solution.groups:
            if g == group:
                new_group = LicenseGroup(new_license, g.owner, g.additional_members)
                new_groups.append(new_group)
            else:
                new_groups.append(g)

        return self._create_solution_from_groups(new_groups)

    def _mutate_reassign_member(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Mutation: reassign member between groups"""
        if len(solution.groups) < 2:
            return solution

        # Find eligible groups
        can_lose = [g for g in solution.groups if g.size > g.license_type.min_capacity and g.additional_members]
        can_gain = [g for g in solution.groups if g.size < g.license_type.max_capacity]

        if not can_lose or not can_gain:
            return solution

        from_group = random.choice(can_lose)
        potential_to_groups = [g for g in can_gain if g != from_group]

        if not potential_to_groups:
            return solution

        to_group = random.choice(potential_to_groups)

        if not from_group.additional_members:
            return solution

        member = random.choice(list(from_group.additional_members))

        # Check connectivity
        to_owner_neighbors = set(graph.neighbors(to_group.owner)) | {to_group.owner}
        if member not in to_owner_neighbors:
            return solution

        # Create mutated solution
        new_groups = []
        for g in solution.groups:
            if g == from_group:
                new_additional = g.additional_members - {member}
                new_groups.append(LicenseGroup(g.license_type, g.owner, new_additional))
            elif g == to_group:
                new_additional = g.additional_members | {member}
                new_groups.append(LicenseGroup(g.license_type, g.owner, new_additional))
            else:
                new_groups.append(g)

        return self._create_solution_from_groups(new_groups)

    def _mutate_merge_groups(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Mutation: merge two adjacent groups"""
        if len(solution.groups) < 2:
            return solution

        group1, group2 = random.sample(solution.groups, 2)
        all_members = group1.all_members | group2.all_members

        # Find compatible license types
        compatible_licenses = [lt for lt in license_types if lt.min_capacity <= len(all_members) <= lt.max_capacity]

        if not compatible_licenses:
            return solution

        license_type = random.choice(compatible_licenses)

        # Try owners from both groups
        for owner in [group1.owner, group2.owner]:
            owner_neighbors = set(graph.neighbors(owner)) | {owner}
            if all_members.issubset(owner_neighbors):
                new_additional = all_members - {owner}
                new_group = LicenseGroup(license_type, owner, new_additional)

                new_groups = [g for g in solution.groups if g not in [group1, group2]]
                new_groups.append(new_group)

                return self._create_solution_from_groups(new_groups)

        return solution

    def _mutate_split_group(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Mutation: split a group into smaller groups"""
        if not solution.groups:
            return solution

        # Find groups that can be split
        splittable = [g for g in solution.groups if g.size > 2]
        if not splittable:
            return solution

        group = random.choice(splittable)
        members = list(group.all_members)

        # Try a few random splits
        for _ in range(3):
            random.shuffle(members)
            split_point = random.randint(1, len(members) - 1)

            members1 = members[:split_point]
            members2 = members[split_point:]

            # Find valid license types for both parts
            for lt1 in license_types:
                if lt1.min_capacity <= len(members1) <= lt1.max_capacity:
                    for lt2 in license_types:
                        if lt2.min_capacity <= len(members2) <= lt2.max_capacity:
                            # Check connectivity for both parts
                            owner1 = random.choice(members1)
                            owner1_neighbors = set(graph.neighbors(owner1)) | {owner1}

                            owner2 = random.choice(members2)
                            owner2_neighbors = set(graph.neighbors(owner2)) | {owner2}

                            if set(members1).issubset(owner1_neighbors) and set(members2).issubset(owner2_neighbors):
                                additional1 = set(members1) - {owner1}
                                additional2 = set(members2) - {owner2}

                                group1 = LicenseGroup(lt1, owner1, additional1)
                                group2 = LicenseGroup(lt2, owner2, additional2)

                                new_groups = [g for g in solution.groups if g != group]
                                new_groups.extend([group1, group2])

                                return self._create_solution_from_groups(new_groups)

        return solution

    def _create_solution_from_groups(self, groups: List[LicenseGroup]) -> Solution:
        """Create a solution from a list of groups"""
        return SolutionBuilder.create_solution_from_groups(groups)

    def _is_valid_solution(self, solution: Solution, graph: nx.Graph) -> bool:
        """Check if a solution is valid"""
        return self.validator.is_valid_solution(solution, graph)

    def _mutate_destructive_reconstruct(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Destructive mutation: destroy and reconstruct part of the solution"""
        # Choose 20-40% of groups to destroy and reconstruct
        num_groups_to_destroy = random.randint(len(solution.groups) // 5, 2 * len(solution.groups) // 5)

        if num_groups_to_destroy == 0:
            return solution

        # Select groups to destroy
        groups_to_destroy = random.sample(solution.groups, min(num_groups_to_destroy, len(solution.groups)))
        remaining_groups = [g for g in solution.groups if g not in groups_to_destroy]

        # Collect nodes from destroyed groups
        nodes_to_reassign = set()
        for group in groups_to_destroy:
            nodes_to_reassign.update(group.all_members)

        # Reconstruct these nodes using random strategy
        new_groups = []
        uncovered = nodes_to_reassign.copy()

        while uncovered:
            owner = random.choice(list(uncovered))

            # Get available neighbors within uncovered nodes
            owner_neighbors = set(graph.neighbors(owner)) & uncovered
            available = owner_neighbors | {owner}

            # Choose random license type
            license_type = random.choice(license_types)

            # Create group of random size within constraints
            max_size = min(len(available), license_type.max_capacity)
            min_size = license_type.min_capacity

            if max_size >= min_size:
                group_size = random.randint(min_size, max_size)
                members = random.sample(list(available), group_size)
                additional_members = set(members) - {owner}
                group = LicenseGroup(license_type, owner, additional_members)
                new_groups.append(group)
                uncovered -= set(members)
            else:
                # Single node fallback
                fallback_license = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 else float("inf"))
                group = LicenseGroup(fallback_license, owner, set())
                new_groups.append(group)
                uncovered.remove(owner)

        # Combine remaining and new groups
        all_groups = remaining_groups + new_groups
        return self._create_solution_from_groups(all_groups)

    def _mutate_random_shake(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Random shake mutation: make multiple random changes"""
        mutated = solution

        # Apply 3-6 random small changes
        num_changes = random.randint(3, 6)

        for _ in range(num_changes):
            operations = [self._mutate_change_license, self._mutate_reassign_member, self._mutate_split_group]

            operation = random.choice(operations)
            new_mutated = operation(mutated, graph, license_types)

            if new_mutated and self._is_valid_solution(new_mutated, graph):
                mutated = new_mutated

        return mutated

    def _mutate_license_shuffle(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """License shuffle mutation: randomly change license types of groups"""
        new_groups = []

        for group in solution.groups:
            # 30% chance to change license type
            if random.random() < 0.3:
                # Find alternative license types that fit the group size
                compatible_licenses = [lt for lt in license_types if lt.min_capacity <= group.size <= lt.max_capacity]

                if compatible_licenses and len(compatible_licenses) > 1:
                    # Choose different license type
                    new_license_options = [lt for lt in compatible_licenses if lt != group.license_type]
                    if new_license_options:
                        new_license = random.choice(new_license_options)
                        new_group = LicenseGroup(new_license, group.owner, group.additional_members)
                        new_groups.append(new_group)
                        continue

            # Keep original group
            new_groups.append(group)

        return self._create_solution_from_groups(new_groups)

    def _intensive_local_search(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Intensive local search to improve the best solution"""
        current_best = solution

        # Try multiple improvement strategies
        improvements = [self._local_search_license_optimization, self._local_search_group_merging, self._local_search_member_reassignment]

        # Apply each improvement strategy
        for improvement_method in improvements:
            improved = improvement_method(current_best, graph, license_types)
            if improved and improved.total_cost < current_best.total_cost:
                current_best = improved

        return current_best

    def _local_search_license_optimization(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Try to find better license types for each group"""
        best_solution = solution

        for i, group in enumerate(solution.groups):
            for license_type in license_types:
                if license_type != group.license_type and license_type.min_capacity <= group.size <= license_type.max_capacity:
                    # Create new solution with changed license
                    new_groups = solution.groups.copy()
                    new_group = LicenseGroup(license_type, group.owner, group.additional_members)
                    new_groups[i] = new_group

                    candidate = self._create_solution_from_groups(new_groups)
                    if candidate.total_cost < best_solution.total_cost:
                        best_solution = candidate

        return best_solution

    def _local_search_group_merging(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Try to merge adjacent groups to save cost"""
        best_solution = solution

        for i, group1 in enumerate(solution.groups):
            for j, group2 in enumerate(solution.groups[i + 1 :], i + 1):
                # Check if groups can be merged (adjacent nodes)
                if group1.owner in set(graph.neighbors(group2.owner)) | {group2.owner} or group2.owner in set(graph.neighbors(group1.owner)) | {group1.owner}:
                    # Try to merge into a single group
                    merged_members = group1.all_members | group2.all_members

                    # Find best license type for merged group
                    for license_type in license_types:
                        if license_type.min_capacity <= len(merged_members) <= license_type.max_capacity:
                            # Choose better owner
                            owner1_connections = len(set(graph.neighbors(group1.owner)) & merged_members)
                            owner2_connections = len(set(graph.neighbors(group2.owner)) & merged_members)

                            new_owner = group1.owner if owner1_connections >= owner2_connections else group2.owner
                            new_additional = merged_members - {new_owner}

                            # Check connectivity
                            owner_neighbors = set(graph.neighbors(new_owner)) | {new_owner}
                            if new_additional.issubset(owner_neighbors):
                                merged_group = LicenseGroup(license_type, new_owner, new_additional)

                                # Create new solution
                                new_groups = [g for k, g in enumerate(solution.groups) if k != i and k != j]
                                new_groups.append(merged_group)

                                candidate = self._create_solution_from_groups(new_groups)
                                if candidate.total_cost < best_solution.total_cost:
                                    best_solution = candidate
                                break

        return best_solution

    def _local_search_member_reassignment(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        """Try to reassign members between groups"""
        best_solution = solution

        # Try to move members between adjacent groups
        for i, group1 in enumerate(solution.groups):
            for j, group2 in enumerate(solution.groups):
                if i == j:
                    continue

                # Try to move members from group1 to group2
                for member in list(group1.additional_members):
                    # Check if member can be moved (connected to group2's owner)
                    if member in set(graph.neighbors(group2.owner)) | {group2.owner}:
                        # Check capacity constraints
                        if group2.size < group2.license_type.max_capacity and len(group1.additional_members) > group1.license_type.min_capacity - 1:
                            # Create modified groups
                            new_group1_members = group1.additional_members - {member}
                            new_group2_members = group2.additional_members | {member}

                            new_group1 = LicenseGroup(group1.license_type, group1.owner, new_group1_members)
                            new_group2 = LicenseGroup(group2.license_type, group2.owner, new_group2_members)

                            # Create new solution
                            new_groups = solution.groups.copy()
                            new_groups[i] = new_group1
                            new_groups[j] = new_group2

                            candidate = self._create_solution_from_groups(new_groups)
                            if candidate.total_cost < best_solution.total_cost:
                                best_solution = candidate

        return best_solution
