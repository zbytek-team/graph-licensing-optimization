import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class GeneticAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.4,  # Increased from 0.15
        crossover_rate: float = 0.85,
        elite_size: int = 5,
        seed: int | None = None,
    ) -> None:
        super().__init__("Genetic")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.seed = seed
        self._best_solution = None

    def supports_warm_start(self) -> bool:
        return True

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        population = self._initialize_population(graph, config, warm_start)

        best_fitness = float("inf")
        stagnation_count = 0
        max_stagnation = 50  # Increased from 20 to allow more exploration

        for generation in range(self.generations):
            fitness_scores = [self._evaluate_fitness(individual, config) for individual in population]

            current_best = min(fitness_scores)
            if current_best < best_fitness:
                best_fitness = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= max_stagnation:
                break

            population = self._evolve_population(population, fitness_scores, graph, config)

        best_individual = min(population, key=lambda x: self._evaluate_fitness(x, config))
        self._best_solution = self._decode_individual(best_individual, graph, config)
        return self._best_solution

    def _initialize_population(
        self, graph: "nx.Graph", config: "LicenseConfig", warm_start: Optional["LicenseSolution"] = None
    ) -> List[Dict[str, Any]]:
        population = []

        if warm_start:
            warm_individual = self._encode_solution(warm_start)
            population.append(warm_individual)

        from ..greedy import GreedyAlgorithm

        greedy_algo = GreedyAlgorithm()
        greedy_solution = greedy_algo.solve(graph, config)
        greedy_individual = self._encode_solution(greedy_solution)
        population.append(greedy_individual)

        while len(population) < self.population_size:
            random_individual = self._create_random_individual(graph, config)
            population.append(random_individual)

        return population

    def _encode_solution(self, solution: "LicenseSolution") -> Dict[str, Any]:
        assignments = {}

        for license_type, groups in solution.licenses.items():
            for owner, members in groups.items():
                for member in members:
                    assignments[member] = (license_type, owner)

        return {"assignments": assignments}

    def _decode_individual(
        self, individual: Dict[str, Any], graph: "nx.Graph", config: "LicenseConfig"
    ) -> "LicenseSolution":
        from ...models.license import LicenseSolution

        licenses = {}
        assignments = individual["assignments"]

        for node, (license_type, owner) in assignments.items():
            if license_type not in licenses:
                licenses[license_type] = {}
            if owner not in licenses[license_type]:
                licenses[license_type][owner] = []
            if node not in licenses[license_type][owner]:
                licenses[license_type][owner].append(node)

        for license_type in licenses:
            for owner in list(licenses[license_type].keys()):
                members = licenses[license_type][owner]
                if owner not in members:
                    # Owner must be in their own group
                    members.insert(0, owner)

        return LicenseSolution(licenses=licenses)

    def _create_random_individual(self, graph: "nx.Graph", config: "LicenseConfig") -> Dict[str, Any]:
        nodes = list(graph.nodes())
        assignments = {}
        unassigned = set(nodes)

        # Strategy: mix of random group formation and cost-aware decisions
        strategy = random.choice(["random", "cost_aware", "high_degree_groups"])
        
        if strategy == "cost_aware":
            # Prioritize cost-effective groups
            nodes_by_degree = sorted(nodes, key=lambda x: graph.degree(x), reverse=True)
            for node in nodes_by_degree:
                if node not in unassigned:
                    continue
                    
                # Try to form cost-effective groups
                best_assignment = self._find_best_assignment_for_node(node, graph, config, unassigned)
                if best_assignment:
                    license_type, owner, members = best_assignment
                    for member in members:
                        assignments[member] = (license_type, owner)
                        unassigned.discard(member)
                else:
                    # Fallback to solo
                    cheapest_solo = self._get_cheapest_solo_license(config)
                    if cheapest_solo:
                        assignments[node] = (cheapest_solo, node)
                    unassigned.discard(node)
                    
        elif strategy == "high_degree_groups":
            # Focus on creating groups around high-degree nodes
            high_degree_nodes = sorted([n for n in nodes if n in unassigned], 
                                     key=lambda x: graph.degree(x), reverse=True)[:len(nodes)//4]
            
            for node in high_degree_nodes:
                if node not in unassigned:
                    continue
                    
                neighbors = [n for n in graph.neighbors(node) if n in unassigned]
                if len(neighbors) >= 1:
                    # Try to create larger groups
                    best_group_license = self._find_best_group_license(config, min(4, len(neighbors) + 1))
                    if best_group_license:
                        group_size = min(config.license_types[best_group_license].max_size, len(neighbors) + 1)
                        selected_neighbors = random.sample(neighbors, min(group_size - 1, len(neighbors)))
                        group_members = [node] + selected_neighbors
                        
                        for member in group_members:
                            assignments[member] = (best_group_license, node)
                            unassigned.discard(member)
                        continue
                
                # Fallback to solo
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    assignments[node] = (cheapest_solo, node)
                unassigned.discard(node)
                
            # Handle remaining nodes
            for node in list(unassigned):
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    assignments[node] = (cheapest_solo, node)
                unassigned.discard(node)
                
        else:
            # Original random strategy (but improved)
            while unassigned:
                node = random.choice(list(unassigned))
                
                # More bias towards groups when beneficial
                if random.random() < 0.6:  # 60% chance to try group
                    available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]
                    if available_neighbors:
                        # Try different group sizes
                        for license_type, license_config in config.license_types.items():
                            max_possible = min(license_config.max_size, len(available_neighbors) + 1)
                            if max_possible >= license_config.min_size:
                                group_size = random.randint(license_config.min_size, max_possible)
                                if group_size > 1:
                                    selected_neighbors = random.sample(available_neighbors, 
                                                                     min(group_size - 1, len(available_neighbors)))
                                    group_members = [node] + selected_neighbors
                                    
                                    for member in group_members:
                                        assignments[member] = (license_type, node)
                                        unassigned.discard(member)
                                    break
                        
                        # If no group was formed, fall through to solo
                        if node not in assignments:
                            cheapest_solo = self._get_cheapest_solo_license(config)
                            if cheapest_solo:
                                assignments[node] = (cheapest_solo, node)
                            unassigned.discard(node)
                    else:
                        # No neighbors, assign solo
                        cheapest_solo = self._get_cheapest_solo_license(config)
                        if cheapest_solo:
                            assignments[node] = (cheapest_solo, node)
                        unassigned.discard(node)
                else:
                    # Solo assignment
                    cheapest_solo = self._get_cheapest_solo_license(config)
                    if cheapest_solo:
                        assignments[node] = (cheapest_solo, node)
                    unassigned.discard(node)

        return {"assignments": assignments}

    def _find_best_assignment_for_node(self, node: int, graph: "nx.Graph", config: "LicenseConfig", unassigned: set) -> tuple | None:
        """Find the most cost-effective assignment for a node."""
        best_cost_per_person = float("inf")
        best_assignment = None
        
        available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]
        
        for license_type, license_config in config.license_types.items():
            max_possible_size = min(license_config.max_size, len(available_neighbors) + 1)
            
            for group_size in range(license_config.min_size, max_possible_size + 1):
                if group_size == 1:
                    cost_per_person = license_config.price
                    if cost_per_person < best_cost_per_person:
                        best_cost_per_person = cost_per_person
                        best_assignment = (license_type, node, [node])
                        
                elif group_size > 1 and available_neighbors:
                    selected_members = available_neighbors[:group_size - 1]  # Simple selection
                    if len(selected_members) == group_size - 1:
                        members = [node] + selected_members
                        cost_per_person = license_config.price / len(members)
                        
                        if cost_per_person < best_cost_per_person:
                            best_cost_per_person = cost_per_person
                            best_assignment = (license_type, node, members)
        
        return best_assignment

    def _evaluate_fitness(self, individual: Dict[str, Any], config: "LicenseConfig") -> float:
        assignments = individual["assignments"]
        licenses = {}

        for node, (license_type, owner) in assignments.items():
            if license_type not in licenses:
                licenses[license_type] = {}
            if owner not in licenses[license_type]:
                licenses[license_type][owner] = []
            licenses[license_type][owner].append(node)

        total_cost = 0.0
        for license_type, groups in licenses.items():
            if license_type in config.license_types:
                license_config = config.license_types[license_type]
                for owner, members in groups.items():
                    if license_config.is_valid_size(len(members)):
                        total_cost += license_config.price
                    else:
                        # Use large penalty instead of infinity to allow recovery
                        penalty = license_config.price * len(members) * 10  # 10x cost penalty
                        total_cost += penalty

        return total_cost

    def _evolve_population(
        self, population: List[Dict[str, Any]], fitness_scores: List[float], graph: "nx.Graph", config: "LicenseConfig"
    ) -> List[Dict[str, Any]]:
        new_population = []

        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[: self.elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, graph, config)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1, graph, config)
                # Debug: check if mutation actually changed something
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2, graph, config)

            new_population.extend([child1, child2])

        return new_population[: self.population_size]

    def _tournament_selection(
        self, population: List[Dict[str, Any]], fitness_scores: List[float], tournament_size: int = 3
    ) -> Dict[str, Any]:
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()

    def _crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any], graph: "nx.Graph", config: "LicenseConfig"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        nodes = list(graph.nodes())

        child1_assignments = {}
        child2_assignments = {}

        for node in nodes:
            if random.random() < 0.5:
                child1_assignments[node] = parent1["assignments"].get(node, self._get_default_assignment(node, config))
                child2_assignments[node] = parent2["assignments"].get(node, self._get_default_assignment(node, config))
            else:
                child1_assignments[node] = parent2["assignments"].get(node, self._get_default_assignment(node, config))
                child2_assignments[node] = parent1["assignments"].get(node, self._get_default_assignment(node, config))

        child1 = {"assignments": child1_assignments}
        child2 = {"assignments": child2_assignments}

        child1 = self._repair_individual(child1, graph, config)
        child2 = self._repair_individual(child2, graph, config)

        return child1, child2

    def _mutate(self, individual: Dict[str, Any], graph: "nx.Graph", config: "LicenseConfig") -> Dict[str, Any]:
        mutated = {"assignments": individual["assignments"].copy()}
        nodes = list(graph.nodes())

        # Different mutation strategies
        mutation_type = random.choice(["merge_solos", "split_group", "reassign_nodes", "create_group"])
        
        if mutation_type == "merge_solos":
            # Try to merge solo licenses into groups
            solo_nodes = [node for node, (lt, owner) in mutated["assignments"].items() if owner == node]
            if len(solo_nodes) >= 2:
                # Find connected solo nodes
                for i in range(len(solo_nodes)):
                    for j in range(i + 1, len(solo_nodes)):
                        node1, node2 = solo_nodes[i], solo_nodes[j]
                        if graph.has_edge(node1, node2):
                            # Create group
                            best_license = self._find_best_group_license(config, 2)
                            if best_license and random.random() < 0.7:  # Increased from 0.3 to 0.7
                                mutated["assignments"][node1] = (best_license, node1)
                                mutated["assignments"][node2] = (best_license, node1)
                                break
                                
        elif mutation_type == "split_group":
            # Split some groups into solo licenses
            groups = {}
            for node, (license_type, owner) in mutated["assignments"].items():
                key = (license_type, owner)
                if key not in groups:
                    groups[key] = []
                groups[key].append(node)
            
            group_keys = [(lt, owner) for (lt, owner), members in groups.items() if len(members) > 1]
            if group_keys and random.random() < 0.2:
                license_type, owner = random.choice(group_keys)
                members = groups[(license_type, owner)]
                if len(members) > 2:  # Only split if group has more than 2 members
                    # Split one member to solo
                    member_to_split = random.choice([m for m in members if m != owner])
                    cheapest_solo = self._get_cheapest_solo_license(config)
                    if cheapest_solo:
                        mutated["assignments"][member_to_split] = (cheapest_solo, member_to_split)
                        
        elif mutation_type == "reassign_nodes":
            # Reassign some nodes to different groups or solo
            num_mutations = max(1, int(len(nodes) * 0.05))  # Reduced to 5%
            for _ in range(num_mutations):
                node = random.choice(nodes)
                
                # Find neighbors with different assignments
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    neighbor = random.choice(neighbors)
                    neighbor_license, neighbor_owner = mutated["assignments"][neighbor]
                    
                    # Check if we can join neighbor's group
                    neighbor_group = [n for n, (lt, owner) in mutated["assignments"].items() 
                                    if lt == neighbor_license and owner == neighbor_owner]
                    
                    if (neighbor_license in config.license_types and 
                        len(neighbor_group) < config.license_types[neighbor_license].max_size):
                        mutated["assignments"][node] = (neighbor_license, neighbor_owner)
                        
        elif mutation_type == "create_group":
            # Try to create new groups from scratch
            high_degree_nodes = sorted(nodes, key=lambda n: graph.degree(n), reverse=True)[:10]
            for node in high_degree_nodes[:3]:  # Try top 3 high-degree nodes
                if random.random() < 0.3:
                    neighbors = [n for n in graph.neighbors(node) if n in nodes]
                    if len(neighbors) >= 1:
                        # Create group with some neighbors
                        group_size = min(3, len(neighbors) + 1)  # Max size 3
                        best_license = self._find_best_group_license(config, group_size)
                        if best_license:
                            selected_neighbors = random.sample(neighbors, min(group_size - 1, len(neighbors)))
                            mutated["assignments"][node] = (best_license, node)
                            for neighbor in selected_neighbors:
                                mutated["assignments"][neighbor] = (best_license, node)

        return self._intelligent_repair(mutated, graph, config)

    def _find_best_group_license(self, config: "LicenseConfig", group_size: int) -> str | None:
        """Find the most cost-effective license type for a given group size."""
        best_license = None
        best_cost_per_person = float('inf')
        
        for license_type, license_config in config.license_types.items():
            if license_config.is_valid_size(group_size):
                cost_per_person = license_config.cost_per_person(group_size)
                if cost_per_person < best_cost_per_person:
                    best_cost_per_person = cost_per_person
                    best_license = license_type
                    
        return best_license

    def _intelligent_repair(self, individual: Dict[str, Any], graph: "nx.Graph", config: "LicenseConfig") -> Dict[str, Any]:
        """Repair individual with cost optimization in mind."""
        
        # First, apply basic connectivity and size constraints
        repaired_individual = self._repair_individual(individual, graph, config)
        
        # Then, try to optimize cost by merging beneficial solo licenses
        solo_nodes = [node for node, (lt, owner) in repaired_individual["assignments"].items() if owner == node]
        
        for i in range(len(solo_nodes)):
            for j in range(i + 1, len(solo_nodes)):
                node1, node2 = solo_nodes[i], solo_nodes[j]
                if graph.has_edge(node1, node2):
                    # Calculate cost of current solo licenses
                    lt1, _ = repaired_individual["assignments"][node1]
                    lt2, _ = repaired_individual["assignments"][node2]
                    current_cost = config.license_types[lt1].price + config.license_types[lt2].price
                    
                    # Check if group license would be cheaper
                    best_group_license = self._find_best_group_license(config, 2)
                    if best_group_license:
                        group_cost = config.license_types[best_group_license].price
                        if group_cost < current_cost:
                            repaired_individual["assignments"][node1] = (best_group_license, node1)
                            repaired_individual["assignments"][node2] = (best_group_license, node1)
        
        return repaired_individual

    def _repair_individual(
        self, individual: Dict[str, Any], graph: "nx.Graph", config: "LicenseConfig"
    ) -> Dict[str, Any]:
        assignments = individual["assignments"].copy()

        groups = {}
        for node, (license_type, owner) in assignments.items():
            key = (license_type, owner)
            if key not in groups:
                groups[key] = []
            groups[key].append(node)

        for (license_type, owner), members in groups.items():
            if license_type not in config.license_types:
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    for member in members:
                        assignments[member] = (cheapest_solo, member)
                continue

            license_config = config.license_types[license_type]

            if not license_config.is_valid_size(len(members)):
                if len(members) > license_config.max_size:
                    excess_members = members[license_config.max_size :]
                    cheapest_solo = self._get_cheapest_solo_license(config)
                    if cheapest_solo:
                        for member in excess_members:
                            assignments[member] = (cheapest_solo, member)
                elif len(members) < license_config.min_size:
                    cheapest_solo = self._get_cheapest_solo_license(config)
                    if cheapest_solo:
                        for member in members:
                            assignments[member] = (cheapest_solo, member)

            if len(members) > 1:
                valid_members = [owner]  # Owner is always valid
                for member in members:
                    if member != owner and graph.has_edge(owner, member):
                        valid_members.append(member)

                disconnected = set(members) - set(valid_members)
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    for member in disconnected:
                        assignments[member] = (cheapest_solo, member)

        return {"assignments": assignments}

    def _get_default_assignment(self, node: int, config: "LicenseConfig") -> Tuple[str, int]:
        cheapest_solo = self._get_cheapest_solo_license(config)
        return (cheapest_solo, node) if cheapest_solo else (list(config.license_types.keys())[0], node)

    def _get_cheapest_solo_license(self, config: "LicenseConfig") -> str | None:
        cheapest_type = None
        cheapest_cost = float("inf")

        for license_type, license_config in config.license_types.items():
            if license_config.is_valid_size(1) and license_config.price < cheapest_cost:
                cheapest_cost = license_config.price
                cheapest_type = license_type

        return cheapest_type
