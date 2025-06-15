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
        mutation_rate: float = 0.15,
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
        max_stagnation = 20

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
            for owner in licenses[license_type]:
                if owner not in licenses[license_type][owner]:
                    licenses[license_type][owner].insert(0, owner)

        return LicenseSolution(licenses=licenses)

    def _create_random_individual(self, graph: "nx.Graph", config: "LicenseConfig") -> Dict[str, Any]:
        nodes = list(graph.nodes())
        assignments = {}
        unassigned = set(nodes)

        while unassigned:
            node = random.choice(list(unassigned))

            license_type = random.choice(list(config.license_types.keys()))
            license_config = config.license_types[license_type]

            available_neighbors = [n for n in graph.neighbors(node) if n in unassigned]
            max_group_size = min(license_config.max_size, len(available_neighbors) + 1)

            if max_group_size >= license_config.min_size:
                group_size = random.randint(license_config.min_size, max_group_size)

                if group_size == 1:
                    assignments[node] = (license_type, node)
                    unassigned.remove(node)
                else:
                    selected_neighbors = random.sample(
                        available_neighbors, min(group_size - 1, len(available_neighbors))
                    )
                    group_members = [node] + selected_neighbors

                    for member in group_members:
                        assignments[member] = (license_type, node)
                        unassigned.discard(member)
            else:
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    assignments[node] = (cheapest_solo, node)
                unassigned.remove(node)

        return {"assignments": assignments}

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
                        total_cost += float("inf")  # Invalid group size penalty

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

        num_mutations = max(1, int(len(nodes) * 0.1))  # Mutate 10% of nodes

        for _ in range(num_mutations):
            node = random.choice(nodes)

            available_types = list(config.license_types.keys())
            new_license_type = random.choice(available_types)

            if random.random() < 0.5 or not list(graph.neighbors(node)):
                mutated["assignments"][node] = (new_license_type, node)
            else:
                neighbor = random.choice(list(graph.neighbors(node)))
                mutated["assignments"][node] = (new_license_type, neighbor)

        return self._repair_individual(mutated, graph, config)

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
