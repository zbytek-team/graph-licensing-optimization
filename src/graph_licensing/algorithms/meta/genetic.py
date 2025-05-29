"""Genetic Algorithm for licensing optimization."""

import random
from typing import TYPE_CHECKING

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class GeneticAlgorithm(BaseAlgorithm):
    """Genetic Algorithm for licensing optimization."""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        seed: int | None = None,
    ) -> None:
        """Initialize the genetic algorithm.

        Args:
            population_size: Size of the population.
            generations: Number of generations to evolve.
            mutation_rate: Probability of mutation.
            crossover_rate: Probability of crossover.
            seed: Random seed for reproducibility.
        """
        super().__init__("Genetic")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self._best_solution = None

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using genetic algorithm.

        Args:
            graph: The social network graph.
            config: License configuration.
            **kwargs: Additional parameters (ignored).

        Returns:
            Best licensing solution found.
        """
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution(solo_nodes=[], group_owners={})

        # Initialize population
        population = [self._create_random_individual(graph, config) for _ in range(self.population_size)]

        # Evolution loop
        for _generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual, config) for individual in population]

            # Selection and reproduction
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, graph, config)
                else:
                    child = parent1.copy() if random.random() < 0.5 else parent2.copy()

                if random.random() < self.mutation_rate:
                    child = self._mutate(child, graph, config)

                new_population.append(child)

            population = new_population

        # Return best solution
        best_individual = min(
            population,
            key=lambda x: self._evaluate_fitness(x, config),
        )
        self._best_solution = self._decode_individual(best_individual, nodes)
        return self._best_solution

    def solve_dynamic(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        iterations: int,
        modification_prob: float = 0.1,
        **kwargs,
    ) -> list["LicenseSolution"]:
        """Solve dynamic version using warm starts from previous solutions.

        Args:
            graph: Initial graph.
            config: License configuration.
            iterations: Number of iterations.
            modification_prob: Graph modification probability.
            **kwargs: Additional parameters.

        Returns:
            List of solutions for each iteration.
        """
        solutions = []
        current_graph = graph.copy()

        for i in range(iterations):
            # Use previous solution as seed for population if available
            if self._best_solution is not None and i > 0:
                # Adapt previous solution to current graph
                kwargs["warm_start"] = self._best_solution

            solution = self.solve(current_graph, config, **kwargs)
            solutions.append(solution)

            if i < iterations - 1:
                current_graph = self._modify_graph(current_graph, modification_prob)

        return solutions

    def _create_random_individual(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> list[int]:
        """Create a random individual (chromosome).

        Each individual is represented as a list where:
        - 0 = solo license
        - i > 0 = member of group owned by node i

        Args:
            graph: The social network graph.
            config: License configuration.

        Returns:
            Random individual representation.
        """
        nodes = list(graph.nodes())
        individual = [0] * len(nodes)  # Start with all solo

        # Randomly assign some nodes to groups
        for i, node in enumerate(nodes):
            if random.random() < 0.3:  # 30% chance to be in a group
                # Find potential group owners (neighbors)
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    owner = random.choice([*neighbors, node])  # Can own group too
                    individual[i] = nodes.index(owner) + 1  # +1 to avoid 0 (solo)

        return self._repair_individual(individual, graph, config)

    def _repair_individual(
        self,
        individual: list[int],
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> list[int]:
        """Repair an individual to ensure validity.

        Args:
            individual: Individual to repair.
            graph: The social network graph.
            config: License configuration.

        Returns:
            Repaired individual.
        """
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
                # Remove excess members
                members = [i for i, assignment in enumerate(repaired) if assignment == owner_idx + 1]
                excess = size - config.group_size
                for _ in range(excess):
                    member_idx = random.choice(members)
                    repaired[member_idx] = 0  # Convert to solo
                    members.remove(member_idx)

        # Fix connectivity violations
        for i, assignment in enumerate(repaired):
            if assignment > 0:
                owner_idx = assignment - 1
                node = nodes[i]
                owner_node = nodes[owner_idx]

                if node != owner_node and not graph.has_edge(node, owner_node):
                    repaired[i] = 0  # Convert to solo if not connected

        return repaired

    def _evaluate_fitness(
        self,
        individual: list[int],
        config: "LicenseConfig",
    ) -> float:
        """Evaluate fitness of an individual (lower is better).

        Args:
            individual: Individual to evaluate.
            config: License configuration.

        Returns:
            Fitness score (total cost).
        """
        # Count solo licenses
        solo_count = sum(1 for assignment in individual if assignment == 0)

        # Count group licenses
        group_owners = set()
        for assignment in individual:
            if assignment > 0:
                group_owners.add(assignment - 1)
        group_count = len(group_owners)

        return solo_count * config.solo_price + group_count * config.group_price

    def _tournament_selection(
        self,
        population: list[list[int]],
        fitness_scores: list[float],
    ) -> list[int]:
        """Select individual using tournament selection.

        Args:
            population: Current population.
            fitness_scores: Fitness scores for each individual.

        Returns:
            Selected individual.
        """
        tournament_size = 3
        tournament_indices = random.sample(
            range(len(population)),
            min(tournament_size, len(population)),
        )

        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def _crossover(
        self,
        parent1: list[int],
        parent2: list[int],
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> list[int]:
        """Perform crossover between two parents.

        Args:
            parent1: First parent.
            parent2: Second parent.
            graph: The social network graph.
            config: License configuration.

        Returns:
            Offspring individual.
        """
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]

        return self._repair_individual(child, graph, config)

    def _mutate(
        self,
        individual: list[int],
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> list[int]:
        """Mutate an individual.

        Args:
            individual: Individual to mutate.
            graph: The social network graph.
            config: License configuration.

        Returns:
            Mutated individual.
        """
        mutated = individual.copy()
        nodes = list(graph.nodes())

        # Random mutation: change assignment of a random node
        node_idx = random.randint(0, len(nodes) - 1)
        node = nodes[node_idx]

        # Get possible assignments (solo or neighbors as group owners)
        possible_assignments = [0]  # Solo
        for neighbor in graph.neighbors(node):
            neighbor_idx = nodes.index(neighbor)
            possible_assignments.append(neighbor_idx + 1)

        # Also consider making this node a group owner
        possible_assignments.append(node_idx + 1)

        mutated[node_idx] = random.choice(possible_assignments)

        return self._repair_individual(mutated, graph, config)

    def _decode_individual(
        self,
        individual: list[int],
        nodes: list[int],
    ) -> "LicenseSolution":
        """Decode individual to LicenseSolution.

        Args:
            individual: Individual to decode.
            nodes: List of graph nodes.

        Returns:
            Corresponding LicenseSolution.
        """
        from ...models.license import LicenseSolution

        solo_nodes = []
        group_owners = {}

        # Build groups
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
