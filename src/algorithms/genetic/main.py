"""Moduł implementuje algorytm main dla dystrybucji licencji.

Wejście zwykle obejmuje obiekt `networkx.Graph` oraz konfiguracje licencji (`LicenseType`, `LicenseGroup`).
"""

from src.core import LicenseType, Solution, Algorithm
from src.core import SolutionValidator
from .population import PopulationManager
from .operators import GeneticOperators
from typing import Any, List
import random
import networkx as nx


class GeneticAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "genetic"

    def solve(
        self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any
    ) -> Solution:
        self.validator = SolutionValidator()
        self.population_manager = PopulationManager(self.validator)
        self.operators = GeneticOperators(self.validator)

        population_size = kwargs.get("population_size", 50)
        max_generations = kwargs.get("max_generations", 200)
        mutation_rate = kwargs.get("mutation_rate", 0.3)
        crossover_rate = kwargs.get("crossover_rate", 0.7)
        elitism_count = kwargs.get("elitism_count", 3)

        population = self.population_manager.initialize_population(
            graph, license_types, population_size
        )
        best_solution = min(population, key=lambda sol: sol.total_cost)
        generations_without_improvement = 0
        max_stagnation = 20

        for generation in range(max_generations):
            fitness_scores = [
                self._calculate_fitness(solution) for solution in population
            ]
            current_best = min(population, key=lambda sol: sol.total_cost)

            if current_best.total_cost < best_solution.total_cost:
                best_solution = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                if generations_without_improvement > 8 and generation % 3 == 0:
                    improved_best = self.operators.intensive_local_search(
                        best_solution, graph, license_types
                    )
                    if improved_best.total_cost < best_solution.total_cost:
                        best_solution = improved_best
                        generations_without_improvement = 0
                        worst_idx = min(
                            range(len(population)), key=lambda i: fitness_scores[i]
                        )
                        population[worst_idx] = best_solution

            adaptive_mutation_rate = mutation_rate
            if generations_without_improvement > 10:
                adaptive_mutation_rate = min(0.9, mutation_rate * 3)
            elif generations_without_improvement > 5:
                adaptive_mutation_rate = min(0.7, mutation_rate * 2)

            if generations_without_improvement > max_stagnation:
                break

            new_population = []
            elitism_size = elitism_count
            if generations_without_improvement > 5:
                elitism_size = min(elitism_count * 2, population_size // 4)

            elite_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True,
            )[:elitism_size]
            for idx in elite_indices:
                new_population.append(population[idx])

            while len(new_population) < population_size:
                parent1 = self.operators.tournament_selection(
                    population, fitness_scores
                )
                parent2 = self.operators.tournament_selection(
                    population, fitness_scores
                )

                if random.random() < crossover_rate:
                    child1, child2 = self.operators.crossover(
                        parent1, parent2, graph, license_types
                    )
                else:
                    child1, child2 = parent1, parent2

                if random.random() < adaptive_mutation_rate:
                    mutated_child1 = self.operators.mutate(child1, graph, license_types)
                    if mutated_child1 and self.validator.is_valid_solution(
                        mutated_child1, graph
                    ):
                        child1 = mutated_child1

                if random.random() < adaptive_mutation_rate:
                    mutated_child2 = self.operators.mutate(child2, graph, license_types)
                    if mutated_child2 and self.validator.is_valid_solution(
                        mutated_child2, graph
                    ):
                        child2 = mutated_child2

                if child1 and self.validator.is_valid_solution(child1, graph):
                    new_population.append(child1)
                if (
                    child2
                    and self.validator.is_valid_solution(child2, graph)
                    and len(new_population) < population_size
                ):
                    new_population.append(child2)

            while len(new_population) < population_size:
                attempts = 0
                max_attempts = 10

                while attempts < max_attempts and len(new_population) < population_size:
                    solution = self.population_manager.generate_truly_random_solution(
                        graph, license_types
                    )
                    if solution and self.validator.is_valid_solution(solution, graph):
                        new_population.append(solution)
                        break
                    attempts += 1

                if len(new_population) < population_size:
                    base_solution = random.choice(
                        new_population[:elitism_count]
                        if new_population
                        else [best_solution]
                    )
                    attempts = 0
                    while (
                        attempts < max_attempts
                        and len(new_population) < population_size
                    ):
                        mutated = self.operators.intensive_mutate(
                            base_solution, graph, license_types
                        )
                        if mutated and self.validator.is_valid_solution(mutated, graph):
                            new_population.append(mutated)
                            break
                        attempts += 1

                    if len(new_population) < population_size:
                        new_population.append(base_solution)

            population = new_population[:population_size]

        # Final validation and repair of best solution
        if not self.validator.is_valid_solution(best_solution, graph):
            # Try to repair the best solution
            repaired_solution = self.operators._force_repair_solution(
                best_solution, graph, license_types
            )
            if self.validator.is_valid_solution(repaired_solution, graph):
                best_solution = repaired_solution
            else:
                # If repair fails, use a greedy fallback
                from ..greedy import GreedyAlgorithm

                greedy = GreedyAlgorithm()
                best_solution = greedy.solve(graph, license_types)

        return best_solution

    def _calculate_fitness(self, solution: Solution) -> float:
        return 1.0 / (solution.total_cost + 1.0)
