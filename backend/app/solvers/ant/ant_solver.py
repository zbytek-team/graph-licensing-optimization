import numpy as np
from app.models.solve import License, Assignments, AntSolverType
from app.models.graph import Graph
from app.solvers.ant.ant import Ant, ant_solvers_args_validator

def ant_solver(
    graph: Graph,
    licenses: list[License],
    ants: int,
    iterations: int = 0,
    alpha: float = 1.0,
    beta: float = 8.0,
    evaporation: float = 0.75,
    stagnation_limit: int = 30,
    solution_type: AntSolverType = AntSolverType.PATH,
) -> Assignments:
    '''
    Podstawowa implementacja algorytmu mrówkowego dla problemu przypisania licencji.

    :param graph: Graf
    :param licenses: Lista licencji
    :param ants: Liczba mrówek
    :param iterations: Maksymalna liczba iteracji
    :param alpha: Waga feromonów
    :param beta: Waga heurystyki
    :param evaporation: Współczynnik parowania feromonów
    :param stagnation_limit: Limit stagnacji
    :param solution_type: Typ rozwiązania

    :return: Przypisanie licenc
    '''

    ant_solvers_args_validator(graph, licenses, ants, iterations, alpha, beta, evaporation, stagnation_limit, solution_type)

    pheromones = {node: np.ones(len(licenses)) for node in graph.nodes}
    best_solution = None
    best_cost = float("inf")
    stagnation_counter = 0
    iteration = 0

    while iterations == 0 or iteration < iterations:
        results= [
            Ant(graph, licenses, pheromones, alpha, beta, solution_type)
            for _ in range(ants)
        ]

        for ant in results:
            ant.construct_solution()

        results = [(ant.solution, ant.cost) for ant in results]
        
        results.sort(key=lambda x: x[1])

        best_ant_solution, best_ant_cost = results[0]

        if best_ant_cost < best_cost:
            best_solution = best_ant_solution
            best_cost = best_ant_cost
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        for node in pheromones:
            pheromones[node] *= evaporation

        for license, assignments in best_ant_solution.items():
            for assignment in assignments:
                node = assignment.license_holder
                license_index = licenses.index(
                    next(lic for lic in licenses if lic.name == license)
                )
                pheromones[node][license_index] += 1.0 / best_ant_cost

        iteration += 1
        if iterations == 0 and stagnation_counter >= stagnation_limit:
            break

    return best_solution if best_solution else {}