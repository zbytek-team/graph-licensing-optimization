import pytest
import networkx as nx

from src import GraphGeneratorFactory, LicenseConfigFactory, SolutionValidator
from src.algorithms import (
    ILPSolver,
    GreedyAlgorithm,
    DominatingSetAlgorithm,
    RandomizedAlgorithm,
    GeneticAlgorithm,
    SimulatedAnnealing,
    TabuSearch,
    AntColonyOptimization,
    NaiveAlgorithm,
)

GRAPH_SPECS = {
    "random": {"p": 0.1, "seed": 42},
    "small_world": {"k": 4, "p": 0.1, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
}

LICENSE_CFG = "roman_domination"

validator = SolutionValidator(debug=False)

def generate_graph(name: str, n: int) -> nx.Graph:
    params = GRAPH_SPECS[name]
    gen = GraphGeneratorFactory.get(name)
    return gen(n_nodes=n, **params)

def solve_cost(algo, graph: nx.Graph, license_types, **kwargs):
    solution = algo.solve(graph=graph, license_types=license_types, **kwargs)
    ok, issues = validator.validate(solution, graph)
    assert ok, f"{algo.name} invalid: {issues}"
    return float(solution.total_cost)

@pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
def test_algorithms_against_baselines(graph_name: str):
    license_types = LicenseConfigFactory.get_config(LICENSE_CFG)
    graph = generate_graph(graph_name, 30)

    ilp_cost = solve_cost(ILPSolver(), graph, license_types)
    greedy_cost = solve_cost(GreedyAlgorithm(), graph, license_types)
    assert greedy_cost >= ilp_cost

    algos = [
        (DominatingSetAlgorithm(), {}),
        (RandomizedAlgorithm(seed=42), {}),
        (GeneticAlgorithm(population_size=20, generations=20, seed=42), {}),
        (SimulatedAnnealing(max_iterations=200, max_stall=50), {}),
        (TabuSearch(), {"max_iterations": 100, "neighbors_per_iter": 5, "tabu_tenure": 7}),
        (AntColonyOptimization(num_ants=5, max_iterations=20), {}),
    ]

    for algo, kwargs in algos:
        cost = solve_cost(algo, graph, license_types, **kwargs)
        assert cost >= ilp_cost
        if not isinstance(algo, RandomizedAlgorithm):
            assert cost <= greedy_cost

@pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
def test_naive_against_baselines(graph_name: str):
    license_types = LicenseConfigFactory.get_config(LICENSE_CFG)
    graph = generate_graph(graph_name, 10)

    ilp_cost = solve_cost(ILPSolver(), graph, license_types)
    greedy_cost = solve_cost(GreedyAlgorithm(), graph, license_types)
    assert greedy_cost >= ilp_cost

    naive_cost = solve_cost(NaiveAlgorithm(), graph, license_types)
    assert naive_cost >= ilp_cost
    assert naive_cost <= greedy_cost
