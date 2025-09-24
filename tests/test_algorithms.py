import networkx as nx
import pytest

from glopt.algorithms import (
    AntColonyOptimization,
    DominatingSetAlgorithm,
    GeneticAlgorithm,
    GreedyAlgorithm,
    ILPSolver,
    NaiveAlgorithm,
    RandomizedAlgorithm,
    SimulatedAnnealing,
    TabuSearch,
)
from glopt.core.solution_validator import SolutionValidator
from glopt.io.graph_generator import GraphGeneratorFactory
from glopt.license_config import LicenseConfigFactory

GRAPH_SPECS = {
    "random": {"p": 0.1, "seed": 42},
    "small_world": {"k": 4, "p": 0.1, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
}

LICENSE_CFGS = ["roman_domination", "duolingo_super", "spotify"]

ALGOS = [
    ("ilp", lambda: ILPSolver(), {}, 25),
    ("greedy", lambda: GreedyAlgorithm(), {}, 4000),
    ("dominating_set", lambda: DominatingSetAlgorithm(), {}, 1500),
    ("randomized", lambda: RandomizedAlgorithm(seed=42), {}, 4000),
    ("genetic", lambda: GeneticAlgorithm(population_size=20, generations=20, seed=42), {}, 600),
    ("simulated_annealing", lambda: SimulatedAnnealing(max_iterations=200, max_stall=50), {}, 1000),
    ("tabu_search", lambda: TabuSearch(), {"max_iterations": 100, "neighbors_per_iter": 5, "tabu_tenure": 7}, 1500),
    ("ant_colony", lambda: AntColonyOptimization(num_ants=5, max_iterations=20), {}, 700),
    ("naive", lambda: NaiveAlgorithm(), {}, 10),
]


validator = SolutionValidator(debug=False)


def generate_graph(name: str, n: int) -> nx.Graph:
    params = GRAPH_SPECS[name]
    gen = GraphGeneratorFactory.get(name)
    return gen(n_nodes=n, **params)


@pytest.mark.parametrize("license_cfg", LICENSE_CFGS)
@pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
@pytest.mark.parametrize(("algo_id", "algo_factory", "algo_kwargs", "n_nodes"), ALGOS, ids=[a[0] for a in ALGOS])
def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int) -> None:
    license_types = LicenseConfigFactory.get_config(license_cfg)
    graph = generate_graph(graph_name, n_nodes)
    algo = algo_factory()
    solution = algo.solve(graph=graph, license_types=license_types, **algo_kwargs)
    ok, issues = validator.validate(solution, graph)
    assert ok, f"{algo.name} invalid: {issues}"
