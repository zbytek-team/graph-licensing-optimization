import pytest

from src.algorithms.static.greedy_basic import GreedyBasicSolver
from src.algorithms.static.ilp_solver import ILPSolver
from src.utils.solution_utils import validate_solution, calculate_cost
from src.data.generators import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
)

SOLVERS = [GreedyBasicSolver, ILPSolver]

GRAPH_PARAMS = [
    #  (generator, kwargs, c_single, c_group, group_size)
    (generate_erdos_renyi, {"n": 50, "p": 0.05, "seed": 11}, 3.0, 5.0, 3),
    (generate_erdos_renyi, {"n": 100, "p": 0.03, "seed": 12}, 2.0, 4.0, 4),
    (generate_barabasi_albert, {"n": 80, "m": 4, "seed": 13}, 1.5, 3.5, 5),
    (generate_barabasi_albert, {"n": 150, "m": 3, "seed": 14}, 2.0, 4.0, 6),
    (generate_watts_strogatz, {"n": 60, "k": 6, "p": 0.2, "seed": 15}, 3.0, 6.0, 4),
    (generate_watts_strogatz, {"n": 120, "k": 8, "p": 0.2, "seed": 16}, 2.5, 5.0, 6),
]


@pytest.mark.parametrize("solver_cls", SOLVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("gen, kwargs, c_single, c_group, gsize", GRAPH_PARAMS)
def test_solver_validity(gen, kwargs, c_single, c_group, gsize, solver_cls):
    graph = gen(**kwargs)
    solver = solver_cls()
    sol = solver.solve(graph, c_single, c_group, gsize)
    errors = validate_solution(graph, sol, gsize)
    assert errors == []


def test_ilp_not_worse_than_greedy():
    for gen, kwargs, c_single, c_group, gsize in GRAPH_PARAMS:
        graph = gen(**kwargs)
        greedy_sol = GreedyBasicSolver().solve(graph, c_single, c_group, gsize)
        ilp_sol = ILPSolver().solve(graph, c_single, c_group, gsize)
        greedy_cost = calculate_cost(greedy_sol, c_single, c_group)
        ilp_cost = calculate_cost(ilp_sol, c_single, c_group)
        assert ilp_cost <= greedy_cost + 1e-6
