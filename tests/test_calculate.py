import networkx as nx
from src.utils.solution_utils import calculate_cost, validate_solution
from src.algorithms.static.greedy_basic import GreedyBasicSolver


def test_calculate_cost_and_validation():
    graph = nx.gnp_random_graph(20, 0.1, seed=1)
    solver = GreedyBasicSolver()
    sol = solver.solve(graph, c_single=3.0, c_group=5.0, group_size=3)
    cost = calculate_cost(sol, 3.0, 5.0)
    assert cost > 0
    assert validate_solution(graph, sol, 3) == []
