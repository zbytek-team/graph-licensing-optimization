from src.algorithms.base import Solution


def calculate_cost(solution: Solution, c_single: float, c_group: float) -> float:
    cost = 0.0
    cost += len(solution["singles"]) * c_single
    for _ in solution["groups"]:
        cost += c_group
    return cost
