import networkx as nx
from src.algorithms.static.ant_colony import AntColonySolver
from src.algorithms.base import BaseSolver
from src.algorithms.static.greedy_basic import GreedyBasicSolver
from src.algorithms.static.ilp_solver import ILPSolver
from src.utils.calculate_cost import calculate_cost
from src.data.generators import generate_erdos_renyi
from src.utils.graph_vis import visualize_graph


def run_solver(solver: BaseSolver, graph: nx.Graph, c_single: float, c_group: float, group_size: int):
    solution = solver.solve(graph, c_single, c_group, group_size)

    solver_name = solver.__class__.__name__

    print(f"\nSolver: {solver.__class__.__name__}")

    print(f"\tSingles: {solution['singles']}\n")

    print("\tGroups:")
    for group in solution["groups"]:
        print(f"\t\tHolder {group['license_holder']} with {group['members']}")

    cost = calculate_cost(solution, c_single, c_group)

    print(f"\n\tCost: §{cost:.2f}")

    visualize_graph(
        graph,
        solution,
        title=f"Solution by {solver_name}",
        show=False,
        save_path=f"results/images/{solver_name.lower()}.png",
    )


def main():
    graph = generate_erdos_renyi(50, 0.05, seed=42)
    c_single = 3.0
    c_group = 5.0
    group_size = 3

    greedy_basic_solver = GreedyBasicSolver()
    run_solver(greedy_basic_solver, graph, c_single, c_group, group_size)

    ilp_solver = ILPSolver()
    run_solver(ilp_solver, graph, c_single, c_group, group_size)

    ant_colony_solver = AntColonySolver(
        ant_count=10,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5,
        iterations=100,
    )
    run_solver(ant_colony_solver, graph, c_single, c_group, group_size)


if __name__ == "__main__":
    main()
