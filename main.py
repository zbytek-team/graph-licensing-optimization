import networkx as nx
from src.solvers import TabuSolver, GreedySolver, MIPSolver, AntColonySolver
from src.solvers.base import Solver
from src.visualize import visualize_graph

SOLVERS: list[tuple[str, type[Solver]]] = [
    ("Greedy", GreedySolver),
    ("MIP", MIPSolver),
    ("Tabu", TabuSolver),
    ("Ant Colony", AntColonySolver),
]

INDIVIDUAL_COST = 5.0
GROUP_COST = 8.0
GROUP_SIZE = 6


def main():
    print("\n=== OPTIMAL LICENSE DISTRIBUTION SOLVER ===\n")

    graph = nx.erdos_renyi_graph(100, 0.05, seed=42)
    results: list[tuple[str, float]] = []

    for solver_name, solver_class in SOLVERS:
        print(f"\n--- Running {solver_name} Solver ---\n")

        solver = solver_class(INDIVIDUAL_COST, GROUP_COST, GROUP_SIZE)
        result = solver.run(graph)

        assignment = result["assignment"]
        total_cost = result["total_cost"]
        results.append((solver_name, total_cost))

        visualize_graph(graph, assignment, f"images/{solver_name}.png")

        print("Solution:")
        for node in assignment["individual"]:
            print(f"Node {node}: Individual")
        for holder, members in assignment["group"].items():
            print(f"Group led by {holder}: {members}")

        print(f"\nTotal cost: §{total_cost}\n")

    sorted_results = sorted(results, key=lambda x: x[1])

    print("\n=== SOLVER RANKINGS ===")
    for rank, (solver_name, cost) in enumerate(sorted_results, start=1):
        print(f"{rank}. {solver_name} - Total Cost: §{cost}")


if __name__ == "__main__":
    main()
