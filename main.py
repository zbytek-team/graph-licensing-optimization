import networkx as nx
from src.solvers.greedy import GreedySolver
from src.solvers.mip import MIPSolver

SOLVERS = [
    ("Greedy", GreedySolver),
    ("MIP", MIPSolver)
]

INDIVIDUAL_COST = 5.0
GROUP_COST = 8.0
GROUP_SIZE = 6

def main():
    print("\n=== OPTIMAL LICENSE DISTRIBUTION SOLVER ===\n")

    graph = nx.erdos_renyi_graph(20, 0.2, seed=42)

    for solver_name, solver_class in SOLVERS:
        print(f"\n--- Running {solver_name} Solver ---\n")

        solver = solver_class(INDIVIDUAL_COST, GROUP_COST, GROUP_SIZE)
        solution, cost = solver.run(graph)

        print("Solution:")
        for node, license_type in solution.items():
            print(f"Node {node}: {license_type}")

        print(f"\nTotal cost: {cost}\n")

if __name__ == "__main__":
    main()
