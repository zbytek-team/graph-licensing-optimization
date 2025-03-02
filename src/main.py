import os


from src.generators import ScaleFreeGenerator
from src.solvers.base import StaticSolver
from src.solvers.static import AntColonySolver, GreedySolver, MIPSolver, TabuSolver
from src.utils.visualization import visualize_graph

SOLVERS: list[tuple[str, type[StaticSolver]]] = [
    ("Greedy", GreedySolver),
    ("MIP", MIPSolver),
    ("Tabu", TabuSolver),
    ("Ant Colony", AntColonySolver),
]

INDIVIDUAL_COST = 5.0
GROUP_COST = 8.0
GROUP_SIZE = 6

os.makedirs("images", exist_ok=True)


def main():
    print("\n=== OPTIMAL LICENSE DISTRIBUTION SOLVER ===\n")

    generator = ScaleFreeGenerator()
    graph = generator.generate(100, m=2)
    results: list[tuple[str, float, float]] = []

    for solver_name, solver_class in SOLVERS:
        print(f"\n--- Running {solver_name} Solver ---\n")

        solver = solver_class(INDIVIDUAL_COST, GROUP_COST, GROUP_SIZE)
        result = solver.run(graph)

        assignment = result["assignment"]
        total_cost = result["total_cost"]
        time_taken = result["time_taken"]

        results.append((solver_name, total_cost, time_taken))

        visualize_graph(graph, assignment, f"images/{solver_name}.png")

        if len(assignment["individual"]) < 50 and len(assignment["group"]) < 10:
            print("\nSolution:")
            print(f"\t- Individual assignments: {assignment['individual']}")
            print("\t- Group assignments:")
            for holder, members in assignment["group"].items():
                print(f"\t\t- {holder}:\t{members}")

        print(f"\nTotal Cost: §{total_cost}.")
        print(f"Time Taken: {time_taken:.4f} seconds.")

    sorted_results = sorted(results, key=lambda x: x[1])

    print("\n=== SOLVER RANKINGS ===")
    for rank, (solver_name, cost, time_taken) in enumerate(sorted_results, start=1):
        print(f"{rank}. {solver_name} - Total Cost: §{cost} - Time Taken: {time_taken:.4f} seconds.")


if __name__ == "__main__":
    main()
