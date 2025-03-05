import os

from src.generators import WattsStrogatzGenerator
from src.solvers import (
    AntColonySolver,
    GreedySolver,
    MIPSolver,
    TabuSolver,
)
from src.solvers.base import BaseStaticSolver
from src.utils.visualization import create_layout, visualize_graph

BASE_PARAMS = {"individual_cost": 1, "group_cost": 1.2, "group_size": 6}

SOLVERS: list[BaseStaticSolver] = [
    GreedySolver(**BASE_PARAMS),
    MIPSolver(**BASE_PARAMS),
    TabuSolver(**BASE_PARAMS, tabu_size=256, iterations=128, max_neighbor_solutions=128),
    AntColonySolver(**BASE_PARAMS, ant_count=64, alpha=1.0, beta=2.0, evaporation_rate=0.1, iterations=128),
]

os.makedirs("visualizations", exist_ok=True)


def main():
    print("\n=== OPTIMAL LICENSE DISTRIBUTION ===\n")

    generator = WattsStrogatzGenerator(8, 0.3)
    graph = generator.generate(100)
    layout = create_layout(graph)

    results: list[tuple[str, float, float]] = []

    for solver in SOLVERS:
        solver_name = str(solver)
        print(f"\n--- Running {solver} Solver ---\n")

        result = solver.run(graph)

        assignment = result["assignment"]
        total_cost = result["total_cost"]
        execution_time = result["execution_time"]

        results.append((solver_name, total_cost, execution_time))

        visualize_graph(graph, assignment, f"visualizations/{solver_name}.png", layout)

        assignment_count = len(assignment["individual"]) + len(assignment["group"])

        if assignment_count < 101:
            print("\nSolution:")
            print(f"\t- Individual assignments: {assignment['individual'] or 'No individual assignments.'}")
            print("\t- Group assignments:")
            for holder, members in assignment["group"].items():
                print(f"\t\t- {holder}:\t{members}")

            if len(assignment["group"]) == 0:
                print("\t\tNo group assignments.")

        print(f"\nAssignment count: {assignment_count}.")
        print(f"Total Cost: §{total_cost}.")
        print(f"Execution Time: {execution_time:.4f} s.")

    sorted_results = sorted(results, key=lambda x: (x[1], x[2]))

    print("\n=== SOLVER RANKINGS ===")
    for rank, (solver_name, cost, execution_time) in enumerate(sorted_results, start=1):
        print(f"{rank}. {solver_name} - Total Cost: §{round(cost, 2)} - Execution Time: {execution_time:.4f} s.")


if __name__ == "__main__":
    main()
