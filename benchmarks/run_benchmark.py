import os

from src.solvers.base import StaticSolver
from src.solvers.static import AntColonySolver, GreedySolver, MIPSolver, TabuSolver
from src.generators import ScaleFreeGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOLVERS: list[tuple[str, type[StaticSolver]]] = [
    ("Greedy", GreedySolver),
    ("MIP", MIPSolver),
    ("Ant Colony", AntColonySolver),
]

INDIVIDUAL_COST = 5.0
GROUP_COST = 8.0
GROUP_SIZE = 6

os.makedirs("images", exist_ok=True)


def main():
    print("\n=== OPTIMAL LICENSE DISTRIBUTION SOLVER ===\n")

    generator = ScaleFreeGenerator()

    results = []

    for size in range(100, 1001, 100):
        graph = generator.generate(size, m=2)

        for solver_name, solver_class in SOLVERS:
            print(f"\n--- Running {solver_name} Solver ---\n")

            solver = solver_class(INDIVIDUAL_COST, GROUP_COST, GROUP_SIZE)
            result = solver.run(graph)

            total_cost = result["total_cost"]
            time_taken = result["time_taken"]

            results.append((solver_name, size, total_cost, time_taken))

            logger.info(
                f"Solver: {solver_name}, Size: {size}, Total Cost: §{total_cost}, Time Taken: {time_taken:.4f} seconds."
            )

    results.sort(key=lambda x: (x[1], x[2]))
    print("\n=== BENCHMARK RESULTS ===")
    print(f"{'Solver':<10} {'Nodes':<10} {'Total Cost':<15} {'Time Taken (s)':<15}")
    print("-" * 50)

    for solver_name, size, cost, time_taken in results:
        print(f"{solver_name:<10} {size:<10} {cost:<15.2f} {time_taken:<15.4f}")


if __name__ == "__main__":
    main()
