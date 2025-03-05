import os

import matplotlib.pyplot as plt

from src.generators import WattsStrogatzGenerator
from src.solvers import (
    AntColonySolver,
    GreedySolver,
    MIPSolver,
    TabuSolver,
)
from src.solvers.base import BaseStaticSolver
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOLVERS: list[tuple[str, type[BaseStaticSolver]]] = [
    ("MIP", MIPSolver),
    ("Greedy", GreedySolver),
    ("Ant Colony", AntColonySolver),
    ("Tabu", TabuSolver),
]

INDIVIDUAL_COST = 5.0
GROUP_COST = 8.0
GROUP_SIZE = 6

os.makedirs("benchmarks/plots", exist_ok=True)


def main():
    print("\n=== OPTIMAL LICENSE DISTRIBUTION SOLVER ===\n")

    generator = WattsStrogatzGenerator(12, 0.3)

    results = []

    for size in range(20, 301, 10):
        graph = generator.generate(size)

        for solver_name, solver_class in SOLVERS:
            print(f"\n--- Running {solver_name} Solver ---\n")

            solver = solver_class(INDIVIDUAL_COST, GROUP_COST, GROUP_SIZE)
            result = solver.run(graph)

            total_cost = result["total_cost"]
            execution_time = result["execution_time"]

            results.append((solver_name, size, total_cost, execution_time))

            logger.info(
                f"Solver: {solver_name}, Size: {size}, Total Cost: §{total_cost}, Time Taken: {execution_time:.4f} seconds."
            )

    results.sort(key=lambda x: (x[1], x[2]))
    print("\n=== BENCHMARK RESULTS ===")
    print(f"{'Solver':<10} {'Nodes':<10} {'Total Cost':<15} {'Time Taken (s)':<15}")
    print("-" * 50)

    for solver_name, size, cost, execution_time in results:
        print(f"{solver_name:<10} {size:<10} {cost:<15.2f} {execution_time:<15.4f}")

    plt.figure(figsize=(8, 5))
    for solver_name, _ in SOLVERS:
        solver_results = [r for r in results if r[0] == solver_name]
        sizes = [r[1] for r in solver_results]
        times = [r[3] for r in solver_results]

        plt.plot(sizes, times, label=solver_name)

    plt.xlabel("Number of Nodes")
    plt.ylabel("Time Taken (s)")
    plt.title("Solver Performance Benchmark")
    plt.legend()
    plt.grid()

    plt.savefig("benchmarks/plots/solver_performance.png")
    plt.show()

    quality_data = {}
    for solver_name, _ in SOLVERS:
        quality_data[solver_name] = {"sizes": [], "ratios": []}

    results_by_size = {}
    for solver_name, size, cost, _ in results:
        if size not in results_by_size:
            results_by_size[size] = {}
        results_by_size[size][solver_name] = cost

    for size, cost_dict in results_by_size.items():
        mip_cost = cost_dict["MIP"]
        for solver_name, cost in cost_dict.items():
            ratio = cost / mip_cost
            quality_data[solver_name]["sizes"].append(size)
            quality_data[solver_name]["ratios"].append(ratio)

    plt.figure(figsize=(8, 5))
    for solver_name in quality_data:
        sizes = quality_data[solver_name]["sizes"]
        ratios = quality_data[solver_name]["ratios"]
        plt.plot(sizes, ratios, label=solver_name)

    plt.xlabel("Number of Nodes")
    plt.ylabel("Solution Quality (Solver Cost / MIP Cost)")
    plt.title("Solver Quality Benchmark")
    plt.legend()
    plt.grid()

    plt.savefig("benchmarks/plots/solver_quality.png")
    plt.show()


if __name__ == "__main__":
    main()
