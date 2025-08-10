#!/usr/bin/env python3

from src.algorithms import (
    ILPSolver,
    GreedyAlgorithm,
    NaiveAlgorithm,
    DominatingSetAlgorithm,
)
from src.graphs import GraphGeneratorFactory, GraphVisualizer
from src.core import LicenseConfigFactory, SolutionValidator
from datetime import datetime
import os

GRAPH_TYPES = ["complete", "small_world", "random", "cycle", "path"]
GRAPH_NODES = [4, 6, 8, 10]  # Small graphs only for naive algorithm
GRAPH_SEED = 42
LICENSE_CONFIGS = ["spotify", "duolingo_super", "roman_domination"]

ALGORITHMS = [
    ("Naive Algorithm", lambda: NaiveAlgorithm()),
    ("Dominating Set", lambda: DominatingSetAlgorithm()),
    ("Greedy", lambda: GreedyAlgorithm()),
    ("ILP Solver", lambda: ILPSolver()),
]


def test_algorithm(algorithm, algorithm_name, graph, license_types, visualizer=None, timestamp_folder=None):
    print(f"\n{'=' * 20} {algorithm_name} {'=' * 20}")
    try:
        solution = algorithm.solve(graph, license_types)
        print("Solution found!")
        print(f"- Total cost: {solution.total_cost}")
        print(f"- Number of groups: {len(solution.groups)}")
        print(f"- Covered nodes: {len(solution.covered_nodes)}/{len(graph.nodes())}")

        validator = SolutionValidator()
        try:
            is_valid = validator.is_valid_solution(solution, graph)
            print(f"- Solution is valid: {is_valid}")
        except Exception as validation_error:
            print(f"- Solution validation FAILED: {validation_error}")
            return float("inf")

        if len(solution.groups) <= 5:
            print("\nLicense groups:")
            for i, group in enumerate(solution.groups, 1):
                print(f"  Group {i}: {group.license_type.name}")
                print(f"    Owner: {group.owner}, Members: {sorted(group.additional_members)}")

        return solution.total_cost
    except Exception as e:
        print(f"\n🚨 ALGORITHM FAILURE: {algorithm_name}")
        print(f"📍 Error Type: {type(e).__name__}")
        print(f"💬 Error Message: {str(e)}")
        return float("inf")


def main():
    print("Small Graph Testing with Naive Algorithm (Reference)")
    print("=" * 60)

    now = datetime.now()
    timestamp_folder = now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"results/graphs/{timestamp_folder}", exist_ok=True)

    visualizer = GraphVisualizer()

    for license_config in LICENSE_CONFIGS:
        for graph_type in GRAPH_TYPES:
            for nodes in GRAPH_NODES:
                print(f"\n\n{'=' * 80}")
                print(f"Testing: {graph_type} graph with {nodes} nodes, {license_config} licenses")
                print(f"{'=' * 80}")

                try:
                    # Generate graph
                    generator = GraphGeneratorFactory.get_generator(graph_type)

                    if graph_type == "small_world":
                        k = min(4, nodes - 1)
                        if k < 2:
                            k = 2
                        if k >= nodes:
                            k = nodes - 1
                        graph = generator(n_nodes=nodes, k=k, p=0.3, seed=GRAPH_SEED)
                    elif graph_type == "random":
                        graph = generator(n_nodes=nodes, p=0.3, seed=GRAPH_SEED)
                    else:
                        graph = generator(n_nodes=nodes, seed=GRAPH_SEED)

                    license_types = LicenseConfigFactory.get_config(license_config)

                    print(f"Generated {graph_type} graph:")
                    print(f"- Nodes: {len(graph.nodes())}")
                    print(f"- Edges: {len(graph.edges())}")
                    if len(graph.nodes()) > 0:
                        print(f"- Average degree: {sum(dict(graph.degree()).values()) / len(graph.nodes()):.2f}")

                    results = {}

                    # Test all algorithms
                    for name, algorithm_factory in ALGORITHMS:
                        algorithm_instance = algorithm_factory()
                        cost = test_algorithm(
                            algorithm_instance,
                            name,
                            graph,
                            license_types,
                            visualizer,
                            timestamp_folder,
                        )
                        results[name] = cost

                    # Compare results
                    print(f"\n{'=' * 40}")
                    print("RESULTS COMPARISON:")
                    print(f"{'=' * 40}")

                    sorted_results = sorted(results.items(), key=lambda x: x[1])
                    best_cost = sorted_results[0][1] if sorted_results else float("inf")

                    for name, cost in sorted_results:
                        if cost == float("inf"):
                            print(f"{name:20}: FAILED")
                        else:
                            optimality_gap = ((cost - best_cost) / best_cost * 100) if best_cost > 0 else 0
                            marker = "🏆" if cost == best_cost else "📈"
                            print(f"{name:20}: {cost:8.2f} {marker} (+{optimality_gap:5.1f}%)")

                    # Check if naive found optimal
                    naive_cost = results.get("Naive Algorithm", float("inf"))
                    ilp_cost = results.get("ILP Solver", float("inf"))

                    if naive_cost != float("inf") and ilp_cost != float("inf"):
                        if abs(naive_cost - ilp_cost) < 0.01:
                            print("\n✅ Naive algorithm found OPTIMAL solution!")
                        else:
                            print(f"\n❌ Naive algorithm suboptimal: {naive_cost:.2f} vs optimal {ilp_cost:.2f}")

                except Exception as e:
                    print(f"Failed to process {graph_type} with {nodes} nodes: {e}")

    print(f"\n\nTesting completed! Visualizations saved to: results/graphs/{timestamp_folder}/")


if __name__ == "__main__":
    main()
