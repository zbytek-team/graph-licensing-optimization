from src.algorithms.ilp import ILPSolver
from src.algorithms.greedy import GreedyAlgorithm
from src.algorithms.tabu_search import TabuSearch
from src.utils.graphs.generator import GraphGeneratorFactory
from src.utils.graphs.visualization import GraphVisualizer
from src.utils.licenses import LicenseConfigFactory, LicenseConfigInfo
from datetime import datetime
import os


def test_algorithm(algorithm, algorithm_name, graph, license_types, visualizer=None, timestamp_folder=None):
    print(f"\n{'=' * 20} {algorithm_name} {'=' * 20}")

    try:
        solution = algorithm.solve(graph, license_types)

        print("Solution found!")
        print(f"- Total cost: {solution.total_cost}")
        print(f"- Number of groups: {len(solution.groups)}")
        print(f"- Covered nodes: {len(solution.covered_nodes)}/{len(graph.nodes())}")

        # Debug: Print groups before validation for ILP
        if algorithm_name == "ILP Solver":
            print("\nDEBUG: Groups before validation:")
            for i, group in enumerate(solution.groups, 1):
                print(f"  Group {i}: {group.license_type.name} (capacity {group.license_type.min_capacity}-{group.license_type.max_capacity})")
                print(f"    Owner: {group.owner}")
                print(f"    Additional members: {sorted(group.additional_members)}")
                print(f"    Actual size: {group.size}")

        # Validate solution
        all_nodes = set(graph.nodes())
        is_valid = solution.is_valid(graph, all_nodes)
        print(f"- Solution is valid: {is_valid}")

        if len(solution.groups) <= 10:  # Only show details for small solutions
            print("\nLicense groups:")
            for i, group in enumerate(solution.groups, 1):
                print(f"  Group {i}: {group.license_type.name}")
                print(f"    Owner: {group.owner}")
                print(f"    Additional members: {sorted(group.additional_members)}")
                print(f"    Size: {group.size}")
                print(f"    Cost: {group.license_type.cost}")
        else:
            print("\n(Too many groups to display - showing summary only)")

        # Create visualization for this algorithm
        if visualizer:
            solver_name = algorithm_name.lower().replace(" ", "_")
            visualizer.visualize_solution(graph, solution, solver_name=solver_name, timestamp_folder=timestamp_folder)

        return solution.total_cost

    except Exception as e:
        print(f"Error solving with {algorithm_name}: {e}")
        return float("inf")


def main():
    print("Graph Licensing Optimization - Algorithm Comparison")
    print("=" * 60)

    # Create small world graph with 50 nodes
    generator = GraphGeneratorFactory.get_generator("small_world")
    graph = generator.generate(n_nodes=50, k=4, p=0.3, seed=42)

    print("Generated small world graph:")
    print(f"- Nodes: {len(graph.nodes())}")
    print(f"- Edges: {len(graph.edges())}")
    print(f"- Average degree: {sum(dict(graph.degree()).values()) / len(graph.nodes()):.2f}")

    # Get license configuration
    license_config = "roman_domination"  # Change this to test different configs: duolingo_super, spotify, roman_domination
    license_types = LicenseConfigFactory.get_config(license_config)

    print(f"\nUsing license configuration: {license_config.upper()}")
    LicenseConfigInfo.print_config_info(license_config)

    # Create results directory and timestamp folder
    now = datetime.now()
    timestamp_folder = now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"results/graphs/{timestamp_folder}", exist_ok=True)

    # Test algorithms
    algorithms = [(GreedyAlgorithm(), "Greedy Algorithm"), (TabuSearch(), "Tabu Search"), (ILPSolver(), "ILP Solver")]

    results = {}
    visualizer = GraphVisualizer()

    for algorithm, name in algorithms:
        cost = test_algorithm(algorithm, name, graph, license_types, visualizer=visualizer, timestamp_folder=timestamp_folder)
        results[name] = cost

    # Compare results
    print(f"\n{'=' * 20} COMPARISON {'=' * 20}")
    best_cost = min(results.values())
    for name, cost in results.items():
        if cost == float("inf"):
            print(f"{name}: FAILED")
        else:
            gap = ((cost - best_cost) / best_cost * 100) if best_cost > 0 else 0
            print(f"{name}: cost={cost:.1f} (gap: {gap:.1f}%)")


if __name__ == "__main__":
    main()
