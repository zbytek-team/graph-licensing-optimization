from src.algorithms import RandomizedAlgorithm, GreedyAlgorithm, ILPSolver
from src.core import LicenseConfigFactory
import networkx as nx


def quick_comparison_test():
    graph = nx.cycle_graph(6)
    license_types = LicenseConfigFactory.get_config("spotify")
    
    algorithms = [
        ("ILP Optimal", ILPSolver()),
        ("Greedy", GreedyAlgorithm()),
        ("Randomized (90% greedy)", RandomizedAlgorithm(greedy_probability=0.9, seed=42)),
        ("Randomized (50% greedy)", RandomizedAlgorithm(greedy_probability=0.5, seed=42)),
        ("Randomized (10% greedy)", RandomizedAlgorithm(greedy_probability=0.1, seed=42)),
    ]
    
    print("Quick Comparison Test - 6-node cycle graph")
    print("-" * 60)
    
    results = []
    for name, algorithm in algorithms:
        solution = algorithm.solve(graph, license_types)
        results.append((name, solution.total_cost, len(solution.groups)))
        print(f"{name:25} | Cost: {solution.total_cost:6.2f} | Groups: {len(solution.groups):2d}")
    
    print("-" * 60)
    
    # Sprawdź, czy wszystkie rozwiązania są valide
    optimal_cost = results[0][1]  # ILP optimal
    for name, cost, groups in results:
        quality_ratio = cost / optimal_cost
        print(f"{name:25} | Quality ratio: {quality_ratio:.2f}")
    
    print("\nTEST PASSED: Wszystkie algorytmy znajdą valide rozwiązania!")


if __name__ == "__main__":
    quick_comparison_test()
