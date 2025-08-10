#!/usr/bin/env python3

from src.algorithms.dominating_set import DominatingSetAlgorithm
from src.algorithms.greedy import GreedyAlgorithm
from src.algorithms.ilp import ILPSolver
from src.algorithms.naive import NaiveAlgorithm
from src.core import LicenseConfigFactory, SolutionValidator
from src.graphs import GraphGeneratorFactory
import networkx as nx


def test_dominating_set_algorithm():
    print("Testing Dominating Set Algorithm...")

    validator = SolutionValidator()
    dominating_set_algo = DominatingSetAlgorithm()

    # Test 1: Star graph - idealny przypadek dla zbioru dominującego
    print("\n=== Test 1: Star graph (6 nodes) ===")
    graph = nx.star_graph(5)  # 1 central + 5 leaves
    license_types = LicenseConfigFactory.get_config("spotify")

    solution = dominating_set_algo.solve(graph, license_types)
    is_valid = validator.is_valid_solution(solution, graph)

    print(f"Solution valid: {is_valid}")
    print(f"Total cost: {solution.total_cost}")
    print(f"Groups: {len(solution.groups)}")
    for i, group in enumerate(solution.groups):
        print(f"  Group {i + 1}: {group.license_type.name}, owner: {group.owner}, members: {sorted(group.additional_members)}")

    # Test 2: Complete graph
    print("\n=== Test 2: Complete graph (5 nodes) ===")
    graph = nx.complete_graph(5)

    solution = dominating_set_algo.solve(graph, license_types)
    is_valid = validator.is_valid_solution(solution, graph)

    print(f"Solution valid: {is_valid}")
    print(f"Total cost: {solution.total_cost}")
    print(f"Groups: {len(solution.groups)}")
    for i, group in enumerate(solution.groups):
        print(f"  Group {i + 1}: {group.license_type.name}, owner: {group.owner}, members: {sorted(group.additional_members)}")

    # Test 3: Path graph
    print("\n=== Test 3: Path graph (6 nodes) ===")
    graph = nx.path_graph(6)

    solution = dominating_set_algo.solve(graph, license_types)
    is_valid = validator.is_valid_solution(solution, graph)

    print(f"Solution valid: {is_valid}")
    print(f"Total cost: {solution.total_cost}")
    print(f"Groups: {len(solution.groups)}")
    for i, group in enumerate(solution.groups):
        print(f"  Group {i + 1}: {group.license_type.name}, owner: {group.owner}, members: {sorted(group.additional_members)}")

    # Test 4: Porównanie z innymi algorytmami
    print("\n=== Test 4: Comparison with other algorithms ===")
    graph = GraphGeneratorFactory.get_generator("small_world")(n_nodes=8, k=2, p=0.3, seed=42)
    license_types = LicenseConfigFactory.get_config("duolingo_super")

    print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    results = {}

    # Dominating Set
    ds_solution = dominating_set_algo.solve(graph, license_types)
    ds_valid = validator.is_valid_solution(ds_solution, graph)
    results["Dominating Set"] = ds_solution.total_cost
    print(f"Dominating Set: cost={ds_solution.total_cost:.2f}, valid={ds_valid}, groups={len(ds_solution.groups)}")

    # Greedy
    greedy = GreedyAlgorithm()
    greedy_solution = greedy.solve(graph, license_types)
    greedy_valid = validator.is_valid_solution(greedy_solution, graph)
    results["Greedy"] = greedy_solution.total_cost
    print(f"Greedy: cost={greedy_solution.total_cost:.2f}, valid={greedy_valid}, groups={len(greedy_solution.groups)}")

    # Naive (jeśli graf jest wystarczająco mały)
    if len(graph.nodes()) <= 10:
        try:
            naive = NaiveAlgorithm()
            naive_solution = naive.solve(graph, license_types)
            naive_valid = validator.is_valid_solution(naive_solution, graph)
            results["Naive (Optimal)"] = naive_solution.total_cost
            print(f"Naive (Optimal): cost={naive_solution.total_cost:.2f}, valid={naive_valid}, groups={len(naive_solution.groups)}")
        except Exception as e:
            print(f"Naive failed: {e}")

    # ILP (optimal)
    try:
        ilp = ILPSolver()
        ilp_solution = ilp.solve(graph, license_types)
        ilp_valid = validator.is_valid_solution(ilp_solution, graph)
        results["ILP (Optimal)"] = ilp_solution.total_cost
        print(f"ILP (Optimal): cost={ilp_solution.total_cost:.2f}, valid={ilp_valid}, groups={len(ilp_solution.groups)}")
    except Exception as e:
        print(f"ILP failed: {e}")

    # Analiza wyników
    print("\n--- Performance Analysis ---")
    if results:
        best_cost = min(results.values())
        print(f"Best cost: {best_cost:.2f}")

        for algo_name, cost in results.items():
            gap = ((cost - best_cost) / best_cost * 100) if best_cost > 0 else 0
            print(f"{algo_name}: {cost:.2f} (+{gap:.1f}% from optimal)")

    # Test 5: Różne konfiguracje licencji
    print("\n=== Test 5: Different license configurations ===")
    graph = nx.cycle_graph(8)

    configs = ["spotify", "duolingo_super", "roman_domination"]

    for config in configs:
        license_types = LicenseConfigFactory.get_config(config)
        solution = dominating_set_algo.solve(graph, license_types)
        is_valid = validator.is_valid_solution(solution, graph)
        print(f"{config}: cost={solution.total_cost:.2f}, valid={is_valid}, groups={len(solution.groups)}")

    # Test 6: Edge cases
    print("\n=== Test 6: Edge cases ===")

    # Single node
    single_graph = nx.Graph()
    single_graph.add_node(0)
    solution = dominating_set_algo.solve(single_graph, license_types)
    is_valid = validator.is_valid_solution(solution, single_graph)
    print(f"Single node: cost={solution.total_cost:.2f}, valid={is_valid}, groups={len(solution.groups)}")

    # Empty graph
    empty_graph = nx.Graph()
    solution = dominating_set_algo.solve(empty_graph, license_types)
    print(f"Empty graph: cost={solution.total_cost:.2f}, groups={len(solution.groups)}")

    # Disconnected graph
    disconnected = nx.Graph()
    disconnected.add_edges_from([(0, 1), (2, 3)])  # Two separate edges
    solution = dominating_set_algo.solve(disconnected, license_types)
    is_valid = validator.is_valid_solution(solution, disconnected)
    print(f"Disconnected graph: cost={solution.total_cost:.2f}, valid={is_valid}, groups={len(solution.groups)}")


if __name__ == "__main__":
    test_dominating_set_algorithm()
