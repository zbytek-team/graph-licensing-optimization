#!/usr/bin/env python3

from src.algorithms.naive import NaiveAlgorithm
from src.algorithms.greedy import GreedyAlgorithm
from src.algorithms.ilp import ILPSolver
from src.core import LicenseConfigFactory, SolutionValidator
from src.graphs import GraphGeneratorFactory
import networkx as nx


def test_naive_algorithm():
    print("Testing Naive Algorithm...")

    # Test 1: Very small graph (3 nodes)
    print("\n=== Test 1: Triangle graph (3 nodes) ===")
    graph = nx.complete_graph(3)
    license_types = LicenseConfigFactory.get_config("spotify")

    naive = NaiveAlgorithm()
    solution = naive.solve(graph, license_types)

    validator = SolutionValidator()
    is_valid = validator.is_valid_solution(solution, graph)

    print(f"Solution valid: {is_valid}")
    print(f"Total cost: {solution.total_cost}")
    print(f"Groups: {len(solution.groups)}")
    for i, group in enumerate(solution.groups):
        print(f"  Group {i + 1}: {group.license_type.name}, owner: {group.owner}, members: {group.additional_members}")

    # Test 2: Path graph (4 nodes)
    print("\n=== Test 2: Path graph (4 nodes) ===")
    graph = nx.path_graph(4)

    solution = naive.solve(graph, license_types)
    is_valid = validator.is_valid_solution(solution, graph)

    print(f"Solution valid: {is_valid}")
    print(f"Total cost: {solution.total_cost}")
    print(f"Groups: {len(solution.groups)}")
    for i, group in enumerate(solution.groups):
        print(f"  Group {i + 1}: {group.license_type.name}, owner: {group.owner}, members: {group.additional_members}")

    # Test 3: Compare with greedy and ILP on small graph
    print("\n=== Test 3: Comparison with other algorithms ===")
    graph = GraphGeneratorFactory.get_generator("small_world")(n_nodes=6, k=2, p=0.3, seed=42)
    license_types = LicenseConfigFactory.get_config("duolingo_super")

    print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    # Naive
    naive_solution = naive.solve(graph, license_types)
    naive_valid = validator.is_valid_solution(naive_solution, graph)
    print(f"Naive: cost={naive_solution.total_cost}, valid={naive_valid}, groups={len(naive_solution.groups)}")

    # Greedy
    greedy = GreedyAlgorithm()
    greedy_solution = greedy.solve(graph, license_types)
    greedy_valid = validator.is_valid_solution(greedy_solution, graph)
    print(f"Greedy: cost={greedy_solution.total_cost}, valid={greedy_valid}, groups={len(greedy_solution.groups)}")

    # ILP (optimal)
    try:
        ilp = ILPSolver()
        ilp_solution = ilp.solve(graph, license_types)
        ilp_valid = validator.is_valid_solution(ilp_solution, graph)
        print(f"ILP: cost={ilp_solution.total_cost}, valid={ilp_valid}, groups={len(ilp_solution.groups)}")

        # Verify naive finds optimal solution
        if naive_solution.total_cost == ilp_solution.total_cost:
            print("✅ Naive algorithm found optimal solution!")
        else:
            print(f"❌ Naive algorithm suboptimal: {naive_solution.total_cost} vs {ilp_solution.total_cost}")
    except Exception as e:
        print(f"ILP failed: {e}")

    # Test 4: Edge case - single node
    print("\n=== Test 4: Single node ===")
    graph = nx.Graph()
    graph.add_node(0)

    solution = naive.solve(graph, license_types)
    is_valid = validator.is_valid_solution(solution, graph)
    print(f"Solution valid: {is_valid}")
    print(f"Total cost: {solution.total_cost}")
    print(f"Groups: {len(solution.groups)}")

    # Test 5: Test size limit
    print("\n=== Test 5: Size limit test ===")
    try:
        large_graph = nx.complete_graph(11)  # Too large
        naive.solve(large_graph, license_types)
        print("❌ Should have failed for graph > 10 nodes")
    except ValueError as e:
        print(f"✅ Correctly rejected large graph: {e}")


if __name__ == "__main__":
    test_naive_algorithm()
