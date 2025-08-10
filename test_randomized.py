import networkx as nx
from src.algorithms import RandomizedAlgorithm, ILPSolver
from src.core import LicenseType
from src.graphs import GraphGeneratorFactory


def test_randomized_algorithm_basic():
    graph = nx.path_graph(5)
    license_types = [
        LicenseType("individual", cost=1.0, min_capacity=1, max_capacity=1),
        LicenseType("small_group", cost=2.5, min_capacity=2, max_capacity=3),
    ]
    
    algorithm = RandomizedAlgorithm(greedy_probability=0.5, seed=42)
    solution = algorithm.solve(graph, license_types)
    
    assert solution.covered_nodes == set(graph.nodes())
    assert solution.total_cost > 0
    assert all(group.license_type in license_types for group in solution.groups)


def test_randomized_algorithm_reproducibility():
    graph = nx.star_graph(4)
    license_types = [
        LicenseType("individual", cost=1.0, min_capacity=1, max_capacity=1),
        LicenseType("group", cost=3.0, min_capacity=2, max_capacity=5),
    ]
    
    algorithm1 = RandomizedAlgorithm(greedy_probability=0.7, seed=123)
    algorithm2 = RandomizedAlgorithm(greedy_probability=0.7, seed=123)
    
    solution1 = algorithm1.solve(graph, license_types)
    solution2 = algorithm2.solve(graph, license_types)
    
    assert solution1.total_cost == solution2.total_cost
    assert solution1.covered_nodes == solution2.covered_nodes


def test_randomized_algorithm_pure_greedy():
    graph = nx.complete_graph(4)
    license_types = [
        LicenseType("individual", cost=1.0, min_capacity=1, max_capacity=1),
        LicenseType("group", cost=2.0, min_capacity=2, max_capacity=4),
    ]
    
    # Pure greedy (p=1.0)
    algorithm = RandomizedAlgorithm(greedy_probability=1.0, seed=42)
    solution = algorithm.solve(graph, license_types)
    
    assert solution.covered_nodes == set(graph.nodes())
    # W grafie pełnym optymalne jest jedna grupa
    assert len(solution.groups) == 1
    assert solution.groups[0].license_type.name == "group"


def test_randomized_algorithm_pure_random():
    graph = nx.cycle_graph(4)
    license_types = [
        LicenseType("individual", cost=1.0, min_capacity=1, max_capacity=1),
        LicenseType("pair", cost=1.8, min_capacity=2, max_capacity=2),
    ]
    
    # Pure random (p=0.0)
    algorithm = RandomizedAlgorithm(greedy_probability=0.0, seed=42)
    solution = algorithm.solve(graph, license_types)
    
    assert solution.covered_nodes == set(graph.nodes())
    assert solution.total_cost > 0


def test_randomized_algorithm_edge_cases():
    # Pusty graf
    empty_graph = nx.Graph()
    license_types = [LicenseType("individual", cost=1.0, min_capacity=1, max_capacity=1)]
    
    algorithm = RandomizedAlgorithm(seed=42)
    solution = algorithm.solve(empty_graph, license_types)
    
    assert solution.covered_nodes == set()
    assert solution.total_cost == 0.0
    assert len(solution.groups) == 0
    
    # Jeden węzeł
    single_graph = nx.Graph()
    single_graph.add_node(0)
    
    solution = algorithm.solve(single_graph, license_types)
    assert solution.covered_nodes == {0}
    assert solution.total_cost == 1.0


def test_randomized_algorithm_vs_optimal():
    """Test porównujący z optymalnym rozwiązaniem ILP dla małych grafów."""
    graph = nx.path_graph(4)
    license_types = [
        LicenseType("individual", cost=1.0, min_capacity=1, max_capacity=1),
        LicenseType("pair", cost=1.9, min_capacity=2, max_capacity=2),
    ]
    
    # Rozwiązanie optymalne
    ilp_solver = ILPSolver()
    optimal_solution = ilp_solver.solve(graph, license_types)
    
    # Rozwiązanie randomized z wysoką tendencją zachłanną
    randomized_algorithm = RandomizedAlgorithm(greedy_probability=0.9, seed=42)
    randomized_solution = randomized_algorithm.solve(graph, license_types)
    
    # Sprawdź pokrycie
    assert randomized_solution.covered_nodes == set(graph.nodes())
    
    # Rozwiązanie nie powinno być gorsze niż 2x optimal (rozumny bound)
    quality_ratio = randomized_solution.total_cost / optimal_solution.total_cost
    assert quality_ratio <= 2.0


def test_randomized_algorithm_different_probabilities():
    """Test zachowania dla różnych wartości prawdopodobieństwa."""
    graph = nx.cycle_graph(6)
    license_types = [
        LicenseType("individual", cost=1.0, min_capacity=1, max_capacity=1),
        LicenseType("triple", cost=2.7, min_capacity=3, max_capacity=3),
    ]
    
    # Test z różnymi prawdopodobieństwami
    probabilities = [0.0, 0.3, 0.7, 1.0]
    results = []
    
    for prob in probabilities:
        algorithm = RandomizedAlgorithm(greedy_probability=prob, seed=123)
        solution = algorithm.solve(graph, license_types)
        results.append((prob, solution.total_cost))
        assert solution.covered_nodes == set(graph.nodes())
    
    # Wszystkie rozwiązania powinny być valide
    assert all(cost > 0 for _, cost in results)


def test_randomized_algorithm_large_graph():
    """Test na większym grafie."""
    graph = GraphGeneratorFactory.random(15, p=0.3, seed=42)
    license_types = [
        LicenseType("individual", cost=1.0, min_capacity=1, max_capacity=1),
        LicenseType("small_group", cost=2.5, min_capacity=2, max_capacity=3),
        LicenseType("large_group", cost=4.0, min_capacity=4, max_capacity=6),
    ]
    
    algorithm = RandomizedAlgorithm(greedy_probability=0.6, seed=42)
    solution = algorithm.solve(graph, license_types)
    
    assert solution.covered_nodes == set(graph.nodes())
    assert solution.total_cost > 0
    assert len(solution.groups) > 0


if __name__ == "__main__":
    test_randomized_algorithm_basic()
    test_randomized_algorithm_reproducibility()
    test_randomized_algorithm_pure_greedy()
    test_randomized_algorithm_pure_random()
    test_randomized_algorithm_edge_cases()
    test_randomized_algorithm_vs_optimal()
    test_randomized_algorithm_different_probabilities()
    test_randomized_algorithm_large_graph()
    print("Wszystkie testy algorytmu losowego przeszły pomyślnie!")
