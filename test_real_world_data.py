from src.graphs import RealWorldDataLoader
from src.algorithms import ILPSolver, GreedyAlgorithm, RandomizedAlgorithm
from src.core import LicenseConfigFactory
import networkx as nx


def test_facebook_data_loader():
    """Test podstawowy ładowarki danych Facebook."""
    loader = RealWorldDataLoader()
    
    # Test załadowania pojedynczego ego network
    graph = loader.load_facebook_ego_network("0")
    
    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes()) > 0
    assert len(graph.edges()) > 0
    
    # Sprawdź czy węzeł ego istnieje
    assert 0 in graph.nodes()
    assert graph.nodes[0].get('is_ego', False)
    
    print(f"Network 0: {len(graph.nodes())} węzłów, {len(graph.edges())} krawędzi")


def test_facebook_network_stats():
    """Test statystyk sieci Facebook."""
    loader = RealWorldDataLoader()
    
    stats = loader.get_facebook_network_stats()
    
    assert isinstance(stats, dict)
    assert len(stats) > 0
    
    # Sprawdź format statystyk
    for ego_id, stat in stats.items():
        assert 'nodes' in stat
        assert 'edges' in stat
        assert 'density' in stat
        assert stat['nodes'] > 0
        assert stat['edges'] >= 0
        
        print(f"Network {ego_id}: {stat['nodes']} nodes, density={stat['density']:.3f}")


def test_suitable_networks_for_testing():
    """Test znajdowania odpowiednich sieci do testowania."""
    loader = RealWorldDataLoader()
    
    # Znajdź sieci średniej wielkości
    suitable = loader.get_suitable_networks_for_testing(min_nodes=30, max_nodes=150)
    
    assert isinstance(suitable, list)
    print(f"Znaleziono {len(suitable)} odpowiednich sieci: {suitable}")
    
    # Sprawdź czy rzeczywiście mają odpowiedni rozmiar
    for ego_id in suitable[:3]:  # Sprawdź pierwsze 3
        graph = loader.load_facebook_ego_network(ego_id)
        node_count = len(graph.nodes())
        assert 30 <= node_count <= 150
        print(f"Network {ego_id}: {node_count} węzłów - OK")


def test_algorithms_on_real_data():
    """Test algorytmów na rzeczywistych danych Facebook."""
    loader = RealWorldDataLoader()
    
    # Znajdź małą sieć do szybkiego testowania
    suitable = loader.get_suitable_networks_for_testing(min_nodes=50, max_nodes=200)
    
    if not suitable:
        # Fallback - weź najmniejszą dostępną sieć
        stats = loader.get_facebook_network_stats()
        ego_id = min(stats.keys(), key=lambda x: stats[x]['nodes'])
    else:
        ego_id = suitable[0]
    graph = loader.load_facebook_ego_network(ego_id)
    license_types = LicenseConfigFactory.get_config("spotify")
    
    print(f"\nTestowanie algorytmów na sieci {ego_id} ({len(graph.nodes())} węzłów)")
    
    algorithms = [
        ("Greedy", GreedyAlgorithm()),
        ("Randomized", RandomizedAlgorithm(greedy_probability=0.7, seed=42)),
        ("ILP Optimal", ILPSolver()),
    ]
    
    results = []
    for name, algorithm in algorithms:
        try:
            solution = algorithm.solve(graph, license_types)
            results.append((name, solution.total_cost, len(solution.groups)))
            
            # Sprawdź poprawność rozwiązania
            assert solution.covered_nodes == set(graph.nodes())
            print(f"{name:15} | Cost: {solution.total_cost:7.2f} | Groups: {len(solution.groups):2d}")
            
        except Exception as e:
            print(f"{name:15} | ERROR: {e}")
    
    assert len(results) > 0, "Przynajmniej jeden algorytm powinien działać"


def test_facebook_features_and_circles():
    """Test ładowania cech węzłów i kręgów społecznych."""
    loader = RealWorldDataLoader()
    
    # Załaduj sieć z cechami
    graph = loader.load_facebook_ego_network("0")
    
    # Sprawdź czy niektóre węzły mają cechy
    nodes_with_features = [n for n in graph.nodes() if 'features' in graph.nodes[n]]
    print(f"Węzły z cechami: {len(nodes_with_features)}/{len(graph.nodes())}")
    
    # Sprawdź czy są informacje o kręgach
    if 'circles' in graph.graph:
        circles = graph.graph['circles']
        print(f"Liczba kręgów: {len(circles)}")
        
        # Sprawdź czy węzły mają przypisane kręgi
        nodes_with_circles = [n for n in graph.nodes() if graph.nodes[n].get('circles')]
        print(f"Węzły w kręgach: {len(nodes_with_circles)}/{len(graph.nodes())}")


def test_combined_facebook_network():
    """Test tworzenia połączonego grafu z wielu ego networks."""
    loader = RealWorldDataLoader()
    
    # Utwórz połączony graf z maksymalnie 3 sieci
    combined = loader.create_combined_facebook_network(max_networks=3)
    
    assert isinstance(combined, nx.Graph)
    assert len(combined.nodes()) > 0
    assert len(combined.edges()) > 0
    
    print(f"Połączony graf: {len(combined.nodes())} węzłów, {len(combined.edges())} krawędzi")
    
    # Sprawdź czy graf jest spójny lub ma niewiele komponentów
    components = nx.number_connected_components(combined)
    print(f"Liczba komponentów spójnych: {components}")


def test_real_world_vs_generated_comparison():
    """Porównanie algorytmów na danych rzeczywistych vs. wygenerowanych."""
    loader = RealWorldDataLoader()
    
    # Dane rzeczywiste - weź sieć średniej wielkości
    suitable = loader.get_suitable_networks_for_testing(min_nodes=60, max_nodes=200)
    if not suitable:
        # Fallback - weź sieć o rozmiarze około 150 węzłów
        stats = loader.get_facebook_network_stats()
        ego_id = min(stats.keys(), key=lambda x: abs(stats[x]['nodes'] - 150))
    else:
        ego_id = suitable[0]
    
    real_graph = loader.load_facebook_ego_network(ego_id)
    
    # Dane wygenerowane o podobnym rozmiarze
    from src.graphs import GraphGeneratorFactory
    generated_graph = GraphGeneratorFactory.small_world(
        len(real_graph.nodes()), 
        k=4, 
        p=0.3, 
        seed=42
    )
    
    license_types = LicenseConfigFactory.get_config("duolingo_super")
    algorithm = RandomizedAlgorithm(greedy_probability=0.8, seed=42)
    
    # Porównaj wyniki
    real_solution = algorithm.solve(real_graph, license_types)
    generated_solution = algorithm.solve(generated_graph, license_types)
    
    print("\nPorównanie graf rzeczywisty vs. wygenerowany:")
    print(f"Rzeczywisty:    {len(real_graph.nodes())} węzłów, koszt: {real_solution.total_cost:.2f}")
    print(f"Wygenerowany:   {len(generated_graph.nodes())} węzłów, koszt: {generated_solution.total_cost:.2f}")
    
    # Sprawdź czy oba rozwiązania są poprawne
    assert real_solution.covered_nodes == set(real_graph.nodes())
    assert generated_solution.covered_nodes == set(generated_graph.nodes())


if __name__ == "__main__":
    test_facebook_data_loader()
    test_facebook_network_stats()
    test_suitable_networks_for_testing()
    test_algorithms_on_real_data()
    test_facebook_features_and_circles()
    test_combined_facebook_network()
    test_real_world_vs_generated_comparison()
    print("\nWszystkie testy ładowarki danych rzeczywistych przeszły pomyślnie!")
