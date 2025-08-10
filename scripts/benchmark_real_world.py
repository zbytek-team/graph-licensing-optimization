from src.algorithms import (
    ILPSolver, GreedyAlgorithm, RandomizedAlgorithm, DominatingSetAlgorithm, NaiveAlgorithm
)
from src.graphs import RealWorldDataLoader, GraphGeneratorFactory
from src.core import LicenseConfigFactory
from datetime import datetime
import csv
import os
import time


def benchmark_real_world_networks():
    """Benchmark algorytmów na rzeczywistych sieciach Facebook."""
    
    loader = RealWorldDataLoader()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Przygotuj algorytmy
    algorithms = [
        ("Naive", lambda: NaiveAlgorithm()),
        ("Dominating Set", lambda: DominatingSetAlgorithm()),
        ("Randomized (90% greedy)", lambda: RandomizedAlgorithm(greedy_probability=0.9, seed=42)),
        ("Randomized (70% greedy)", lambda: RandomizedAlgorithm(greedy_probability=0.7, seed=42)),
        ("Randomized (50% greedy)", lambda: RandomizedAlgorithm(greedy_probability=0.5, seed=42)),
        ("Greedy", lambda: GreedyAlgorithm()),
        ("ILP Optimal", lambda: ILPSolver()),
    ]
    
    # Konfiguracje licencji do testowania
    license_configs = ["spotify", "duolingo_super", "roman_domination"]
    
    # Znajdź odpowiednie sieci do testowania
    suitable_networks = loader.get_suitable_networks_for_testing(min_nodes=20, max_nodes=200)
    print(f"Testowanie na {len(suitable_networks)} sieciach Facebook: {suitable_networks}")
    
    # Przygotuj plik wyników
    results_dir = "results/real_world"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/facebook_benchmark_{timestamp}.csv"
    
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = [
            'network_id', 'nodes', 'edges', 'density', 'avg_clustering',
            'license_config', 'algorithm', 'cost', 'groups', 'execution_time',
            'quality_ratio', 'is_optimal'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for network_id in suitable_networks:
            print(f"\nTesting network {network_id}...")
            
            # Załaduj sieć
            try:
                graph = loader.load_facebook_ego_network(network_id)
                
                # Statystyki sieci
                import networkx as nx
                nodes = len(graph.nodes())
                edges = len(graph.edges())
                density = nx.density(graph)
                avg_clustering = nx.average_clustering(graph)
                
                print(f"  {nodes} węzłów, {edges} krawędzi, density={density:.3f}")
                
                for license_config in license_configs:
                    license_types = LicenseConfigFactory.get_config(license_config)
                    optimal_cost = None
                    
                    for algo_name, algo_factory in algorithms:
                        # Skip naive dla dużych grafów
                        if algo_name == "Naive" and nodes > 10:
                            continue
                        
                        try:
                            algorithm = algo_factory()
                            start_time = time.time()
                            solution = algorithm.solve(graph, license_types)
                            execution_time = time.time() - start_time
                            
                            # Sprawdź czy to jest rozwiązanie optymalne
                            is_optimal = (algo_name == "ILP Optimal" or 
                                        (algo_name == "Naive" and nodes <= 10))
                            
                            if is_optimal:
                                optimal_cost = solution.total_cost
                            
                            # Oblicz quality ratio
                            quality_ratio = None
                            if optimal_cost and optimal_cost > 0:
                                quality_ratio = solution.total_cost / optimal_cost
                            
                            # Zapisz wyniki
                            writer.writerow({
                                'network_id': network_id,
                                'nodes': nodes,
                                'edges': edges,
                                'density': density,
                                'avg_clustering': avg_clustering,
                                'license_config': license_config,
                                'algorithm': algo_name,
                                'cost': solution.total_cost,
                                'groups': len(solution.groups),
                                'execution_time': execution_time,
                                'quality_ratio': quality_ratio,
                                'is_optimal': is_optimal
                            })
                            
                            print(f"    {algo_name:25} | {license_config:15} | "
                                  f"Cost: {solution.total_cost:7.2f} | "
                                  f"Time: {execution_time:6.3f}s")
                            
                        except Exception as e:
                            print(f"    {algo_name:25} | {license_config:15} | ERROR: {e}")
                            
            except Exception as e:
                print(f"  ERROR loading network {network_id}: {e}")
    
    print(f"\nWyniki zapisane w: {results_file}")


def compare_real_vs_generated():
    """Porównanie algorytmów na danych rzeczywistych vs. wygenerowanych."""
    
    loader = RealWorldDataLoader()
    
    # Wybierz reprezentatywne sieci rzeczywiste
    suitable_networks = loader.get_suitable_networks_for_testing(min_nodes=50, max_nodes=150)
    
    if len(suitable_networks) < 3:
        print("Za mało odpowiednich sieci do porównania")
        return
    
    # Weź 3 różne rozmiary sieci
    test_networks = suitable_networks[:3]
    
    algorithms = [
        ("Dominating Set", lambda: DominatingSetAlgorithm()),
        ("Randomized", lambda: RandomizedAlgorithm(greedy_probability=0.7, seed=42)),
        ("Greedy", lambda: GreedyAlgorithm()),
    ]
    
    license_types = LicenseConfigFactory.get_config("spotify")
    
    print("Porównanie: Sieci Rzeczywiste vs. Wygenerowane")
    print("=" * 60)
    
    for network_id in test_networks:
        real_graph = loader.load_facebook_ego_network(network_id)
        real_nodes = len(real_graph.nodes())
        
        # Utwórz graf wygenerowany o podobnym rozmiarze
        generated_graph = GraphGeneratorFactory.small_world(
            real_nodes, k=min(4, real_nodes-1), p=0.3, seed=42
        )
        
        print(f"\nSieć {network_id} ({real_nodes} węzłów):")
        print("-" * 40)
        
        for algo_name, algo_factory in algorithms:
            algorithm = algo_factory()
            
            # Test na sieci rzeczywistej
            real_solution = algorithm.solve(real_graph, license_types)
            
            # Test na sieci wygenerowanej
            generated_solution = algorithm.solve(generated_graph, license_types)
            
            cost_ratio = real_solution.total_cost / generated_solution.total_cost
            
            print(f"{algo_name:15} | Rzeczywista: {real_solution.total_cost:6.2f} | "
                  f"Wygenerowana: {generated_solution.total_cost:6.2f} | "
                  f"Ratio: {cost_ratio:.2f}")


def analyze_facebook_network_properties():
    """Analiza właściwości sieci Facebook."""
    
    loader = RealWorldDataLoader()
    stats = loader.get_facebook_network_stats()
    
    print("Analiza właściwości sieci Facebook")
    print("=" * 50)
    
    # Sortuj według rozmiaru
    sorted_networks = sorted(stats.items(), key=lambda x: x[1]['nodes'])
    
    print(f"{'Network ID':<10} {'Nodes':<6} {'Edges':<6} {'Density':<8} {'Clustering':<10} {'Components':<10}")
    print("-" * 70)
    
    for network_id, stat in sorted_networks:
        print(f"{network_id:<10} {stat['nodes']:<6} {stat['edges']:<6} "
              f"{stat['density']:<8.3f} {stat['avg_clustering']:<10.3f} {stat['components']:<10}")
    
    # Statystyki ogólne
    all_nodes = [stat['nodes'] for stat in stats.values()]
    all_densities = [stat['density'] for stat in stats.values()]
    all_clusterings = [stat['avg_clustering'] for stat in stats.values()]
    
    print("\nStatystyki ogólne:")
    print(f"Liczba sieci: {len(stats)}")
    print(f"Rozmiar węzłów - min: {min(all_nodes)}, max: {max(all_nodes)}, avg: {sum(all_nodes)/len(all_nodes):.1f}")
    print(f"Gęstość - min: {min(all_densities):.3f}, max: {max(all_densities):.3f}, avg: {sum(all_densities)/len(all_densities):.3f}")
    print(f"Clustering - min: {min(all_clusterings):.3f}, max: {max(all_clusterings):.3f}, avg: {sum(all_clusterings)/len(all_clusterings):.3f}")


if __name__ == "__main__":
    print("=== Benchmark algorytmów na rzeczywistych danych Facebook ===\n")
    
    # 1. Analiza właściwości sieci
    analyze_facebook_network_properties()
    
    print("\n" + "="*60 + "\n")
    
    # 2. Porównanie real vs generated
    compare_real_vs_generated()
    
    print("\n" + "="*60 + "\n")
    
    # 3. Pełny benchmark
    benchmark_real_world_networks()
    
    print("\n=== Benchmark zakończony ===")
